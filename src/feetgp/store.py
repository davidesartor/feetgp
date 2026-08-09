"""Per-run storage: a jsonl path summary plus one npz of arrays per fit."""

from typing import TYPE_CHECKING, Any, NamedTuple

import json
import os
import numpy as np

# reading a path summary should not pull jax in, so the state import stays lazy
if TYPE_CHECKING:
    from feetgp.glasso_admm import ADMMState

PATH_FILE = "path.jsonl"
STATE_DIR = "states"


def state_to_arrays(state: "ADMMState") -> dict[str, np.ndarray]:
    """Flatten an ADMMState, its aux tuple spread over numbered keys."""
    arrays = {
        f"admm.{field}": np.asarray(getattr(state, field))
        for field in state._fields
        if field != "aux"
    }
    arrays.update({f"admm.aux.{i}": np.asarray(a) for i, a in enumerate(state.aux)})
    return arrays


def state_from_arrays(arrays: dict[str, np.ndarray]) -> "ADMMState":
    from feetgp.glasso_admm import ADMMState

    aux_keys = sorted(k for k in arrays if k.startswith("admm.aux."))
    fields = {
        field: arrays[f"admm.{field}"]
        for field in ADMMState._fields
        if f"admm.{field}" in arrays
    }
    return ADMMState(**fields, aux=tuple(arrays[k] for k in aux_keys))


def model_to_arrays(model: NamedTuple) -> dict[str, np.ndarray]:
    """Fitted arrays only: tags live in meta.json, training data is reloadable."""
    return {
        f"model.{field}": np.asarray(value)
        for field, value in zip(model._fields, model)
        if hasattr(value, "shape") and field not in ("x_train", "y_train")
    }


class RunStore:
    """Append-only record of a penalty path, resumable by matching lambda."""

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.state_dir = os.path.join(save_dir, STATE_DIR)
        os.makedirs(self.state_dir, exist_ok=True)
        self.path_file = os.path.join(save_dir, PATH_FILE)
        self.rows = self.read_rows(save_dir)

    @staticmethod
    def read_rows(save_dir: str) -> list[dict[str, Any]]:
        path_file = os.path.join(save_dir, PATH_FILE)
        if not os.path.exists(path_file):
            return []
        with open(path_file) as f:
            rows = [json.loads(line) for line in f if line.strip()]

        # a rerun of the same lambda supersedes the earlier row
        latest: dict[float, dict[str, Any]] = {}
        for row in rows:
            latest[row["l1_penalty"]] = row
        return sorted(latest.values(), key=lambda row: row["l1_penalty"])

    def find(self, l1_penalty: float) -> dict[str, Any] | None:
        for row in self.rows:
            if np.isclose(row["l1_penalty"], l1_penalty, rtol=1e-6, atol=0.0):
                return row
        return None

    def state_path(self, index: int) -> str:
        return os.path.join(self.state_dir, f"{index:05d}.npz")

    def append(self, row: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict:
        index = max((r["index"] for r in self.rows), default=-1) + 1
        row = dict(row, index=index)
        np.savez_compressed(self.state_path(index), **arrays)
        with open(self.path_file, "a") as f:
            f.write(json.dumps(row) + "\n")
        self.rows = sorted(
            [r for r in self.rows if r["l1_penalty"] != row["l1_penalty"]] + [row],
            key=lambda r: r["l1_penalty"],
        )
        return row

    def load_state(self, index: int) -> "ADMMState | None":
        with np.load(self.state_path(index)) as data:
            arrays = {k: data[k] for k in data.files}
        if not any(k.startswith("admm.") for k in arrays):
            return None
        return state_from_arrays(arrays)
