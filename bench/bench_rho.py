
import argparse
import pickle
import re
from pathlib import Path

import numpy as np

from feetgp.gp import admm_state_from_pickle

parser = argparse.ArgumentParser()
parser.add_argument("--run_dir", type=str, required=True)
args = parser.parse_args()


def group_norms(a: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a, axis=-1)


print(
    f"{'lambda':>12} {'rho':>9} {'l1/rho':>9} {'max|z|_g':>9} {'max|x+u|_g':>11} {'thresh':>8} {'nz':>4}"
)
for path in sorted(
    Path(args.run_dir).glob("lambda=*.pkl"),
    key=lambda p: float(re.search(r"lambda=([0-9.e+-]+)\.pkl", p.name).group(1)),
):
    l1 = float(re.search(r"lambda=([0-9.e+-]+)\.pkl", path.name).group(1))
    with open(path, "rb") as f:
        state = admm_state_from_pickle(pickle.load(f))
    rho = float(state.rho)
    xu = group_norms(np.asarray(state.x) + np.asarray(state.u))
    zn = group_norms(np.asarray(state.z))
    thresh = l1 / (rho * xu.max()) if xu.max() > 0 else np.inf
    print(
        f"{l1:12.4f} {rho:9.3g} {l1 / rho:9.3g} {zn.max():9.3f} {xu.max():11.3f} "
        f"{thresh:8.3f} {int((zn > 1e-8).sum()):4d}"
    )
