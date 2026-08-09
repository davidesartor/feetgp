import glob
import json
import os
import sys

import numpy as np

from feetgp.store import PATH_FILE, RunStore


def summarize(run_dir):
    rows = RunStore.read_rows(run_dir)
    if not rows:
        return None
    meta_path = os.path.join(run_dir, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    r2 = [float(np.median(row["r2_test"])) for row in rows]
    kkt = [row["max_kkt"] for row in rows]
    first_death = next(
        (row["l1_penalty"] for row in rows if row["n_active"] < rows[0]["n_active"]),
        None,
    )
    usable = [(row, median) for row, median in zip(rows, r2) if median > 0.5]
    return dict(
        run=run_dir,
        n_lambda=len(rows),
        groups=rows[0]["n_groups"],
        lambda_max=rows[-1]["l1_penalty"],
        active_first=rows[0]["n_active"],
        active_last=rows[-1]["n_active"],
        first_death=first_death,
        r2_at_zero=r2[0],
        r2_min_usable=min((median for _, median in usable), default=float("nan")),
        smallest_support_usable=min(
            (row["n_active"] for row, _ in usable), default=None
        ),
        unconverged=sum(1 for row in rows if not row["converged"]),
        kkt_median=float(np.median(kkt)),
        kkt_max=float(np.max(kkt)),
        trajectory=[(round(row["l1_penalty"], 3), row["n_active"]) for row in rows],
        git=meta.get("git_revision"),
        dirty=meta.get("git_dirty"),
    )


if __name__ == "__main__":
    out = []
    for root in sys.argv[1:]:
        for path in sorted(glob.glob(f"{root}/model=*/target=*/feet=*/*/{PATH_FILE}")):
            summary = summarize(os.path.dirname(path))
            if summary:
                out.append(summary)
                print(json.dumps(summary), flush=True)
    print(f"# {len(out)} runs summarized", file=sys.stderr)
