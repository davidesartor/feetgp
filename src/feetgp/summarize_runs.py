
import glob
import json
import os
import pickle
import re
import sys

import numpy as np

from feetgp.gp import CERTIFICATE_TOLERANCE


def read_run(run_dir):
    paths = sorted(
        glob.glob(run_dir + "/lambda=*.pkl"),
        key=lambda p: float(re.search(r"lambda=([0-9.e+-]+)\.pkl", p).group(1)),
    )
    rows = []
    for path in paths:
        l1 = float(re.search(r"lambda=([0-9.e+-]+)\.pkl", path).group(1))
        try:
            result = pickle.load(open(path, "rb"))
        except Exception as error:
            rows.append(dict(l1=l1, error=type(error).__name__))
            continue
        r2 = result.get("r2_test", result.get("r2"))
        info = result.get("info", {})
        certificate = info.get("certificate") if info else None
        rows.append(
            dict(
                l1=l1,
                active=int((np.asarray(result["group_norms"]) > 0).sum()),
                groups=int(np.asarray(result["group_norms"]).size),
                r2=float(np.median(r2)) if r2 is not None else float("nan"),
                converged=bool(info.get("converged", False)) if info else None,
                iterations=int(info.get("iterations", 0)) if info else None,
                max_live_kkt=(
                    float(certificate["max_live_kkt"]) if certificate else None
                ),
                winner=info.get("winner") if info else None,
            )
        )
    return rows


def summarize(run_dir):
    rows = [row for row in read_run(run_dir) if "error" not in row]
    if not rows:
        return None
    meta_path = os.path.join(run_dir, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    have_info = [row for row in rows if row["converged"] is not None]
    kkt = [row["max_live_kkt"] for row in rows if row["max_live_kkt"] is not None]
    winners = [row["winner"] for row in rows if row["winner"] is not None]
    first_death = next(
        (row["l1"] for row in rows if row["active"] < rows[0]["active"]), None
    )
    usable = [row for row in rows if row["r2"] > 0.5]
    return dict(
        run=run_dir,
        n_lambda=len(rows),
        groups=rows[0]["groups"],
        lambda_max=rows[-1]["l1"],
        active_first=rows[0]["active"],
        active_last=rows[-1]["active"],
        first_death=first_death,
        r2_at_zero=rows[0]["r2"],
        r2_min_usable=min((row["r2"] for row in usable), default=float("nan")),
        smallest_support_usable=min((row["active"] for row in usable), default=None),
        unconverged=sum(1 for row in have_info if not row["converged"]),
        n_with_info=len(have_info),
        kkt_median=float(np.median(kkt)) if kkt else None,
        kkt_max=float(np.max(kkt)) if kkt else None,
        kkt_failures=sum(1 for value in kkt if value > CERTIFICATE_TOLERANCE),
        n_certified=len(kkt),
        winners={label: winners.count(label) for label in sorted(set(winners))},
        trajectory=[(round(row["l1"], 3), row["active"]) for row in rows],
        git=meta.get("git_revision"),
        dirty=meta.get("git_dirty"),
    )


if __name__ == "__main__":
    out = []
    for root in sys.argv[1:]:
        for run_dir in sorted(glob.glob(root + "/model=*/target=*/feet=*/*")):
            summary = summarize(run_dir)
            if summary:
                out.append(summary)
                print(json.dumps(summary), flush=True)
    print(f"# {len(out)} runs summarized", file=sys.stderr)
