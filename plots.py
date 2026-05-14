import os
import pickle
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange


MARKERS = [
    "CAL1", "CUB", "LCAL", "LMAL", "MCAL", "MMAL",
    "MT1B", "MT1H", "MT2H", "MT5B", "MT5H", "NAV", "TOE",
]


def load_run(run_dir: str) -> dict | None:
    files = glob.glob(os.path.join(run_dir, "lambda=*.pkl"))
    if not files:
        return None
    results = []
    for f in files:
        with open(f, "rb") as fp:
            r = pickle.load(fp)
        results.append({
            "l1_penalty": r["l1_penalty"],
            "group_norms": np.asarray(r["group_norms"]),
            "r2":          np.asarray(r["r2"]),
        })
    results.sort(key=lambda r: r["l1_penalty"])
    return {
        "lambdas":     np.array([r["l1_penalty"] for r in results]),
        "group_norms": np.stack([r["group_norms"] for r in results]),
        "r2":          np.stack([r["r2"] for r in results]),
    }


def plot_run(run_dir: str):
    # parse feet and ungrouped from path
    feet = "both"
    ungrouped = False
    for part in run_dir.split(os.sep):
        if part.startswith("feet="):
            feet = part.split("=", 1)[1].replace("_ungrouped", "")
            ungrouped = "ungrouped" in part
    group_size = 3 if ungrouped or feet != "both" else 6

    data = load_run(run_dir)
    if data is None:
        print(f"  No results found, skipping.")
        return

    lambdas     = data["lambdas"]
    group_norms = data["group_norms"]
    r2          = data["r2"]
    n_groups    = group_norms.shape[1]

    if feet == "both":
        labels = [f"L {m}" for m in MARKERS] + [f"R {m}" for m in MARKERS]
    elif feet == "left_only":
        labels = [f"L {m}" for m in MARKERS]
    else:
        labels = [f"R {m}" for m in MARKERS]

    colors_per_marker = plt.cm.tab20(np.linspace(0, 1, len(MARKERS)))
    colors = np.tile(colors_per_marker, (n_groups // len(MARKERS) + 1, 1))[:n_groups]
    r2_per_marker = rearrange(r2, "f (m g) -> f m g", g=group_size).mean(-1)

    fig, (ax_norm, ax_r2) = plt.subplots(1, 2, figsize=(12, 5))

    for j in range(n_groups):
        ax_norm.plot(lambdas, group_norms[:, j], color=colors[j], label=labels[j])
    ax_norm.set_xscale("log")
    ax_norm.set_yscale("log")
    ax_norm.set_xlabel(r"$\lambda$")
    ax_norm.set_ylabel("group norm")
    ax_norm.grid(True, which="both", alpha=0.3)
    ax_norm.legend(fontsize=7, ncol=2)

    for j in range(r2_per_marker.shape[1]):
        ax_r2.plot(lambdas, r2_per_marker[:, j], color=colors_per_marker[j % len(MARKERS)], label=labels[j])
    ax_r2.set_xscale("log")
    ax_r2.set_ylim(-0.05, 1.05)
    ax_r2.set_xlabel(r"$\lambda$")
    ax_r2.set_ylabel(r"$R^2$")
    ax_r2.grid(True, which="both", alpha=0.3)
    ax_r2.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "summary.pdf"))
    plt.close(fig)
    print(f"  Saved {os.path.join(run_dir, 'summary.pdf')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    run_dirs = sorted(set(
        os.path.dirname(f)
        for f in glob.glob(os.path.join(args.results_dir, "*/*/*/*/lambda=*.pkl"))
    ))
    if not run_dirs:
        print(f"No results found in {args.results_dir}")
    for run_dir in run_dirs:
        print(f"Plotting {run_dir} ...")
        plot_run(run_dir)