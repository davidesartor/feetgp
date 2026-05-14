import os
import pickle
import glob
import argparse
import numpy as np
from einops import rearrange
import plotly.graph_objects as go
from plotly.subplots import make_subplots


MARKERS = [
    "CAL1", "CUB", "LCAL", "LMAL", "MCAL", "MMAL",
    "MT1B", "MT1H", "MT2H", "MT5B", "MT5H", "NAV", "TOE",
]

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a",
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


def run_name_from_dir(run_dir: str, results_dir: str) -> str:
    rel = os.path.relpath(run_dir, results_dir)
    return rel.replace(os.sep, "_")


def plot_run(run_dir: str, results_dir: str):
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
        group_colors = [COLORS[i % len(COLORS)] for i in range(len(MARKERS))] * 2
    elif feet == "left_only":
        labels = [f"L {m}" for m in MARKERS]
        group_colors = [COLORS[i % len(COLORS)] for i in range(len(MARKERS))]
    else:
        labels = [f"R {m}" for m in MARKERS]
        group_colors = [COLORS[i % len(COLORS)] for i in range(len(MARKERS))]

    r2_per_marker = rearrange(r2, "f (m g) -> f m g", g=group_size).mean(-1)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Group norm per marker", "R² per marker"))

    for j in range(n_groups):
        fig.add_trace(go.Scatter(
            x=lambdas,
            y=group_norms[:, j],
            mode="lines",
            name=labels[j],
            line=dict(color=group_colors[j]),
            legendgroup=labels[j],
            showlegend=True,
        ), row=1, col=1)

    for j in range(r2_per_marker.shape[1]):
        fig.add_trace(go.Scatter(
            x=lambdas,
            y=r2_per_marker[:, j],
            mode="lines",
            name=labels[j],
            line=dict(color=group_colors[j]),
            legendgroup=labels[j],
            showlegend=False,  # legend already shown in left plot
        ), row=1, col=2)

    fig.update_xaxes(type="log", title_text="λ")
    fig.update_yaxes(type="log", title_text="group norm", row=1, col=1)
    fig.update_yaxes(range=[-0.05, 1.05], title_text="R²", row=1, col=2)
    fig.update_layout(
        width=1200,
        height=500,
        legend=dict(font=dict(size=10), tracegroupgap=0),
    )

    name = run_name_from_dir(run_dir, results_dir)
    out = os.path.join(results_dir, f"{name}.html")
    fig.write_html(out)
    print(f"  Saved {out}")


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
        plot_run(run_dir, args.results_dir)