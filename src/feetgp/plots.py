import os
import json
import pickle
import glob
import argparse
import numpy as np
from einops import rearrange
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from feetgp.gp import CERTIFICATE_TOLERANCE

MARKERS = [
    "CAL1",
    "CUB",
    "LCAL",
    "LMAL",
    "MCAL",
    "MMAL",
    "MT1B",
    "MT1H",
    "MT2H",
    "MT5B",
    "MT5H",
    "NAV",
    "TOE",
]

COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
]


def load_run(run_dir: str) -> dict | None:
    files = glob.glob(os.path.join(run_dir, "lambda=*.pkl"))
    if not files:
        return None
    results = []
    for f in files:
        with open(f, "rb") as fp:
            r = pickle.load(fp)
        r2 = np.asarray(r["r2_test"])
        r2_train = (
            np.asarray(r["r2_train"]) if "r2_train" in r else np.full_like(r2, np.nan)
        )
        info = r.get("info", {})
        certificate = info.get("certificate")
        if certificate is not None:
            trusted = float(certificate["max_live_kkt"]) <= CERTIFICATE_TOLERANCE
        else:
            trusted = info.get("converged", True)
        results.append(
            {
                "l1_penalty": r["l1_penalty"],
                "group_norms": np.asarray(r["group_norms"]),
                "r2": r2,
                "r2_train": r2_train,
                "trusted": trusted,
            }
        )
    results.sort(key=lambda r: r["l1_penalty"])
    return {
        "lambdas": np.array([r["l1_penalty"] for r in results]),
        "group_norms": np.stack([r["group_norms"] for r in results]),
        "r2": np.stack([r["r2"] for r in results]),
        "r2_train": np.stack([r["r2_train"] for r in results]),
        "trusted": np.array([r["trusted"] for r in results]),
    }


def run_name_from_dir(run_dir: str, results_dir: str) -> str:
    rel = os.path.relpath(run_dir, results_dir)
    return rel.replace(os.sep, "_")


FORCE_LABELS = ["Fx", "Fy", "Fz"]


def labels_from_run_dir(run_dir: str) -> tuple[int, list[str], str]:
    feet = "both"
    ungrouped = False
    relative_marker = None
    target = "markers"
    for part in run_dir.split(os.sep):
        if part.startswith("feet="):
            feet = part.split("=", 1)[1].replace("_ungrouped", "")
            ungrouped = "ungrouped" in part
        if part.startswith("target="):
            target = part.split("=", 1)[1]
        if "relative=" in part:
            relative_marker = part.split("relative=")[1]
    group_size = 3 if ungrouped or feet != "both" else 6

    active_markers = [m for m in MARKERS if m != relative_marker]
    if feet != "both":
        side = "L" if feet == "left_only" else "R"
        group_labels = [f"{side} {m}" for m in active_markers]
    elif ungrouped:
        group_labels = [f"{side} {m}" for m in active_markers for side in ("L", "R")]
    else:
        group_labels = active_markers
    return group_size, group_labels, target


def plot_run(run_dir: str, results_dir: str):
    meta_path = os.path.join(run_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        group_size = meta["group_size"]
        group_labels = meta["group_labels"]
        target = meta["args"]["target"]
        force_labels = meta["y_columns"]
    else:
        group_size, group_labels, target = labels_from_run_dir(run_dir)
        force_labels = FORCE_LABELS

    data = load_run(run_dir)
    if data is None:
        print(f"  No results found, skipping.")
        return

    lambdas = data["lambdas"]
    group_norms = data["group_norms"]
    r2 = data["r2"]
    r2_train = data["r2_train"]
    n_groups = group_norms.shape[1]

    untrusted = ~data["trusted"]
    if untrusted.any():
        print(f"  {int(untrusted.sum())}/{len(lambdas)} lambdas failed certification")
    marker_sizes = np.where(untrusted, 7, 0)
    marker_style = dict(size=marker_sizes, symbol="x-thin", line=dict(width=1.5))

    group_colors = [COLORS[j % len(COLORS)] for j in range(n_groups)]

    if target == "forces":
        r2_labels = force_labels
        r2_colors = [COLORS[j % len(COLORS)] for j in range(len(force_labels))]
        r2_summary = r2
        r2_train_summary = r2_train
    else:
        r2_labels = group_labels
        r2_colors = group_colors
        r2_summary = rearrange(r2, "f (m g) -> f m g", g=group_size).mean(-1)
        r2_train_summary = rearrange(r2_train, "f (m g) -> f m g", g=group_size).mean(
            -1
        )

    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Group norm per marker", "R² per output")
    )

    for j in range(n_groups):
        fig.add_trace(
            go.Scatter(
                x=lambdas,
                y=group_norms[:, j],
                mode="lines+markers",
                marker=dict(**marker_style, color=group_colors[j]),
                name=group_labels[j],
                line=dict(color=group_colors[j]),
                legendgroup=group_labels[j],
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    train_trace_indices = []
    for j in range(r2_summary.shape[1]):
        fig.add_trace(
            go.Scatter(
                x=lambdas,
                y=r2_summary[:, j],
                mode="lines+markers",
                marker=dict(**marker_style, color=r2_colors[j]),
                name=r2_labels[j],
                line=dict(color=r2_colors[j]),
                legendgroup=r2_labels[j],
                showlegend=True,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=lambdas,
                y=r2_train_summary[:, j],
                mode="lines",
                name=f"{r2_labels[j]} (train)",
                line=dict(color=r2_colors[j], dash="dot"),
                legendgroup=r2_labels[j],
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        train_trace_indices.append(len(fig.data) - 1)

    fig.update_xaxes(type="log", title_text="λ")
    fig.update_yaxes(type="log", autorange=True, title_text="group norm", row=1, col=1)
    fig.update_yaxes(range=[-0.05, 1.05], title_text="R²", row=1, col=2)
    fig.update_layout(
        width=1200,
        height=500,
        legend=dict(font=dict(size=10), tracegroupgap=0),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=1.0,
                y=1.12,
                xanchor="right",
                yanchor="bottom",
                buttons=[
                    dict(
                        label="Toggle train",
                        method="restyle",
                        args=[{"visible": "legendonly"}, train_trace_indices],
                        args2=[{"visible": True}, train_trace_indices],
                    ),
                ],
            )
        ],
    )

    name = run_name_from_dir(run_dir, results_dir)
    out = os.path.join(results_dir, f"{name}.html")
    fig.write_html(out)
    print(f"  Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    run_dirs = sorted(
        set(
            os.path.dirname(f)
            for f in glob.glob(os.path.join(args.results_dir, "*/*/*/*/lambda=*.pkl"))
        )
    )
    if not run_dirs:
        print(f"No results found in {args.results_dir}")
    for run_dir in run_dirs:
        print(f"Plotting {run_dir} ...")
        plot_run(run_dir, args.results_dir)
