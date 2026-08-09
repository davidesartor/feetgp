import os
import json
import pickle
import glob
import argparse
import numpy as np
from einops import rearrange
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

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

# same hue order, re-stepped for a dark surface
COLORS_DARK = [
    "#5aa9e6",
    "#ffa14f",
    "#5fd06a",
    "#ff6b6b",
    "#b18ce0",
    "#c98f77",
    "#f492d1",
    "#b0b0b0",
    "#dade4c",
    "#45d3e8",
    "#9fd0f0",
    "#ffd08a",
    "#b3e89b",
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
        results.append(
            {
                "l1_penalty": r["l1_penalty"],
                "group_norms": np.asarray(r["group_norms"]),
                "r2": r2,
                "r2_train": r2_train,
            }
        )
    results.sort(key=lambda r: r["l1_penalty"])
    return {
        "lambdas": np.array([r["l1_penalty"] for r in results]),
        "group_norms": np.stack([r["group_norms"] for r in results]),
        "r2": np.stack([r["r2"] for r in results]),
        "r2_train": np.stack([r["r2_train"] for r in results]),
    }


def run_name_from_dir(run_dir: str, results_dir: str) -> str:
    rel = os.path.relpath(run_dir, results_dir)
    return rel.replace(os.sep, "_")


def plot_run(run_dir: str, results_dir: str):
    meta_path = os.path.join(run_dir, "meta.json")
    if not os.path.exists(meta_path):
        print("  No meta.json, skipping.")
        return
    with open(meta_path) as f:
        meta = json.load(f)
    group_size = meta["group_size"]
    group_labels = meta["group_labels"]
    target = meta["args"]["target"]
    force_labels = meta["y_columns"]

    data = load_run(run_dir)
    if data is None:
        print(f"  No results found, skipping.")
        return

    lambdas = data["lambdas"]
    group_norms = data["group_norms"]
    r2 = data["r2"]
    r2_train = data["r2_train"]
    n_groups = group_norms.shape[1]

    group_colors = [COLORS_DARK[j % len(COLORS_DARK)] for j in range(n_groups)]
    group_colors_light = [COLORS[j % len(COLORS)] for j in range(n_groups)]

    if target == "forces":
        r2_labels = force_labels
        r2_colors = [
            COLORS_DARK[j % len(COLORS_DARK)] for j in range(len(force_labels))
        ]
        r2_colors_light = [COLORS[j % len(COLORS)] for j in range(len(force_labels))]
        r2_summary = r2
        r2_train_summary = r2_train
    else:
        r2_labels = group_labels
        r2_colors = group_colors
        r2_colors_light = group_colors_light
        r2_summary = rearrange(r2, "f (m g) -> f m g", g=group_size).mean(-1)
        r2_train_summary = rearrange(r2_train, "f (m g) -> f m g", g=group_size).mean(
            -1
        )

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.5, 0.5],
        horizontal_spacing=0.16,
        subplot_titles=("Group norm per marker", "R² per output"),
    )

    for j in range(n_groups):
        fig.add_trace(
            go.Scatter(
                x=lambdas,
                y=group_norms[:, j],
                mode="lines",
                name=group_labels[j],
                line=dict(color=group_colors[j], width=2),
                legend="legend",
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
                mode="lines",
                name=r2_labels[j],
                line=dict(color=r2_colors[j], width=2),
                legend="legend2",
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
                line=dict(color=r2_colors[j], dash="dot", width=2),
                legend="legend2",
                legendgroup=r2_labels[j],
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        train_trace_indices.append(len(fig.data) - 1)

    dark_by_trace = group_colors + [
        c for c in r2_colors[: r2_summary.shape[1]] for _ in range(2)
    ]
    light_by_trace = group_colors_light + [
        c for c in r2_colors_light[: r2_summary.shape[1]] for _ in range(2)
    ]
    all_traces = list(range(len(fig.data)))

    fig.update_xaxes(type="log", title_text="λ", title_standoff=8)
    fig.update_yaxes(
        type="log",
        autorange=True,
        title_text="group norm",
        title_standoff=8,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        range=[-0.05, 1.05], title_text="R²", title_standoff=8, row=1, col=2
    )
    legend_style = dict(font=dict(size=10), tracegroupgap=0, yanchor="top", y=1.0)
    fig.update_layout(
        template="plotly_dark",
        autosize=True,
        margin=dict(l=70, r=150, t=110, b=70),
        hovermode="x unified",
        legend=dict(**legend_style, x=0.415, xanchor="left", title_text="marker"),
        legend2=dict(**legend_style, x=1.02, xanchor="left", title_text="output"),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.0,
                y=1.16,
                xanchor="left",
                yanchor="bottom",
                bgcolor="#eeeeee",
                bordercolor="#999999",
                font=dict(color="#111111"),
                buttons=[
                    dict(
                        label="Toggle train",
                        method="restyle",
                        args=[{"visible": "legendonly"}, train_trace_indices],
                        args2=[{"visible": True}, train_trace_indices],
                    ),
                    dict(
                        label="Light",
                        method="update",
                        args=[
                            {"line.color": light_by_trace},
                            {
                                "template": pio.templates[
                                    "plotly_white"
                                ].to_plotly_json()
                            },
                            all_traces,
                        ],
                        args2=[
                            {"line.color": dark_by_trace},
                            {"template": pio.templates["plotly_dark"].to_plotly_json()},
                            all_traces,
                        ],
                    ),
                ],
            ),
        ],
    )

    name = run_name_from_dir(run_dir, results_dir)
    out = os.path.join(results_dir, f"{name}.html")
    fig.write_html(
        out,
        default_width="100%",
        default_height="100vh",
        config={"responsive": True},
    )
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
