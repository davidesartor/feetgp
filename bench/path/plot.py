"""Plot per-marker group norms along the regularization path, one panel per path."""

import argparse
import os

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, "results.jsonl")

SERIES_COLORS = [
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
    "#5ab4d6",
    "#b07cc6",
    "#8a9a5b",
    "#cf7a30",
    "#7f8c9a",
]

SURFACE, GRID, AXIS_LINE = "#141412", "#2e2d29", "#3d3c36"
TEXT_PRIMARY, TEXT_SECONDARY = "#f4f3ef", "#bab8b0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default=RESULTS_PATH)
    parser.add_argument("--out", type=str, default=os.path.join(HERE, "plot.html"))
    parser.add_argument("--n_columns", type=int, default=3)
    return parser.parse_args()


def long_norms(results: pd.DataFrame) -> pd.DataFrame:
    """One row per (panel, lambda, group) with the group norm."""
    frames = []
    for _, row in results.iterrows():
        panel = "cold" if row.kind == "cold" else f"path {int(row.path)}"
        frames.append(
            pd.DataFrame(
                dict(
                    panel=panel,
                    l1_penalty=row.l1_penalty,
                    group=row.groups,
                    norm=row.norms,
                )
            )
        )
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    args = parse_args()
    results = pd.read_json(args.results, lines=True)
    results = results[results.norms.notna()]

    norms = long_norms(results)
    panels = sorted(norms.panel.unique(), key=lambda p: (p == "cold", p))
    groups = list(dict.fromkeys(norms.group))
    colors = {g: SERIES_COLORS[i % len(SERIES_COLORS)] for i, g in enumerate(groups)}

    n_columns = min(args.n_columns, len(panels))
    n_rows = -(-len(panels) // n_columns)
    figure = make_subplots(
        rows=n_rows,
        cols=n_columns,
        subplot_titles=panels,
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.04,
        vertical_spacing=0.08,
    )

    # one legend entry per marker, shared across every panel
    for index, panel in enumerate(panels):
        row_index, col_index = divmod(index, n_columns)
        for group, trace in norms[norms.panel == panel].groupby("group", sort=False):
            trace = trace.sort_values("l1_penalty", ascending=False)
            figure.add_trace(
                go.Scatter(
                    x=trace.l1_penalty,
                    y=trace.norm,
                    name=group,
                    legendgroup=group,
                    showlegend=index == 0,
                    mode="lines+markers",
                    line=dict(color=colors[group], width=1.8),
                    marker=dict(size=4, color=colors[group]),
                    hovertemplate=(
                        f"<b>{group}</b> · {panel}<br>"
                        "lambda=%{x:.4g}<br>norm=%{y:.4g}<extra></extra>"
                    ),
                ),
                row=row_index + 1,
                col=col_index + 1,
            )

    figure.update_xaxes(
        title="lambda",
        type="log",
        autorange="reversed",
        gridcolor=GRID,
        zeroline=False,
        linecolor=AXIS_LINE,
    )
    figure.update_yaxes(
        title="group norm", gridcolor=GRID, zeroline=False, linecolor=AXIS_LINE
    )
    figure.update_layout(
        title=dict(
            text="regularization path: group norms per marker",
            font=dict(color=TEXT_PRIMARY, size=17),
            x=0,
            xref="paper",
        ),
        font=dict(color=TEXT_SECONDARY, size=12),
        legend=dict(yanchor="top", y=1.0, xanchor="left", x=1.02),
        hovermode="closest",
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        margin=dict(l=70, r=220, t=70, b=60),
        autosize=True,
    )

    figure.write_html(
        args.out,
        include_plotlyjs="cdn",
        default_width="100%",
        default_height="100vh",
        config={"responsive": True},
    )
    print(f"wrote {args.out}")
