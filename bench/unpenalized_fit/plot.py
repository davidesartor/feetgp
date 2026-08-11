"""Plot unpenalized-fit wall time against training set size, one line per device."""

import argparse
import os
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go

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
]
# chip owns the hue, so a series keeps its color when other series come and go
CHIP_COLORS = {
    "A100-SXM4-80GB": SERIES_COLORS[0],
    "RTX 2080 Ti": SERIES_COLORS[1],
    "GTX 1080 Ti": SERIES_COLORS[2],
}
DTYPE_DASHES = {"float64": "solid", "float32": "dash"}
PROFILE_SYMBOLS = {"rbf": "circle", "matern52": "diamond"}

SURFACE, GRID, AXIS_LINE = "#141412", "#2e2d29", "#3d3c36"
TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED = "#f4f3ef", "#bab8b0", "#807e76"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default=RESULTS_PATH)
    parser.add_argument("--out", type=str, default=os.path.join(HERE, "plot.html"))
    return parser.parse_args()


def short_chip(chip: str) -> str:
    return re.sub(r"NVIDIA GeForce |NVIDIA |\(R\)|\(TM\)| CPU @.*", "", chip).strip()


def series_label(chip: str, dtype: str, profile: str) -> str:
    return f"{short_chip(chip)} fp{dtype[-2:]} {profile}"


def chip_color(chip: str) -> str:
    """Hues left over from the fixed map go to whatever chip shows up next."""
    short = short_chip(chip)
    if short not in CHIP_COLORS:
        spare = [c for c in SERIES_COLORS if c not in CHIP_COLORS.values()]
        CHIP_COLORS[short] = spare[0] if spare else SERIES_COLORS[-1]
    return CHIP_COLORS[short]


def local_exponent(group: pd.DataFrame) -> float:
    """Cost exponent between the two largest measured sizes."""
    tail = group.iloc[-2:]
    return float(np.diff(np.log(tail.fit_s))[0] / np.diff(np.log(tail.n))[0])


if __name__ == "__main__":
    args = parse_args()
    results = pd.read_json(args.results, lines=True)

    # repeat 0 carries the compile, so the fastest observation is the honest one
    ok = results[results.status == "ok"]
    ok = ok.loc[ok.groupby(["chip", "dtype", "profile", "n"]).fit_s.idxmin()]
    oom = results[results.status != "ok"]

    figure = go.Figure()
    grouped = ok.groupby(["chip", "dtype", "profile"])
    for (chip, dtype, profile), group in grouped:
        group = group.sort_values("n")
        color = chip_color(chip)
        label = series_label(chip, dtype, profile)
        figure.add_trace(
            go.Scatter(
                x=group.n,
                y=group.fit_s,
                name=label,
                mode="lines+markers",
                line=dict(color=color, width=2, dash=DTYPE_DASHES[dtype]),
                marker=dict(
                    size=9,
                    color=color,
                    symbol=PROFILE_SYMBOLS[profile],
                    line=dict(color=SURFACE, width=2),
                ),
                customdata=np.stack([group.admm_iters], axis=-1),
                hovertemplate=(
                    f"<b>{label}</b><br>n=%{{x}}<br>"
                    "%{y:.2f}s full fit<br>"
                    "%{customdata[0]} ADMM iterations"
                    "<extra></extra>"
                ),
            )
        )
        if len(group) > 1:
            print(f"{label}: local exponent {local_exponent(group):.2f}")

    # cubic reference anchored at the cheapest measured point
    if ok.n.nunique() > 1:
        anchor = ok.loc[ok.n.idxmin()]
        reference_n = np.array([anchor.n, ok.n.max()])
        figure.add_trace(
            go.Scatter(
                x=reference_n,
                y=anchor.fit_s * (reference_n / anchor.n) ** 3,
                name="n³ reference",
                mode="lines",
                line=dict(color=TEXT_MUTED, width=1.5, dash="longdash"),
                hoverinfo="skip",
            )
        )

    # shapes take data coordinates on a log axis, annotations take the exponent
    for n in sorted(oom.n.unique()):
        figure.add_vline(x=n, line=dict(color=TEXT_MUTED, width=1, dash="dot"))
        figure.add_annotation(
            x=np.log10(n),
            y=1.0,
            yref="paper",
            text=f"OOM at n={int(n)}",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(color=TEXT_SECONDARY, size=11),
        )

    figure.update_layout(
        title=dict(
            text="Unpenalized GP fit: wall time",
            font=dict(color=TEXT_PRIMARY, size=17),
            x=0,
            xref="paper",
        ),
        xaxis=dict(
            title="training points n",
            type="log",
            dtick=1,
            gridcolor=GRID,
            zeroline=False,
            linecolor=AXIS_LINE,
        ),
        yaxis=dict(
            title="fit seconds",
            type="log",
            dtick=1,
            gridcolor=GRID,
            zeroline=False,
            linecolor=AXIS_LINE,
        ),
        font=dict(color=TEXT_SECONDARY, size=12),
        legend=dict(yanchor="top", y=1.0, xanchor="left", x=1.02),
        hovermode="closest",
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        margin=dict(l=70, r=250, t=70, b=60),
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
