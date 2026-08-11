"""Plot lambda_max wall time and penalized-fit wall time against training set size."""

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
PROFILE_SYMBOLS = {"rbf": "circle", "matern52": "diamond"}

SURFACE, GRID, AXIS_LINE = "#141412", "#2e2d29", "#3d3c36"
TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED = "#f4f3ef", "#bab8b0", "#807e76"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default=RESULTS_PATH)
    parser.add_argument("--out", type=str, default=os.path.join(HERE, "plot.html"))
    parser.add_argument("--ratio", type=float, default=None)
    return parser.parse_args()


def short_chip(chip: str) -> str:
    return re.sub(r"NVIDIA GeForce |NVIDIA |\(R\)|\(TM\)| CPU @.*", "", chip).strip()


def series_label(chip: str, profile: str) -> str:
    return f"{short_chip(chip)} {profile}"


def chip_color(chip: str) -> str:
    """Hues left over from the fixed map go to whatever chip shows up next."""
    short = short_chip(chip)
    if short not in CHIP_COLORS:
        spare = [c for c in SERIES_COLORS if c not in CHIP_COLORS.values()]
        CHIP_COLORS[short] = spare[0] if spare else SERIES_COLORS[-1]
    return CHIP_COLORS[short]


def local_exponent(group: pd.DataFrame, column: str) -> float:
    """Cost exponent between the two largest measured sizes."""
    tail = group.iloc[-2:]
    return float(np.diff(np.log(tail[column]))[0] / np.diff(np.log(tail.n))[0])


if __name__ == "__main__":
    args = parse_args()
    results = pd.read_json(args.results, lines=True)

    ok = results[results.status == "ok"]
    if args.ratio is not None:
        ok = ok[ok.ratio == args.ratio]

    # repeat 0 carries the compile, so the fastest observation is the honest one
    keys = ["chip", "profile", "n"]
    fits = ok.loc[ok.groupby(keys).fit_s.idxmin()]
    lambdas = ok.loc[ok.groupby(keys).lambda_max_s.idxmin()]

    figure = go.Figure()
    for (chip, profile), group in fits.groupby(["chip", "profile"]):
        group = group.sort_values("n")
        color = chip_color(chip)
        label = series_label(chip, profile)
        figure.add_trace(
            go.Scatter(
                x=group.n,
                y=group.fit_s,
                name=label,
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(
                    size=9,
                    color=color,
                    symbol=PROFILE_SYMBOLS[profile],
                    line=dict(color=SURFACE, width=2),
                ),
                customdata=np.stack([group.admm_iters, group.ratio], axis=-1),
                hovertemplate=(
                    f"<b>{label}</b><br>n=%{{x}}<br>"
                    "%{y:.2f}s penalized fit<br>"
                    "%{customdata[0]} ADMM iterations<br>"
                    "lambda / lambda_max = %{customdata[1]}"
                    "<extra></extra>"
                ),
            )
        )
        if len(group) > 1:
            print(f"fit {label}: local exponent {local_exponent(group, 'fit_s'):.2f}")

    # lambda_max is one bounded nugget search, so it should stay far below the fit
    for (chip, profile), group in lambdas.groupby(["chip", "profile"]):
        group = group.sort_values("n")
        label = series_label(chip, profile)
        figure.add_trace(
            go.Scatter(
                x=group.n,
                y=group.lambda_max_s,
                name=f"{label} · lambda_max",
                mode="lines+markers",
                visible="legendonly",
                line=dict(color=chip_color(chip), width=1.5, dash="dot"),
                marker=dict(size=6, symbol=PROFILE_SYMBOLS[profile]),
                hovertemplate=(
                    f"<b>{label} lambda_max</b><br>n=%{{x}}<br>"
                    "%{y:.2f}s<extra></extra>"
                ),
            )
        )
        if len(group) > 1:
            exponent = local_exponent(group, "lambda_max_s")
            print(f"lambda_max {label}: local exponent {exponent:.2f}")

    # cubic reference anchored at the cheapest measured point
    if fits.n.nunique() > 1:
        anchor = fits.loc[fits.n.idxmin()]
        reference_n = np.array([anchor.n, fits.n.max()])
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

    figure.update_layout(
        title=dict(
            text="lambda_max bench: wall time",
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
            title="seconds",
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
