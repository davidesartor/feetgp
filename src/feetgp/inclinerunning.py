from typing import Literal
import numpy as np

import os
import pandas as pd
from tqdm import tqdm


class InclineRunning:
    marker_names: list[str] = [
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
    file_names: list[str] = [
        "inc0_10kmh",
        "inc0_12kmh",
        "inc0_14kmh",
        "inc5_10kmh",
        "inc5_12kmh",
        "inc5_14kmh",
        "inc10_10kmh",
        "inc10_12kmh",
        "inc10_14kmh",
    ]

    def __init__(
        self,
        path: str = "data/Incline Running",
        subsample: int = 1,
        feet: Literal["both", "left_only", "right_only"] = "both",
        target: Literal["markers", "forces"] = "markers",
        inclines: Literal["all", "inc0", "inc5", "inc10"] = "all",
        relative: str | None = None,
    ):
        self.path = path
        if feet == "both":
            # make sure right and left markers appear one after the other in the list
            # this is important for the group lasso to work properly!!!
            self.markers = [
                prefix + name for name in self.marker_names for prefix in ("L", "R")
            ]
        elif feet == "left_only":
            self.markers = ["L" + name for name in self.marker_names]
        elif feet == "right_only":
            self.markers = ["R" + name for name in self.marker_names]
        else:
            raise ValueError(f"feet must be 'both|left_only|right_only', got {feet}")

        # load marker data for the input features
        df_markers = self.load_marker_data(inclines, relative)
        x = df_markers.values
        self.x_columns = list(df_markers.columns)
        print("Loaded marker data with shape:", x.shape, x.dtype)

        # load data for the target variable
        if target == "forces":
            df_forces = self.load_ground_reaction_forces(inclines)
            y = np.cbrt(
                df_forces.values
            )  # cube root to make the distribution more normal
            self.y_columns = list(df_forces.columns)
        elif target == "markers":
            y = x.copy()
            self.y_columns = list(self.x_columns)
        print("Loaded target data with shape:", y.shape, y.dtype)

        # subsample the data and drop rows with NaN values
        x, y = x[::subsample], y[::subsample]
        valid = ~np.isnan(x).any(axis=1) & ~np.isnan(y).any(axis=1)
        x, y = x[valid], y[valid]

        # DEBT: even/odd rows are consecutive mocap frames, so test is not
        # independent of train and every R² here is optimistic. Kept
        # deliberately while sanity-checking; replace with a blocked split
        # before believing any result.
        self.x_train, self.x_test = x[::2, :], x[1::2, :]
        self.y_train, self.y_test = y[::2, :], y[1::2, :]
        print("train:", self.x_train.shape, self.y_train.shape)
        print("test:", self.x_test.shape, self.y_test.shape)

        # normalize input features; constant columns (e.g. reference marker in relative mode) stay 0
        x_min = np.min(self.x_train, axis=0, keepdims=True)
        x_max = np.max(self.x_train, axis=0, keepdims=True)
        x_range = np.where(x_max == x_min, 1, x_max - x_min)
        self.x_train = (self.x_train - x_min) / x_range
        self.x_test = (self.x_test - x_min) / x_range

        # standardize target variable; constant columns stay 0
        y_mean = np.mean(self.y_train, axis=0, keepdims=True)
        y_std = np.std(self.y_train, axis=0, keepdims=True)
        y_std = np.where(y_std == 0, 1, y_std)
        self.y_train = (self.y_train - y_mean) / y_std
        self.y_test = (self.y_test - y_mean) / y_std

    def load_marker_data(
        self,
        inclines: Literal["all", "inc0", "inc5", "inc10"] = "all",
        relative: str | None = None,
    ):
        def load_tsv_file(filepath):
            df = pd.read_csv(filepath, sep="\t", skiprows=10)
            df = df.iloc[:, :-1]
            df = df[df.columns[df.columns.str.contains("|".join(self.markers))]]
            df = df[sorted(df.columns, key=lambda x: self.markers.index(x[:-2]))]
            return df

        files = self.file_names
        if inclines != "all":
            files = [f for f in files if f.startswith(inclines)]
        dfs = [
            load_tsv_file(os.path.join(self.path, f"{f}.tsv"))
            for f in tqdm(files, desc="Loading Marker Data")
        ]
        df = pd.concat(dfs, ignore_index=True)
        if relative is not None:
            for prefix in ("L", "R"):
                for coord in ("X", "Y", "Z"):
                    cols = [c for c in df.columns if c.startswith(prefix)]
                    cols = [c for c in cols if c.endswith(coord)]
                    if not cols:
                        continue
                    if relative == "midpoint":
                        # reference is the midpoint between the LMAL and MMAL markers
                        ref = (
                            df[f"{prefix}LMAL {coord}"] + df[f"{prefix}MMAL {coord}"]
                        ) / 2
                    else:
                        ref = df[f"{prefix}{relative} {coord}"]
                    df[cols] = df[cols].subtract(ref, axis=0)
            if relative != "midpoint":
                # drop the reference marker (now identically zero); the midpoint
                # is not a marker column, so nothing is dropped in that mode
                df = df.drop(
                    columns=[
                        c
                        for c in df.columns
                        if any(c.startswith(f"{p}{relative} ") for p in ("L", "R"))
                    ]
                )
        return df

    def load_ground_reaction_forces(
        self, inclines: Literal["all", "inc0", "inc5", "inc10"] = "all"
    ):
        def load_xlsx(filepath):
            df = pd.read_excel(filepath)
            df = df.iloc[:, 1:]
            df = df.groupby(df.index // 5).mean()
            return df

        files = self.file_names
        if inclines != "all":
            files = [f for f in files if f.startswith(inclines)]
        dfs = [
            load_xlsx(os.path.join(self.path, f"{f}_f_1.xlsx"))
            for f in tqdm(files, desc="Loading Ground Reaction Forces")
        ]
        return pd.concat(dfs, ignore_index=True)
