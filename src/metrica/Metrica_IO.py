#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 11:18:49 2020

Module for reading in Metrica sample data.

Data can be found at: https://github.com/metrica-sports/sample-data

Refactor + cache safety fixes: 2026-01-14

@author: Laurie Shaw (@EightyFivePoint)
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------

PathLike = Union[str, Path]
FieldDimen = Tuple[float, float]
FileSig = Tuple[str, int]  # (absolute_path_str, mtime_ns)


# ---------------------------------------------------------------------
# Public API (preserved names; caching is session-local and SAFE)
# ---------------------------------------------------------------------


def read_match_data(
    DATADIR: PathLike, gameid: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read all Metrica match data (tracking home/away, and event data)."""
    tracking_home = tracking_data(DATADIR, gameid, "Home")
    tracking_away = tracking_data(DATADIR, gameid, "Away")
    events = read_event_data(DATADIR, gameid)
    return tracking_home, tracking_away, events


def read_event_data(DATADIR: PathLike, game_id: int) -> pd.DataFrame:
    """Read Metrica event data for `game_id` and return as a DataFrame.

    Notes
    -----
    Returned DataFrame is always a *fresh copy* so downstream code can safely mutate it.
    """
    datadir = Path(DATADIR)
    eventfile = (
        datadir / f"Sample_Game_{game_id}" / f"Sample_Game_{game_id}_RawEventsData.csv"
    )
    sig = _file_sig(eventfile)
    df = _read_csv_cached(sig)
    return df.copy(deep=True)


def tracking_data(DATADIR: PathLike, game_id: int, teamname: str) -> pd.DataFrame:
    """Read Metrica tracking data for `game_id` and return as a DataFrame."""
    datadir = Path(DATADIR)
    teamfile = (
        datadir
        / f"Sample_Game_{game_id}"
        / f"Sample_Game_{game_id}_RawTrackingData_{teamname}_Team.csv"
    )

    sig = _file_sig(teamfile)
    columns, teamnamefull = _tracking_columns_cached(sig, teamname)

    print(f"Reading team: {teamnamefull}")

    df = pd.read_csv(teamfile, names=columns, index_col="Frame", skiprows=3)
    return df.copy(deep=True)


def merge_tracking_data(home: pd.DataFrame, away: pd.DataFrame) -> pd.DataFrame:
    """Merge home & away tracking data into a single DataFrame."""
    home_wo_ball = home.drop(columns=["ball_x", "ball_y"], errors="ignore")
    return pd.concat([home_wo_ball, away], axis=1)


def to_metric_coordinates(
    data: pd.DataFrame,
    field_dimen: FieldDimen = (106.0, 68.0),
    *,
    inplace: bool = False,
) -> pd.DataFrame:
    """Convert positions from Metrica units to meters (origin at centre circle).

    IMPORTANT
    ---------
    Default behavior is **not** to mutate inputs (inplace=False). This prevents accidental
    double-conversion and prevents poisoning any cached DataFrame objects.
    """
    df = data if inplace else data.copy(deep=True)

    # Works for both tracking columns like "Home_1_x" and event columns like "Start X"
    x_cols = [c for c in df.columns if str(c).strip().lower().endswith("x")]
    y_cols = [c for c in df.columns if str(c).strip().lower().endswith("y")]

    if x_cols:
        df.loc[:, x_cols] = (df.loc[:, x_cols] - 0.5) * field_dimen[0]
    if y_cols:
        df.loc[:, y_cols] = -1.0 * (df.loc[:, y_cols] - 0.5) * field_dimen[1]

    return df


def to_single_playing_direction(
    home: pd.DataFrame,
    away: pd.DataFrame,
    events: pd.DataFrame,
    *,
    period_col: str = "Period",
    second_half_value: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Flip coordinates in the second half so each team attacks in the same direction all match.

    NOTE: This intentionally mutates inputs (matches original tutorial style).
    """

    def _coord_cols(df: pd.DataFrame) -> list[str]:
        cols: list[str] = []
        for c in df.columns:
            cl = str(c).lower()
            if (
                cl.endswith(("_x", "_y"))
                or cl.endswith((" x", " y"))
                or cl in ("x", "y")
            ):
                cols.append(c)
        return cols

    def flip_df(df: pd.DataFrame) -> None:
        if period_col not in df.columns:
            raise KeyError(f"'{period_col}' column not found")

        mask_2h = df[period_col] == second_half_value
        if not mask_2h.any():
            return

        start_idx = df.index[mask_2h][0]
        cols = _coord_cols(df)
        if cols:
            df.loc[start_idx:, cols] = df.loc[start_idx:, cols] * -1.0

    for df in (home, away, events):
        flip_df(df)

    return home, away, events


def find_playing_direction(team: pd.DataFrame, teamname: str) -> float:
    """Find direction of play (+1 left->right, -1 right->left) based on GK x at kickoff."""
    gk_num = find_goalkeeper(team)
    gk_col_x = f"{teamname}_{gk_num}_x"
    x0 = float(team.iloc[0][gk_col_x])
    return float(-np.sign(x0) if x0 != 0 else 1.0)


def find_goalkeeper(team: pd.DataFrame) -> str:
    """Identify goalkeeper jersey number as the player closest to goal at kickoff."""
    x_cols = [
        c
        for c in team.columns
        if str(c).lower().endswith("_x") and str(c)[:4] in ("Home", "Away")
    ]
    if not x_cols:
        raise ValueError(
            "No player x-columns found (expected columns like 'Home_1_x')."
        )

    row0 = team.iloc[0][x_cols]
    abs_x = row0.astype(float).abs().fillna(-np.inf)
    gk_col = str(abs_x.idxmax())
    return gk_col.split("_")[1]


# ---------------------------------------------------------------------
# File-level caching helpers (SAFE within a Python session)
# ---------------------------------------------------------------------


def _file_sig(path: Path) -> FileSig:
    """Return a stable cache key that changes when the file changes."""
    p = path.resolve()
    mtime_ns = int(p.stat().st_mtime_ns)
    return (str(p), mtime_ns)


@lru_cache(maxsize=32)
def _read_csv_cached(sig: FileSig) -> pd.DataFrame:
    """Cached CSV load keyed by (absolute_path, mtime_ns)."""
    path_str, _mtime_ns = sig
    return pd.read_csv(path_str)


@lru_cache(maxsize=64)
def _tracking_columns_cached(sig: FileSig, teamname: str) -> tuple[list[str], str]:
    """Parse tracking header rows to build DataFrame column names (cached)."""
    teamfile = Path(sig[0])

    with teamfile.open("r", newline="") as f:
        reader = csv.reader(f)

        row1 = next(reader)
        teamnamefull = str(row1[3]).lower() if len(row1) > 3 else teamname.lower()

        jerseys = [x for x in next(reader) if x != ""]
        columns = next(reader)

    for i, j in enumerate(jerseys):
        base = 3 + i * 2
        if base < len(columns):
            columns[base] = f"{teamname}_{j}_x"
        if base + 1 < len(columns):
            columns[base + 1] = f"{teamname}_{j}_y"

    if len(columns) >= 2:
        columns[-2] = "ball_x"
        columns[-1] = "ball_y"

    return columns, teamnamefull


def clear_caches() -> None:
    """Clear in-memory file/header caches used for faster repeated loads."""
    _read_csv_cached.cache_clear()
    _tracking_columns_cached.cache_clear()
