#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for measuring player velocities, smoothed using a Savitzky-Golay filter, with Metrica tracking data.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd
from scipy import signal

FilterName = Literal["Savitzky-Golay", "moving average"]


def calc_player_velocities(
    team: pd.DataFrame,
    smoothing: bool = True,
    filter_: FilterName = "Savitzky-Golay",
    window: int = 7,
    polyorder: int = 1,
    maxspeed: float = 12,
) -> pd.DataFrame:
    """Calculate player velocities and speed for a tracking DataFrame.

    Adds, per player prefix (e.g. "Home_7"), the columns:
      - "{prefix}_vx", "{prefix}_vy", "{prefix}_speed"

    Notes
    -----
    - Uses Time [s] to compute dt via diff(). Typical Metrica sample data is 25 Hz (dt ~ 0.04s).
    - Smoothing is applied separately within each half (Period==1 and Period==2) to avoid halftime discontinuity.
    - Savitzky-Golay smoothing is made robust by interpolating NaNs just for filtering, then restoring NaNs.

    Parameters
    ----------
    team:
        Tracking DataFrame with columns like "Home_1_x", "Home_1_y" and "Time [s]", "Period".
    smoothing:
        Whether to smooth velocity signals.
    filter_:
        Either "Savitzky-Golay" or "moving average".
    window:
        Smoothing window size in frames (SavGol requires odd window >= polyorder+2).
    polyorder:
        Polynomial order for SavGol.
    maxspeed:
        If > 0, velocity samples whose instantaneous speed exceeds maxspeed are set to NaN (likely tracking glitches).

    Returns
    -------
    pd.DataFrame
        New DataFrame with velocity and speed columns added.
    """
    # Copy/drop old velocity columns first (original behavior: returns a modified frame, not in-place)
    team = remove_player_velocities(team)

    # Infer player prefixes once (fast, cached per schema)
    prefixes = _infer_player_prefixes(tuple(team.columns))
    if not prefixes:
        return team

    x_cols = [f"{p}_x" for p in prefixes if f"{p}_x" in team.columns]
    y_cols = [f"{p}_y" for p in prefixes if f"{p}_y" in team.columns]
    if not x_cols or not y_cols:
        return team

    if "Time [s]" not in team.columns:
        raise KeyError("Missing required column: 'Time [s]'")
    if "Period" not in team.columns:
        raise KeyError("Missing required column: 'Period'")

    # dt is per-row; dividing a DataFrame by a Series (axis=0) is vectorized.
    dt = team["Time [s]"].diff()

    # Vectorized velocity estimates across all players
    vx = team[x_cols].diff().div(dt, axis=0)
    vy = team[y_cols].diff().div(dt, axis=0)

    # Optionally remove outliers that exceed maxspeed
    if maxspeed and maxspeed > 0:
        speed_raw = np.sqrt(vx.to_numpy() ** 2 + vy.to_numpy() ** 2)
        outlier = speed_raw > float(maxspeed)
        # set vx/vy where outlier -> NaN
        vx = vx.mask(outlier)
        vy = vy.mask(outlier)

    if smoothing:
        period = team["Period"]
        mask_1h = period == 1
        mask_2h = period == 2

        vx = _smooth_by_half(
            vx, mask_1h, mask_2h, filter_=filter_, window=window, polyorder=polyorder
        )
        vy = _smooth_by_half(
            vy, mask_1h, mask_2h, filter_=filter_, window=window, polyorder=polyorder
        )

    # Compute speed after smoothing/outlier removal
    speed = np.sqrt(vx.to_numpy() ** 2 + vy.to_numpy() ** 2)
    speed_df = pd.DataFrame(
        speed, index=team.index, columns=[f"{p}_speed" for p in prefixes]
    )

    # Write outputs back to team (aligned by index)
    team = team.copy()
    for p in prefixes:
        px, py = f"{p}_x", f"{p}_y"
        if px in x_cols:
            team[f"{p}_vx"] = vx[px]
        if py in y_cols:
            team[f"{p}_vy"] = vy[py]
        team[f"{p}_speed"] = speed_df[f"{p}_speed"]

    return team


def remove_player_velocities(team: pd.DataFrame) -> pd.DataFrame:
    """Remove player velocity/acceleration/speed columns if present."""
    suffixes = {"vx", "vy", "ax", "ay", "speed", "acceleration"}
    drop_cols = [c for c in team.columns if c.split("_")[-1] in suffixes]
    if not drop_cols:
        return team
    return team.drop(columns=drop_cols, errors="ignore")


# -------------------------
# Internals
# -------------------------


@lru_cache(maxsize=32)
def _infer_player_prefixes(columns: tuple[str, ...]) -> tuple[str, ...]:
    """Infer unique player prefixes like 'Home_7' from tracking columns.

    We consider columns matching '<Team>_<id>_x' or '<Team>_<id>_y' for Team in {Home, Away}.
    """
    prefixes: set[str] = set()
    for c in columns:
        if not (c.startswith("Home_") or c.startswith("Away_")):
            continue
        if c.endswith("_x") or c.endswith("_y"):
            parts = c.split("_")
            if len(parts) >= 3:
                prefixes.add(f"{parts[0]}_{parts[1]}")
    return tuple(sorted(prefixes))


def _smooth_by_half(
    v: pd.DataFrame,
    mask_1h: pd.Series,
    mask_2h: pd.Series,
    *,
    filter_: FilterName,
    window: int,
    polyorder: int,
) -> pd.DataFrame:
    """Smooth velocity DataFrame by half, robust to NaNs."""
    v_out = v.copy()

    if filter_ == "Savitzky-Golay":
        v_out.loc[mask_1h] = _savgol_df(
            v_out.loc[mask_1h], window=window, polyorder=polyorder
        )
        v_out.loc[mask_2h] = _savgol_df(
            v_out.loc[mask_2h], window=window, polyorder=polyorder
        )
        return v_out

    if filter_ == "moving average":
        v_out.loc[mask_1h] = _moving_average_df(v_out.loc[mask_1h], window=window)
        v_out.loc[mask_2h] = _moving_average_df(v_out.loc[mask_2h], window=window)
        return v_out

    # If someone passes a new string, fail loudly (better than silently doing nothing)
    raise ValueError(f"Unknown filter_: {filter_!r}")


def _valid_savgol_window(n: int, window: int, polyorder: int) -> int:
    """Return a SavGol-safe odd window length, or 0 if smoothing should be skipped."""
    if n <= 0:
        return 0

    wl = int(window)
    po = int(polyorder)

    # Need at least polyorder+2 points to fit stably
    if n < (po + 2):
        return 0

    if wl < (po + 2):
        wl = po + 2
    if wl > n:
        wl = n

    # Must be odd and >= 3 to be meaningful
    if wl % 2 == 0:
        wl -= 1
    if wl < 3:
        return 0

    return wl


def _savgol_df(df: pd.DataFrame, *, window: int, polyorder: int) -> pd.DataFrame:
    """SavGol smooth all columns in a DataFrame at once (axis=0), with NaN safety."""
    if df.empty:
        return df

    n = df.shape[0]
    wl = _valid_savgol_window(n, window=window, polyorder=polyorder)
    if wl == 0:
        return df

    # Remember original NaNs and fill temporarily for filtering
    nan_mask = ~np.isfinite(df.to_numpy(dtype=float))

    # Vectorized interpolation per column (pandas is fast here)
    filled = df.astype(float).interpolate(
        method="linear", limit_direction="both", axis=0
    )

    x = filled.to_numpy(dtype=float)

    # SavGol across time axis for each column
    try:
        y = signal.savgol_filter(x, window_length=wl, polyorder=int(polyorder), axis=0)
    except Exception:
        # If anything weird happens, fall back to unsmoothed
        return df

    y[nan_mask] = np.nan
    return pd.DataFrame(y, index=df.index, columns=df.columns)


def _moving_average_df(df: pd.DataFrame, *, window: int) -> pd.DataFrame:
    """Moving-average smooth all columns in a DataFrame at once (axis=0), with NaN safety."""
    if df.empty:
        return df

    w = int(window)
    if w <= 1:
        return df

    # Fill NaNs temporarily
    nan_mask = ~np.isfinite(df.to_numpy(dtype=float))
    filled = df.astype(float).interpolate(
        method="linear", limit_direction="both", axis=0
    )
    x = filled.to_numpy(dtype=float)

    # Convolve along axis=0 using an FIR kernel.
    kernel = np.ones(w, dtype=float) / float(w)
    # Apply per-column with FFT-free convolution (w is small; this is fine and predictable)
    y = np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="same"), 0, x)

    y[nan_mask] = np.nan
    return pd.DataFrame(y, index=df.index, columns=df.columns)
