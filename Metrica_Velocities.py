#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for measuring player velocities, smoothed using a Savitzky-Golay filter, with Metrica tracking data.
"""

from __future__ import annotations

import numpy as np
import scipy.signal as signal


def calc_player_velocities(
    team,
    smoothing: bool = True,
    filter_: str = "Savitzky-Golay",
    window: int = 7,
    polyorder: int = 1,
    maxspeed: float = 12,
):
    """calc_player_velocities( tracking_data )

    Calculate player velocities in x & y direction, and total player speed at each timestamp of the tracking data.
    """

    # remove any velocity data already in the dataframe
    team = remove_player_velocities(team)

    # Get the player ids (prefixes like 'Home_1', 'Away_7')
    player_ids = np.unique([c[:-2] for c in team.columns if c[:4] in ["Home", "Away"]])

    # Calculate timestep between frames; ~0.04s within halves at 25 Hz
    dt = team["Time [s]"].diff()

    # Identify halves robustly
    period = team["Period"]
    mask_1h = period == 1
    mask_2h = period == 2

    # Helper: make SavGol safe in the presence of NaNs and short segments
    def _safe_savgol(series, *, window_length: int, polyorder_: int):
        """
        Apply SavGol to a pandas Series that may contain NaNs.
        We interpolate just for filtering, then restore original NaNs.
        """
        # Convert to numpy float array
        x = series.to_numpy(dtype=float)
        n = x.size

        # If too short, just return as-is
        if n == 0:
            return series
        if n < (polyorder_ + 2):
            return series  # cannot fit polynomial reliably

        # Ensure window_length is valid: odd, <= n, and >= polyorder+2
        wl = int(window_length)
        if wl < (polyorder_ + 2):
            wl = polyorder_ + 2
        if wl > n:
            wl = n
        if wl % 2 == 0:
            wl = wl - 1 if wl > 1 else 1
        if wl < 3:
            return series  # no meaningful smoothing possible

        # If all NaN or too few finite points, return as-is
        finite = np.isfinite(x)
        if finite.sum() < (polyorder_ + 2):
            return series

        # Interpolate NaNs just for smoothing
        # (use linear interpolation across gaps, then forward/back fill edges)
        x_filled = x.copy()
        idx = np.arange(n)
        x_filled[~finite] = np.interp(idx[~finite], idx[finite], x[finite])

        # Apply SavGol
        try:
            y = signal.savgol_filter(x_filled, window_length=wl, polyorder=polyorder_)
        except Exception:
            # If SavGol still fails for any reason, fall back to unsmoothed series
            return series

        # Restore original NaNs (keep missingness where it truly existed)
        y[~finite] = np.nan
        return series.__class__(y, index=series.index, name=series.name)

    # Helper: moving average safe for NaNs
    def _safe_moving_average(series, window_length: int):
        x = series.to_numpy(dtype=float)
        n = x.size
        if n == 0:
            return series
        w = int(window_length)
        if w <= 1:
            return series

        finite = np.isfinite(x)
        if finite.sum() == 0:
            return series

        # Fill NaNs via interpolation for convolution, then restore
        x_filled = x.copy()
        idx = np.arange(n)
        x_filled[~finite] = np.interp(idx[~finite], idx[finite], x[finite])

        kernel = np.ones(w, dtype=float) / w
        y = np.convolve(x_filled, kernel, mode="same")
        y[~finite] = np.nan
        return series.__class__(y, index=series.index, name=series.name)

    # estimate velocities for players in team
    for player in player_ids:
        # Unsmooth velocity estimate
        vx = team[player + "_x"].diff() / dt
        vy = team[player + "_y"].diff() / dt

        if maxspeed and maxspeed > 0:
            raw_speed = np.sqrt(vx**2 + vy**2)
            vx[raw_speed > maxspeed] = np.nan
            vy[raw_speed > maxspeed] = np.nan

        if smoothing:
            if filter_ == "Savitzky-Golay":
                # Smooth each half independently to avoid halftime discontinuity issues
                vx.loc[mask_1h] = _safe_savgol(vx.loc[mask_1h], window_length=window, polyorder_=polyorder)
                vy.loc[mask_1h] = _safe_savgol(vy.loc[mask_1h], window_length=window, polyorder_=polyorder)
                vx.loc[mask_2h] = _safe_savgol(vx.loc[mask_2h], window_length=window, polyorder_=polyorder)
                vy.loc[mask_2h] = _safe_savgol(vy.loc[mask_2h], window_length=window, polyorder_=polyorder)

            elif filter_ == "moving average":
                vx.loc[mask_1h] = _safe_moving_average(vx.loc[mask_1h], window_length=window)
                vy.loc[mask_1h] = _safe_moving_average(vy.loc[mask_1h], window_length=window)
                vx.loc[mask_2h] = _safe_moving_average(vx.loc[mask_2h], window_length=window)
                vy.loc[mask_2h] = _safe_moving_average(vy.loc[mask_2h], window_length=window)

        # put player speed in x,y direction, and total speed back in the data frame
        team[player + "_vx"] = vx
        team[player + "_vy"] = vy
        team[player + "_speed"] = np.sqrt(vx**2 + vy**2)

    return team


def remove_player_velocities(team):
    # remove player velocities and acceleration measures already in dataframe
    columns = [c for c in team.columns if c.split("_")[-1] in ["vx", "vy", "ax", "ay", "speed", "acceleration"]]
    if columns:
        team = team.drop(columns=columns, errors="ignore")
    return team