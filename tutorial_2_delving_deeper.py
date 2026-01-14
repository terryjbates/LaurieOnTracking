#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tutorial 2 (FoT Lesson 5): Physical data (velocities, distance bands, sprints)

Original: Mon Apr 13 11:34:26 2020 (Laurie Shaw, @EightyFivePoint)
Refactor: Jan 14, 2026

Goals of this refactor
- Runs top-to-bottom without “run selection”.
- Uses the refactored Metrica_IO cache automatically (read_event_data / tracking_data).
- Ensures event + tracking coordinates are numeric and converted correctly (no blank-pitch traps).
- Keeps original outputs/plots but removes/guards lines that can error (e.g., regex pitfalls).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import metrica.Metrica_IO as mio
import metrica.Metrica_Viz as mviz
import metrica.Metrica_Velocities as mvel


# ----------------------------
# Config
# ----------------------------

DATADIR = os.environ.get("METRICA_DATADIR", r"C:\Users\lover\github\sample-data\data")
game_id = 2

FPS = 25.0  # Metrica sample data is 25 Hz

# Movie clip settings (safe defaults)
CLIP_START = 73600
CLIP_LEN = 500
CLIP_NAME = "home_goal_2"
PLOTDIR = DATADIR  # original tutorial uses DATADIR


# ----------------------------
# Helpers
# ----------------------------

EVENT_COORDS = ["Start X", "Start Y", "End X", "End Y"]


def require_datadir(path: str) -> None:
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"DATADIR does not exist:\n  {path}\n"
            "Fix DATADIR or set METRICA_DATADIR to your local sample-data/data folder."
        )


def coerce_cols_numeric(df: pd.DataFrame, cols: Iterable[str]) -> None:
    """In-place numeric coercion for selected columns that may be dtype=object."""
    present = [c for c in cols if c in df.columns]
    if present:
        df.loc[:, present] = df.loc[:, present].apply(pd.to_numeric, errors="coerce")


def player_ids_from_tracking(tracking: pd.DataFrame, team_prefix: str) -> list[str]:
    """Return jersey numbers as strings for columns like Home_10_x."""
    pat = re.compile(rf"^{re.escape(team_prefix)}_(\d+)_x$")
    ids = sorted({m.group(1) for c in tracking.columns if (m := pat.match(c))})
    return ids


def minutes_played(tracking: pd.DataFrame, team_prefix: str, player_id: str, fps: float = FPS) -> float:
    """Minutes played based on first/last valid x observation."""
    col = f"{team_prefix}_{player_id}_x"
    first = tracking[col].first_valid_index()
    last = tracking[col].last_valid_index()
    if first is None or last is None:
        return 0.0
    frames = (last - first + 1)
    return float(frames / fps / 60.0)


def total_distance_km_from_speed(tracking: pd.DataFrame, speed_col: str, fps: float = FPS) -> float:
    """
    The FoT velocity module's *_speed is meters/second.
    Distance per frame is speed * dt. Summing speed and dividing by fps gives meters.
    """
    speed = pd.to_numeric(tracking[speed_col], errors="coerce")
    meters = float(speed.sum(skipna=True) / fps)
    return meters / 1000.0


def distance_bands_km(tracking: pd.DataFrame, speed_col: str, fps: float = FPS) -> Tuple[float, float, float, float]:
    """Return (walking, jogging, running, sprinting) distances in km."""
    speed = pd.to_numeric(tracking[speed_col], errors="coerce")

    def km(mask: pd.Series) -> float:
        meters = float(speed.loc[mask].sum(skipna=True) / fps)
        return meters / 1000.0

    walking = km(speed < 2)
    jogging = km((speed >= 2) & (speed < 4))
    running = km((speed >= 4) & (speed < 7))
    sprinting = km(speed >= 7)
    return walking, jogging, running, sprinting


def sustained_sprints_count(speed: pd.Series, *, threshold: float, window_frames: int) -> int:
    """
    Count sustained sprint occurrences using the original convolution trick.
    Returns number of segments where speed >= threshold for at least window_frames.
    """
    s = pd.to_numeric(speed, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(s) & (s >= threshold)
    ok = ok.astype(int)

    sustained = np.convolve(ok, np.ones(window_frames), mode="same") >= window_frames
    edges = np.diff(sustained.astype(int))
    return int(np.sum(edges == 1))


def sustained_sprint_windows(speed: pd.Series, *, threshold: float, window_frames: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return arrays of (start_indices, end_indices) for sustained sprint windows."""
    s = pd.to_numeric(speed, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(s) & (s >= threshold)
    ok = ok.astype(int)

    sustained = np.convolve(ok, np.ones(window_frames), mode="same") >= window_frames
    edges = np.diff(sustained.astype(int))

    starts = np.where(edges == 1)[0] - int(window_frames / 2) + 1
    ends = np.where(edges == -1)[0] + int(window_frames / 2) + 1

    # clip to valid range
    starts = np.clip(starts, 0, len(s) - 1)
    ends = np.clip(ends, 0, len(s) - 1)

    # ensure pairs align
    n = min(len(starts), len(ends))
    return starts[:n], ends[:n]


def print_top_speeds(tracking: pd.DataFrame, team_prefix: str) -> None:
    """Print max speed per player using reliable column parsing (fixes original regex bug)."""
    pat = re.compile(rf"^{re.escape(team_prefix)}_(\d+)_speed$")
    for col in tracking.columns:
        m = pat.match(col)
        if not m:
            continue
        pid = m.group(1)
        vmax = pd.to_numeric(tracking[col], errors="coerce").max(skipna=True)
        print(f"Maximum speed for {team_prefix} player{pid} is {float(vmax):.2f} meters per second")


# ----------------------------
# Load data (cached by your Metrica_IO refactor)
# ----------------------------

require_datadir(DATADIR)

events = mio.read_event_data(DATADIR, game_id)
tracking_home = mio.tracking_data(DATADIR, game_id, "Home")
tracking_away = mio.tracking_data(DATADIR, game_id, "Away")

# Coerce event coords numeric BEFORE conversion
coerce_cols_numeric(events, EVENT_COORDS)

# Convert to metric coordinates
tracking_home = mio.to_metric_coordinates(tracking_home)
tracking_away = mio.to_metric_coordinates(tracking_away)
events = mio.to_metric_coordinates(events)

# Single playing direction (mutates; we rebind for clarity)
tracking_home, tracking_away, events = mio.to_single_playing_direction(tracking_home, tracking_away, events)

# ----------------------------
# Make a movie clip (optional but runs)
# ----------------------------

# Ensure output dir exists
Path(PLOTDIR).mkdir(parents=True, exist_ok=True)

mviz.save_match_clip(
    tracking_home.iloc[CLIP_START : CLIP_START + CLIP_LEN],
    tracking_away.iloc[CLIP_START : CLIP_START + CLIP_LEN],
    PLOTDIR,
    fname=CLIP_NAME,
    include_player_velocities=False,
)

# ----------------------------
# Velocities
# ----------------------------

# Primary path (original lesson)
try:
    tracking_home = mvel.calc_player_velocities(tracking_home, smoothing=True)
    tracking_away = mvel.calc_player_velocities(tracking_away, smoothing=True)
except TypeError:
    # Compatibility fallback noted in original tutorial
    tracking_home = mvel.calc_player_velocities(tracking_home, smoothing=True, filter_="moving_average")
    tracking_away = mvel.calc_player_velocities(tracking_away, smoothing=True, filter_="moving_average")

# Plot a random frame with velocities
mviz.plot_frame(tracking_home.loc[10000], tracking_away.loc[10000], include_player_velocities=True, annotate=True)

# ----------------------------
# HOME physical summary
# ----------------------------

home_players = player_ids_from_tracking(tracking_home, "Home")
home_summary = pd.DataFrame(index=home_players)

home_summary["Minutes Played"] = [minutes_played(tracking_home, "Home", pid, fps=FPS) for pid in home_players]
home_summary = home_summary.sort_values(["Minutes Played"], ascending=False)

home_summary["Distance [km]"] = [
    total_distance_km_from_speed(tracking_home, f"Home_{pid}_speed", fps=FPS) for pid in home_summary.index
]

# Frame 51: used in original as a “look at player 11” diagnostic
mviz.plot_frame(tracking_home.loc[51], tracking_away.loc[51], include_player_velocities=True, annotate=True)

# Bar chart of total distance
plt.subplots()
ax = home_summary["Distance [km]"].plot.bar(rot=0)
ax.set_xlabel("Player")
ax.set_ylabel("Distance covered [km]")
ax.set_title("Home: total distance covered")

# Positions at KO-ish frame used in original (51)
mviz.plot_frame(tracking_home.loc[51], tracking_away.loc[51], include_player_velocities=False, annotate=True)

# Distance bands
bands = [distance_bands_km(tracking_home, f"Home_{pid}_speed", fps=FPS) for pid in home_summary.index]
home_summary["Walking [km]"] = [b[0] for b in bands]
home_summary["Jogging [km]"] = [b[1] for b in bands]
home_summary["Running [km]"] = [b[2] for b in bands]
home_summary["Sprinting [km]"] = [b[3] for b in bands]

ax = home_summary[["Walking [km]", "Jogging [km]", "Running [km]", "Sprinting [km]"]].plot.bar(colormap="coolwarm")
ax.set_xlabel("Player")
ax.set_ylabel("Distance covered [km]")
ax.set_title("Home: distance by speed band")

# Sustained sprints
sprint_threshold = 7.0
sprint_window = int(1 * FPS)  # 1 second

home_summary["# sprints"] = [
    sustained_sprints_count(tracking_home[f"Home_{pid}_speed"], threshold=sprint_threshold, window_frames=sprint_window)
    for pid in home_summary.index
]

# Plot sprint trajectories for Home player 10 (original)
player = "10"
speed_col = f"Home_{player}_speed"
x_col = f"Home_{player}_x"
y_col = f"Home_{player}_y"

starts, ends = sustained_sprint_windows(
    tracking_home[speed_col], threshold=sprint_threshold, window_frames=sprint_window
)

fig, ax = mviz.plot_pitch()
for s, e in zip(starts, ends):
    ax.plot(tracking_home[x_col].iloc[s], tracking_home[y_col].iloc[s], "ro")
    ax.plot(tracking_home[x_col].iloc[s : e + 1], tracking_home[y_col].iloc[s : e + 1], "r")
ax.set_title(f"Home player {player}: sustained sprints (>{sprint_threshold} m/s for >=1s)")

# Top speeds (fixed parsing)
print("\n--- Top speeds (Away) ---")
print_top_speeds(tracking_away, "Away")
print("\n--- Top speeds (Home) ---")
print_top_speeds(tracking_home, "Home")

# Acceleration example (Away_26)
dt = pd.to_numeric(tracking_away["Time [s]"].diff(), errors="coerce")
ax_acc = pd.to_numeric(tracking_away["Away_26_vx"], errors="coerce").diff() / dt
ay_acc = pd.to_numeric(tracking_away["Away_26_vy"], errors="coerce").diff() / dt

acc = np.sqrt(ax_acc**2 + ay_acc**2)
acc = acc.mask(acc > 11.0)  # cap spikes
max_acc = float(acc.max(skipna=True))
print(f"\nMax acceleration of player 26 Away: {max_acc:.2f} meters per sec^2")

# ----------------------------
# AWAY physical summary (mirrors home)
# ----------------------------

away_players = player_ids_from_tracking(tracking_away, "Away")
away_summary = pd.DataFrame(index=away_players)

away_summary["Minutes Played"] = [minutes_played(tracking_away, "Away", pid, fps=FPS) for pid in away_players]
away_summary = away_summary.sort_values(["Minutes Played"], ascending=False)

away_summary["Distance [km]"] = [
    total_distance_km_from_speed(tracking_away, f"Away_{pid}_speed", fps=FPS) for pid in away_summary.index
]

plt.subplots()
ax = away_summary["Distance [km]"].plot.bar(rot=0)
ax.set_xlabel("Player")
ax.set_ylabel("Distance covered [km]")
ax.set_title("Away: total distance covered")

bands = [distance_bands_km(tracking_away, f"Away_{pid}_speed", fps=FPS) for pid in away_summary.index]
away_summary["Walking [km]"] = [b[0] for b in bands]
away_summary["Jogging [km]"] = [b[1] for b in bands]
away_summary["Running [km]"] = [b[2] for b in bands]
away_summary["Sprinting [km]"] = [b[3] for b in bands]

ax = away_summary[["Walking [km]", "Jogging [km]", "Running [km]", "Sprinting [km]"]].plot.bar(colormap="coolwarm")
ax.set_xlabel("Player")
ax.set_ylabel("Distance covered [km]")
ax.set_title("Away: distance by speed band")

away_summary["# sprints"] = [
    sustained_sprints_count(tracking_away[f"Away_{pid}_speed"], threshold=sprint_threshold, window_frames=sprint_window)
    for pid in away_summary.index
]

# Plot sprint trajectories for Away player 26 (as in your modified version)
player = "26"
speed_col = f"Away_{player}_speed"
x_col = f"Away_{player}_x"
y_col = f"Away_{player}_y"

starts, ends = sustained_sprint_windows(
    tracking_away[speed_col], threshold=sprint_threshold, window_frames=sprint_window
)

fig, ax = mviz.plot_pitch()
for s, e in zip(starts, ends):
    ax.plot(tracking_away[x_col].iloc[s], tracking_away[y_col].iloc[s], "ro")
    ax.plot(tracking_away[x_col].iloc[s : e + 1], tracking_away[y_col].iloc[s : e + 1], "r")
ax.set_title(f"Away player {player}: sustained sprints (>{sprint_threshold} m/s for >=1s)")

print("\nEND")
