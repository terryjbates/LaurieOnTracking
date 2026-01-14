#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tutorial 1: Getting Started with Metrica sample data (Events + Tracking)

Original: Sun Apr  5 13:19:08 2020 (Laurie Shaw, @EightyFivePoint)
Refactor: Jan 14, 2026

Runs top-to-bottom (no Spyder “run selection” assumptions).
Assumes Metrica_IO.py and Metrica_Viz.py are importable as shown below.
"""

from __future__ import annotations

import os
import re
from typing import Iterable

import numpy as np
import pandas as pd

import metrica.Metrica_IO as mio
import metrica.Metrica_Viz as mviz


# ----------------------------
# Config
# ----------------------------

DATADIR = os.environ.get("METRICA_DATADIR", r"C:\Users\lover\github\sample-data\data")
game_id = 2

EVENT_ID_GOAL_1 = 198
EVENT_ID_GOAL_3 = 1118

EVENT_COORDS = ["Start X", "Start Y", "End X", "End Y"]


# ----------------------------
# Small helpers (minimal + deterministic)
# ----------------------------

def require_datadir(path: str) -> None:
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"DATADIR does not exist:\n  {path}\n"
            "Fix DATADIR or set METRICA_DATADIR to your local sample-data/data folder."
        )


def coerce_event_coords_numeric(events: pd.DataFrame, cols: Iterable[str] = EVENT_COORDS) -> pd.DataFrame:
    """Ensure event coordinate columns are floats (not 'object'). Operates in-place and returns df."""
    present = [c for c in cols if c in events.columns]
    if not present:
        return events
    events.loc[:, present] = events.loc[:, present].apply(pd.to_numeric, errors="coerce")
    return events


def safe_for_arrows(events_slice: pd.DataFrame) -> pd.DataFrame:
    """Arrows need all 4 coords. Markers only need Start X/Y."""
    return events_slice.dropna(subset=EVENT_COORDS).copy()


def plot_single_event_with_arrow(events: pd.DataFrame, event_id: int, *, title: str = "") -> None:
    """Always plots something if Start X/Y exist; draws arrow only if End X/Y exist."""
    row = events.loc[event_id]
    sx, sy = float(row["Start X"]), float(row["Start Y"])

    fig, ax = mviz.plot_pitch()
    ax.plot(sx, sy, "ro")

    ex, ey = row.get("End X", np.nan), row.get("End Y", np.nan)
    if np.isfinite(ex) and np.isfinite(ey):
        ax.annotate(
            "",
            xy=(float(ex), float(ey)),
            xytext=(sx, sy),
            alpha=0.6,
            arrowprops=dict(arrowstyle="->", color="r"),
        )

    if title:
        ax.set_title(title)


# ----------------------------
# EVENTS DATA
# ----------------------------

require_datadir(DATADIR)

# Load (cached by your mio.read_event_data implementation)
events = mio.read_event_data(DATADIR, game_id)

# IMPORTANT: force numeric BEFORE to_metric_coordinates (prevents "object" garbage)
coerce_event_coords_numeric(events)

print(events["Type"].value_counts())

# Convert to meters (keep ONE dataframe: events)
# Your mio.to_metric_coordinates mutates and also returns df; we keep the canonical name "events".
events = mio.to_metric_coordinates(events)

# Quick sanity print: should now be within pitch scale
print("\nRAW/METRIC sanity for event", EVENT_ID_GOAL_1)
print(events.loc[EVENT_ID_GOAL_1, EVENT_COORDS])
print("Abs max:", events[EVENT_COORDS].abs().max())

# Plot the first goal (single event)
plot_single_event_with_arrow(events, EVENT_ID_GOAL_1, title=f"Goal event {EVENT_ID_GOAL_1}")

# Run-up to goal: show markers+arrows for rows that support arrows
runup_190_198 = safe_for_arrows(events.loc[190:198])
mviz.plot_events(runup_190_198, indicators=["Marker", "Arrow"], annotate=True)

runup_189_197 = safe_for_arrows(events.loc[189:197])
mviz.plot_events(runup_189_197, indicators=["Marker", "Arrow"], annotate=True)

# Split by team
home_events = events[events["Team"] == "Home"]
away_events = events[events["Team"] == "Away"]

print("\nHome event type counts:\n", home_events["Type"].value_counts())
print("\nAway event type counts:\n", away_events["Type"].value_counts())

# Shots
shots = events[events["Type"] == "SHOT"]
home_shots = home_events[home_events["Type"] == "SHOT"]
away_shots = away_events[away_events["Type"] == "SHOT"]

print("\nHome shot subtype counts:\n", home_shots["Subtype"].value_counts())
print("\nAway shot subtype counts:\n", away_shots["Subtype"].value_counts())
print("\nHome shots by player:\n", home_shots["From"].value_counts())

# Goals
home_goals = home_shots[home_shots["Subtype"].str.contains("-GOAL", na=False)].copy()
away_goals = away_shots[away_shots["Subtype"].str.contains("-GOAL", na=False)].copy()
home_goals["Minute"] = home_goals["Start Time [s]"] / 60.0

print("\nHome goals rows:\n", home_goals[["Start Time [s]", "Minute", "From", "Subtype"]].head())

# Plot the third goal (single event)
plot_single_event_with_arrow(events, EVENT_ID_GOAL_3, title=f"Goal event {EVENT_ID_GOAL_3}")

# NOTE: original tutorial had an invalid "annotate on slice" pattern.
# We keep the intent: plot the start points for a longer window.
fig, ax = mviz.plot_pitch()
ax.plot(events.loc[EVENT_ID_GOAL_3:1681]["Start X"], events.loc[EVENT_ID_GOAL_3:1681]["Start Y"], "ro")
ax.set_title(f"Start positions from event {EVENT_ID_GOAL_3} to 1681")

# small slice near goal (safe for arrows)
mviz.plot_events(safe_for_arrows(events.loc[1116:1117]), indicators=["Marker", "Arrow"], annotate=True)


# ----------------------------
# TRACKING DATA
# ----------------------------

tracking_home = mio.tracking_data(DATADIR, game_id, "Home")
tracking_away = mio.tracking_data(DATADIR, game_id, "Away")

print("\nTracking home columns:\n", tracking_home.columns)

tracking_home = mio.to_metric_coordinates(tracking_home)
tracking_away = mio.to_metric_coordinates(tracking_away)

# Player trajectories (first 1500 frames)
fig, ax = mviz.plot_pitch()
ax.plot(tracking_home["Home_11_x"].iloc[:1500], tracking_home["Home_11_y"].iloc[:1500], "r.", markersize=1)
ax.plot(tracking_home["Home_1_x"].iloc[:1500], tracking_home["Home_1_y"].iloc[:1500], "b.", markersize=1)
ax.plot(tracking_home["Home_2_x"].iloc[:1500], tracking_home["Home_2_y"].iloc[:1500], "g.", markersize=1)
ax.plot(tracking_home["Home_3_x"].iloc[:1500], tracking_home["Home_3_y"].iloc[:1500], "k.", markersize=1)
ax.plot(tracking_home["Home_4_x"].iloc[:1500], tracking_home["Home_4_y"].iloc[:1500], "c.", markersize=1)
ax.set_title("Home player trajectories (first 1500 frames)")

# Plot kickoff frame
KO_Frame = int(events.loc[0]["Start Frame"])
mviz.plot_frame(tracking_home.loc[KO_Frame], tracking_away.loc[KO_Frame])

# Plot positions at goal 1 with event overlay
fig, ax = mviz.plot_events(events.loc[EVENT_ID_GOAL_1:EVENT_ID_GOAL_1], indicators=["Marker", "Arrow"], annotate=True)
goal_frame = int(events.loc[EVENT_ID_GOAL_1]["Start Frame"])
mviz.plot_frame(tracking_home.loc[goal_frame], tracking_away.loc[goal_frame], figax=(fig, ax))

# Plot positions at goal 3 with event overlay
fig, ax = mviz.plot_events(events.loc[EVENT_ID_GOAL_3:EVENT_ID_GOAL_3], indicators=["Marker", "Arrow"], annotate=True)
goal_frame = int(events.loc[EVENT_ID_GOAL_3]["Start Frame"])
mviz.plot_frame(tracking_home.loc[goal_frame], tracking_away.loc[goal_frame], figax=(fig, ax))


# ----------------------------
# Run-up diagnostics (kept but safe)
# ----------------------------

runup = events.loc[1110:EVENT_ID_GOAL_3].copy()
print("\nRun-up head:\n", runup[["Type", "Subtype"] + EVENT_COORDS].head(30))

bad = runup[EVENT_COORDS].isna().any(axis=1)
print("\nBad rows:\n", runup.loc[bad, ["Type", "Subtype"] + EVENT_COORDS])
print("\ndtypes:\n", runup[EVENT_COORDS].dtypes)

arrow_ok = safe_for_arrows(runup)
mviz.plot_events(arrow_ok, indicators=["Marker", "Arrow"], annotate=True)


# ----------------------------
# Run-up plotting helpers (used below)
# ----------------------------

def plot_runup_v2(idx: int, window: int = 10) -> None:
    runup_local = events.loc[idx - window : idx].copy()
    runup_local = runup_local[runup_local["Type"].isin(["PASS", "SHOT"])].copy()

    players = runup_local["From"].dropna().unique().tolist()
    print(f"\nRun-up idx={idx}: players involved: {players}")

    # Use safe arrow slice so it never “looks blank” due to NaN end coords
    mviz.plot_events(safe_for_arrows(runup_local), indicators=["Marker", "Arrow"], annotate=True)


for idx in home_goals.index:
    plot_runup_v2(int(idx))


# ----------------------------
# Player 9 shot map + highlight goals
# ----------------------------

player_9_mask = home_shots["From"] == "Player9"
player_shots = home_shots.loc[player_9_mask].copy()

# Need coords for arrows
player_shots = safe_for_arrows(player_shots)

fig, ax = mviz.plot_pitch()
mviz.plot_events(player_shots, figax=(fig, ax), indicators=["Marker", "Arrow"], annotate=False)
ax.set_title("Player9 shots (all)")

is_goal = player_shots["Subtype"].str.contains("-GOAL", na=False)
shots_nongoal = player_shots[~is_goal]
shots_goal = player_shots[is_goal]

fig, ax = mviz.plot_pitch()
mviz.plot_events(shots_nongoal, figax=(fig, ax), indicators=["Marker", "Arrow"], annotate=False, alpha=0.25)
mviz.plot_events(shots_goal, figax=(fig, ax), indicators=["Marker", "Arrow"], annotate=False, alpha=0.9)
ax.set_title("Player9 shots: goals highlighted")


# ----------------------------
# Distance covered by team
# ----------------------------

def distance_covered_by_team(tracking: pd.DataFrame, team_prefix: str, *, step_cap_m: float = 2.0) -> pd.DataFrame:
    pat = re.compile(rf"^{re.escape(team_prefix)}_(\d+)_x$")
    player_ids = sorted({int(m.group(1)) for c in tracking.columns if (m := pat.match(c))})

    out_rows = []
    for pid in player_ids:
        x = pd.to_numeric(tracking[f"{team_prefix}_{pid}_x"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(tracking[f"{team_prefix}_{pid}_y"], errors="coerce").to_numpy(dtype=float)

        dx = np.diff(x)
        dy = np.diff(y)
        step = np.sqrt(dx * dx + dy * dy)

        if step_cap_m and step_cap_m > 0:
            step = np.minimum(step, step_cap_m)

        valid = np.isfinite(x[:-1]) & np.isfinite(y[:-1]) & np.isfinite(x[1:]) & np.isfinite(y[1:])
        dist_m = float(step[valid].sum())
        out_rows.append({"player": f"{team_prefix}_{pid}", "distance_m": dist_m, "distance_km": dist_m / 1000.0})

    return pd.DataFrame(out_rows).sort_values("distance_m", ascending=False).reset_index(drop=True)


home_dist = distance_covered_by_team(tracking_home, "Home")
away_dist = distance_covered_by_team(tracking_away, "Away")

print("\nTop 5 home distance:\n", home_dist.head(5))
print("\nTop 5 away distance:\n", away_dist.head(5))

print("\nEND")