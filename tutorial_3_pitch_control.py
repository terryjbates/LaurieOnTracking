#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tutorial 3 (FoT Lesson 6): Pitch Control

Original: Mon Apr 13 11:34:26 2020 (Laurie Shaw, @EightyFivePoint)
Refactor: Jan 14, 2026

Goals of this refactor
- Runs top-to-bottom without “run selection”.
- Uses your refactored Metrica_IO caching automatically (read_event_data / tracking_data).
- Ensures event + tracking coordinates are numeric and converted correctly (prevents blank pitch traps).
- Keeps original analysis flow:
  1) pitch control for passes leading up to goal 2 (events 820–822)
  2) pass-success (pitch control) probability for every successful Home pass
  3) highlight “risky” completed passes (Patt < 0.5) and inspect what followed

Notes / guardrails
- We coerce event coordinate columns to numeric BEFORE to_metric_coordinates.
- We drop NaN pass endpoints for pitch-control computations (target must be finite).
- calc_player_velocities is wrapped in a small try/except fallback (as in tutorial 2).
"""

from __future__ import annotations

import os
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import metrica.Metrica_IO as mio
import metrica.Metrica_Viz as mviz
import metrica.Metrica_Velocities as mvel
import metrica.Metrica_PitchControl as mpc


# ----------------------------
# Config
# ----------------------------

DATADIR = os.environ.get("METRICA_DATADIR", r"C:\Users\lover\github\sample-data\data")
game_id = 2

FIELD_DIMEN = (106.0, 68.0)
N_GRID_CELLS_X = 50

# In the original tutorial these are the 3 passes leading up to the second goal (match 2)
LEADUP_EVENT_IDS = [820, 821, 822]


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


def safe_calc_player_velocities(tracking: pd.DataFrame) -> pd.DataFrame:
    """Match the tutorial note: try default smoothing, then fallback filter_ arg."""
    try:
        return mvel.calc_player_velocities(tracking, smoothing=True)
    except TypeError:
        return mvel.calc_player_velocities(tracking, smoothing=True, filter_="moving_average")


def safe_plot_pitchcontrol_for_event(event_id: int, PPCF: np.ndarray, events: pd.DataFrame,
                                     tracking_home: pd.DataFrame, tracking_away: pd.DataFrame) -> None:
    """
    Your refactored Metrica_Viz.plot_pitchcontrol_for_event does NOT require xgrid/ygrid.
    This wrapper just makes the intent explicit.
    """
    mviz.plot_pitchcontrol_for_event(
        event_id,
        events,
        tracking_home,
        tracking_away,
        PPCF,
        annotate=True,
        field_dimen=FIELD_DIMEN,
    )


# ----------------------------
# Load data (cached)
# ----------------------------

require_datadir(DATADIR)

events = mio.read_event_data(DATADIR, game_id)
tracking_home = mio.tracking_data(DATADIR, game_id, "Home")
tracking_away = mio.tracking_data(DATADIR, game_id, "Away")

# Make sure event coords are numeric BEFORE coordinate conversion (prevents “blank pitch / huge numbers” issues)
coerce_cols_numeric(events, EVENT_COORDS)

# Convert positions to meters
tracking_home = mio.to_metric_coordinates(tracking_home, field_dimen=FIELD_DIMEN)
tracking_away = mio.to_metric_coordinates(tracking_away, field_dimen=FIELD_DIMEN)
events = mio.to_metric_coordinates(events, field_dimen=FIELD_DIMEN)

# Single playing direction (mutates; rebind for clarity)
tracking_home, tracking_away, events = mio.to_single_playing_direction(tracking_home, tracking_away, events)

# Velocities (needed by pitch control model)
tracking_home = safe_calc_player_velocities(tracking_home)
tracking_away = safe_calc_player_velocities(tracking_away)


# ----------------------------
# Pitch control for passes leading up to goal 2
# ----------------------------

shots = events[events["Type"] == "SHOT"]
goals = shots[shots["Subtype"].str.contains("-GOAL", na=False)].copy()

print("\n--- Goals (from SHOT Subtype contains '-GOAL') ---")
print(goals)

print("\n--- Plot events leading up to goal 2 (820:823) ---")
mviz.plot_events(events.loc[820:823], color="k", indicators=["Marker", "Arrow"], annotate=True)

# Pitch control model params
params = mpc.default_model_params()

# Goalkeepers for offside calcs
GK_numbers = [mio.find_goalkeeper(tracking_home), mio.find_goalkeeper(tracking_away)]
print("\nGK numbers (Home, Away):", GK_numbers)

for eid in LEADUP_EVENT_IDS:
    PPCF, xgrid, ygrid = mpc.generate_pitch_control_for_event(
        eid,
        events,
        tracking_home,
        tracking_away,
        params,
        GK_numbers,
        field_dimen=FIELD_DIMEN,
        n_grid_cells_x=N_GRID_CELLS_X,
    )
    safe_plot_pitchcontrol_for_event(eid, PPCF, events, tracking_home, tracking_away)


# ----------------------------
# Pass success probability for every Home PASS
# ----------------------------

# Home passes (original tutorial says “successful pass”, but uses all PASS rows in events)
home_passes = events[(events["Type"].isin(["PASS"])) & (events["Team"] == "Home")].copy()

# Need finite start/end coords + start frame
needed = ["Start X", "Start Y", "End X", "End Y", "Start Frame"]
missing = [c for c in needed if c not in home_passes.columns]
if missing:
    raise KeyError(f"Events missing columns needed for pass probability: {missing}")

# Some passes can have NaNs for end coords; those cannot be evaluated.
home_passes = home_passes.dropna(subset=["Start X", "Start Y", "End X", "End Y", "Start Frame"]).copy()

pass_success_probability: List[Tuple[int, float]] = []

for i, row in home_passes.iterrows():
    pass_start_pos = np.array([row["Start X"], row["Start Y"]], dtype=float)
    pass_target_pos = np.array([row["End X"], row["End Y"]], dtype=float)
    pass_frame = int(row["Start Frame"])

    # Initialise players at the pass frame
    attacking_players = mpc.initialise_players(tracking_home.loc[pass_frame], "Home", params, GK_numbers[0])
    defending_players = mpc.initialise_players(tracking_away.loc[pass_frame], "Away", params, GK_numbers[1])

    Patt, Pdef = mpc.calculate_pitch_control_at_target(
        pass_target_pos,
        attacking_players,
        defending_players,
        pass_start_pos,
        params,
    )

    pass_success_probability.append((int(i), float(Patt)))

# Histogram of pass success probabilities
fig, ax = plt.subplots()
ax.hist([p[1] for p in pass_success_probability], bins=np.arange(0, 1.1, 0.1))
ax.set_xlabel("Pass success probability (Home control at target)")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of pitch-control-based pass success probabilities")


# ----------------------------
# Risky passes (Patt < 0.5) + what happened next
# ----------------------------

pass_success_probability_sorted = sorted(pass_success_probability, key=lambda x: x[1])

risky_ids = [pid for (pid, p) in pass_success_probability_sorted if p < 0.5]
risky_passes = events.loc[risky_ids].copy() if risky_ids else events.iloc[0:0].copy()

print(f"\nNumber of Home passes evaluated: {len(pass_success_probability)}")
print(f"Number of risky completed Home passes (Patt < 0.5): {len(risky_passes)}")

if len(risky_passes) > 0:
    mviz.plot_events(risky_passes, color="k", indicators=["Marker", "Arrow"], annotate=True)
else:
    print("No risky passes found under the Patt < 0.5 threshold in this match slice.")

print("\nEvent following a risky (completed) pass (lowest Patt first)")
# Print a limited number like the original tutorial (first 20 after sorting)
for pid, patt in pass_success_probability_sorted[:20]:
    next_id = pid + 1
    if next_id in events.index:
        outcome = str(events.loc[next_id].Type)
    else:
        outcome = "<no next event>"
    print(f"{patt:.3f} -> {outcome}")

print("\nEND")
