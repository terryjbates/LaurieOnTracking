#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tutorial 4: EPV + Pitch Control (Metrica sample data)

Original: Mon Apr 13 11:34:26 2020 (Laurie Shaw, @EightyFivePoint)
Refactor: Jan 14, 2026

Goals of this refactor
- Runs top-to-bottom (no Spyder “run selection” required).
- Uses the refactored metrica modules (including safe file-level caching in Metrica_IO).
- Removes duplicate imports and conflicting DATADIR assignments.
- Keeps original outputs/plots while making paths + assumptions explicit and robust.

Requirements
- Metrica sample-data repo downloaded locally (DATADIR points to .../sample-data/data)
- metrica package on PYTHONPATH (metrica/Metrica_IO.py etc.)
- EPV_grid.csv available (default: alongside this script, or set EPV_GRID_PATH env var)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

import metrica.Metrica_IO as mio
import metrica.Metrica_Viz as mviz
import metrica.Metrica_Velocities as mvel
import metrica.Metrica_PitchControl as mpc
import metrica.Metrica_EPV as mepv


# ----------------------------
# Config
# ----------------------------

DATADIR = os.environ.get("METRICA_DATADIR", r"C:\Users\lover\github\sample-data\data")
game_id = 2

# EPV grid path: prefer env var, else "./EPV_grid.csv" next to this script.
_THIS_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
EPV_GRID_PATH = Path(os.environ.get("EPV_GRID_PATH", str(_THIS_DIR / "EPV_grid.csv")))

# Example events from the original tutorial
EVENT_AWAY_GOAL1_ASSIST = 822  # away team first goal pass leading to shot/goal sequence

# Examples later in tutorial
EVENT_HOME_ASSIST_HEADER_OFFTARGET = 1753
EVENT_AWAY_ASSIST_BLOCKED_SHOT = 1663
EVENT_RETAIN_POSSESSION = 195
EVENT_ASSIST_EXAMPLE = 1680
CROSSFIELD_EXAMPLES = [403, 68, 829]


def _require_dir(path: str) -> None:
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"DATADIR does not exist or is not a directory:\n  {path}\n"
            "Fix DATADIR (or set METRICA_DATADIR env var) to your local sample-data/data folder."
        )


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required file not found:\n  {path}\n"
            "Set EPV_GRID_PATH env var, or place EPV_grid.csv next to this script."
        )


# ----------------------------
# Load data
# ----------------------------

_require_dir(DATADIR)
_require_file(EPV_GRID_PATH)

# If you keep modules next to your data dir, this helps some IDE setups.
if DATADIR not in sys.path:
    sys.path.append(DATADIR)

# Cached CSV loads happen inside mio.read_event_data / mio.tracking_data
events = mio.read_event_data(DATADIR, game_id)
tracking_home = mio.tracking_data(DATADIR, game_id, "Home")
tracking_away = mio.tracking_data(DATADIR, game_id, "Away")

# Convert to meters
tracking_home = mio.to_metric_coordinates(tracking_home)
tracking_away = mio.to_metric_coordinates(tracking_away)
events = mio.to_metric_coordinates(events)

# Single playing direction (home always attacks right->left, per tutorial convention)
tracking_home, tracking_away, events = mio.to_single_playing_direction(tracking_home, tracking_away, events)

# Velocities (required for pitch control model)
tracking_home = mvel.calc_player_velocities(tracking_home, smoothing=True)
tracking_away = mvel.calc_player_velocities(tracking_away, smoothing=True)
# If you hit a numpy-specific error in your environment, use:
# tracking_home = mvel.calc_player_velocities(tracking_home, smoothing=True, filter_="moving_average")
# tracking_away = mvel.calc_player_velocities(tracking_away, smoothing=True, filter_="moving_average")


# ----------------------------
# Pitch control model setup (offsides enabled)
# ----------------------------

params = mpc.default_model_params()

# GK numbers used for offsides logic
GK_numbers = [mio.find_goalkeeper(tracking_home), mio.find_goalkeeper(tracking_away)]
print("GK_numbers:", GK_numbers)


# ----------------------------
# EPV surface
# ----------------------------

home_attack_direction = mio.find_playing_direction(tracking_home, "Home")  # +1 or -1
EPV = mepv.load_EPV_grid(str(EPV_GRID_PATH))

# Plot EPV surface
mviz.plot_EPV(EPV, field_dimen=(106.0, 68.0), attack_direction=home_attack_direction)


# ----------------------------
# Example: event sequence leading to away goal 1 (820–823)
# ----------------------------

mviz.plot_events(events.loc[820:823], color="k", indicators=["Marker", "Arrow"], annotate=True)

# Compute EPV added for the assist-like pass
event_number = EVENT_AWAY_GOAL1_ASSIST
EEPV_added, EPV_diff = mepv.calculate_epv_added(
    event_number, events, tracking_home, tracking_away, GK_numbers, EPV, params
)

PPCF, xgrid, ygrid = mpc.generate_pitch_control_for_event(
    event_number,
    events,
    tracking_home,
    tracking_away,
    params,
    GK_numbers,
    field_dimen=(106.0, 68.0),
    n_grid_cells_x=50,
    offsides=True,
)

fig, ax = mviz.plot_EPV_for_event(
    event_number,
    events,
    tracking_home,
    tracking_away,
    PPCF,
    EPV,
    annotate=True,
    autoscale=True,
)
fig.suptitle(f"Pass EPV added: {EEPV_added:1.3f}", y=0.95)

mviz.plot_pitchcontrol_for_event(
    event_number, events, tracking_home, tracking_away, PPCF, annotate=True
)


# ----------------------------
# Value-added for all passes (home and away)
# ----------------------------

shots = events[events["Type"] == "SHOT"]
home_shots = shots[shots["Team"] == "Home"]
away_shots = shots[shots["Team"] == "Away"]

home_passes = events[(events["Type"] == "PASS") & (events["Team"] == "Home")]
away_passes = events[(events["Type"] == "PASS") & (events["Team"] == "Away")]

home_pass_value_added = []
for i, _pass in home_passes.iterrows():
    EEPV_added, EPV_diff = mepv.calculate_epv_added(i, events, tracking_home, tracking_away, GK_numbers, EPV, params)
    home_pass_value_added.append((i, EEPV_added, EPV_diff))

away_pass_value_added = []
for i, _pass in away_passes.iterrows():
    EEPV_added, EPV_diff = mepv.calculate_epv_added(i, events, tracking_home, tracking_away, GK_numbers, EPV, params)
    away_pass_value_added.append((i, EEPV_added, EPV_diff))

home_pass_value_added = sorted(home_pass_value_added, key=lambda x: x[1], reverse=True)
away_pass_value_added = sorted(away_pass_value_added, key=lambda x: x[1], reverse=True)

print("\nTop 5 home team passes by expected EPV-added")
print(home_pass_value_added[:5])

print("\nTop 5 away team passes by expected EPV-added")
print(away_pass_value_added[:5])


# ----------------------------
# Specific examples (as in original tutorial)
# ----------------------------

def plot_event_epv_and_pitchcontrol(event_number: int, *, autoscale: bool = False, contours: bool = False) -> None:
    """Compute EPV-added + pitch control for an event and plot surfaces."""
    EEPV_added, EPV_diff = mepv.calculate_epv_added(
        event_number, events, tracking_home, tracking_away, GK_numbers, EPV, params
    )
    PPCF, xgrid, ygrid = mpc.generate_pitch_control_for_event(
        event_number,
        events,
        tracking_home,
        tracking_away,
        params,
        GK_numbers,
        field_dimen=(106.0, 68.0),
        n_grid_cells_x=50,
        offsides=True,
    )

    fig, ax = mviz.plot_EPV_for_event(
        event_number,
        events,
        tracking_home,
        tracking_away,
        PPCF,
        EPV,
        annotate=True,
        autoscale=autoscale,
        contours=contours,
    )
    fig.suptitle(f"Pass EPV added: {EEPV_added:1.3f}", y=0.95)

    mviz.plot_pitchcontrol_for_event(
        event_number, events, tracking_home, tracking_away, PPCF, annotate=True
    )


# home team assist to header off target
plot_event_epv_and_pitchcontrol(EVENT_HOME_ASSIST_HEADER_OFFTARGET)

# away team assist to blocked shot
plot_event_epv_and_pitchcontrol(EVENT_AWAY_ASSIST_BLOCKED_SHOT)

# retaining possession
plot_event_epv_and_pitchcontrol(EVENT_RETAIN_POSSESSION)

# assist example with autoscale + contours
plot_event_epv_and_pitchcontrol(EVENT_ASSIST_EXAMPLE, autoscale=True, contours=True)

# cross-field passes
for event_number in CROSSFIELD_EXAMPLES:
    plot_event_epv_and_pitchcontrol(event_number, autoscale=True, contours=True)

print("\nEND")
