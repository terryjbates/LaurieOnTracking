#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:41:01 2020

Module for exploring expected possession value (EPV) surfaces using MetricaSports's tracking & event data.

EPV is the probability that a possession will end with a goal given the current location of the ball. Multiplying by a
pitch control surface gives the expected value of moving the ball to any location, accounting for the probability that the 
ball move (pass/carry) is successful.

The EPV surface is saved in the FoT github repo and can be loaded using load_EPV_grid()

A detailed description of EPV can be found in the accompanying video tutorial here: 
    
GitHub repo for this code can be found here:
https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking

Data can be found at: https://github.com/metrica-sports/sample-data

Main Functions
----------

load_EPV_grid(): load pregenerated EPV surface from file. 
calculate_epv_added(): Calculates the expected possession value added by a pass
find_max_value_added_target(): Finds the *maximum* expected possession value that could have been achieved for a pass (defined by the event_id) by searching the entire field for the best target.
    

@author: Laurie Shaw (@EightyFivePoint)

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple

import numpy as np
import pandas as pd

import Metrica_IO as mio
import Metrica_PitchControl as mpc

FieldDimen = Tuple[float, float]


# -----------------------------
# Public API (unchanged)
# -----------------------------

def load_EPV_grid(fname: str = "EPV_grid.csv") -> np.ndarray:
    """Load pregenerated EPV surface from file."""
    return np.loadtxt(fname, delimiter=",")


def get_EPV_at_location(
    position: Tuple[float, float] | np.ndarray,
    EPV: np.ndarray,
    attack_direction: int,
    field_dimen: FieldDimen = (106.0, 68.0),
) -> float:
    """Return the EPV value at a given (x, y) location.

    Notes
    -----
    - Returns 0.0 if position is off the field.
    - If attack_direction == -1, EPV grid is mirrored left-right.
    """
    x, y = float(position[0]), float(position[1])
    half_len, half_wid = field_dimen[0] / 2.0, field_dimen[1] / 2.0
    if (abs(x) > half_len) or (abs(y) > half_wid):
        return 0.0

    epv = np.fliplr(EPV) if attack_direction == -1 else EPV
    ny, nx = epv.shape
    dx = field_dimen[0] / float(nx)
    dy = field_dimen[1] / float(ny)

    eps = 1e-4
    ix = int((x + half_len - eps) / dx)
    iy = int((y + half_wid - eps) / dy)

    ix = 0 if ix < 0 else (nx - 1 if ix >= nx else ix)
    iy = 0 if iy < 0 else (ny - 1 if iy >= ny else iy)

    return float(epv[iy, ix])


# -----------------------------
# Caching layer (new, opt-in)
# -----------------------------

@dataclass
class EPVCache:
    """In-process cache to speed up repeated EPV queries within a single match/session.

    This cache is SAFE to reuse across many calls as long as you keep using the same:
      - events / tracking_home / tracking_away DataFrames
      - GK_numbers and params
      - EPV grid

    It does NOT persist across Python restarts (for persistence, cache outputs to disk yourself).
    """
    home_attack_direction: Optional[int] = None

    # Pre-flipped EPV grids (avoid repeated np.fliplr)
    epv_lr: Optional[np.ndarray] = None  # attack left->right
    epv_rl: Optional[np.ndarray] = None  # attack right->left (flipped)

    # Player initialisations are expensive. Cache them per (frame, pass_team).
    # Value: (attacking_players, defending_players, attack_direction)
    players_by_frame: Dict[Tuple[int, str], Tuple[Any, Any, int]] = field(default_factory=dict)

    def prime_epv(self, EPV: np.ndarray) -> None:
        """Precompute both EPV orientations once."""
        self.epv_lr = EPV
        self.epv_rl = np.fliplr(EPV)

    def epv_for_direction(self, attack_direction: int) -> np.ndarray:
        """Return cached EPV grid for the given attack direction."""
        if self.epv_lr is None or self.epv_rl is None:
            raise ValueError("EPVCache not primed. Call cache.prime_epv(EPV) first.")
        return self.epv_lr if attack_direction == 1 else self.epv_rl


def calculate_epv_added(
    event_id: int,
    events: pd.DataFrame,
    tracking_home: pd.DataFrame,
    tracking_away: pd.DataFrame,
    GK_numbers: Tuple[int, int],
    EPV: np.ndarray,
    params: Mapping[str, Any],
    cache: Optional[EPVCache] = None,
) -> Tuple[float, float]:
    """Calculate expected possession value (EPV) added by a pass.

    Parameters are identical to the original implementation, with one optional addition:
    cache: EPVCache | None
        If provided, expensive intermediate computations are reused across calls.

    Returns
    -------
    EEPV_added:
        Pitch-control-weighted EPV value-added of pass defined by event_id
    EPV_difference:
        Raw EPV change (ignoring pitch control)
    """
    ctx = _build_pass_context(event_id, events, tracking_home, tracking_away, GK_numbers, params, cache)

    Patt_start, _ = mpc.calculate_pitch_control_at_target(
        ctx.pass_start_pos, ctx.attacking_players, ctx.defending_players, ctx.pass_start_pos, params
    )
    Patt_target, _ = mpc.calculate_pitch_control_at_target(
        ctx.pass_target_pos, ctx.attacking_players, ctx.defending_players, ctx.pass_start_pos, params
    )

    # Use pre-flipped EPV if cache is primed; otherwise fall back to flipping inside get_EPV_at_location.
    if cache is not None and cache.epv_lr is not None and cache.epv_rl is not None:
        epv_grid = cache.epv_for_direction(ctx.attack_direction)
        EPV_start = _get_EPV_at_location_with_grid(ctx.pass_start_pos, epv_grid)
        EPV_target = _get_EPV_at_location_with_grid(ctx.pass_target_pos, epv_grid)
    else:
        EPV_start = get_EPV_at_location(ctx.pass_start_pos, EPV, attack_direction=ctx.attack_direction)
        EPV_target = get_EPV_at_location(ctx.pass_target_pos, EPV, attack_direction=ctx.attack_direction)

    EEPV_start = Patt_start * EPV_start
    EEPV_target = Patt_target * EPV_target

    return (EEPV_target - EEPV_start), (EPV_target - EPV_start)


def find_max_value_added_target(
    event_id: int,
    events: pd.DataFrame,
    tracking_home: pd.DataFrame,
    tracking_away: pd.DataFrame,
    GK_numbers: Tuple[int, int],
    EPV: np.ndarray,
    params: Mapping[str, Any],
    cache: Optional[EPVCache] = None,
) -> Tuple[float, Tuple[float, float]]:
    """Find the *maximum* expected possession value achievable for a pass, given the instant of the event.

    Parameters are identical to the original implementation, with one optional addition:
    cache: EPVCache | None
        If provided, expensive intermediate computations are reused across calls.
    """
    ctx = _build_pass_context(event_id, events, tracking_home, tracking_away, GK_numbers, params, cache)

    Patt_start, _ = mpc.calculate_pitch_control_at_target(
        ctx.pass_start_pos, ctx.attacking_players, ctx.defending_players, ctx.pass_start_pos, params
    )

    if cache is not None and cache.epv_lr is not None and cache.epv_rl is not None:
        epv_grid = cache.epv_for_direction(ctx.attack_direction)
        EPV_start = _get_EPV_at_location_with_grid(ctx.pass_start_pos, epv_grid)
    else:
        EPV_start = get_EPV_at_location(ctx.pass_start_pos, EPV, attack_direction=ctx.attack_direction)

    EEPV_start = Patt_start * EPV_start

    PPCF, xgrid, ygrid = mpc.generate_pitch_control_for_event(
        event_id,
        events,
        tracking_home,
        tracking_away,
        params,
        GK_numbers,
        field_dimen=(106.0, 68.0),
        n_grid_cells_x=50,
        offsides=True,
    )

    if cache is not None and cache.epv_lr is not None and cache.epv_rl is not None:
        epv_grid = cache.epv_for_direction(ctx.attack_direction)
        EEPV_surface = epv_grid * PPCF
    else:
        epv_grid = np.fliplr(EPV) if ctx.attack_direction == -1 else EPV
        EEPV_surface = epv_grid * PPCF

    flat_idx = int(np.nanargmax(EEPV_surface))
    iy, ix = np.unravel_index(flat_idx, EEPV_surface.shape)

    maxEEPV = float(EEPV_surface[iy, ix])
    maxEPV_added = maxEEPV - float(EEPV_start)
    max_target_location = (float(xgrid[ix]), float(ygrid[iy]))

    return maxEPV_added, max_target_location


# -----------------------------
# Internals
# -----------------------------

@dataclass(frozen=True)
class _PassContext:
    pass_start_pos: np.ndarray
    pass_target_pos: np.ndarray
    pass_frame: int
    pass_team: str
    attack_direction: int
    attacking_players: Any
    defending_players: Any


def _get_home_attack_direction(
    tracking_home: pd.DataFrame,
    cache: Optional[EPVCache],
) -> int:
    if cache is not None and cache.home_attack_direction is not None:
        return cache.home_attack_direction

    direction = int(mio.find_playing_direction(tracking_home, "Home"))
    if cache is not None:
        cache.home_attack_direction = direction
    return direction


def _get_EPV_at_location_with_grid(
    position: np.ndarray,
    epv_grid: np.ndarray,
    field_dimen: FieldDimen = (106.0, 68.0),
) -> float:
    """Fast EPV lookup when you already have the correctly oriented EPV grid."""
    x, y = float(position[0]), float(position[1])
    half_len, half_wid = field_dimen[0] / 2.0, field_dimen[1] / 2.0
    if (abs(x) > half_len) or (abs(y) > half_wid):
        return 0.0

    ny, nx = epv_grid.shape
    dx = field_dimen[0] / float(nx)
    dy = field_dimen[1] / float(ny)
    eps = 1e-4

    ix = int((x + half_len - eps) / dx)
    iy = int((y + half_wid - eps) / dy)

    ix = 0 if ix < 0 else (nx - 1 if ix >= nx else ix)
    iy = 0 if iy < 0 else (ny - 1 if iy >= ny else iy)

    return float(epv_grid[iy, ix])


def _build_pass_context(
    event_id: int,
    events: pd.DataFrame,
    tracking_home: pd.DataFrame,
    tracking_away: pd.DataFrame,
    GK_numbers: Tuple[int, int],
    params: Mapping[str, Any],
    cache: Optional[EPVCache],
) -> _PassContext:
    row = events.loc[event_id]

    pass_start_pos = np.array([row["Start X"], row["Start Y"]], dtype=float)
    pass_target_pos = np.array([row["End X"], row["End Y"]], dtype=float)
    pass_frame = int(row["Start Frame"])
    pass_team = str(row["Team"])

    home_attack_direction = _get_home_attack_direction(tracking_home, cache)

    # Reuse expensive player initializations per frame/team if possible
    if cache is not None:
        key = (pass_frame, pass_team)
        cached = cache.players_by_frame.get(key)
        if cached is not None:
            attacking_players, defending_players, attack_direction = cached
            return _PassContext(
                pass_start_pos=pass_start_pos,
                pass_target_pos=pass_target_pos,
                pass_frame=pass_frame,
                pass_team=pass_team,
                attack_direction=attack_direction,
                attacking_players=attacking_players,
                defending_players=defending_players,
            )

    if pass_team == "Home":
        attack_direction = home_attack_direction
        attacking_players = mpc.initialise_players(
            tracking_home.loc[pass_frame], "Home", params, GK_numbers[0]
        )
        defending_players = mpc.initialise_players(
            tracking_away.loc[pass_frame], "Away", params, GK_numbers[1]
        )
    else:  # "Away"
        attack_direction = -home_attack_direction
        defending_players = mpc.initialise_players(
            tracking_home.loc[pass_frame], "Home", params, GK_numbers[0]
        )
        attacking_players = mpc.initialise_players(
            tracking_away.loc[pass_frame], "Away", params, GK_numbers[1]
        )

    attacking_players = mpc.check_offsides(
        attacking_players, defending_players, pass_start_pos, GK_numbers
    )

    if cache is not None:
        cache.players_by_frame[(pass_frame, pass_team)] = (
            attacking_players,
            defending_players,
            attack_direction,
        )

    return _PassContext(
        pass_start_pos=pass_start_pos,
        pass_target_pos=pass_target_pos,
        pass_frame=pass_frame,
        pass_team=pass_team,
        attack_direction=attack_direction,
        attacking_players=attacking_players,
        defending_players=defending_players,
    )
