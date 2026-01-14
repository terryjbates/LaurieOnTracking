#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:52:19 2020

Module for calculating a Pitch Control surface using MetricaSports's tracking & event data.

Pitch control (at a given location on the field) is the probability that a team will gain
possession if the ball is moved to that location on the field.

Methdology is described in "Off the ball scoring opportunities" by William Spearman:
http://www.sloansportsconference.com/wp-content/uploads/2018/02/2002.pdf

GitHub repo for this code can be found here:
https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking

Data can be found at: https://github.com/metrica-sports/sample-data

Functions
----------

calculate_pitch_control_at_target(): calculate the pitch control probability for the attacking and defending teams at a specified target position on the ball.

generate_pitch_control_for_event(): this function evaluates pitch control surface over the entire field at the moment
of the given event (determined by the index of the event passed as an input)

Classes
---------

The 'player' class collects and stores trajectory information for each player required by the pitch control calculations.

@author: Laurie Shaw (@EightyFivePoint)

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Types
# -----------------------------

Array2 = np.ndarray  # shape (2,)
FieldDimen = Tuple[float, float]
GKNumbers = Tuple[int, int]
Params = Mapping[str, float]


# -----------------------------
# Optional caching (opt-in)
# -----------------------------

@dataclass
class PitchControlCache:
    """In-process cache to speed up repeated pitch-control calls in the same match/session.

    What it caches
    -------------
    - expensive `initialise_players(...)` results per (frame, teamname, GKid)

    Notes
    -----
    - This is *in-memory* only (resets on kernel restart).
    - Safe as long as you reuse the same tracking DataFrames and params.
    """
    players_by_frame: Dict[Tuple[int, str, int], List["player"]] = field(default_factory=dict)


# -----------------------------
# Core helpers (fast column parsing)
# -----------------------------

def _player_ids_from_row(team_row: pd.Series, teamname: str) -> np.ndarray:
    """Extract player jersey IDs from tracking row column names like 'Home_12_x'."""
    # Only columns for this team, and only x/y/vx/vy columns.
    ids: set[str] = set()
    prefix = f"{teamname}_"
    for c in team_row.index:
        if not c.startswith(prefix):
            continue
        # expected: Home_12_x / Home_12_vx etc.
        parts = c.split("_")
        if len(parts) >= 3 and parts[1]:
            ids.add(parts[1])
    return np.array(sorted(ids), dtype=object)


# -----------------------------
# Public API (same names; optional cache args added with defaults)
# -----------------------------

def initialise_players(
    team: pd.Series,
    teamname: str,
    params: Params,
    GKid: int,
    *,
    cache: Optional[PitchControlCache] = None,
) -> List["player"]:
    """Create player objects holding position/velocity from a single tracking row.

    Parameters
    ----------
    team:
        A single row (instant) of tracking data for the team.
    teamname:
        "Home" or "Away".
    params:
        Pitch control model parameters (see default_model_params()).
    GKid:
        Goalkeeper jersey number for this team.
    cache:
        Optional PitchControlCache for reuse across calls.

    Returns
    -------
    List[player]
        Player objects present in-frame.
    """
    # Cache key: use the *tracking frame index* if present; else fall back to object id.
    # In generate_pitch_control_for_event we pass `tracking.loc[frame]` so name is usually the frame.
    frame = int(team.name) if getattr(team, "name", None) is not None else -1
    if cache is not None and frame >= 0:
        key = (frame, teamname, int(GKid))
        cached = cache.players_by_frame.get(key)
        if cached is not None:
            return cached

    player_ids = _player_ids_from_row(team, teamname)
    out: List[player] = []
    for pid in player_ids:
        p = player(str(pid), team, teamname, params, int(GKid))
        if p.inframe:
            out.append(p)

    if cache is not None and frame >= 0:
        cache.players_by_frame[(frame, teamname, int(GKid))] = out

    return out


def check_offsides(
    attacking_players: List["player"],
    defending_players: List["player"],
    ball_position: Array2,
    GK_numbers: GKNumbers,
    verbose: bool = False,
    tol: float = 0.2,
) -> List["player"]:
    """Remove offside attacking players from the pitch control calculation."""
    if not attacking_players:
        return attacking_players

    # Which GK is defending (to infer attack direction)?
    defending_gk_id = GK_numbers[1] if attacking_players[0].teamname == "Home" else GK_numbers[0]

    # Pull the defending GK object (fast single pass)
    # --- Robustly identify defending GK (ID match, then fallback heuristic) ---

    # Determine which GK number we should be looking for based on defending team.
    # GK_numbers is expected like: [home_gk_num, away_gk_num]
    try:
        home_gk_num = str(GK_numbers[0])
        away_gk_num = str(GK_numbers[1])
    except Exception:
        home_gk_num = None
        away_gk_num = None

    defending_team = getattr(defending_players[0], "team", None) if defending_players else None
    target_gk_num = home_gk_num if defending_team == "Home" else away_gk_num

    defending_gk = None

    # 1) Try to match by jersey/id (string-normalized)
    if target_gk_num is not None:
        for p in defending_players:
            if str(getattr(p, "id", "")) == target_gk_num:
                defending_gk = p
                break

    # 2) Fallback: pick the defending player closest to their own goal (max abs(x))
    # This handles cases where GK is missing in tracking at this frame (NaNs → dropped).
    if defending_gk is None and len(defending_players) > 0:
        finite = [p for p in defending_players if np.isfinite(p.position[0]) and np.isfinite(p.position[1])]
        if len(finite) > 0:
            defending_gk = max(finite, key=lambda p: abs(float(p.position[0])))

            if verbose:
                print(
                    f"[check_offsides] GK id {target_gk_num} not found for {defending_team}; "
                    f"falling back to heuristic GK={getattr(defending_gk, 'id', '?')}"
                )

    if defending_gk is None:
        # At this point, we truly cannot determine defending GK; skip offside removal rather than crash.
        if verbose:
            print("[check_offsides] Could not identify defending GK; skipping offside check for this frame.")
        return attacking_players

    defending_half = float(np.sign(defending_gk.position[0]))  # -1: left goal, +1: right goal
    if defending_half == 0.0:
        # Rare edge case: GK at x==0. Default to "right goal" to avoid division by zero-ish logic.
        defending_half = 1.0

    # Second deepest defender (including GK). Multiply by defending_half so we can just "max".
    # Sort descending so [0] deepest, [1] second deepest.
    depths = sorted((defending_half * float(p.position[0]) for p in defending_players), reverse=True)
    if len(depths) < 2:
        # If tracking is weird and only one defender is present, fall back to that one.
        second_deepest = depths[0]
    else:
        second_deepest = depths[1]

    ball_x = float(ball_position[0]) if ball_position is not None else 0.0
    offside_line = max(second_deepest, defending_half * ball_x, 0.0) + float(tol)

    if verbose:
        for p in attacking_players:
            if float(p.position[0]) * defending_half > offside_line:
                print(f"player {p.id} in {p.playername} is offside")

    return [p for p in attacking_players if float(p.position[0]) * defending_half <= offside_line]


class player:
    """Player object for pitch control.

    Stores position/velocity, time-to-intercept, and cumulative pitch control contribution (PPCF).
    """

    __slots__ = (
        "id",
        "id_int",
        "is_gk",
        "teamname",
        "playername",
        "vmax",
        "reaction_time",
        "tti_sigma",
        "lambda_att",
        "lambda_def",
        "position",
        "velocity",
        "inframe",
        "PPCF",
        "time_to_intercept",
    )

    def __init__(self, pid: str, team: pd.Series, teamname: str, params: Params, GKid: int):
        self.id = pid
        self.id_int = int(pid) if pid.isdigit() else -1
        self.is_gk = (self.id_int == int(GKid))
        self.teamname = teamname
        self.playername = f"{teamname}_{pid}_"

        self.vmax = float(params["max_player_speed"])
        self.reaction_time = float(params["reaction_time"])
        self.tti_sigma = float(params["tti_sigma"])

        self.lambda_att = float(params["lambda_att"])
        self.lambda_def = float(params["lambda_gk"] if self.is_gk else params["lambda_def"])

        self.position = np.array([np.nan, np.nan], dtype=float)
        self.velocity = np.array([0.0, 0.0], dtype=float)
        self.inframe = False

        self.PPCF = 0.0
        self.time_to_intercept = np.inf

        self._set_position(team)
        self._set_velocity(team)

    def _set_position(self, team: pd.Series) -> None:
        x = float(team.get(self.playername + "x", np.nan))
        y = float(team.get(self.playername + "y", np.nan))
        self.position = np.array([x, y], dtype=float)
        self.inframe = not np.isnan(self.position).any()

    def _set_velocity(self, team: pd.Series) -> None:
        vx = team.get(self.playername + "vx", 0.0)
        vy = team.get(self.playername + "vy", 0.0)
        vx = 0.0 if (vx is None or (isinstance(vx, float) and np.isnan(vx))) else float(vx)
        vy = 0.0 if (vy is None or (isinstance(vy, float) and np.isnan(vy))) else float(vy)
        self.velocity = np.array([vx, vy], dtype=float)

    def simple_time_to_intercept(self, r_final: Array2) -> float:
        """Spearman-style simple time to intercept (reaction + full-speed run)."""
        self.PPCF = 0.0
        r_reaction = self.position + self.velocity * self.reaction_time
        self.time_to_intercept = self.reaction_time + (np.linalg.norm(r_final - r_reaction) / self.vmax)
        return float(self.time_to_intercept)

    def probability_intercept_ball(self, T: float) -> float:
        """Probability of intercept by time T (sigmoid around expected arrival time)."""
        # Keep as float ops (fast); no allocations.
        k = -np.pi / np.sqrt(3.0) / self.tti_sigma
        return float(1.0 / (1.0 + np.exp(k * (T - self.time_to_intercept))))


# -----------------------------
# Model parameters
# -----------------------------

def default_model_params(time_to_control_veto: float = 3.0) -> Dict[str, float]:
    """Return default parameters for Spearman (2018)-style pitch control model."""
    params: Dict[str, float] = {}
    params["max_player_accel"] = 7.0
    params["max_player_speed"] = 5.0
    params["reaction_time"] = 0.7
    params["tti_sigma"] = 0.45
    params["kappa_def"] = 1.0
    params["lambda_att"] = 4.3
    params["lambda_def"] = 4.3 * params["kappa_def"]
    params["lambda_gk"] = params["lambda_def"] * 3.0
    params["average_ball_speed"] = 15.0
    params["int_dt"] = 0.04
    params["max_int_time"] = 10.0
    params["model_converge_tol"] = 0.01

    # Shortcut thresholds
    common = np.log(10.0) * (np.sqrt(3.0) * params["tti_sigma"] / np.pi)
    params["time_to_control_att"] = time_to_control_veto * (common + 1.0 / params["lambda_att"])
    params["time_to_control_def"] = time_to_control_veto * (common + 1.0 / params["lambda_def"])
    return params


# -----------------------------
# Pitch control computations
# -----------------------------

def generate_pitch_control_for_event(
    event_id: int,
    events: pd.DataFrame,
    tracking_home: pd.DataFrame,
    tracking_away: pd.DataFrame,
    params: Params,
    GK_numbers: GKNumbers,
    field_dimen: FieldDimen = (106.0, 68.0),
    n_grid_cells_x: int = 50,
    offsides: bool = True,
    *,
    cache: Optional[PitchControlCache] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate pitch control surface over the entire field at the moment of `event_id`."""
    row = events.loc[event_id]
    pass_frame = int(row["Start Frame"])
    pass_team = str(row["Team"])
    ball_start_pos = np.array([row["Start X"], row["Start Y"]], dtype=float)

    # Grid: centers of cells across field
    n_grid_cells_y = int(n_grid_cells_x * field_dimen[1] / field_dimen[0])
    dx = field_dimen[0] / float(n_grid_cells_x)
    dy = field_dimen[1] / float(n_grid_cells_y)

    xgrid = (np.arange(n_grid_cells_x, dtype=float) * dx) - (field_dimen[0] / 2.0) + (dx / 2.0)
    ygrid = (np.arange(n_grid_cells_y, dtype=float) * dy) - (field_dimen[1] / 2.0) + (dy / 2.0)

    PPCFa = np.zeros((n_grid_cells_y, n_grid_cells_x), dtype=float)
    PPCFd = np.zeros((n_grid_cells_y, n_grid_cells_x), dtype=float)

    # Initialise players once at this frame (this is where caching helps the most)
    if pass_team == "Home":
        attacking_players = initialise_players(tracking_home.loc[pass_frame], "Home", params, GK_numbers[0], cache=cache)
        defending_players = initialise_players(tracking_away.loc[pass_frame], "Away", params, GK_numbers[1], cache=cache)
    elif pass_team == "Away":
        defending_players = initialise_players(tracking_home.loc[pass_frame], "Home", params, GK_numbers[0], cache=cache)
        attacking_players = initialise_players(tracking_away.loc[pass_frame], "Away", params, GK_numbers[1], cache=cache)
    else:
        raise ValueError("Team in possession must be either 'Home' or 'Away'")

    if offsides:
        attacking_players = check_offsides(attacking_players, defending_players, ball_start_pos, GK_numbers)

    # Evaluate surface (still the dominant cost; keep it tight)
    # We avoid allocating target_position in the hot loop by reusing an array.
    target = np.empty(2, dtype=float)
    for i, y in enumerate(ygrid):
        target[1] = y
        for j, x in enumerate(xgrid):
            target[0] = x
            a, d = calculate_pitch_control_at_target(target, attacking_players, defending_players, ball_start_pos, params)
            PPCFa[i, j] = a
            PPCFd[i, j] = d

    checksum = float(np.sum(PPCFa + PPCFd) / (n_grid_cells_y * n_grid_cells_x))
    if (1.0 - checksum) >= float(params["model_converge_tol"]):
        raise AssertionError(f"Checksum failed: {1.0 - checksum:1.3f}")

    return PPCFa, xgrid, ygrid


def calculate_pitch_control_at_target(
    target_position: Array2,
    attacking_players: List[player],
    defending_players: List[player],
    ball_start_pos: Optional[Array2],
    params: Params,
) -> Tuple[float, float]:
    """Calculate pitch control probability for attacking and defending teams at a target location."""
    # Ball travel time
    if ball_start_pos is None or np.isnan(ball_start_pos).any():
        ball_travel_time = 0.0
    else:
        ball_travel_time = float(np.linalg.norm(target_position - ball_start_pos) / params["average_ball_speed"])

    # Compute time-to-intercept for all players and find minima
    tau_min_att = np.inf
    for p in attacking_players:
        tti = p.simple_time_to_intercept(target_position)
        if tti < tau_min_att:
            tau_min_att = tti

    tau_min_def = np.inf
    for p in defending_players:
        tti = p.simple_time_to_intercept(target_position)
        if tti < tau_min_def:
            tau_min_def = tti

    # Short-circuit if a team has a decisive head-start
    max_bt_def = max(ball_travel_time, tau_min_def)
    max_bt_att = max(ball_travel_time, tau_min_att)

    if (tau_min_att - max_bt_def) >= float(params["time_to_control_def"]):
        return 0.0, 1.0
    if (tau_min_def - max_bt_att) >= float(params["time_to_control_att"]):
        return 1.0, 0.0

    # Filter players too far behind the best arrival time (reduces work)
    ttc_att = float(params["time_to_control_att"])
    ttc_def = float(params["time_to_control_def"])
    att = [p for p in attacking_players if (p.time_to_intercept - tau_min_att) < ttc_att]
    deff = [p for p in defending_players if (p.time_to_intercept - tau_min_def) < ttc_def]

    int_dt = float(params["int_dt"])
    max_int_time = float(params["max_int_time"])
    tol = float(params["model_converge_tol"])

    # Integration times (start slightly before ball arrives)
    dT = np.arange(ball_travel_time - int_dt, ball_travel_time + max_int_time, int_dt, dtype=float)

    # Running totals (scalar is enough; arrays were unnecessary overhead in the original)
    p_att = 0.0
    p_def = 0.0
    p_tot = 0.0

    # Integrate until convergence
    # Note: player.PPCF is cumulative; we follow the original logic but keep operations tight.
    for i in range(1, dT.size):
        if (1.0 - p_tot) <= tol:
            break

        T = float(dT[i])
        # Use previous totals (p_att, p_def) for (1 - ... ) term
        one_minus = 1.0 - p_att - p_def
        if one_minus < 0.0:
            one_minus = 0.0  # numerical guard

        # Attacking contributions
        sum_att = 0.0
        for pl in att:
            dP = one_minus * pl.probability_intercept_ball(T) * pl.lambda_att
            if dP < 0.0:
                raise AssertionError("Invalid attacking player probability (calculate_pitch_control_at_target)")
            pl.PPCF += dP * int_dt
            sum_att += pl.PPCF

        # Defending contributions
        sum_def = 0.0
        for pl in deff:
            dP = one_minus * pl.probability_intercept_ball(T) * pl.lambda_def
            if dP < 0.0:
                raise AssertionError("Invalid defending player probability (calculate_pitch_control_at_target)")
            pl.PPCF += dP * int_dt
            sum_def += pl.PPCF

        p_att = sum_att
        p_def = sum_def
        p_tot = p_att + p_def

    # If we didn’t converge, keep the best estimate
    if (1.0 - p_tot) > tol:
        # Keep a single print (original had one). You can silence by removing.
        print(f"Integration failed to converge: {p_tot:1.3f}")

    return float(p_att), float(p_def)
