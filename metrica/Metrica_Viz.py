#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 09:10:58 2020

Module for visualising Metrica tracking and event data

Data can be found at: https://github.com/metrica-sports/sample-data

UPDATE for tutorial 4: plot_pitchcontrol_for_event no longer requires 'xgrid' and 'ygrid' as inputs.

@author: Laurie Shaw (@EightyFivePoint)

Refactor notes (2026):
- Preserves original public function names + signatures (including figax).
- Faster plotting paths (itertuples, precomputed column lists where possible).
- More robust: silently skips NaN event coordinates instead of “plotting nothing”.
- Type hinted, cleaner internals, and consistent behavior for Spyder “run sections”.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Sequence, Tuple, Union, Optional, List

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import Metrica_IO as mio

try:
    import pandas as pd
    from pandas import DataFrame, Series
except Exception:  # pragma: no cover (keeps tutorial friendliness)
    pd = None
    DataFrame = object  # type: ignore
    Series = object  # type: ignore


# ----------------------------
# Internal helpers
# ----------------------------

FieldDimen = Tuple[float, float]
FigAx = Tuple["plt.Figure", "plt.Axes"]


def _is_nan(x: object) -> bool:
    try:
        return bool(np.isnan(x))  # type: ignore[arg-type]
    except Exception:
        return False


@lru_cache(maxsize=32)
def _pitch_geometry(field_dimen: FieldDimen) -> dict:
    """
    Precompute pitch geometry arrays for the given field dimensions.

    Returns a dict of arrays (in meters) that plot_pitch uses.
    """
    meters_per_yard = 0.9144

    half_L = field_dimen[0] / 2.0
    half_W = field_dimen[1] / 2.0

    # Standard markings (converted from yards)
    goal_line_width = 8 * meters_per_yard
    box_width = 20 * meters_per_yard
    box_length = 6 * meters_per_yard
    area_width = 44 * meters_per_yard
    area_length = 18 * meters_per_yard
    penalty_spot = 12 * meters_per_yard
    corner_radius = 1 * meters_per_yard
    D_length = 8 * meters_per_yard
    D_radius = 10 * meters_per_yard
    D_pos = 12 * meters_per_yard
    centre_circle_radius = 10 * meters_per_yard

    # Circle + arcs samples
    y_cc = np.linspace(-1, 1, 80) * centre_circle_radius
    x_cc = np.sqrt(np.maximum(centre_circle_radius**2 - y_cc**2, 0.0))

    y_corner = np.linspace(0, 1, 60) * corner_radius
    x_corner = np.sqrt(np.maximum(corner_radius**2 - y_corner**2, 0.0))

    y_D = np.linspace(-1, 1, 80) * D_length
    x_D = np.sqrt(np.maximum(D_radius**2 - y_D**2, 0.0)) + D_pos

    return {
        "half_L": half_L,
        "half_W": half_W,
        "goal_line_width": goal_line_width,
        "box_width": box_width,
        "box_length": box_length,
        "area_width": area_width,
        "area_length": area_length,
        "penalty_spot": penalty_spot,
        "corner_radius": corner_radius,
        "D_length": D_length,
        "D_radius": D_radius,
        "D_pos": D_pos,
        "centre_circle_radius": centre_circle_radius,
        "x_cc": x_cc,
        "y_cc": y_cc,
        "x_corner": x_corner,
        "y_corner": y_corner,
        "x_D": x_D,
        "y_D": y_D,
    }


def _player_xy_columns(row: "Series") -> Tuple[List[str], List[str]]:
    """
    From a tracking row, identify player x/y columns (excluding ball_x/ball_y).
    Matches columns like Home_11_x, Away_2_y.
    """
    keys = row.keys()
    x_cols = [c for c in keys if str(c).lower().endswith("_x") and c != "ball_x"]
    y_cols = [c for c in keys if str(c).lower().endswith("_y") and c != "ball_y"]
    # Keep ordering stable (x_cols corresponds to y_cols by prefix)
    x_cols.sort()
    y_cols.sort()
    return x_cols, y_cols


def _velocity_columns_from_xy(x_cols: Sequence[str], y_cols: Sequence[str]) -> Tuple[List[str], List[str]]:
    vx_cols = [f"{c[:-2]}_vx" for c in x_cols]
    vy_cols = [f"{c[:-2]}_vy" for c in y_cols]
    return vx_cols, vy_cols


def _ensure_events_df(events: Union["DataFrame", "Series"]) -> "DataFrame":
    """Accept a single event row (Series) or event DataFrame, return DataFrame."""
    if pd is None:
        raise RuntimeError("pandas is required for plot_events (events is a DataFrame/Series).")
    if isinstance(events, pd.Series):
        return events.to_frame().T
    return events


# ----------------------------
# Public API (preserved)
# ----------------------------

def plot_pitch(field_dimen: FieldDimen = (106.0, 68.0), field_color: str = "green", linewidth: float = 2,
               markersize: float = 20) -> FigAx:
    """plot_pitch

    Plots a soccer pitch. All distance units converted to meters.

    Parameters
    -----------
        field_dimen: (length, width) of field in meters. Default is (106,68)
        field_color: color of field. options are {'green','white'}
        linewidth  : width of lines. default = 2
        markersize : size of markers (e.g. penalty spot, centre spot, posts). default = 20

    Returrns
    -----------
       fig,ax : figure and axis objects (so that other data can be plotted onto the pitch)
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    if field_color == "green":
        ax.set_facecolor("mediumseagreen")
        lc = "whitesmoke"
        pc = "w"
    elif field_color == "white":
        lc = "k"
        pc = "k"
    else:
        # keep behavior permissive: default to green-like lines
        ax.set_facecolor("mediumseagreen")
        lc = "whitesmoke"
        pc = "w"

    g = _pitch_geometry(field_dimen)
    half_L = g["half_L"]
    half_W = g["half_W"]

    # Border around field
    border_dimen = (3.0, 3.0)
    xmax = half_L + border_dimen[0]
    ymax = half_W + border_dimen[1]

    # Halfway line + centre spot + centre circle
    ax.plot([0, 0], [-half_W, half_W], lc, linewidth=linewidth)
    ax.scatter(0.0, 0.0, marker="o", facecolor=lc, linewidth=0, s=markersize)
    ax.plot(g["x_cc"], g["y_cc"], lc, linewidth=linewidth)
    ax.plot(-g["x_cc"], g["y_cc"], lc, linewidth=linewidth)

    # Ends
    for s in (-1, 1):
        # Pitch boundary
        ax.plot([-half_L, half_L], [s * half_W, s * half_W], lc, linewidth=linewidth)
        ax.plot([s * half_L, s * half_L], [-half_W, half_W], lc, linewidth=linewidth)

        # Goal posts
        ax.plot(
            [s * half_L, s * half_L],
            [-g["goal_line_width"] / 2.0, g["goal_line_width"] / 2.0],
            pc + "s",
            markersize=6 * markersize / 20.0,
            linewidth=linewidth,
        )

        # 6-yard box
        ax.plot([s * half_L, s * half_L - s * g["box_length"]], [g["box_width"] / 2.0, g["box_width"] / 2.0], lc,
                linewidth=linewidth)
        ax.plot([s * half_L, s * half_L - s * g["box_length"]], [-g["box_width"] / 2.0, -g["box_width"] / 2.0], lc,
                linewidth=linewidth)
        ax.plot([s * half_L - s * g["box_length"], s * half_L - s * g["box_length"]],
                [-g["box_width"] / 2.0, g["box_width"] / 2.0], lc, linewidth=linewidth)

        # Penalty area
        ax.plot([s * half_L, s * half_L - s * g["area_length"]], [g["area_width"] / 2.0, g["area_width"] / 2.0], lc,
                linewidth=linewidth)
        ax.plot([s * half_L, s * half_L - s * g["area_length"]], [-g["area_width"] / 2.0, -g["area_width"] / 2.0], lc,
                linewidth=linewidth)
        ax.plot([s * half_L - s * g["area_length"], s * half_L - s * g["area_length"]],
                [-g["area_width"] / 2.0, g["area_width"] / 2.0], lc, linewidth=linewidth)

        # Penalty spot
        ax.scatter(s * half_L - s * g["penalty_spot"], 0.0, marker="o", facecolor=lc, linewidth=0, s=markersize)

        # Corner arcs
        ax.plot(s * half_L - s * g["x_corner"], -half_W + g["y_corner"], lc, linewidth=linewidth)
        ax.plot(s * half_L - s * g["x_corner"], half_W - g["y_corner"], lc, linewidth=linewidth)

        # The D
        ax.plot(s * half_L - s * g["x_D"], g["y_D"], lc, linewidth=linewidth)

    # Clean axes
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([-xmax, xmax])
    ax.set_ylim([-ymax, ymax])
    ax.set_axisbelow(True)

    return fig, ax


def plot_frame(
    hometeam,
    awayteam,
    figax=None,
    team_colors: Tuple[str, str] = ("r", "b"),
    field_dimen: FieldDimen = (106.0, 68.0),
    include_player_velocities: bool = False,
    Playermarkersize: float = 10,
    PlayerAlpha: float = 0.7,
    annotate: bool = False,
):
    """plot_frame( hometeam, awayteam )

    Plots a frame of Metrica tracking data (player positions and the ball) on a football pitch. All distances should be in meters.
    """
    if figax is None:
        fig, ax = plot_pitch(field_dimen=field_dimen)
    else:
        fig, ax = figax

    # Plot home & away teams
    for team, color in zip((hometeam, awayteam), team_colors):
        x_cols, y_cols = _player_xy_columns(team)
        x = np.asarray(team[x_cols], dtype=float)
        y = np.asarray(team[y_cols], dtype=float)

        ax.plot(x, y, color + "o", markersize=Playermarkersize, alpha=PlayerAlpha)

        if include_player_velocities:
            vx_cols, vy_cols = _velocity_columns_from_xy(x_cols, y_cols)
            vx = np.asarray(team[vx_cols], dtype=float)
            vy = np.asarray(team[vy_cols], dtype=float)
            ax.quiver(
                x, y, vx, vy,
                color=color,
                scale_units="inches",
                scale=10.0,
                width=0.0015,
                headlength=5,
                headwidth=3,
                alpha=PlayerAlpha,
            )

        if annotate:
            # Annotate jersey numbers based on column name pattern: Team_#_x
            for xc, yc in zip(x_cols, y_cols):
                xv = team[xc]
                yv = team[yc]
                if _is_nan(xv) or _is_nan(yv):
                    continue
                try:
                    jersey = str(xc).split("_")[1]
                except Exception:
                    jersey = str(xc)
                ax.text(float(xv) + 0.5, float(yv) + 0.5, jersey, fontsize=10, color=color)

    # Ball
    bx = hometeam.get("ball_x", np.nan)
    by = hometeam.get("ball_y", np.nan)
    if not (_is_nan(bx) or _is_nan(by)):
        ax.plot(bx, by, "ko", markersize=6, alpha=1.0, linewidth=0)

    return fig, ax


def save_match_clip(
    hometeam,
    awayteam,
    fpath: str,
    fname: str = "clip_test",
    figax=None,
    frames_per_second: int = 25,
    team_colors: Tuple[str, str] = ("r", "b"),
    field_dimen: FieldDimen = (106.0, 68.0),
    include_player_velocities: bool = False,
    Playermarkersize: float = 10,
    PlayerAlpha: float = 0.7,
):
    """save_match_clip( hometeam, awayteam, fpath )

    Generates a movie from Metrica tracking data, saving it in the 'fpath' directory with name 'fname'
    """
    assert np.all(hometeam.index == awayteam.index), "Home and away team Dataframe indices must be the same"
    index = hometeam.index

    FFMpegWriter = animation.writers["ffmpeg"]
    metadata = dict(title="Tracking Data", artist="Matplotlib", comment="Metrica tracking data clip")
    writer = FFMpegWriter(fps=frames_per_second, metadata=metadata)

    out_path = f"{fpath}/{fname}.mp4"

    if figax is None:
        fig, ax = plot_pitch(field_dimen=field_dimen)
    else:
        fig, ax = figax
    fig.set_tight_layout(True)

    # Precompute columns once (from first row)
    h0 = hometeam.loc[index[0]]
    a0 = awayteam.loc[index[0]]
    h_x_cols, h_y_cols = _player_xy_columns(h0)
    a_x_cols, a_y_cols = _player_xy_columns(a0)

    # Create artists we will update
    home_pts, = ax.plot([], [], team_colors[0] + "o", markersize=Playermarkersize, alpha=PlayerAlpha)
    away_pts, = ax.plot([], [], team_colors[1] + "o", markersize=Playermarkersize, alpha=PlayerAlpha)
    ball_pt, = ax.plot([], [], "ko", markersize=6, alpha=1.0, linewidth=0)
    time_txt = ax.text(-2.5, field_dimen[1] / 2.0 + 1.0, "", fontsize=14)

    # Quivers (if enabled) are recreated each frame (updating quiver cleanly is annoying + brittle)
    home_quiv = None
    away_quiv = None

    print("Generating movie...", end="")
    with writer.saving(fig, out_path, 100):
        for i in index:
            ht = hometeam.loc[i]
            at = awayteam.loc[i]

            hx = np.asarray(ht[h_x_cols], dtype=float)
            hy = np.asarray(ht[h_y_cols], dtype=float)
            axx = np.asarray(at[a_x_cols], dtype=float)
            ayy = np.asarray(at[a_y_cols], dtype=float)

            home_pts.set_data(hx, hy)
            away_pts.set_data(axx, ayy)

            bx = ht.get("ball_x", np.nan)
            by = ht.get("ball_y", np.nan)
            if _is_nan(bx) or _is_nan(by):
                ball_pt.set_data([], [])
            else:
                ball_pt.set_data([bx], [by])

            # Time label
            t = float(ht.get("Time [s]", 0.0))
            frame_minute = int(t / 60.0)
            frame_second = (t / 60.0 - frame_minute) * 60.0
            time_txt.set_text(f"{frame_minute:d}:{frame_second:0.2f}")

            # Optional velocities
            if include_player_velocities:
                # remove previous quivers
                if home_quiv is not None:
                    home_quiv.remove()
                if away_quiv is not None:
                    away_quiv.remove()

                h_vx_cols, h_vy_cols = _velocity_columns_from_xy(h_x_cols, h_y_cols)
                a_vx_cols, a_vy_cols = _velocity_columns_from_xy(a_x_cols, a_y_cols)

                hvx = np.asarray(ht[h_vx_cols], dtype=float)
                hvy = np.asarray(ht[h_vy_cols], dtype=float)
                avx = np.asarray(at[a_vx_cols], dtype=float)
                avy = np.asarray(at[a_vy_cols], dtype=float)

                home_quiv = ax.quiver(
                    hx, hy, hvx, hvy,
                    color=team_colors[0],
                    scale_units="inches",
                    scale=10.0,
                    width=0.0015,
                    headlength=5,
                    headwidth=3,
                    alpha=PlayerAlpha,
                )
                away_quiv = ax.quiver(
                    axx, ayy, avx, avy,
                    color=team_colors[1],
                    scale_units="inches",
                    scale=10.0,
                    width=0.0015,
                    headlength=5,
                    headwidth=3,
                    alpha=PlayerAlpha,
                )

            writer.grab_frame()

    print("done")
    plt.clf()
    plt.close(fig)


def plot_events(
    events,
    figax=None,
    field_dimen: FieldDimen = (106.0, 68),
    indicators: Sequence[str] = ("Marker", "Arrow"),
    color: str = "r",
    marker_style: str = "o",
    alpha: float = 0.5,
    annotate: bool = False,
):
    """plot_events( events )

    Plots Metrica event positions on a football pitch. events can be a single row or several rows of a DataFrame.
    All distances should be in meters.

    IMPORTANT: This function intentionally skips rows with missing coordinates (NaNs).
    That avoids the “blank pitch” failure mode when your slice contains incomplete rows.
    """
    if pd is None:
        raise RuntimeError("pandas is required for plot_events.")

    if figax is None:
        fig, ax = plot_pitch(field_dimen=field_dimen)
    else:
        fig, ax = figax

    df = _ensure_events_df(events)

    want_marker = "Marker" in indicators
    want_arrow = "Arrow" in indicators

    # itertuples is much faster than iterrows
    # We only touch the columns we need
    cols_needed = ["Start X", "Start Y"]
    if want_arrow:
        cols_needed += ["End X", "End Y"]
    if annotate:
        cols_needed += ["Type", "From"]
    # if any missing columns, fail loudly (matches tutorial expectations)
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        raise KeyError(f"plot_events() missing required event columns: {missing}")

    for row in df[cols_needed].itertuples(index=False, name=None):
        # Unpack based on what was requested
        if annotate and want_arrow:
            sx, sy, ex, ey, typ, frm = row
        elif annotate and not want_arrow:
            sx, sy, typ, frm = row
            ex = ey = np.nan
        elif (not annotate) and want_arrow:
            sx, sy, ex, ey = row
            typ = frm = ""
        else:
            sx, sy = row
            ex = ey = np.nan
            typ = frm = ""

        if want_marker:
            if not (_is_nan(sx) or _is_nan(sy)):
                ax.plot(sx, sy, color + marker_style, alpha=alpha)

        if want_arrow:
            if not (_is_nan(sx) or _is_nan(sy) or _is_nan(ex) or _is_nan(ey)):
                ax.annotate(
                    "",
                    xy=(ex, ey),
                    xytext=(sx, sy),
                    alpha=alpha,
                    arrowprops=dict(alpha=alpha, width=0.5, headlength=4.0, headwidth=4.0, color=color),
                    annotation_clip=False,
                )

        if annotate:
            if not (_is_nan(sx) or _is_nan(sy)):
                textstring = f"{typ}: {frm}"
                ax.text(sx, sy, textstring, fontsize=10, color=color)

    return fig, ax


def plot_pitchcontrol_for_event(
    event_id,
    events,
    tracking_home,
    tracking_away,
    PPCF,
    alpha: float = 0.7,
    include_player_velocities: bool = True,
    annotate: bool = False,
    field_dimen: FieldDimen = (106.0, 68),
):
    """plot_pitchcontrol_for_event(...)

    Plots the pitch control surface at the instant of the event given by event_id.
    Player and ball positions are overlaid.
    """
    pass_frame = events.loc[event_id]["Start Frame"]
    pass_team = events.loc[event_id].Team

    fig, ax = plot_pitch(field_color="white", field_dimen=field_dimen)
    plot_frame(
        tracking_home.loc[pass_frame],
        tracking_away.loc[pass_frame],
        figax=(fig, ax),
        PlayerAlpha=alpha,
        include_player_velocities=include_player_velocities,
        annotate=annotate,
    )
    plot_events(
        events.loc[event_id:event_id],
        figax=(fig, ax),
        indicators=("Marker", "Arrow"),
        annotate=False,
        color="k",
        alpha=1.0,
    )

    cmap = "bwr" if pass_team == "Home" else "bwr_r"
    ax.imshow(
        np.flipud(PPCF),
        extent=(-field_dimen[0] / 2.0, field_dimen[0] / 2.0, -field_dimen[1] / 2.0, field_dimen[1] / 2.0),
        interpolation="spline36",
        vmin=0.0,
        vmax=1.0,
        cmap=cmap,
        alpha=0.5,
    )

    return fig, ax


def plot_EPV_for_event(
    event_id,
    events,
    tracking_home,
    tracking_away,
    PPCF,
    EPV,
    alpha: float = 0.7,
    include_player_velocities: bool = True,
    annotate: bool = False,
    autoscale: Union[bool, float] = 0.1,
    contours: bool = False,
    field_dimen: FieldDimen = (106.0, 68),
):
    """plot_EPV_for_event(...)

    Plots the EPVxPitchControl surface at the instant of the event given by event_id.
    Player and ball positions are overlaid.
    """
    pass_frame = events.loc[event_id]["Start Frame"]
    pass_team = events.loc[event_id].Team

    fig, ax = plot_pitch(field_color="white", field_dimen=field_dimen)
    plot_frame(
        tracking_home.loc[pass_frame],
        tracking_away.loc[pass_frame],
        figax=(fig, ax),
        PlayerAlpha=alpha,
        include_player_velocities=include_player_velocities,
        annotate=annotate,
    )
    plot_events(
        events.loc[event_id:event_id],
        figax=(fig, ax),
        indicators=("Marker", "Arrow"),
        annotate=False,
        color="k",
        alpha=1.0,
    )

    # Determine EPV orientation based on playing direction
    if pass_team == "Home":
        cmap = "Reds"
        lcolor = "r"
        EPV_use = np.fliplr(EPV) if mio.find_playing_direction(tracking_home, "Home") == -1 else EPV
    else:
        cmap = "Blues"
        lcolor = "b"
        EPV_use = np.fliplr(EPV) if mio.find_playing_direction(tracking_away, "Away") == -1 else EPV

    EPVxPPCF = PPCF * EPV_use

    if autoscale is True:
        vmax = float(np.max(EPVxPPCF) * 2.0)
    elif isinstance(autoscale, (int, float)) and 0.0 <= float(autoscale) <= 1.0:
        vmax = float(autoscale)
    else:
        raise ValueError("'autoscale' must be either True or a float between 0 and 1")

    ax.imshow(
        np.flipud(EPVxPPCF),
        extent=(-field_dimen[0] / 2.0, field_dimen[0] / 2.0, -field_dimen[1] / 2.0, field_dimen[1] / 2.0),
        interpolation="spline36",
        vmin=0.0,
        vmax=vmax,
        cmap=cmap,
        alpha=0.7,
    )

    if contours:
        ax.contour(
            EPVxPPCF,
            extent=(-field_dimen[0] / 2.0, field_dimen[0] / 2.0, -field_dimen[1] / 2.0, field_dimen[1] / 2.0),
            levels=np.array([0.75]) * np.max(EPVxPPCF),
            colors=lcolor,
            alpha=1.0,
        )

    return fig, ax


def plot_EPV(EPV, field_dimen: FieldDimen = (106.0, 68), attack_direction: int = 1):
    """plot_EPV( EPV, field_dimen, attack_direction )

    Plots the pre-generated Expected Possession Value surface.
    """
    EPV_use = np.fliplr(EPV) if attack_direction == -1 else EPV

    fig, ax = plot_pitch(field_color="white", field_dimen=field_dimen)
    ax.imshow(
        EPV_use,
        extent=(-field_dimen[0] / 2.0, field_dimen[0] / 2.0, -field_dimen[1] / 2.0, field_dimen[1] / 2.0),
        vmin=0.0,
        vmax=0.6,
        cmap="Blues",
        alpha=0.6,
    )
    return fig, ax
