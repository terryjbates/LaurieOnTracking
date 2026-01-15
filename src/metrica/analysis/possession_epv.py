# metrica/analysis/possession_epv.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd

import metrica.Metrica_IO as mio
import metrica.Metrica_PitchControl as mpc
import metrica.Metrica_EPV as mepv

Team = Literal["Home", "Away"]


@dataclass(frozen=True)
class PossessionConfig:
    """
    Configuration for possession-level EPV and physical metrics.
    """
    fps: float = 25.0
    speed_threshold_mps: float = 3.0  # distance accumulated for frames where speed >= threshold
    include_period_in_time_filter: bool = True  # honor Period when slicing tracking
    passes_only: bool = True  # build possession sequences using only PASS events


def add_possession_sequence(
    pass_events: pd.DataFrame,
    *,
    team_col: str = "Team",
    out_col: str = "Poss_Seq",
) -> pd.DataFrame:
    """
    Add a possession sequence id based on changes in the `Team` column.

    This mirrors the original logic:
      pass_events['Team'].ne(pass_events['Team'].shift()).cumsum()

    Returns a copy (does not mutate input).
    """
    out = pass_events.copy()
    out[out_col] = out[team_col].ne(out[team_col].shift()).cumsum()
    return out


def _possession_bounds(
    poss: pd.DataFrame,
    *,
    start_time_col: str = "Start Time [s]",
    end_time_col: str = "End Time [s]",
    period_col: str = "Period",
) -> Tuple[float, float, np.ndarray]:
    """
    Return (start_time, end_time, periods_array) for a possession slice.
    """
    start_time = float(pd.to_numeric(poss[start_time_col], errors="coerce").min())
    end_time = float(pd.to_numeric(poss[end_time_col], errors="coerce").max())
    periods = poss[period_col].dropna().unique()
    return start_time, end_time, periods


def _speed_cols(tracking: pd.DataFrame, team: Team) -> list[str]:
    """
    Return list of speed columns like 'Home_5_speed' for a given team.
    """
    prefix = f"{team}_"
    cols = [c for c in tracking.columns if c.startswith(prefix) and c.endswith("_speed")]
    return cols


def distance_above_speed_threshold_km(
    tracking: pd.DataFrame,
    team: Team,
    *,
    start_time_s: float,
    end_time_s: float,
    periods: Optional[np.ndarray],
    speed_threshold_mps: float,
    fps: float,
    time_col: str = "Time [s]",
    period_col: str = "Period",
    include_period: bool = True,
) -> float:
    """
    Total distance (km) covered by a team for frames within [start_time_s, end_time_s],
    counting only frames where player speed >= speed_threshold_mps.

    Vectorized across players:
      - slice the tracking rows once
      - subset speed columns
      - threshold to keep speeds or 0
      - sum all speeds / fps / 1000
    """
    time = pd.to_numeric(tracking[time_col], errors="coerce")
    mask = (time >= start_time_s) & (time <= end_time_s)

    if include_period and periods is not None and len(periods) > 0:
        mask &= tracking[period_col].isin(periods)

    seg = tracking.loc[mask]
    speed_cols = _speed_cols(seg, team)
    if not speed_cols:
        return float("nan")

    speeds = seg[speed_cols].apply(pd.to_numeric, errors="coerce")
    speeds = speeds.where(speeds >= speed_threshold_mps, 0.0)

    # Sum of m/s samples -> meters by dividing by fps -> km /1000
    dist_km = float(speeds.to_numpy(dtype=float).sum() / fps / 1000.0)
    return dist_km


def possession_epv_added(
    pass_event_ids: np.ndarray,
    *,
    events: pd.DataFrame,
    tracking_home: pd.DataFrame,
    tracking_away: pd.DataFrame,
    gk_numbers: Tuple[str, str],
    epv_grid: np.ndarray,
    params: dict,
) -> float:
    """
    Total EPV added across a set of pass event indices.

    NOTE: This is the expensive part. We keep it explicit and isolated.
    """
    total = 0.0
    for event_id in pass_event_ids:
        eepv_added, _ = mepv.calculate_epv_added(
            int(event_id),
            events,
            tracking_home,
            tracking_away,
            list(gk_numbers),
            epv_grid,
            params,
        )
        total += float(eepv_added)
    return total


def build_possession_physical_epv_table(
    *,
    events: pd.DataFrame,
    tracking_home: pd.DataFrame,
    tracking_away: pd.DataFrame,
    epv_grid: np.ndarray,
    params: dict,
    gk_numbers: Tuple[str, str],
    team: Team = "Home",
    cfg: Optional[PossessionConfig] = None,
    poss_col: str = "Poss_Seq",
) -> pd.DataFrame:
    """
    Build a possession-level table for a team:

      HomeDist_km  (speed>=threshold during possession)
      AwayDist_km
      EEPV

    Possessions are defined over PASS events.
    """
    cfg = cfg or PossessionConfig()

    pass_events = events[events["Type"] == "PASS"].copy() if cfg.passes_only else events.copy()
    pass_events = add_possession_sequence(pass_events, out_col=poss_col)

    team_passes = pass_events[pass_events["Team"] == team].copy()
    if team_passes.empty:
        return pd.DataFrame(columns=["HomeDist", "AwayDist", "EEPV", "Poss_Seq"])

    rows: list[dict] = []

    for poss_id, poss in team_passes.groupby(poss_col, sort=True):
        start_time, end_time, periods = _possession_bounds(poss)

        home_dist = distance_above_speed_threshold_km(
            tracking_home,
            "Home",
            start_time_s=start_time,
            end_time_s=end_time,
            periods=periods,
            speed_threshold_mps=cfg.speed_threshold_mps,
            fps=cfg.fps,
            include_period=cfg.include_period_in_time_filter,
        )
        away_dist = distance_above_speed_threshold_km(
            tracking_away,
            "Away",
            start_time_s=start_time,
            end_time_s=end_time,
            periods=periods,
            speed_threshold_mps=cfg.speed_threshold_mps,
            fps=cfg.fps,
            include_period=cfg.include_period_in_time_filter,
        )

        pass_ids = poss.index.to_numpy()
        total_eepv = possession_epv_added(
            pass_ids,
            events=events,
            tracking_home=tracking_home,
            tracking_away=tracking_away,
            gk_numbers=gk_numbers,
            epv_grid=epv_grid,
            params=params,
        )

        rows.append(
            {
                "Poss_Seq": int(poss_id),
                "HomeDist": float(home_dist),
                "AwayDist": float(away_dist),
                "EEPV": float(total_eepv),
                "StartTime_s": float(start_time),
                "EndTime_s": float(end_time),
            }
        )

    out = pd.DataFrame(rows).sort_values("Poss_Seq").reset_index(drop=True)
    return out


def fit_linear_model_home_dist_to_eepv(df: pd.DataFrame) -> tuple[float, np.ndarray]:
    """
    Fit a simple linear model: EEPV ~ HomeDist

    Returns
    -------
    r2 : float
    yhat : np.ndarray
    """
    from sklearn.linear_model import LinearRegression

    x = df["HomeDist"].to_numpy(dtype=float).reshape(-1, 1)
    y = df["EEPV"].to_numpy(dtype=float).reshape(-1, 1)

    model = LinearRegression().fit(x, y)
    r2 = float(model.score(x, y))
    yhat = model.predict(x).reshape(-1)
    return r2, yhat
