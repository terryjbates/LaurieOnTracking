# metrica/analysis/physical.py
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
import scipy as sp
import seaborn as sns
import statsmodels.formula.api as smf

import metrica.Metrica_IO as mio
import metrica.Metrica_Velocities as mvel


Team = Literal["Home", "Away"]
FieldDimen = Tuple[float, float]


@dataclass(frozen=True)
class PhysicalConfig:
    """
    Configuration for physical/rolling metrics.

    Notes
    -----
    The Metrica sample-data is 25Hz, but we use the Time [s] column whenever
    we need dt; fps is primarily used for distance conversions and rolling windows.
    """
    fps: float = 25.0

    # sanity caps (optical tracking can spike)
    max_speed_mps: float = 12.0
    max_acc_mps2: float = 6.0

    # speed bands (m/s)
    walk_lt: float = 2.0
    jog_lt: float = 4.0
    run_lt: float = 7.0
    sprint_ge: float = 7.0

    # high-speed distance threshold (m/s)
    hsd_speed_ge: float = 5.0

    # SPI / rolling windows
    spi_window_frames: int = 1500        # 60s at 25Hz
    mp_rolling_frames: int = 7500        # 5 min at 25Hz

    # acc/dec segmentation
    high_acc_threshold: float = 2.0      # classify |acc| >= 2 as "High"
    min_high_acc_duration_s: float = 0.75

    # plotting defaults
    default_plot_n_frames: int = 9000


@dataclass(frozen=True)
class MatchData:
    """Convenience container for a fully-prepped match load."""
    events: pd.DataFrame
    tracking_home: pd.DataFrame
    tracking_away: pd.DataFrame
    gk_numbers: Tuple[str, str]
    home_attack_direction: float


def require_dir(path: str) -> None:
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"DATADIR not found:\n  {path}\n"
            "Set METRICA_DATADIR or pass datadir explicitly."
        )


def player_numbers_from_tracking(tracking: pd.DataFrame, team: Team) -> list[str]:
    """
    Extract jersey numbers as strings from columns like 'Home_5_x', 'Away_26_x'.

    Returns
    -------
    Sorted unique list of jersey numbers (as strings).
    """
    pat = re.compile(rf"^{team}_(\d+)_x$")
    nums = sorted({m.group(1) for c in tracking.columns if (m := pat.match(c))})
    return nums


def compute_unsmoothed_speeds(
    tracking: pd.DataFrame,
    *,
    max_speed_mps: float,
) -> pd.DataFrame:
    """
    Compute per-player speed without smoothing (for comparison / teaching).

    Adds columns like:
      Home_5_speed_raw

    Uses Time [s] for dt and caps raw spikes above max_speed_mps.
    """
    out = tracking.copy()
    dt = pd.to_numeric(out["Time [s]"], errors="coerce").diff()

    # Iterate only players that have *_x columns; avoids broken parsing.
    for team in ("Home", "Away"):
        for num in player_numbers_from_tracking(out, team):  # jersey number
            base = f"{team}_{num}"
            x = pd.to_numeric(out.get(f"{base}_x"), errors="coerce")
            y = pd.to_numeric(out.get(f"{base}_y"), errors="coerce")
            if x is None or y is None:
                continue

            vx = x.diff() / dt
            vy = y.diff() / dt
            raw_speed = np.sqrt(vx * vx + vy * vy)

            if max_speed_mps and max_speed_mps > 0:
                bad = raw_speed > max_speed_mps
                raw_speed = raw_speed.mask(bad)

            out[f"{base}_speed_raw"] = raw_speed

    return out


def load_match(
    datadir: str,
    game_id: int,
    *,
    field_dimen: FieldDimen = (106.0, 68.0),
    smoothing: bool = True,
    config: Optional[PhysicalConfig] = None,
) -> MatchData:
    """
    Load events + tracking and apply the same prep you used in the tutorials:
    - read files
    - convert to metric coordinates
    - single playing direction
    - compute smoothed velocities (optional)

    Returns a MatchData container.
    """
    cfg = config or PhysicalConfig()
    require_dir(datadir)

    events = mio.read_event_data(datadir, game_id)
    tracking_home = mio.tracking_data(datadir, game_id, "Home")
    tracking_away = mio.tracking_data(datadir, game_id, "Away")

    tracking_home = mio.to_metric_coordinates(tracking_home, field_dimen=field_dimen)
    tracking_away = mio.to_metric_coordinates(tracking_away, field_dimen=field_dimen)
    events = mio.to_metric_coordinates(events, field_dimen=field_dimen)

    tracking_home, tracking_away, events = mio.to_single_playing_direction(tracking_home, tracking_away, events)

    if smoothing:
        tracking_home = mvel.calc_player_velocities(tracking_home, smoothing=True, maxspeed=cfg.max_speed_mps)
        tracking_away = mvel.calc_player_velocities(tracking_away, smoothing=True, maxspeed=cfg.max_speed_mps)

    gk_numbers = (mio.find_goalkeeper(tracking_home), mio.find_goalkeeper(tracking_away))
    home_attack_direction = mio.find_playing_direction(tracking_home, "Home")

    return MatchData(
        events=events,
        tracking_home=tracking_home,
        tracking_away=tracking_away,
        gk_numbers=gk_numbers,
        home_attack_direction=home_attack_direction,
    )


def minutes_played(tracking: pd.DataFrame, team: Team, *, fps: float) -> pd.Series:
    """Minutes played based on first/last valid x observation for each player."""
    mins: dict[str, float] = {}
    for num in player_numbers_from_tracking(tracking, team):
        xcol = f"{team}_{num}_x"
        first = tracking[xcol].first_valid_index()
        last = tracking[xcol].last_valid_index()
        if first is None or last is None:
            mins[num] = 0.0
        else:
            mins[num] = (last - first + 1) / fps / 60.0
    return pd.Series(mins, name="Minutes Played").sort_values(ascending=False)


def distance_km_from_speed(tracking: pd.DataFrame, team: Team, *, fps: float) -> pd.Series:
    """Distance in km from per-frame speed (m/s): sum(speed)/fps/1000."""
    dist: dict[str, float] = {}
    for num in player_numbers_from_tracking(tracking, team):
        scol = f"{team}_{num}_speed"
        if scol not in tracking.columns:
            dist[num] = float("nan")
            continue
        s = pd.to_numeric(tracking[scol], errors="coerce")
        dist[num] = float(s.sum() / fps / 1000.0)
    return pd.Series(dist, name="Distance [km]")


def speed_band_distances_km(tracking: pd.DataFrame, team: Team, *, cfg: PhysicalConfig) -> pd.DataFrame:
    """Distance (km) in each speed band for a team."""
    rows: dict[str, dict[str, float]] = {}
    for num in player_numbers_from_tracking(tracking, team):
        scol = f"{team}_{num}_speed"
        s = pd.to_numeric(tracking[scol], errors="coerce")

        walking = float(s[s < cfg.walk_lt].sum() / cfg.fps / 1000.0)
        jogging = float(s[(s >= cfg.walk_lt) & (s < cfg.jog_lt)].sum() / cfg.fps / 1000.0)
        running = float(s[(s >= cfg.jog_lt) & (s < cfg.run_lt)].sum() / cfg.fps / 1000.0)
        sprinting = float(s[s >= cfg.sprint_ge].sum() / cfg.fps / 1000.0)

        rows[num] = {
            "Walking [km]": walking,
            "Jogging [km]": jogging,
            "Running [km]": running,
            "Sprinting [km]": sprinting,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def summarize_team_physical(
    tracking: pd.DataFrame,
    team: Team,
    *,
    cfg: Optional[PhysicalConfig] = None,
    sub_minutes_threshold: float = 90.0,
) -> pd.DataFrame:
    """
    Build a per-player physical summary table.

    Returns columns:
      Minutes Played, Distance [km], DPM, isSub, (plus speed bands)
    Index is jersey number as string ('5','26',...).
    """
    cfg = cfg or PhysicalConfig()

    mins = minutes_played(tracking, team, fps=cfg.fps)
    dist = distance_km_from_speed(tracking, team, fps=cfg.fps)
    bands = speed_band_distances_km(tracking, team, cfg=cfg)

    summary = pd.concat([mins, dist], axis=1)
    summary["DPM"] = 1000.0 * (summary["Distance [km]"] / summary["Minutes Played"].replace({0.0: np.nan}))
    summary["Team"] = team
    summary["isSub"] = (summary["Minutes Played"] < sub_minutes_threshold).astype(int)

    summary = summary.join(bands, how="left")
    return summary.sort_values("Distance [km]", ascending=False)


def plot_team_distance_bar(game_summary: pd.DataFrame, *, title: str = "Distance Covered by Player [km]") -> None:
    """
    Plot a Home vs Away distance bar chart using seaborn.

    Expects a combined table with columns:
      Team, Distance [km], isSub
    and index = jersey number.
    """
    df = game_summary.reset_index(names="Player")
    df["PlayerLabel"] = np.where(df["isSub"] == 0, df["Player"], df["Player"] + "*")
    df = df.sort_values("Distance [km]", ascending=False)

    sns.set_theme()
    g = sns.catplot(
        data=df,
        x="PlayerLabel",
        y="Distance [km]",
        hue="Team",
        kind="bar",
        height=5,
        aspect=2.2,
    )
    g.set_axis_labels("Player", "Distance [km]")
    g.fig.suptitle(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def plot_speed_comparison(
    *,
    tracking_raw: pd.DataFrame,
    tracking_smoothed: pd.DataFrame,
    player: str,
    cfg: Optional[PhysicalConfig] = None,
    n_frames: Optional[int] = None,
) -> None:
    """
    Plot unsmoothed vs smoothed speed for a single player id like 'Home_5' or 'Away_26'.
    """
    cfg = cfg or PhysicalConfig()
    n = n_frames or cfg.default_plot_n_frames

    raw_col = f"{player}_speed_raw"
    smooth_col = f"{player}_speed"

    if raw_col not in tracking_raw.columns:
        raise KeyError(f"Missing '{raw_col}'. Run compute_unsmoothed_speeds(...) first.")
    if smooth_col not in tracking_smoothed.columns:
        raise KeyError(f"Missing '{smooth_col}'. Run mvel.calc_player_velocities(...) first.")

    x = np.arange(1, n + 1)
    raw = tracking_raw.loc[1:n, raw_col]
    sm = tracking_smoothed.loc[1:n, smooth_col]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, raw, label="Unsmoothed")
    ax.plot(x, sm, label="Smoothed")
    ax.set_title(f"{player}: Unsmoothed vs Smoothed Speed")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Speed (m/s)")
    ax.legend()
    plt.show()


def acc_dec_table(
    tracking: pd.DataFrame,
    team: Team,
    *,
    cfg: Optional[PhysicalConfig] = None,
) -> pd.DataFrame:
    """
    Extract sustained high-acceleration/high-deceleration segments.

    Returns tidy table:
      Player (jersey number), Duration_s, Type ('Acc'/'Dec')
    """
    cfg = cfg or PhysicalConfig()

    dt = pd.to_numeric(tracking["Time [s]"], errors="coerce").diff()
    min_frames = int(np.ceil(cfg.min_high_acc_duration_s * cfg.fps))

    records: list[dict] = []

    for num in player_numbers_from_tracking(tracking, team):
        scol = f"{team}_{num}_speed"
        speed = pd.to_numeric(tracking[scol], errors="coerce")
        acc = speed.diff() / dt
        acc = acc.mask(acc.abs() > cfg.max_acc_mps2)

        high = acc.abs() >= cfg.high_acc_threshold
        grp = (high != high.shift()).cumsum()

        for g, seg in tracking.groupby(grp, sort=False):
            idx0 = seg.index[0]
            if not bool(high.loc[idx0]):
                continue
            if len(seg) < min_frames:
                continue

            t = pd.to_numeric(seg["Time [s]"], errors="coerce")
            dur_s = float(max(0.0, t.iloc[-1] - t.iloc[0]))

            seg_acc = float(pd.to_numeric(acc.loc[seg.index], errors="coerce").mean())
            kind = "Acc" if seg_acc > 0 else "Dec"

            records.append({"Player": num, "Duration_s": dur_s, "Type": kind})

    return pd.DataFrame(records)


def add_acc_dec_ratio(summary: pd.DataFrame, acc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'AccDec' ratio column to a team summary:
      n_acc / n_dec

    If n_dec == 0, ratio becomes NaN.
    """
    if acc_df.empty:
        summary["AccDec"] = np.nan
        return summary

    counts = (
        acc_df.groupby(["Player", "Type"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={"Acc": "n_acc", "Dec": "n_dec"})
    )
    counts["AccDec"] = counts["n_acc"] / counts["n_dec"].replace({0: np.nan})

    out = summary.copy()
    out = out.join(counts["AccDec"], how="left")
    return out


def spi_table(
    tracking: pd.DataFrame,
    team: Team,
    *,
    cfg: Optional[PhysicalConfig] = None,
    top_k: int = 3,
) -> pd.DataFrame:
    """
    Compute SPI-like peaks (top-k) per player for:
      - Dist: rolling distance (m) over 60 seconds
      - HSD: rolling distance (m) for speed>=hsd_speed_ge over 60 seconds

    Returns tidy table:
      Team, Player, Type, SPI, MinAfter
    """
    cfg = cfg or PhysicalConfig()
    rows: list[dict] = []

    for num in player_numbers_from_tracking(tracking, team):
        scol = f"{team}_{num}_speed"
        speed = pd.to_numeric(tracking[scol], errors="coerce").fillna(0.0)

        # rolling distance (m) in window: sum(speed)/fps
        dist_roll = speed.rolling(cfg.spi_window_frames, min_periods=1).sum() / cfg.fps

        peaks, _ = sp.signal.find_peaks(dist_roll.to_numpy(), distance=cfg.spi_window_frames)
        if peaks.size:
            vals = dist_roll.iloc[peaks].to_numpy()
            keep = np.argsort(vals)[-top_k:]
            for idx in peaks[keep]:
                spi_val = float(dist_roll.iloc[idx])
                after = float(speed.iloc[idx + 2 : idx + 2 + cfg.spi_window_frames].sum() / cfg.fps)
                rows.append({"Team": team, "Player": num, "Type": "Dist", "SPI": spi_val, "MinAfter": after})

        hsd_speed = speed.where(speed >= cfg.hsd_speed_ge, 0.0)
        hsd_roll = hsd_speed.rolling(cfg.spi_window_frames, min_periods=1).sum() / cfg.fps

        peaks, _ = sp.signal.find_peaks(hsd_roll.to_numpy(), distance=cfg.spi_window_frames)
        if peaks.size:
            vals = hsd_roll.iloc[peaks].to_numpy()
            keep = np.argsort(vals)[-top_k:]
            for idx in peaks[keep]:
                spi_val = float(hsd_roll.iloc[idx])
                after = float(speed.iloc[idx + 2 : idx + 2 + cfg.spi_window_frames].sum() / cfg.fps)
                rows.append({"Team": team, "Player": num, "Type": "HSD", "SPI": spi_val, "MinAfter": after})

    return pd.DataFrame(rows)


def fit_spi_mixedlm(spi: pd.DataFrame, *, which: Literal["Dist", "HSD"]) -> Optional[object]:
    """
    Fit the simple random-intercept model you had:
      Diff ~ 1, groups=Player

    Expects `spi` includes columns: Player, Type, Diff
    Returns fitted results or None if no data.
    """
    df = spi[spi["Type"] == which].copy()
    if df.empty:
        return None

    md = smf.mixedlm("Diff ~ 1", df, groups=df["Player"])
    return md.fit(method="cg")


def metabolic_cost(acc: float) -> float:
    """
    Experimental metabolic cost model.

    Source referenced in the original script:
    https://jeb.biologists.org/content/221/15/jeb182303

    WARNING: Extremely sensitive to accel noise.
    """
    if acc > 0:
        return float(0.102 * ((acc**2 + 96.2) ** 0.5) * (4.03 * acc + 3.6 * np.exp(-0.408 * acc)))
    if acc < 0:
        return float(0.102 * ((acc**2 + 96.2) ** 0.5) * (-0.85 * acc + 3.6 * np.exp(1.33 * acc)))
    return 0.0


def metabolic_power_roll(
    tracking: pd.DataFrame,
    player_id: str,
    *,
    cfg: Optional[PhysicalConfig] = None,
) -> pd.Series:
    """
    Compute a rough metabolic power proxy for one player (e.g. 'Home_6') and
    return a rolling sum over cfg.mp_rolling_frames.

    MP ~ metabolic_cost(acc) * speed
    """
    cfg = cfg or PhysicalConfig()

    dt = pd.to_numeric(tracking["Time [s]"], errors="coerce").diff()
    speed = pd.to_numeric(tracking[f"{player_id}_speed"], errors="coerce")
    acc = speed.diff() / dt
    acc = acc.mask(acc.abs() > cfg.max_acc_mps2)

    mc = np.vectorize(metabolic_cost, otypes=[float])(acc.to_numpy(dtype=float))
    mp = pd.Series(mc, index=tracking.index) * speed
    return mp.rolling(cfg.mp_rolling_frames, min_periods=1).sum()


def plot_metabolic_changepoints(mp_roll: pd.Series, *, n_bkps: int = 1, min_size: int = 7500) -> None:
    """
    Simple changepoint plots (BinSeg and PELT) for a metabolic power rolling series.
    """
    signal = np.asarray(mp_roll.iloc[min_size:], dtype=float).reshape(-1, 1)
    signal = np.nan_to_num(signal, nan=np.nanmedian(signal))

    algo = rpt.Binseg(model="l2").fit(signal)
    bkps = algo.predict(n_bkps=n_bkps)
    rpt.show.display(signal, bkps, figsize=(10, 4))
    plt.title(f"Metabolic power changepoints (BinSeg, n_bkps={n_bkps})")
    plt.show()

    algo = rpt.Pelt(model="l2", min_size=min_size).fit(signal)
    pen = np.log(len(signal)) * 1.0 * np.std(signal) ** 2
    bkps = algo.predict(pen=pen)
    rpt.show.display(signal, bkps, figsize=(10, 4))
    plt.title("Metabolic power changepoints (PELT)")
    plt.show()
