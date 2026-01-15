# scripts/possession_epv_demo.py
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt

import metrica.Metrica_IO as mio
import metrica.Metrica_Velocities as mvel
import metrica.Metrica_PitchControl as mpc
import metrica.Metrica_EPV as mepv
import metrica.Metrica_Viz as mviz

from metrica.analysis.possession_epv import (
    PossessionConfig,
    build_possession_physical_epv_table,
    fit_linear_model_home_dist_to_eepv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Possession-level EPV added vs physical output demo.")
    p.add_argument(
        "--datadir",
        default=os.environ.get("METRICA_DATADIR", r"C:\Users\lover\github\sample-data\data"),
        help="Path to Metrica sample-data/data directory (or set METRICA_DATADIR).",
    )
    p.add_argument("--game-id", type=int, default=2, help="Sample game id (1 or 2).")
    p.add_argument(
        "--epv-grid",
        default=None,
        help="Path to EPV_grid.csv. If omitted, uses <datadir>/EPV_grid.csv (and then repo root ./EPV_grid.csv).",
    )
    p.add_argument("--speed-threshold", type=float, default=3.0, help="Speed threshold (m/s) for distance tally.")
    p.add_argument("--no-plots", action="store_true", help="Compute only; do not display plots.")
    return p.parse_args()


def resolve_epv_path(datadir: str, epv_arg: str | None) -> str:
    if epv_arg:
        return epv_arg

    candidate1 = os.path.join(datadir, "EPV_grid.csv")
    if os.path.isfile(candidate1):
        return candidate1

    candidate2 = os.path.join(os.getcwd(), "EPV_grid.csv")
    return candidate2


def main() -> None:
    args = parse_args()

    # --- Load match data ---
    events = mio.read_event_data(args.datadir, args.game_id)
    tracking_home = mio.tracking_data(args.datadir, args.game_id, "Home")
    tracking_away = mio.tracking_data(args.datadir, args.game_id, "Away")

    tracking_home = mio.to_metric_coordinates(tracking_home)
    tracking_away = mio.to_metric_coordinates(tracking_away)
    events = mio.to_metric_coordinates(events)

    tracking_home, tracking_away, events = mio.to_single_playing_direction(tracking_home, tracking_away, events)

    # Player velocities (robust: fall back to moving_average if default smoothing hits MKL/LAPACK issues)
    try:
        tracking_home = mvel.calc_player_velocities(tracking_home, smoothing=True)
        tracking_away = mvel.calc_player_velocities(tracking_away, smoothing=True)
    except Exception as e:
        print(f"\nVelocity smoothing failed with: {type(e).__name__}: {e}")
        print("Falling back to filter_='moving_average' ...\n")
        tracking_home = mvel.calc_player_velocities(tracking_home, smoothing=True, filter_="moving_average")
        tracking_away = mvel.calc_player_velocities(tracking_away, smoothing=True, filter_="moving_average")


    gk_numbers = (mio.find_goalkeeper(tracking_home), mio.find_goalkeeper(tracking_away))
    home_attack_direction = mio.find_playing_direction(tracking_home, "Home")

    # --- EPV grid and pitch-control params ---
    params = mpc.default_model_params()
    epv_path = resolve_epv_path(args.datadir, args.epv_grid)
    EPV = mepv.load_EPV_grid(epv_path)

    if not args.no_plots:
        mviz.plot_EPV(EPV, field_dimen=(106.0, 68.0), attack_direction=home_attack_direction)

    # --- Build possession table ---
    cfg = PossessionConfig(speed_threshold_mps=args.speed_threshold)
    df = build_possession_physical_epv_table(
        events=events,
        tracking_home=tracking_home,
        tracking_away=tracking_away,
        epv_grid=EPV,
        params=params,
        gk_numbers=gk_numbers,
        team="Home",
        cfg=cfg,
    )

    print("\nPossession table head:")
    print(df.head(10))
    print(f"\nRows: {len(df)}")

    # --- Fit linear model ---
    r2, yhat = fit_linear_model_home_dist_to_eepv(df)
    print(f"\nLinear model: EEPV ~ HomeDist  |  R^2 = {r2:.3f}")

    if args.no_plots:
        return

    # --- Plot scatter + fit line ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df["HomeDist"], df["EEPV"])
    ax.plot(df["HomeDist"], yhat)
    ax.set_title(f"Total Distance (speed >= {args.speed_threshold:.1f} m/s) vs Total EPV Added (R²={r2:.3f})")
    ax.set_xlabel("HomeDist (km)")
    ax.set_ylabel("Total EEPV Added")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
