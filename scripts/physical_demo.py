# scripts/physical_demo.py
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

from metrica.analysis.physical import (
    PhysicalConfig,
    acc_dec_table,
    add_acc_dec_ratio,
    compute_unsmoothed_speeds,
    fit_spi_mixedlm,
    load_match,
    metabolic_power_roll,
    plot_metabolic_changepoints,
    plot_speed_comparison,
    plot_team_distance_bar,
    spi_table,
    summarize_team_physical,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Physical metrics demo using the metrica package (Metrica sample-data)."
    )
    p.add_argument(
        "--datadir",
        default=os.environ.get("METRICA_DATADIR", r"C:\Users\lover\github\sample-data\data"),
        help="Path to Metrica sample-data/data directory (or set METRICA_DATADIR).",
    )
    p.add_argument("--game-id", type=int, default=2, help="Sample game id (1 or 2).")

    p.add_argument(
        "--player",
        default="Home_5",
        help="Player id for speed/metabolic demos, e.g. 'Home_5' or 'Away_26'.",
    )
    p.add_argument(
        "--frames",
        type=int,
        default=9000,
        help="Number of frames to plot for raw vs smoothed speed.",
    )

    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Run computations but do not display plots.",
    )
    p.add_argument(
        "--skip-speed-compare",
        action="store_true",
        help="Skip raw vs smoothed speed comparison plot.",
    )
    p.add_argument(
        "--do-metabolic",
        action="store_true",
        help="Run metabolic power rolling + changepoint demo for --player.",
    )
    p.add_argument(
        "--do-spi",
        action="store_true",
        help="Compute SPI tables and fit simple mixed models (Dist and HSD).",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PhysicalConfig(default_plot_n_frames=args.frames)

    match = load_match(args.datadir, args.game_id, smoothing=True, config=cfg)

    # --- Team summaries ---
    home_summary = summarize_team_physical(match.tracking_home, "Home", cfg=cfg)
    away_summary = summarize_team_physical(match.tracking_away, "Away", cfg=cfg)
    game_summary = pd.concat([home_summary, away_summary], axis=0)

    # Add acc/dec ratio into each team summary (optional, but it’s useful)
    home_acc = acc_dec_table(match.tracking_home, "Home", cfg=cfg)
    away_acc = acc_dec_table(match.tracking_away, "Away", cfg=cfg)

    home_summary = add_acc_dec_ratio(home_summary, home_acc)
    away_summary = add_acc_dec_ratio(away_summary, away_acc)

    print("\nTop 5 Home distance:")
    print(home_summary[["Minutes Played", "Distance [km]", "DPM"]].head(5))

    print("\nTop 5 Away distance:")
    print(away_summary[["Minutes Played", "Distance [km]", "DPM"]].head(5))

    if not args.no_plots:
        plot_team_distance_bar(game_summary, title="Distance Covered by Player by Team [km]")

    # --- Speed compare (raw vs smoothed) ---
    if not args.skip_speed_compare:
        raw = compute_unsmoothed_speeds(match.tracking_home, max_speed_mps=cfg.max_speed_mps)
        # compute_unsmoothed_speeds operates on one df; if you want Away raw too, call on match.tracking_away.
        if not args.no_plots:
            plot_speed_comparison(
                tracking_raw=raw,
                tracking_smoothed=match.tracking_home,
                player=args.player,
                cfg=cfg,
                n_frames=args.frames,
            )

    # --- Metabolic demo (single-player) ---
    if args.do_metabolic:
        if not args.no_plots:
            mp_roll = metabolic_power_roll(match.tracking_home, args.player, cfg=cfg)
            plt.figure(figsize=(12, 4))
            plt.plot(mp_roll)
            plt.title(f"{args.player}: Metabolic Power Proxy (rolling sum)")
            plt.show()
            plot_metabolic_changepoints(mp_roll, n_bkps=1, min_size=cfg.mp_rolling_frames)

    # --- SPI + mixed models ---
    if args.do_spi:
        home_spi = spi_table(match.tracking_home, "Home", cfg=cfg)
        away_spi = spi_table(match.tracking_away, "Away", cfg=cfg)
        spi = pd.concat([home_spi, away_spi], axis=0, ignore_index=True)

        # join DPM baselines for Diff
        home_dpm = home_summary[["DPM"]].copy()
        away_dpm = away_summary[["DPM"]].copy()
        home_dpm["Team"] = "Home"
        away_dpm["Team"] = "Away"
        dpm = pd.concat([home_dpm, away_dpm], axis=0).reset_index(names="Player")

        spi = spi.merge(dpm, on=["Team", "Player"], how="left")
        spi["MinAfter"] = pd.to_numeric(spi["MinAfter"], errors="coerce")
        spi["Diff"] = spi["MinAfter"] - spi["DPM"]

        for typ in ("Dist", "HSD"):
            res = fit_spi_mixedlm(spi, which=typ)
            if res is None:
                print(f"\nNo SPI rows for Type={typ}")
            else:
                print(f"\nMixedLM results for Type={typ}")
                print(res.summary())

    print("\nEND")


if __name__ == "__main__":
    main()
