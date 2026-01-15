# Using physical.py

## Inside a notebook
```
from metrica.analysis import load_match, summarize_team_physical
match = load_match(DATADIR, 2)
home = summarize_team_physical(match.tracking_home, "Home")
```

## Runner Script

```
# Basic (team distance plot + speed compare)
python .\scripts\physical_demo.py --datadir "$env:METRICA_DATADIR" --game-id 2 --player Home_5

```

```
# Pick a different player and shorter speed plot
python .\scripts\physical_demo.py --player Away_26 --frames 3000
```

```
# Compute-only (no figures)
python .\scripts\physical_demo.py --no-plots --do-spi

```

```
# Metabolic Demo For Player

python .\scripts\physical_demo.py --player Home_6 --do-metabolic

```

# Using possession_epv.py
## Features and Fixes

* Builds possession sequences from PASS events.
* For each Home possession, computes:
* Home distance above a speed threshold (default >= 3 m/s)
* Away distance above a speed threshold
* Total EPV added across passes in that possession
* Returns a tidy DataFrame you can model/plot.
* Keeps the “heavy” part (EPV added per pass) explicit and contained.

* Correct package imports: uses metrica.Metrica_* everywhere.
* No SettingWithCopy: pass_events = ...copy() then adds Poss_Seq in a returned copy.
* Vectorized distance:
    * Instead of looping players and slicing frames each time, we slice tracking once per possession and sum across all speed columns.

* EPV loop isolated:
    * The only remaining loop is “per pass event compute EPV added,” because that function call is inherently per-event.
* EPV_grid path is robust:
    * Looks in datadir/EPV_grid.csv first, then ./EPV_grid.csv.

## Running
### From Repo Root
```
python .\scripts\possession_epv_demo.py --datadir "$env:METRICA_DATADIR" --game-id 2
```

### Change Speed Threshold
```
python ./scripts/possession_epv_demo.py --epv-grid .\src\metrica\data\EPV_grid.csv --speed-threshold 4.0
```

### Compute-Only
```
python ./scripts/possession_epv_demo.py --epv-grid .\src\metrica\data\EPV_grid.csv --speed-threshold 4.0
```
