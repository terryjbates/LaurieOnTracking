# Metrica EPV
## Using cache
Use after loading EPV
```
from metrica_epv import EPVCache, load_EPV_grid, calculate_epv_added

EPV = load_EPV_grid("EPV_grid.csv")

cache = EPVCache()
cache.prime_epv(EPV)  # precomputes the flipped EPV grid (big win)
```

Also, reuse same cache after many calls

```
eevp, raw = calculate_epv_added(
    event_id,
    events,
    tracking_home,
    tracking_away,
    GK_numbers,
    EPV,
    params,
    cache=cache,
)
```

Likewise:

```
max_added, (x, y) = find_max_value_added_target(
    event_id,
    events,
    tracking_home,
    tracking_away,
    GK_numbers,
    EPV,
    params,
    cache=cache,
)
```

### What exactly gets cached
* home_attack_direction (computed once)
* EPV grid flipped for right-to-left attacks (computed once)
* Expensive player initialisation + offside checks for each (Start Frame, Team) pair
(reused if multiple events happen in the same frame)


# Metrica PitchControl
## Create Cache once per match
```
import Metrica_PitchControl as mpc
cache = mpc.PitchControlCache()
```

## Pass into calls that initiate players / compute surfaces
```
PPCF, xgrid, ygrid = mpc.generate_pitch_control_for_event(
    event_id,
    events,
    tracking_home,
    tracking_away,
    params,
    GK_numbers,
    cache=cache,      # <-- enables caching
    offsides=True,
)
```

## What caching buys you (and when it matters)

`initialise_players(...)` is expensive mainly due to per-player object creation and per-column parsing.

When you compute multiple surfaces / multiple targets for the same frame, caching prevents rebuilding those player lists repeatedly.