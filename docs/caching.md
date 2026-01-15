# Caching and performance

`metrica.Metrica_IO` caches certain file reads in-memory to speed up repeated loads in the same Python session.

## What is cached

- Event CSV reads
- Tracking header parsing (column naming)

The cache key includes the file path and file modification time (mtime), so it auto-invalidates when the file changes.

## When it helps

- Re-running tutorials repeatedly
- Iterating in notebooks where you load the same game data many times

## Clearing caches

```
import metrica.Metrica_IO as mio
mio.clear_caches()
```