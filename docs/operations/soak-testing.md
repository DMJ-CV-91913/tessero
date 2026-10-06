# Soak testing

A soak test runs Tessero for hours in one long-lived process, the way a queue
worker or an Octane worker does. It fails if anything accumulates. Unit tests
cannot find a leak of a few bytes per request; a day of requests can.

## The harness

`tools/soak.php` runs a mixed workload in a loop. One iteration covers:

- ufuncs with `out`, including in place;
- views, multi-axis reductions with `keepdims`, boolean masks, JSON;
- sort, argsort, FFT, matrix products;
- `.npy` files created with `openMemmap`, then loaded in every mode (`r`,
  `c`, plain load), plus raw memory maps with `flush`;
- LP and MDP solves;
- error paths: a shape error and a missing file.

Every iteration frees everything it allocates.

```bash
# FFI package, tracing JIT, one hour
php -d ffi.enable=1 -d opcache.enable_cli=1 -d opcache.jit=tracing -d opcache.jit_buffer_size=64M \
    tools/soak.php --backend=ffi --duration=3600 --log=soak-ffi.jsonl

# native extension, 24 hours, 4 OpenMP threads
php -d extension=tessero tools/soak.php --backend=ext --duration=86400 --threads=4 --log=soak-ext.jsonl
```

| Option | Default | Meaning |
|---|---|---|
| `--backend` | `ext` if loaded, else `ffi` | binding under test |
| `--duration` | 60 | seconds of wall-clock time after a 20-iteration warm-up |
| `--threads` | 2 | OpenMP threads for the run |
| `--interval` | 10 | seconds between samples |
| `--log` | none | one JSON sample per interval: iterations, native bytes, mapped bytes, PHP heap, RSS, open descriptors |

## Pass criteria

| Measure | Rule |
|---|---|
| Native bytes in use (`memory_in_use`) | back at the post-warm-up baseline after **every** iteration (exact) |
| Mapped bytes (`memory_mapped`) | back at baseline after every iteration (exact) |
| PHP heap (`memory_get_usage()`) | slope over the second half of the run < 16 KiB/min |
| RSS | slope over the second half < 64 KiB/min |
| Open file descriptors | growth ≤ 2 |

The first half of the run absorbs allocator arenas, OPcache and JIT warm-up.
Trends are computed in constant memory, so the harness does not grow the heap
it measures. The exit code is 0 on pass and 1 on fail, and a JSON summary is
printed.

## When to run it

- **Nightly CI** (`.github/workflows/nightly.yml`) runs 5.5 hours per backend:
  the extension with JIT off, and the FFI package with the tracing JIT.
- **Before every release**, run 24–72 hours on hardware like production's,
  with the thread count production uses. Attach the summaries and logs to the
  release notes.
- **After upgrading PHP**, run at least a few hours. Engine changes (JIT, FFI)
  are the most likely source of new leaks ([Upstream bugs](../project/upstream-bugs.md)).

## Soaking a real deployment

`tools/soak.php` defines `tessero_soak_iteration(string $backend, int $i,
string $dir)`, and the file can be included without running its CLI part. To
soak an actual Octane or Horizon deployment, include it and call the function
from a route or a job, then watch the worker's metrics:

```php
// routes/web.php (staging only)
Route::get('/_soak', function () {
    require_once base_path('vendor/tessero/tessero/tools/soak.php');
    static $i = 0;
    tessero_soak_iteration(extension_loaded('tessero') ? 'ext' : 'ffi', $i++, storage_path('app/soak'));

    return response()->json(Tessero::info());
});
```

Drive it with a load generator (for example
`hey -z 24h -c 4 https://staging.example.com/_soak`). Watch
`memory_in_use`, `memory_mapped` and the worker RSS
([Monitoring](monitoring.md)). With `max_requests` disabled, all three must
stay flat.

## Results so far

Runs on the development machine (2 cores, PHP 8.4.21, both backends at once)
are recorded in the changelog. No accumulation has been seen. These runs lasted
minutes, not days; the 24–72-hour runs belong to the release checklist
([Releasing](../project/releasing.md)).
