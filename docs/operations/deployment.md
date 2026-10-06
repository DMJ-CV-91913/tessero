# Deployment

## 1. Check the machine

```bash
vendor/bin/tessero doctor              # FFI package
php --ri tessero                       # extension
php artisan tessero:doctor             # Laravel: both, plus the selected backend
```

`doctor` exits non-zero when something needs attention, so it can gate a
deployment pipeline. Run it **with the same php.ini as production** (for FPM:
`php -c /etc/php/8.4/fpm/php.ini vendor/bin/tessero doctor`). The CLI
often has a different configuration.

## 2. Pick the backend per runtime

| Runtime | Recommended | Configuration |
|---|---|---|
| CLI scripts, cron, `artisan` commands | either | FFI: `ffi.enable=1`, or `preload` (the CLI SAPI may still call `FFI::cdef`) |
| Queue workers (`queue:work`, Horizon) | extension, or FFI | same as CLI; set `threads` for dedicated heavy queues |
| PHP-FPM / Apache mod_php | extension | FFI needs `ffi.enable=preload` + the preload script (below) |
| Octane (Swoole, RoadRunner, FrankenPHP) | extension | FFI: preload as for FPM, or `ffi.enable=1` if you accept that risk (see [Security](security.md)) |

## 3. PHP-FPM and Apache

### With the extension (simplest)

```ini
; /etc/php/8.4/mods-available/tessero.ini
extension = tessero
tessero.memory_budget = 256M
tessero.threads = 1
```

Enable it for the FPM SAPI and restart FPM. Nothing else is needed.

### With the FFI package

Keep PHP's secure default `ffi.enable=preload` and preload the library:

```ini
ffi.enable = preload
opcache.enable = 1
opcache.preload = /var/www/app/vendor/tessero/tessero/resources/preload.php
opcache.preload_user = www-data
```

`vendor/bin/tessero php-ini` prints these lines with your real paths.

What the preload script does:

1. Writes a small header with `#define FFI_SCOPE "tessero"` and
   `#define FFI_LIB "<absolute path>"` to the temp directory and calls
   `FFI::load()` on it.
2. Does the same for the first LP64 OpenBLAS with LAPACKE it finds (scope
   `tessero_blas`). `TESSERO_BLAS` overrides the search.
3. Preloads every class under `src/`. **This is required.** With
   `ffi.enable=preload`, PHP lets only preloaded code call FFI APIs such as
   `FFI::new` and `FFI::cast`.

Preloaded code is fixed when FPM starts. **Restart (not reload) FPM after
every `composer install/update`** that changes Tessero, or requests will run
the old code against the new library.

Verify on a staging host with the same check CI runs:

```bash
bash vendor/tessero/tessero/tests/e2e/preload.sh
```

It starts `php -S` with `ffi.enable=preload` and `FFI::cdef` forbidden, and
runs arithmetic and LAPACK through the preloaded scopes.

## 4. Octane

- Prefer the extension. Octane workers are long-lived, and the extension has
  no preload step to keep in sync with deployments.
- `tessero/laravel` re-applies threads and the memory budget at the start of
  every request and task. Without Laravel, call
  `Engine::setMaxThreads()`/`Engine::setMemoryBudget()` (or the FFI equivalents)
  at the start of each request.
- Watch for arrays kept in static properties or singletons. They stay for the
  worker's lifetime ([Memory and threads](memory-and-threads.md)).
- Set `--max-requests` as usual. Tessero does not leak, but it bounds the
  damage of application code that does.

## 5. Queue workers and batch jobs

Heavy numerical work (large MILPs, MDPs with millions of states,
simulations) belongs on a dedicated queue with its own worker pool:

```bash
php artisan queue:work --queue=numerics --memory=1024 --timeout=900
```

- `--memory` watches PHP memory only. Native memory is capped by
  `TESSERO_MEMORY_BUDGET` ([Memory and threads](memory-and-threads.md)).
- On a dedicated pool, set `TESSERO_THREADS` to the cores each worker may
  use: for example 4 workers × 2 threads on an 8-core host.
- `--timeout` must exceed the longest solve. Solver calls run in C and cannot
  be interrupted part-way by PHP's `max_execution_time` (see
  [Troubleshooting](troubleshooting.md#a-solve-runs-longer-than-max_execution_time)).
  Bound them with `maxIter`/`nodeLimit` instead.

## 6. Deployment checklist

- [ ] `doctor` passes under the production php.ini of every SAPI you use.
- [ ] A memory budget is set per worker (`tessero.memory_budget` or `TESSERO_MEMORY_BUDGET`).
- [ ] `threads` is 1 under FPM unless you have sized for more.
- [ ] Public endpoints validate array input with a size cap (`NumericArray(maxElements: …)`).
- [ ] FFI + FPM: preload configured, and FPM restarts on deploy.
- [ ] Health check includes `tessero:doctor --json` or `tessero info --json` ([Monitoring](monitoring.md)).
- [ ] The same backend and version is used in staging and production.
