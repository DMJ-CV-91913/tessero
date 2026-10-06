# Live-testing checklist (beta)

This checklist is for validating Tessero outside the Linux review host. Use it
on your own machines, on CI runners, and in a real application. Each
section maps to a P0 item in the [open issues](../project/open-issues.md).
Record the results in the issue: platform, PHP version (`php -v`), the output
of `bin/tessero info --json`, and the pass/fail counts.

## 0. Repository and CI (TSR-001)

1. Push the repository to GitHub and enable Actions.
2. The first push runs `ci.yml`. Expect some jobs to fail on first contact:
   runner images, package names and Windows paths have only been reasoned
   about, not run.
3. Start `nightly.yml` by hand (Actions → nightly → Run workflow) once `ci.yml`
   is green.
4. Import the issue list:

```bash
gh auth login
php tools/beta-issues.php create --repo=OWNER/tessero --dry-run   # review
php tools/beta-issues.php create --repo=OWNER/tessero
```

## 1. The same steps on every platform

Run these from the repository root. Items 1–3 need a C compiler; items 4–6
need PHP with `ffi` and Composer.

| # | Step | Command | Pass when |
|---:|---|---|---|
| 1 | Build the kernel | `make -C csrc` | `csrc/build/libtessero.{so,dylib,dll}` exists |
| 2 | Kernel tests | `make -C csrc test` | 270 checks, 0 failures |
| 3 | Kernel under sanitizers (Linux, macOS) | `make -C csrc sanitize` | same, no ASan/UBSan report |
| 4 | Point PHP at the build | `export TESSERO_LIB=$PWD/csrc/build/libtessero.so` (`.dylib` on macOS, `.dll` on Windows) | `php bin/tessero doctor` reports no errors |
| 5 | FFI suite | `composer install && php -d ffi.enable=1 vendor/bin/phpunit --testsuite unit,parity` | 0 failures; skips are LAPACK-only if no LAPACK is installed |
| 6 | Documentation examples | `php -d ffi.enable=1 tests/docs/examples.php` | 59 OK |

### Native extension (where PHP development headers exist)

```bash
bash tools/sync-ext.sh --check
cd ext && phpize && ./configure --enable-tessero && make -j4
NO_INTERACTION=1 REPORT_EXIT_STATUS=1 make test TESTS=tests     # 14 phpt pass
cd .. && php -d extension=$PWD/ext/modules/tessero.so -d ffi.enable=1 vendor/bin/phpunit --testsuite ext,unit
```

The last command runs the cross-backend parity tests (`BackendParityTest`):
both bindings must give identical bytes.

## 2. Platform notes

**macOS (TSR-003).**

- Apple clang has no OpenMP. Build serially with `OPENMP= make -C csrc`, or
  install `libomp` with Homebrew and pass its include and lib paths.
- LAPACK comes from Accelerate or from `brew install openblas`. Check with
  `bin/tessero doctor`.
- On arm64 this is the first run of the NEON code paths. Watch
  `MathTest` (the exp/log accuracy checks) and `PropertyTest`.

**Windows (TSR-002).**

- The kernel builds with MSYS2 MinGW-w64: `pacman -S make mingw-w64-x86_64-gcc`,
  then `make -C csrc` from the MINGW64 shell.
- The extension needs the PHP SDK (`php-sdk-binary-tools`) and MSVC:
  `phpize && configure --enable-tessero && nmake`.
- This is the first compile of the Win32 memory-map code (`csrc/src/mmap.c`)
  and of the atomics under MSVC. Test every memmap mode (`r`, `r+`, `w+`,
  `c`) and check that the files can be deleted after `unset()`. Windows
  refuses to delete a file that is still mapped.

**Linux arm64 and Alpine/musl (TSR-004).**

- Run section 1 inside `php:8.3-cli-alpine` (`apk add build-base`) and on an
  arm64 host.
- musl's default thread stack is small. If OpenMP code crashes, set
  `OMP_STACKSIZE=8M` and record it in the issue.

**PHP versions (TSR-115).** Repeat section 1, items 5–6, on PHP 8.1, 8.2 and
8.3. Only 8.4 has been tested.

## 3. PHP-FPM with preload (TSR-006)

1. Quick check: run `bash tests/e2e/preload.sh`.
2. Configure a real FPM pool:

```ini
ffi.enable = preload
opcache.enable = 1
opcache.preload = /path/to/vendor/tessero/tessero/resources/preload.php
opcache.preload_user = www-data
```

3. Put load on it: `wrk -t4 -c64 -d10m http://host/route-that-uses-tessero`.
4. Do a graceful reload during the load: `kill -USR2 <fpm master pid>`.
5. Check that there are no 5xx responses and no `tessero-preload-*` or `tessero-blas-*` files left in
   `sys_get_temp_dir()`.
6. Start FPM once with `TESSERO_LIB` pointing at an older build. It must refuse
   to start with the version-mismatch message.

## 4. A real Laravel application (TSR-005)

Use a fresh application on each supported Laravel version (10, 11, 12):

```bash
composer create-project laravel/laravel app && cd app
composer require tessero/laravel      # or a path repository to your checkout
php artisan vendor:publish --tag=tessero-config
php artisan tessero:doctor
```

Then check each of the following:

- [ ] **Casts:** a model with an `AsNDArray` cast round-trips through MySQL,
  PostgreSQL and SQLite.
- [ ] **Queues:** 10 000 jobs that build, compute and serialise arrays run
  under Horizon. `Tessero::info()['memory_in_use']` logged at the end of each
  job stays flat.
- [ ] **Octane:** run with Swoole, RoadRunner and FrankenPHP. A route sets
  `threads` and `memory_budget` and the next request sees the defaults again.
- [ ] **FrankenPHP worker mode, several threads (TSR-009):** run concurrent
  requests that change settings. Nothing leaks between threads beyond what the
  documentation says is process-wide.
- [ ] **Soak inside the application:** call `tessero_soak_iteration()` from a
  route (see [Soak testing](soak-testing.md)) under `wrk` for an hour. RSS
  should be flat.
- [ ] **Both backends:** repeat with the extension loaded
  (`TESSERO_BACKEND=ext`, or `backend => 'ext'` in `config/tessero.php`).

## 5. Long soaks (TSR-007)

```bash
php -d ffi.enable=1 tools/soak.php --backend=ffi --duration=86400 --log=soak-ffi.jsonl
php -d extension=tessero tools/soak.php --backend=ext --duration=86400 --log=soak-ext.jsonl
```

- Pass: exit code 0 and `"result": "PASS"`.
- Attach the summary and the JSONL file to the issue.
- 72-hour runs are required before 1.0.

## 6. Fuzzing (TSR-008)

```bash
CC=clang make -C csrc fuzz FUZZ_TIME=28800      # 8 h per target
```

Commit any new corpus entries. Any crash is a P0 bug: include the
reproducer file from `csrc/build-fuzz/`.

## 7. Benchmarks (for the record, not a gate)

Run the benchmarks and record the CPU model (`lscpu` or
`sysctl -n machdep.cpu.brand_string`) with the results:

```bash
php -d ffi.enable=1 -d memory_limit=2G bench/run.php
```

If Python is available, also run the NumPy comparisons. The gap analysis
figures come from a single AVX-512 host, so results from AVX2 and ARM help
calibrate TSR-206.

## Reporting

- Open failures as new issues with the `beta` label.
- Include:
  - the platform line;
  - the output of `php bin/tessero info --json`;
  - the failing test name;
  - the smallest reproducer you have.
- A crash (segfault, SIGBUS, abort) is always P0. If you can, attach a core
  dump or ASan output ([Troubleshooting](troubleshooting.md) explains how to
  get them).
