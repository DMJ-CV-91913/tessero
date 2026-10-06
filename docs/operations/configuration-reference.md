# Configuration reference

Every setting that affects Tessero, by where it is set.

## php.ini: extension

| Setting | Default | Scope | Effect |
|---|---|---|---|
| `extension=tessero` | | php.ini | loads ext-tessero |
| `tessero.threads` | `1` | `PHP_INI_ALL` | OpenMP threads for element-wise kernels on ≥ 131 072 elements and MDP sweeps on > 4 096 state-action pairs. Clamped to [1, 4 × cores]. 1 when built without OpenMP |
| `tessero.memory_budget` | `0` | `PHP_INI_ALL` | native memory cap for extension arrays and solver workspaces, bytes (`256M`, `2G` accepted). 0 = unlimited |
| `tessero.epsilon` | `1e-6` | `PHP_INI_ALL` | default tolerance of `Engine::mdpValueIteration` |

The extension re-applies `tessero.threads` and `tessero.memory_budget` at every
request start, so runtime changes last for one request.

## php.ini: FFI package

| Setting | Recommended | Effect |
|---|---|---|
| `ffi.enable` | `preload` (FPM), `1` or `preload` (CLI) | `preload` allows FFI only from preloaded code, plus the CLI |
| `opcache.enable` | `1` | required for preloading |
| `opcache.preload` | `…/vendor/tessero/tessero/resources/preload.php` | binds libtessero and BLAS at startup |
| `opcache.preload_user` | the FPM pool user | preload runs as this user |
| `opcache.jit` | any | all suites pass with JIT off, `function` and `tracing` (see [upstream bugs](../project/upstream-bugs.md)) |

## Environment variables

| Variable | Read by | Effect |
|---|---|---|
| `TESSERO_LIB` | FFI package, preload | absolute path of `libtessero` to load instead of the bundled binary |
| `TESSERO_BLAS` | FFI package, preload | absolute path of an LP64 OpenBLAS/LAPACKE library |
| `TESSERO_BLAS_THREADS` | FFI package | OpenBLAS threads (default: 1 outside the CLI) |
| `TESSERO_BACKEND` | Laravel | `auto`, `ext`, `ffi` |
| `TESSERO_THREADS` | Laravel | → `threads` |
| `TESSERO_MEMORY_BUDGET` | Laravel | → `memory_budget` (bytes) |
| `TESSERO_EPSILON` | Laravel | → `epsilon` |
| `TESSERO_CAST_FORMAT` | Laravel | `json` or `binary` |
| `TESSERO_MAX_ELEMENTS` | Laravel | default `max_elements` |
| `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` | OpenBLAS / OpenMP runtimes | read by those libraries at load time. Prefer the Tessero settings, which apply at runtime |

## Laravel: config/tessero.php

| Key | Default | Effect |
|---|---|---|
| `backend` | `auto` | ext when loaded, else FFI; `ext`/`ffi` require one |
| `threads` | 1 | applied to both backends at boot and before every Octane request/task and queued job |
| `memory_budget` | 268 435 456 | bytes per process, both backends (each has its own counter) |
| `blas_threads` | 1 | OpenBLAS threads (FFI) |
| `epsilon` | 1e-6 | MDP tolerance (extension default and `Tessero::solveMdp`) |
| `cast_format` | `json` | default for `AsNDArray` without an argument |
| `max_elements` | 1 000 000 | default cap for request validation |

## Runtime API

| FFI package | Extension | Notes |
|---|---|---|
| `Tessero::setThreads(n)` / `threads()` | `Engine::setMaxThreads(n)` / `getMaxThreads()` | process-wide |
| `Tessero::setMemoryBudget(bytes)` / `memoryBudget()` | `Engine::setMemoryBudget(bytes)` / `memoryBudget()` | process-wide |
| `Tessero::memoryInUse()`, `peakMemory()` | `Engine::memoryInUse()`, `peakMemory()` | bytes |
| `Tessero::setBlasThreads(n)` | | OpenBLAS |
| | `Engine::setEpsilon(x)` / `getEpsilon()` | |
| `Tessero::info()` | `Engine::info()` | diagnostics ([Monitoring](monitoring.md)); the extension also reports `memory_mapped` |

## Build-time options

| Option | Where | Effect |
|---|---|---|
| `OPENMP=` | `make -C csrc` | build libtessero without OpenMP |
| `--disable-tessero-openmp` | `./configure` | build the extension without OpenMP (no libgomp) |
| `CC=clang` | both | compiler choice; GCC gives runtime SIMD dispatch (`target_clones`) |
| `-DTSR_NO_HUGE` | CFLAGS | disable transparent-huge-page hints for blocks ≥ 32 MiB |
