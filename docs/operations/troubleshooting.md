# Troubleshooting

Find the message or symptom, then follow the fix. `tessero doctor` catches
most installation problems.

## Installation and loading

### `FFI API is restricted by "ffi.enable" configuration directive`

The FFI package was called from non-preloaded code with
`ffi.enable=preload` (typically under FPM). Configure the preload script
([Deployment](deployment.md#with-the-ffi-package)) and restart FPM, or install
the extension.

### `LibraryUnavailable: libtessero X does not match tessero/tessero Y`

The binary is from a different release than the PHP code: usually `TESSERO_LIB`
points to an old build, or a deploy replaced the vendor directory but not a
cached library. Tessero refuses to run rather than risk mismatched struct
layouts. Remove `TESSERO_LIB`, or rebuild with `make -C vendor/tessero/tessero/csrc`.
A message ending in `Failed resolving C function` means the same thing.

### `LibraryUnavailable: No prebuilt libtessero for <platform>`

There is no binary for this OS/architecture/libc in `lib/`. Run
`vendor/bin/tessero build` (needs a C compiler), or build elsewhere and set
`TESSERO_LIB=/abs/path/libtessero.so`.

### `LibraryUnavailable` for linear algebra / "no LP64 OpenBLAS with LAPACKE"

Install `libopenblas0` and `liblapacke` (Debian/Ubuntu), `openblas` (macOS
Homebrew), or point `TESSERO_BLAS` at a library. ILP64 builds (64-bit
integers, often named `libopenblas64_`) are refused on purpose.

### `PHP Warning: Module "tessero" is already loaded` / `Module compiled with module API=…`

The INI enables the extension twice, or the `.so` was built for another PHP
version. Check `php --ini` and rebuild with that version's `phpize`.

### `undefined symbol: omp_get_num_procs` or `libgomp.so.1: cannot open shared object file`

The extension was built with OpenMP but the runtime image lacks libgomp.
Install `libgomp1` (Debian) or `libgomp` (Alpine), or rebuild with
`--disable-tessero-openmp`.

### The Laravel app uses FFI although the extension is installed

The extension is not loaded for that SAPI (`php-fpm -m | grep tessero`), or
`TESSERO_BACKEND=ffi` is set. `tessero:doctor` shows the backend.

## Runtime errors

### `MemoryError: Could not allocate … (in use: …, budget: …)` / `Tessero\Ext\MemoryException`

The allocation would exceed the native memory budget. Either the input is
larger than expected (check validation caps), the budget is too small for the
workload ([sizing](memory-and-threads.md#sizing)), or application code keeps
arrays alive (look for statics/singletons, and check `memoryInUse()` between
jobs). The failed call has no side effects. Retrying with a smaller input or a
larger budget is safe.

### `ShapeError: Shapes (2, 3) and (2) cannot be broadcast together`

Operand shapes do not align from the right. Reshape explicitly, e.g.
`$v->reshape(2, 1)` to broadcast along columns.

### `IndexError` on slicing

An integer index is out of range, or there are more indices than
dimensions. Slices (`a:b`) never raise; they clip.

### `DTypeError` when creating an array

The input contains strings, nulls or nested non-rectangular data. Validate
input with `NumericArray` first, or cast numeric strings: `array_map('floatval', …)`.

### `InvalidArgumentException: linprog: c, A_ub, b_ub, A_eq and b_eq must be finite`

The problem data contains NaN or ±INF (often from a division by zero upstream).
Infinite *bounds* are allowed; infinite coefficients are not.

### `Transition probabilities of state S under action A sum to X, not 1`

The MDP row is not a distribution. Common causes: rounding in data exported
from a spreadsheet (normalise rows), or a missing "stay" transition. The
state and action are named in the message.

### `SingularMatrix` / `NotPositiveDefinite`

The matrix is singular or not positive definite in floating point. For
least squares use `Linalg::lstsq`. For nearly singular covariance matrices add
a small ridge (`$C->add(NDArray::eye($n)->mul(1e-10))`) or use `eigh`.

### `the array is read-only (memory-mapped with mode 'r')`

A write (assignment, `assign()`, or use as `out`) reached an array opened with
`NDArray::memmap($file, 'r')`, or a view of one. Open it with `'r+'` to change
the file, `'c'` to change it in memory only, or `clone` it for a heap copy.

### `memmap: ... is outside the allowed path(s) (open_basedir)`

The resolved path is not under `open_basedir`. Move the file or extend the
setting for that pool.

### Worker dies with "Bus error" (SIGBUS) while reading a memmap

The mapped file became shorter than the mapping (another process truncated or
rewrote it). Pages past the new end no longer exist. Coordinate writers
(write to a new file, then rename it into place) instead of truncating files
that are mapped.

### A solve runs longer than `max_execution_time`

PHP checks its timeout only between PHP instructions. A long C call (a large
MILP, an MDP with many states) runs to completion. If it overruns the timeout
by more than PHP's hard-timeout grace period (2 s by default), PHP terminates
the whole process. Under FPM, `request_terminate_timeout` also kills the
worker. Bound solver work explicitly (`nodeLimit`, `maxIter`, `maxiter`) and run
long solves in queue workers with an appropriate `--timeout`.

### Results differ slightly between machines

Element-wise and reduction results are deterministic for a given build and
input, and do not depend on the thread count. Across CPUs, the SIMD path
chosen at load time (baseline, AVX2 with FMA, AVX-512) can change the last
bit of some floating-point results, as in NumPy. BLAS/LAPACK results can differ
between OpenBLAS versions at round-off level. Random streams, integer
arithmetic, sorting, indexing and JSON are bit-identical everywhere.

### A process hangs after `pcntl_fork()`

OpenMP (libgomp) does not support using its thread pool in a forked child once
the parent has used it. Keep `threads = 1` in processes that fork
([Memory and threads](memory-and-threads.md#openmp-and-fork)).

### Wrong results only with the JIT enabled

Please report it (with `opcache.jit` settings and PHP version). Tessero works
around three PHP JIT defects found during development
([Upstream bugs](../project/upstream-bugs.md)), and a new one is possible. As an
immediate mitigation set `opcache.jit=disable`. Results with the JIT off
are the reference.

## Crashes (segmentation fault, bus error)

Tessero's contract is that PHP code cannot crash the process: every
out-of-range shape, index, mode, read-only write and malformed file raises an
exception. A crash is therefore a bug. Please report it with the steps below.
Two known causes are outside that contract:

- **SIGBUS on a memory-mapped array** means the file was truncated, or its
  network filesystem went away, while it was mapped. This is inherent to
  `mmap` ([Memory-mapped arrays](../guide/memmap.md)). Map only files that no
  other process shrinks.
- **A hang or crash after `pcntl_fork()`** with threads > 1 is libgomp
  ([RB-7](runbooks.md#rb-7-hang-after-pcntl_fork-horizon-custom-supervisors)).

To collect what a maintainer needs:

1. **Core dump and backtrace.**
   ```bash
   ulimit -c unlimited                      # or systemd: LimitCORE=infinity
   php script.php                           # reproduce
   gdb "$(command -v php)" core -batch -ex 'bt full' -ex 'info sharedlibrary' > crash.txt
   ```
   Under FPM, set `rlimit_core = unlimited` in the pool configuration.
   Frames in `libtessero.so` or `tessero.so` point at the kernel. Frames in
   `libgomp` point at threads.
2. **Reproduce under AddressSanitizer.** It reports the exact line and the
   allocation history, even when the crash itself is later and elsewhere.
   ```bash
   # FFI package: an instrumented kernel
   make -C vendor/tessero/tessero/csrc CFLAGS="-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -fPIC -std=gnu11" \
        LDLIBS="-lm -fsanitize=address,undefined"
   LD_PRELOAD="$(gcc -print-file-name=libasan.so)" ASAN_OPTIONS=detect_leaks=0 USE_ZEND_ALLOC=0 php script.php

   # extension: build it with sanitizers (as CI does)
   ./configure --enable-tessero CFLAGS="-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer" LDFLAGS="-fsanitize=address,undefined"
   ```
   `USE_ZEND_ALLOC=0` makes PHP use the system allocator, so ASan sees PHP's
   allocations too. `detect_leaks=0` avoids reports about PHP's own
   intentional process-lifetime allocations.
3. Run with `threads = 1`. If the crash disappears, say so in the report.
