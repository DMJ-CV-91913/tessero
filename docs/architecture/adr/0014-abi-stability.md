# 0014. A versioned C ABI: exported surface, soname, capability query and process settings

- Status: Proposed
- Date: 2026-10-23
- Amends: 0004 (budget default), 0006 (ZTS)

## Context

`libtessero.so` is linked directly by other native code (Talario, Kinetix) as well as loaded by the PHP FFI
binding, and the Zend extension embeds a second copy of the same kernel. Before this record:

- The library was built with `-fvisibility=default` and no version script, so it exported 4864 dynamic symbols:
  the 113 `tsr_` functions of `include/tessero.h`, but also the 89 f2c-translated FITPACK routines (`curfit_`,
  `surfit_`, ...), numpy's `random_*` distribution functions, cdflib, QUADPACK, internal helpers
  (`fn_set_error`, `fn_call_routine`, ...) and the whole statically linked C++ runtime. Any of them could
  interpose on, or be interposed by, a same-named symbol of another library in the process (a scipy or numpy
  build, another FITPACK, another libstdc++).
- The soname was the unversioned `libtessero.so`, so nothing stopped a consumer built against one ABI from
  loading another.
- Nothing told a consumer which optional capabilities a given build has.
- The extension never capped the BLAS thread pool, so every php-fpm worker ran an all-core OpenBLAS pool (only the
  FFI binding capped it, in `src/Native/Blas.php`).
- The memory budget defaulted to unlimited, so native arrays in a php-fpm worker were not bounded by
  `memory_limit` at all.
- Under ZTS, `RINIT` wrote each request's thread count and budget into process globals that concurrent requests
  were using.

## Decision

### Exported surface

- The kernel is compiled hidden-by-default (`csrc/Makefile` `VIS`: `-fvisibility=hidden -DTSR_EXPORT_ABI`).
  `src/internal.h` wraps the include of `tessero.h` in `#pragma GCC visibility push(default)`, so exactly the
  declarations of the public header get default visibility. `repro_f32.c` (which does not include
  `internal.h`) is compiled with `-fvisibility=default`; it defines only `tsr_repro_*` ABI functions.
- A GNU ld version script, `csrc/libtessero.map`, exports `tsr_*` in the version node `TESSERO_0.2` and makes
  everything else local, including the internal `tsr_` helpers it names (`tsr_psum`, `tsr_pcg64_bitgen`,
  `tsr_physical_bytes`, `tsr_random_run`, `tsr_cdflib_runaway`, `tsr_effective_cpus`). It is probed, so a
  linker without version scripts (macOS ld64) builds without it; there the visibility rule alone applies.
- The library links with `-Bsymbolic-functions` (probed), so its own calls to its exported functions
  (`tsr_alloc`, `tsr_free`, `tsr_get_threads`, ...) bind inside it. This was observed failing on the reference
  host: Talario's PHP extension loads an installed `libtessero.so` into the global scope, and the FFI-loaded copy
  then resolved its internal calls (allocator, budget, distribution error slot) to that other instance. Three
  unit tests (budget -> MemoryError) and three adversarial stats inputs (expected exceptions) failed there for
  that reason alone; with the hidden build plus `-Bsymbolic-functions` they pass. `make abi-check` proves it
  with `tests/abi_interpose.c`: it loads a renamed copy of the library `RTLD_GLOBAL`, then the build
  `RTLD_LOCAL`, and checks that the build's allocations and budget stay its own (the 0.2 release without the
  flag fails this check).
- `make -C csrc abi-check` fails unless the exported dynamic symbols are exactly the functions declared in
  `tessero.h` (the ifunc resolvers GCC emits for exported `target_clones` functions aside) and the soname is the
  expected one. Result for 0.2: 4864 exported symbols became 118 functions plus 5 resolvers.
- The rule for new code: a function is public if and only if it is declared in `include/tessero.h`. An internal
  helper must not use the `tsr_` prefix unless it is listed as local in the map.

### ABI version, soname and capability query

- One version source: `csrc/include/tessero_version.h` (`TSR_VERSION_STRING`, `TSR_ABI_VERSION`). `tsr_version()`
  and `tsr_abi_version()` return it, the Makefile reads the soname from it and the extension's
  `PHP_TESSERO_VERSION` is defined from it.
- `TSR_ABI_VERSION` is 2 for 0.2.x. It is bumped on any incompatible change: a removed or renamed function, a
  changed signature, a changed layout of `tsr_array`, `tsr_arg` or `tsr_result`, a changed constant or a changed
  documented meaning. A compatible addition (a new function, a new `tsr_feature` name) does not bump it.
- The soname is `libtessero.so.<TSR_ABI_VERSION>` (`libtessero.so.2`). The file keeps its name `libtessero.so`,
  because the FFI loader opens `lib/<platform>/libtessero.so` by path, and the build creates the symlink
  `libtessero.so.2 -> libtessero.so` next to it for programs linked with `-ltessero`. An incompatible change
  bumps `TSR_ABI_VERSION`, which changes the soname and starts a new version node in the map; a program linked
  against the old ABI then fails at load time instead of misbehaving.
- `int tsr_feature(const char *name)` answers 1/0 for named capabilities of the running build: `repro_f32`,
  `gemm_canonical`, `lapack`, `openblas_threads`, `openmp`, `fpenv_check` (0 until the floating-point environment
  check of the FP-environment workstream defines `TSR_HAVE_FPENV`). Unknown names answer 0, so a consumer can ask
  about a capability newer than the library. A binding checks `tsr_abi_version()` once at load and uses
  `tsr_feature` for everything optional.

### Two kernel instances in one process

The Zend extension compiles the kernel into `tessero.so` with every symbol local (`ext/tessero.map` exports only
`get_module`), and Talario or the FFI binding may load `libtessero.so` into the same process. These are two
independent kernels: separate allocators, separate budgets and allocation counters, separate thread settings,
separate error slots. Memory from one must never be freed by the other (`tsr_free` of the other instance would
corrupt both counters), and an array is handed across only as data (copy), never as an owned block. Hidden
symbols guarantee the two never bind to each other's functions by accident.

### Process settings in the extension

- `tessero.blas_threads` (new; `PHP_INI_SYSTEM | PHP_INI_PERDIR`): `auto` (default) caps the BLAS pool at 1 in a
  web SAPI and leaves it alone in the CLI; `0` leaves it alone; `n >= 1` caps it at `n`. The kernel resolves
  `openblas_set_num_threads` / `goto_set_num_threads` / `MKL_Set_Num_Threads` by name at first use (its own
  dependency tree first, then the global scope), so it has no link-time dependency on a particular BLAS
  (`tsr_set_blas_threads`, `tsr_get_blas_threads`, `Engine::setBlasThreads`, `Engine::getBlasThreads`). It is
  applied at MINIT and, when the value changed, at RINIT.
- `tessero.memory_budget`: `0` (default) is now automatic: unlimited in the CLI, and in a web SAPI
  `max(256 MiB, memory_limit)`, or unlimited when `memory_limit` is `-1`. `-1` is an explicit unlimited, a
  positive value is used as given. `Engine::setMemoryBudget()` is unchanged (kernel semantics, `0` = unlimited).
- Under ZTS, `tessero.threads`, `tessero.memory_budget` and `tessero.blas_threads` are `PHP_INI_SYSTEM`: one value
  per process, refused in pool configuration, `.htaccess` and `ini_set`. They are applied once at MINIT and not
  re-applied per request, because a per-request write to process state races with the requests running
  concurrently. The derived budget uses the startup `memory_limit`. `Engine::setMaxThreads` and
  `Engine::setMemoryBudget` still act process-wide, as documented in 0006.
- Fork: a `pthread_atfork` child handler resets the kernel thread count to 1 when the parent ran (or asked for)
  multi-threaded OpenMP regions, because libgomp's worker pool is not usable in the child. The next explicit
  `tsr_set_threads` (the extension's RINIT under NTS) restores it. A parent that only configured threads, such as
  a php-fpm master after MINIT, keeps the setting in its children. Running Tessero compute before a fork (for
  example in opcache preload) is not supported.
- The thread-count clamp (`tsr_set_threads`, 4 x CPUs) counts the CPUs the process may use: the affinity mask
  and the cgroup v1/v2 CPU quota, whichever is smaller, instead of every CPU of the machine.
- The reproducible-f32 regions (`tsr_repro_*`) now pass `num_threads(tsr_get_threads())` like every other kernel
  loop, instead of taking OpenMP's default (all cores, or `OMP_NUM_THREADS`).
- Distribution error messages (`cxx/stats_dist.cpp`) are kept per calling thread, so concurrent ZTS requests no
  longer see or clear each other's errors.

### Checked and counted allocation

- `src/internal.h` provides `tsr_count_mul`, `tsr_bytes_n` and the raw scratch helpers `tsr_scratch_n` /
  `tsr_scratch0_n`; `tsr_alloc_n(n, elsize, pad)` (public) is the counted, overflow-checked allocation. A size
  that is a product of two independent lengths must go through them.
- Migrated: `scipy_cluster_hier.c` (every `n*n`, `(n-1)*4`, `m*2`, `(2n-1)*2` and the `n*n*2` swap-status table);
  its three `n*n` double workspaces (`linkage`, `fclusterdata`, `optimal_leaf_ordering`) are now counted
  `tsr_alloc`/`tsr_free` blocks, so the budget refuses an oversized clustering instead of the OOM killer.
  `scipy_signal.c`: 34 allocations that were used without a NULL check now fail with `TSR_ENOMEM` and free what
  they hold; `zpk2tf_real` reports allocation failure to its five callers.

## The remaining work (not done here)

### Counted allocator for the raw `malloc` sites

About a thousand kernel sites (outside `third_party/`) still use `malloc`/`calloc`/`realloc`/`free` directly,
so their memory is invisible to `tsr_allocated()` and the budget. Redirecting them wholesale through a counted
allocator with a size header (`#define malloc tsr_counted_malloc` in the kernel's own translation units) is only
safe if no pointer ever crosses between redirected and non-redirected code. That is not provable today:

- kernel code frees memory allocated by LAPACKE work queries and by vendored third-party code, and hands buffers
  to third-party routines that `realloc` or `free` them (FITPACK work arrays, numpy random, cdflib, Boost);
- C++ code (`cxx/`) mixes `std::vector` with `malloc`ed buffers across the C boundary.

The plan: (1) migrate file by file, starting with the largest workspaces (`np_linalg.c` LAPACK workspaces, owned
by another workstream, then `scipy_sparse_linalg.c`, `scipy_interpolate.c`, `scipy_ndimage.c`), replacing each
`malloc`/`free` pair with `tsr_alloc_n`/`tsr_free` where both ends are in one function; (2) add a lint that
rejects new raw `malloc` in kernel files that have been migrated; (3) run ASan and both parity backends after each
file. Raw scratch below a few KiB may stay raw.

### Per-context kernel settings

Thread count, budget and allocation counters are process globals. A per-request context (a `tsr_ctx *` holding
threads, budget, counters and the error slot) passed to every kernel call, or installed in thread-local storage
by the binding at request start and inherited by the OpenMP team through a per-call sink, would make ZTS
requests independent. It changes every kernel signature (an ABI bump) or needs every parallel region to forward
the caller's context, so it is deferred until a supported deployment needs per-request settings under ZTS.
Until then: under ZTS the settings are process-wide by construction (`PHP_INI_SYSTEM`).

Related: a failure raised on an OpenMP worker thread of a threaded element loop (`tessero.threads > 1`) still goes
through one process-wide slot in `stats_dist.cpp`, because the worker cannot see the caller's thread-local slot.
The per-call sink above removes that last shared slot.

## Consequences

- Consumers that link `libtessero` must resolve `libtessero.so.2` (the build creates the symlink; packages ship
  it). The FFI binding is unaffected: it opens the file by path.
- `-fvisibility=hidden` lets the compiler bind internal calls directly; nothing outside `tessero.h` can be called
  from outside the library any more.
- php-fpm workers no longer oversubscribe the CPU through BLAS, and their native memory is bounded by default.
- Under ZTS the three settings can no longer be changed per directory or per request.

## Alternatives considered

- Exporting by an explicit symbol list generated from the header: equivalent result, but a second list to keep in
  sync; `abi-check` enforces the same property against the header instead.
- Renaming the file to `libtessero.so.2` with a `libtessero.so` symlink: breaks Windows checkouts and the FFI
  path on systems that do not preserve symlinks.
- Making `Engine::setMaxThreads` throw under ZTS: breaks the Laravel provider, which calls it at boot.
