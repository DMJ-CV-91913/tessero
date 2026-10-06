# Design brief → implementation map

Tessero 0.2 was built from three design briefs: an FFI kernel brief
("phpnum"), a Zend extension brief ("NumPHP"), and a discussion of moving LP
and MDP solvers into C with Laravel integration. A fourth brief (ufuncs and
memory-mapped arrays) extended the extension after 0.2. The briefs used working
names; the project keeps the name Tessero. This page maps every requested
artefact to the code that implements it, and records where the
implementation deliberately differs and why.

## Names

| Brief | Tessero |
|---|---|
| `phpnum_kernel.h` | `csrc/include/tessero.h` |
| `phpnum_kernel.c` | `csrc/src/*.c` (split by subsystem; see [C kernel](c-kernel.md)) |
| `NDArray.php` (FFI binding) | `src/NDArray.php` + `src/Native/{Library,Buffer,Blas,Abi}.php` |
| `php_numphp.h` | `ext/php_tessero.h` |
| `numphp_array.c` | `ext/tessero_ndarray.c` (+ `ext/tessero.c`, `ext/tessero_solvers.c`) |
| `numphp_laravel_bridge.php` | `laravel/` package (`tessero/laravel`) |
| `_php_numphp_array_object` | `tsr_obj` |
| `numphp_array_free_object` | `tsr_free_obj` |
| `numphp_array_do_operation` | `tsr_do_operation` |
| `numphp_array_read_dimension` | `tsr_read_dimension` |
| `numphp_iterator` | `tsr_iter_funcs` (`get_iterator`) |
| `NumPHP\Engine::setMaxThreads` | `Tessero\Ext\Engine::setMaxThreads` (also `Tessero\Tessero::setThreads`) |
| `Engine::setGlobalEpsilon` | `Engine::setEpsilon` / `tessero.epsilon` |
| `config/numphp.php` | `laravel/config/tessero.php` |
| `NumPHPServiceProvider` | `Tessero\Laravel\TesseroServiceProvider` |
| `TensorCast` | `Tessero\Laravel\Casts\TensorCast` (alias of `AsNDArray`) |
| `SciPy::linprog` | `Optimize\LinearProgramming::linprog`, `Engine::linprog`, `Tessero::linprog` |
| `MDP::value_iteration` / `SciPHP::valueIteration` | `Mdp\MarkovDecisionProcess::valueIteration`, `Engine::mdpValueIteration`, `Tessero::solveMdp` |
| `DTYPE_FLOAT64/FLOAT32/INT32/INT64` | dtypes 0–3, plus `uint8`, `bool`, `complex128` |

## FFI kernel brief

| Requirement | Implementation | Evidence |
|---|---|---|
| **Gap 1** element-wise and reductions in C | `tsr_binary`, `tsr_unary`, `tsr_reduce`, `tsr_moments`, `tsr_reduce_mid`, `tsr_cumulative` | parity suite; 1M add 0.9–1.6 ms vs NumPy 1.0–1.3 ms |
| Gap 1: allocator without the zero-fill tax | `tsr_alloc`: 64-byte aligned, not zeroed, huge pages ≥ 32 MiB | C tests; `zeros()` alone uses `tsr_calloc` |
| Gap 1: contiguous iteration | `tsr_iter` coalesces dimensions to one flat loop | [C kernel](c-kernel.md#the-iterator) |
| **Gap 2** N-d metadata struct | `tsr_array` (data, offset, ndim, dtype, shape[32], strides[32] in bytes) | `tessero.h` |
| Gap 2: broadcasting function | `tsr_broadcast_shape`, `tsr_broadcast_strides` (stride 0, no copy) | fuzz suite (300 NumPy cases) |
| Gap 2: views by metadata | `tsr_array_slice` (full slice grammar), `tsr_array_transpose`, `tsr_array_reshape` (fails with −7 when a copy is needed) | `FuzzTest` checks the C parser against NumPy |
| **Gap 3** dtype enum + polymorphic loops | 7 dtypes; macro-generated per-dtype loops selected by `switch` | promotion table test |
| **Gap 4** memory budget with a clean exception | budget in C (atomic counter, checked before allocating) → `MemoryError` | unit tests; see deviation 1 |
| Gap 4: views keep the root alive | FFI: each view holds the `Buffer`. Extension: each view holds the root owner object | lifecycle tests, ASan |
| **Gap 5** `LAPACKE_dgesv`, `LAPACKE_dgelsd` | `Linalg::solve`, `Linalg::lstsq` (and the rest of `Linalg`) | linalg parity at 1e-9 |

## Extension brief

| Phase | Requirement | Implementation |
|---|---|---|
| 1 | object struct integrated with `zend_object` | `tsr_obj` with `zend_object std` last; `zend_object_alloc` |
| 1 | flat contiguous block | one block per owner from `tsr_alloc` (see deviation 2) |
| 1 | stride-based indexing | byte strides; `tsr_array_at` |
| 1 | `free_obj` with view protection | `tsr_free_obj`; views reference the root owner; `get_gc` |
| 2 | backward-stepping broadcast validation | `tsr_broadcast_shape` (right-aligned, 1 stretches) |
| 2 | zero-copy broadcasting via stride 0 | `tsr_broadcast_strides` |
| 2 | `do_operation` for ADD/SUB/MUL/DIV | `tsr_do_operation`: ADD, SUB, MUL, DIV, POW, MOD; arrays, scalars, PHP arrays on either side |
| 3 | `read_dimension` with range strings returning views | `tsr_read_dimension` → C slice parser; masks and integer lists too |
| 3 | native `zend_object_iterator` | `get_iterator`; rows as views |
| 3 | JSON via `smart_str` | `toJson()`/`jsonSerialize()`; byte-identical to `json_encode(…, JSON_PRESERVE_ZERO_FRACTION)` |
| 4 | thread-local configuration | module globals (per thread under ZTS) + RINIT re-application (see deviation 3) |
| 4 | `Engine::setMaxThreads` controlling OpenMP | `Engine::setMaxThreads` → `tsr_set_threads` → `num_threads()` clause (see deviation 4) |
| 4 | Laravel provider + config + `TensorCast` | `tessero/laravel` |
| — | structured errors for invalid dimensions | typed exceptions (`ShapeException`, `IndexException`, …) |

## Solver and Laravel discussion

| Topic | Implementation |
|---|---|
| Simplex entirely in C | `tsr_linprog`: two-phase, Harris ratio test, Bland anti-cycling, SciPy-compatible duals |
| MILP | `tsr_milp`: branch and bound on the simplex |
| MDP value iteration in C | `tsr_mdp_value_iteration` with an ε-optimal stopping rule |
| Policy iteration, finite horizon | `tsr_mdp_policy_iteration` (exact or modified), `tsr_mdp_finite_horizon` |
| Sparse transitions | one CSR matrix with A·S rows; `fromSparse`, `fromTransitions`, CSR input to the extension |
| Octane safety | provider re-applies settings on `RequestReceived`, `TaskReceived`, `JobProcessing`; extension RINIT |
| Eloquent casting of arrays | `AsNDArray`/`TensorCast` with JSON or exact binary storage |

## Ufunc and memmap brief

| Brief | Tessero |
|---|---|
| `numphp_ufunc.c` | `ext/tessero_ufunc.c` (PHP interface) over `csrc/src/ufunc.c` (loops and engine, shared) |
| `numphp_memmap.c` | `ext/tessero_memmap.c` (bindings) over `csrc/src/mmap.c` (mapping, shared with the FFI package) |
| `php_numphp.h`: `is_mmap`, `fd`, `map_size` | `ext/php_tessero.h`: `tsr_obj.flags` (`TSR_F_MMAP`, `TSR_F_READONLY`) and `tsr_obj.map` → `tsr_map {addr, length, data_offset, path, mode, fd}` |
| `NumPHP\Math::sin($matrix)` | `Tessero\Ext\Math::sin($x, ?$out)` and `Tessero\Math::sin(...)` (FFI), 75 functions, `Math::apply()`, `Math::ufuncs()`; loops in `csrc/src/ufunc.c` |
| ufunc engine accepting function pointers or native ops | registry of kernel ufuncs + typed inner-loop tables; one engine for resolution, casting, broadcasting, coalescing, threading |
| zero-stride broadcasting of A and B into a target buffer | `tsr_broadcast_strides`; direct write into `out` when dtype/shape match and no partial overlap |
| loops the compiler can autovectorise | contiguous `restrict` fast paths, `omp simd`, `target_clones` (AVX-512/AVX2/baseline) |
| `NDArray::memmap(string $filename, string $mode, array $shape, string $dtype)` | same, plus `offset`; `shape` optional (inferred 1-D); modes `r r+ w+ c` as numpy.memmap |
| `double *data` = address from `mmap()` | `a.data` = mapping address + in-page offset; any dtype, not only double |
| `numphp_array_free_object` detects memmaps, `munmap` + close | `tsr_free_obj` → `tsr_memmap_release()`; runs after the last view |
| `$matrix->flush()` with `MS_ASYNC` / `MS_SYNC` | `flush()` = `MS_ASYNC`; `flush(sync: true)` = `MS_SYNC` + `fsync` (Windows: `FlushViewOfFile` + `FlushFileBuffers`) |
| clean errors for invalid file paths | typed exceptions naming the resolved path and OS error; `open_basedir`; stream wrappers refused |
| thread-safe (`ZEND_TLS`) | no new global state except an atomic mapped-bytes counter; the registry is `const`; `lgamma` uses `lgamma_r` |

Deviations: mapped bytes are not counted against the memory budget
([ADR-0009](adr/0009-ufunc-engine-and-memmap.md)). The Windows code path is
written against the documented API but has not been compiled yet.

## Deliberate deviations

1. **Memory budget in C, not a PHP static property.** A PHP-side counter
   misses allocations made inside C (solver workspaces, sort buffers) and
   cannot be shared by the extension. An atomic counter inside `tsr_alloc` sees
   every block and is checked before memory is requested.
   ([ADR-0004](adr/0004-native-allocator-and-budget.md))
2. **`tsr_alloc` instead of `emalloc`/`pemalloc`.** The Zend allocator does
   not give 64-byte alignment or huge pages. Its per-request heap would also
   make arrays unusable across requests in Octane workers and invisible to the
   budget. Leak safety comes from `free_obj`, which ASan verifies.
3. **Module globals instead of raw `ZEND_TLS`.** `ZEND_MODULE_GLOBALS` is the
   Zend API's thread-local storage (TSRM under ZTS). In addition, RINIT
   restores INI values at every request, which is what actually protects
   Octane workers from inheriting another request's settings. The kernel's
   thread count and budget are process-wide by nature.
   ([ADR-0006](adr/0006-process-wide-kernel-settings.md))
4. **`num_threads()` instead of `omp_set_num_threads`.** Setting the OpenMP
   global would affect every OpenMP user in the process (including OpenBLAS
   builds that use OpenMP). Passing the count on each parallel region keeps
   Tessero's setting local to Tessero.
5. **One kernel, two bindings.** The FFI and extension briefs describe
   separate engines. Tessero compiles the same C sources into both, so there is
   one implementation of each algorithm to test and maintain.
   ([ADR-0003](adr/0003-one-kernel-two-bindings.md))
