# Memory and threads

## Where the memory goes

| Pool | Holds | Limited by | Seen by |
|---|---|---|---|
| PHP (Zend) heap | PHP values, objects, `toArray()` results, JSON strings | `memory_limit` | `memory_get_usage()`, Laravel `--memory` |
| Native (libtessero) | array data, LP tableaux, MDP workspaces, sort scratch | `memory_budget` | `memoryInUse()`, `peakMemory()` |
| File mappings (`NDArray::memmap`, extension) | pages of mapped files | the OS page cache (evictable) | `Engine::info()['memory_mapped']`, RSS for resident pages |
| OpenBLAS / OpenMP runtimes | thread stacks and buffers | the libraries themselves | the process RSS only |

The array objects themselves are small PHP objects (well under 1 KB). The
data they point to is native. **A 1 000 × 1 000 float64 array uses 8 MB that
`memory_limit` does not see.**

Converting data between the pools briefly needs both: `toList()` on a
1M-element float64 array allocates about 17 MB of PHP memory (a PHP array
costs 16 bytes per element), while the 8 MB native block still exists. Prefer `toJson()`, `toBytes()` or
`serialize()` at API and storage boundaries.

## The budget

```ini
tessero.memory_budget = 256M              ; extension
```
```php
Tessero\Tessero::setMemoryBudget(256 << 20);   // FFI package
```
```dotenv
TESSERO_MEMORY_BUDGET=268435456              # Laravel: both backends
```

An allocation that would take the process past its budget fails cleanly:
`MemoryError` (FFI) or `Tessero\Ext\MemoryException`, with the requested size,
the amount in use and the budget in the message. Nothing is partially
allocated, the process keeps running, and the exception can be caught.

The budget counts every array, and the O(m·n) workspaces of the LP and
MILP solvers, MDP policy evaluation and sorting. Small O(n) solver
bookkeeping uses the system allocator and is not counted.

### Sizing

For PHP-FPM:

```
container / host memory  ≥  pm.max_children × (memory_limit + memory_budget) + OS + opcache
```

Start from the largest array a request legitimately needs, multiply by 3
(input, one temporary, result), and round up. Validation caps
(`NumericArray(maxElements: …)`) should keep requests well under it.

For queue workers doing large solves, size the budget for the model. A dense
LP with m constraints and n variables needs about 8·(m+1)·(n+m+1) bytes for
its tableau. Exact policy iteration on S states needs 8·S² bytes.

### Deterministic release

Native memory is released when the last PHP reference to it goes away (a
variable is reassigned, goes out of scope, or is `unset`). A view keeps its
parent's block alive. Cycles, such as an array stored on an object that
references itself, are collected by PHP's cycle collector: the extension
reports array references to it (`get_gc`).

To check for leaks in a worker, log `memoryInUse()` at the end of each
job. It should return to the same baseline.

## Threads

```ini
tessero.threads = 1                 ; default
```

| Where | Recommendation |
|---|---|
| PHP-FPM, mod_php | **1**. FPM already runs a worker per core, and extra threads compete with other requests. |
| Octane | 1, unless the server runs fewer workers than cores |
| Dedicated queue workers | cores ÷ workers, e.g. 4 workers × 2 threads on 8 cores |
| CLI batch job alone on a host | number of physical cores |

What threads speed up: element-wise operations on arrays of at least 131 072
elements (arithmetic, comparisons, math functions, type conversion, the
extension's `Math` ufuncs), and
Bellman sweeps of MDPs with more than 4 096 state-action pairs. Everything
else runs on one thread: reductions (to keep pairwise summation
deterministic), sort, FFT, LP/MILP and small arrays.

**Results do not depend on the thread count.** Each thread computes a fixed
contiguous chunk with the same code, and reductions are serial. The test suite
checks bit-identical output with 1 and 4 threads (element-wise kernels and MDP
solvers).

### OpenMP and `fork()`

GCC's OpenMP runtime (libgomp) does not survive `fork()` once its thread pool
has started: a forked child that uses OpenMP can hang. This matters for
`pcntl_fork`, Laravel Horizon's process spawning if the master process ran
parallel kernels, and PHP-FPM only if code runs in the master before forking
(it normally does not).

Rules:

- Keep `threads = 1` in any process that forks after doing numerical work.
- Or run numerical work only in children, never in the parent.
- Or build without OpenMP (`--disable-tessero-openmp`, `make OPENMP=`).

### BLAS threads

OpenBLAS keeps its own pool, used by the FFI package's linear algebra and
`matmul`. It is set to 1 thread outside the CLI. Raise it for CLI and worker
processes with `TESSERO_BLAS_THREADS`. Avoid `threads × blas_threads` exceeding the
cores assigned to the process.

### ZTS

Under thread-safe PHP the thread count and the memory counter/budget are
shared by all PHP threads in the process.
