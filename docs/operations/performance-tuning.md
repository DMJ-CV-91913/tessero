# Performance tuning

Measure first: `vendor/bin/tessero bench` runs the benchmark suite
(`bench/run.php`) with its regression gates on your hardware.
`bench/numpy_bench.py` runs the NumPy counterparts for comparison.

## 1. Cross the PHP boundary as rarely as possible

Most slow Tessero code spends its time converting between PHP arrays and
native arrays, or making many calls on small arrays.

| Instead of | Do |
|---|---|
| `foreach` over elements calling `item()` | one vectorised expression |
| building a PHP array in a loop, then `arr()` | build it once, convert once; or `fromBytes()` from a binary source |
| `json_encode($a->toList())` | `$a->toJson()` or `response()->ndarray($a)` |
| storing arrays as JSON columns and decoding per request | the `binary` cast (no float parsing) |
| thousands of calls on 10-element arrays through FFI | the extension (0.3 µs per call against 9 µs), or batch the small arrays into one larger array |

Per-call overhead, measured on a 2-core VM with PHP 8.4:

| | 10-element add | 1M-element add |
|---|---:|---:|
| plain PHP loop | 0.28 µs | ~25 ms |
| extension | 0.28 µs | ~1 ms |
| FFI | 8.7 µs | ~1 ms |

For large arrays the backend does not matter. For small ones it dominates.

## 2. Avoid temporaries in hot loops

`$a->add($b)->mul(2)` allocates two arrays. On the FFI package pass `out:`:

```php
$tmp = zeros($shape);
$a->add($b, out: $tmp);
$tmp->mul(2, out: $tmp);
```

The allocator does not zero memory (except for `zeros()`), and blocks of
32 MiB or more are aligned for transparent huge pages. Large temporaries are
cheap to allocate, but still cost a pass over memory.

## 3. Keep data contiguous for repeated work

Views with unusual strides (transposed, reversed, step > 1) work everywhere,
but repeated heavy operations on them are faster after one `copy()`. Reductions
along the last axis of a C-contiguous array are the fastest case.

## 4. Pick the right solver settings

| Workload | Setting |
|---|---|
| MDP with more than a few thousand states | `policyIteration(evalSweeps: 10–50)` (modified PI) rather than exact PI (O(S³)) or plain VI |
| MDP with γ close to 1 | VI needs about log(ε)/log(γ) sweeps: prefer (modified) policy iteration |
| MILP taking too long | set `mipRelGap` (e.g. 0.01) and `nodeLimit`; tighten variable bounds; add valid inequalities |
| LP larger than a few thousand rows/columns | the dense tableau grows as m·(n+m): use a sparse external solver (see [Roadmap](../project/roadmap.md)) |
| Sparse linear systems | CG for symmetric positive definite matrices, BiCGSTAB otherwise; scale rows to improve conditioning |

## 5. Threads for batch work

On dedicated workers, `threads = cores per worker` speeds up element-wise
operations on large arrays and MDP sweeps. It has no effect on arrays below
131 072 elements. See [Memory and threads](memory-and-threads.md).

## 6. Linear algebra

- Make sure the FFI package found OpenBLAS (`tessero doctor`). Without it
  `matmul` falls back to the C kernel and `Linalg` is unavailable.
- For CLI jobs dominated by large matrix products, raise `TESSERO_BLAS_THREADS`.
- Solve with `solve()` rather than `inv()` followed by `matmul`: it is faster and
  more accurate.

## 7. Enable the JIT (FFI package)

The PHP code around FFI calls (shape and stride arithmetic, argument
marshalling) benefits from `opcache.jit=tracing`. All test suites pass
under the tracing and function JITs. The upstream JIT defects found during
development have workarounds in the code; see [Upstream bugs](../project/upstream-bugs.md).

## Known slow spots

| Operation | vs NumPy | Why |
|---|---|---|
| `exp`, `log`, trig on 1M values | 4–8× slower | scalar libm; NumPy uses AVX-512 SVML-style kernels |
| `sort` 1M float64 | 3–6× slower | radix sort vs NumPy's AVX-512 quicksort (argsort is on par) |
| FFT 2^20 | 1.5–2× slower | straightforward mixed radix vs pocketfft; `rfft` does a full complex transform |
| `toList()` 1M | ~2× slower (FFI), faster (ext) | PHP array construction |

These are the top items on the [roadmap](../project/roadmap.md).
