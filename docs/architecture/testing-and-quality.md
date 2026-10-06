# Testing and quality

NumPy and SciPy are the specification. The main test assets are generated
from them, so "correct" means "what NumPy/SciPy return for the same input",
not "what we believed when writing the test".

## Suites

| Suite | Size | What it checks | Runs in CI |
|---|---:|---|---|
| C unit tests (`make -C csrc test`) | 265 checks | allocator accounting and budget, iterator, every kernel family, FFT vs long-double DFT, real FFT vs complex FFT, RNG streams vs NumPy constants, LP/MILP/MDP, loop ufuncs, memory maps and `.npy` headers, exp/log accuracy (< 1 ulp vs long double), input validation, fuzzer regressions | 6 platforms, and again under ASan + UBSan (GCC and Clang) |
| Parity (`tests/Parity/ParityTest.php`) | 258 cases | fixtures from `tests/fixtures/generate_parity.py` (NumPy 2.4.4, SciPy 1.17.1): element-wise, broadcasting, promotion, reductions, slicing, linalg, FFT, RNG, sparse, optimize, LP/MILP, MDP, `.npy` | every PHP × OS × JIT cell |
| NumPy slice fixtures (`tests/Parity/FuzzTest.php`) | 300 cases | random shapes × random slice strings from NumPy: view contents, the C slice parser, strided arithmetic against broadcast operands, reductions, transposed casts, `toJson` vs `json_encode` | every cell |
| Unit (`tests/Unit`) | NDArray, numerics, solvers, `Math`, files, properties, memory safety | ownership and release, memory budget, promotion table, error mapping, ABI constants in sync, solver behaviour (shadow prices, MILP, MDP validation, thread determinism, non-finite input); `Tessero\Math` against the ufunc fixtures; `.npy` files written by NumPy (load, memory-map, byte-identical save) and memmap modes/errors; property tests (below); memory safety: overflowing and negative shapes, crafted `unserialize` payloads, read-only maps, released buffers, allocation balance, budget (`MemorySafetyTest`) | every cell |
| Extension parity (`tests/Ext/ExtParityTest.php`) | 531 cases (505 run, 26 FFI-only skipped) | the same fixtures through `ext-tessero`, plus 67 ufunc cases from `tests/fixtures/generate_ufunc.py` (NumPy/SciPy), kernel ufuncs vs methods, thread determinism of loop ufuncs | extension job |
| Backend parity (`tests/Ext/BackendParityTest.php`) | > 3 000 comparisons | both backends on the same seeded random arrays (views, all dtypes): every reduction over every axis combination with and without `keepdims`, every loop ufunc; **identical bytes** required | extension job, nightly |
| Extension phpt (`ext/tests`) | 14 files | lifecycle and views, operators, indexing, iteration/JSON/serialize, maths, solvers, userland Operand, INI budget, solver input validation, Math ufuncs (`out`, overlap, casting), memory-mapped arrays (modes, flush, views outliving the root, read-only), `.npy` load/save/openMemmap, memory safety (untrusted shapes, read-only writes, leak check) | extension job, including an ASan build |
| Laravel bridge (`laravel/tests`) | casts, rule, manager | on both backends | laravel job |
| Service provider in an app | boot, config, macro, Octane hooks | a fresh Laravel application | laravel-app job |
| Documentation examples (`tests/docs/examples.php`) | ~30 checks | every numeric claim in the guides | docs job |
| Preload e2e (`tests/e2e/preload.sh`) | 1 | non-CLI SAPI with `ffi.enable=preload` and `FFI::cdef` forbidden | preload job |
| Benchmarks (`bench/run.php`) | 25 cases | loose absolute time gates that catch order-of-magnitude regressions (a kernel falling back to a scalar loop) | bench job |

## Fuzzing

`csrc/fuzz/` holds four coverage-guided targets for the code that reads
untrusted input:

| Target | Input | Oracle (besides ASan/UBSan) |
|---|---|---|
| `fuzz_slice` | slice strings (`$a['...']`) over random shapes | every element of the view lies inside the buffer |
| `fuzz_npy` | `.npy` preambles | data offset inside the input; writing the parsed header back and re-parsing round-trips |
| `fuzz_mmap_plan` | shape, dtype, offset, mode, file size, page size | window aligned, holds exactly the data, never extends a read-only file, no overflow |
| `fuzz_json` | arbitrary bit patterns as arrays | size query = written length, guard bytes intact, output is valid JSON of the right shape, deterministic |

- `make -C csrc fuzz` builds them with libFuzzer (clang + compiler-rt) and runs
  each for `FUZZ_TIME` seconds from the seed corpora in `csrc/fuzz/corpus/`.
  CI runs 60 s per target on every push and 30 minutes per target nightly.
- `make -C csrc fuzz-standalone` links the same targets with a small seeded
  mutation driver (`fuzz/driver.c`) under GCC ASan + UBSan, for machines
  without libFuzzer. It is not coverage-guided.

The first runs found two bugs in the slice parser, both fixed with regression
tests in C and PHP:

- A blank item such as `"0, ,1"` made the parser read past its comma.
- Steps near 2^62 overflowed the element-count and stride arithmetic, in the
  C parser and in the PHP one.

## Property-based tests

`tests/Unit/PropertyTest.php` checks identities on seeded random arrays: random
shapes up to 4-D (including empty), every dtype, and C-order, Fortran-order,
strided and reversed layouts.

- Round trips: bytes, `serialize`, `.npy`, `.npy` memory map, JSON.
- Layout: `T.T`, reshape, reverse twice, expand and squeeze.
- Arithmetic: commutativity, `a+0`, `a*1`, `--a`, exact `(a+b)-b` for
  integers and error-bounded for floats, `sum(all axes) = sum()`,
  `cumsum[-1] = sum`, `where(a>b, a, b) = maximum`.
- Ufuncs: `fmax` commutes, `copysign(|a|, a) = a`, `signbit`, `trunc`
  idempotent, `cbrt^3`, `logaddexp(a, a) = a + ln 2`, `erf + erfc = 1`.
- Sort and FFT: idempotent and ordered, `take(argsort) = sort`, argsort is a
  permutation, `ifft(fft(x)) = x`, `irfft(rfft(x)) = x`.

`TESSERO_PROPERTY_CASES` sets the number of cases (default 40; nightly 2000).

## Soak tests

`tools/soak.php` runs a mixed workload in one process for a set time. The
workload covers ufuncs with `out`, views, multi-axis reductions, masks, sort,
FFT, matmul, `.npy` and memory maps in every mode, solvers and error paths. The
run fails when any of these accumulates:

- native bytes or mapped bytes not back at baseline after an iteration
  (checked exactly, every iteration);
- PHP heap or RSS rising over the second half of the run (least-squares slope
  thresholds);
- open file descriptors.

Nightly CI runs 5.5 hours per backend. Before a release, run 24–72 hours by
hand, as a worker would ([Soak testing](../operations/soak-testing.md)).

## Fixture regeneration

`python tests/fixtures/generate_parity.py` regenerates `parity.json` and
`fuzz.json` (`generate_ufunc.py` and `generate_npy.py` do the same for the ufunc
cases and the `.npy` files) with whatever NumPy/SciPy is installed. The CI `parity-refresh` job
runs it against the **latest** releases and then runs the suite, so a
behaviour change in NumPy or SciPy shows up as a failing job rather than going
unnoticed. Committed fixtures record the versions they came from.

## Tolerances

| Area | Tolerance | Why |
|---|---|---|
| integer ops, slicing, indexing, sort, RNG streams, JSON text | exact | deterministic by definition |
| element-wise float ops, matmul | 1e-12 relative | transcendentals (Tessero's vectorised exp/log are < 1 ulp, NumPy's SIMD or glibc versions differ in the last bit) and FMA contraction |
| reductions | 1e-12 relative | pairwise summation order matches NumPy's closely but not identically |
| linalg | 1e-9 relative (vectors up to sign) | different LAPACK builds |
| FFT | 1e-10 relative | different algorithms |
| LP/MILP objective | 1e-9; duals where unique | alternative optima are legitimate |
| MDP | policies exact; values 1e-10 (exact PI, Q, evaluation), 1e-12 (finite horizon), 1e-6 (value iteration, by its stopping rule) | ties resolved identically (lowest index) |

## JIT matrix

Every PHP suite runs with `opcache.jit` off, `function` and `tracing`. This
matrix found three PHP engine miscompilations during development
([Upstream bugs](../project/upstream-bugs.md)). All are worked around, and the
tests would catch a regression.

## Sanitizers

- C tests under `-fsanitize=address,undefined` with GCC and Clang.
- The extension built with ASan + UBSan runs its phpt suite and the extension
  parity suite (`USE_ZEND_ALLOC=0` so PHP's allocator does not hide errors).
- Integer overflow in kernels is defined behaviour (unsigned arithmetic), so
  UBSan runs with no suppressions for Tessero code. The one suppression is
  a known one-time allocation inside libgomp's initialisation.

## Verification status of this release

What was run for 0.2.0 on the reference machine (Linux x86-64, 2 cores,
PHP 8.4.21 NTS, GCC, NumPy 2.4.4, SciPy 1.17.1) is listed in `CHANGELOG.md`. CI
configuration for the full matrix is committed, but CI has not yet been run. The
[roadmap](../project/roadmap.md) tracks this.
