# Tessero

Numerical computing for PHP 8.2+: n-dimensional arrays, linear algebra, FFT,
NumPy-identical random numbers, sparse matrices, optimisation, linear and
mixed-integer programming, and Markov decision processes. It all runs on one
small SIMD C library, `libtessero`. You reach it through FFI (no compiler
needed) or through a native Zend extension, and the Laravel package
integrates it into applications. Results are checked against NumPy and SciPy.

```php
use Tessero\Ext\NDArray;
use Tessero\Ext\Engine;

$load = NDArray::array($hourlyLoad);               // PHP array -> native float64
$z    = ($load - $load->mean()) / $load->std();    // operators, broadcasting
$peak = $load[$z->gt(2.0)];                        // boolean mask
$day  = $load->reshape(-1, 24)->mean(0);           // hour-of-day profile

$plan = Engine::linprog([-40, -30], [[2, 1], [1, 1]], [100, 80]);
$plan['x'];        // [20, 60]
$plan['ineqlin'];  // shadow prices [-10, -20], SciPy's convention
```

## Packages

| Package | What | Install |
|---|---|---|
| `tessero/tessero` | PHP library; loads prebuilt `libtessero` through FFI | `composer require tessero/tessero` |
| `tessero/tessero-ext` | Zend extension with the kernel compiled in: operators, `$a['1:, ::-1']`, `foreach`, JSON, ~0.3 µs per call | `pie install tessero/tessero-ext` |
| `tessero/laravel` | provider, facade, Eloquent casts, validation rule, Octane-safe settings, `response()->ndarray()` | `composer require tessero/laravel` |

Both backends give identical numbers; see *Choosing a backend* in the docs.

## What is in the box

| Area | API | Notes |
|---|---|---|
| Arrays | `NDArray` | float64, float32, int64, int32, uint8, bool, complex128; byte strides, views share memory |
| Indexing | `$a['1:5, ::2']`, masks, integer arrays, `take/put/where` | Python slice grammar parsed in C, fuzz-tested against NumPy |
| Element-wise | arithmetic, comparisons, logic, 30+ functions | full broadcasting, NEP 50 promotion, integer wrap-around as NumPy; optional OpenMP |
| Reductions | `sum prod min max argmin argmax any all mean var std cumsum cumprod diff` | pairwise summation; mean/var computed as NumPy does (bit-identical on contiguous data) |
| Sorting | `sort`, `argsort` | stable radix sort, NaN last |
| Linear algebra | `Linalg::solve inv det slogdet cholesky qr eigh eig svd pinv lstsq lu solveTriangular norm cond matrixRank` | LAPACKE from OpenBLAS; stacked inputs |
| FFT | `Fft::fft ifft rfft irfft fft2 ifft2 fftfreq rfftfreq fftshift` | any length (mixed radix + Bluestein) |
| Random | `Generator::defaultRng($seed)`, every univariate distribution method | bit-identical to `numpy.random.default_rng`; NumPy's own distribution code |
| Statistics | `Tessero\Stats`: distributions (`norm(...)->cdf`, `gamma(...)->rvs`, ...), `describe`, `pearsonr`, `ttestInd`, ...; `Tessero\Np`: `median`, `quantile`, `histogram`, `cov`, `unique`, ... | `scipy.stats` and NumPy semantics; checked against SciPy/NumPy on both backends ([guide](docs/guide/statistics.md)) |
| Special functions | `Tessero\Special::gammainc erfinv jv ...` | the code SciPy runs (xsf, Boost.Math, SciPy's glue); real arguments |
| Sparse | `CsrMatrix`, `cg`, `bicgstab` | Jacobi preconditioning |
| Optimisation | `minimize` (BFGS, L-BFGS-B, Nelder-Mead), `brentq`, `newton`, `root`, `curveFit` | SciPy's algorithms and conventions |
| LP / MILP | `LinearProgramming::linprog`, `milp` | two-phase simplex and branch and bound in C; shadow prices |
| MDP | `MarkovDecisionProcess` | value/policy/modified-policy iteration, finite horizon, CSR transitions, OpenMP |
| Universal functions | `Tessero\Math::sin cbrt erf gamma logaddexp fmax fmod …` (74; `Tessero\Ext\Math` in the extension) | broadcasting, NumPy type rules, `out:` arrays, OpenMP; both backends, identical results |
| Memory-mapped arrays, `.npy` | `NDArray::memmap($file, 'r+', $shape, $dtype)`, `NDArray::load($f, 'r')`, `openMemmap`, `save` | files larger than RAM, zero-copy views, `flush()`; NumPy's `.npy` format; both backends |
| I/O | `Npy::save load saveZ loadZ`, `toJson`, `serialize` | `.npy` v1/v2 |

## Quick start

```bash
composer require tessero/tessero
vendor/bin/tessero doctor
```

```php
use function Tessero\{arr, arange};
use Tessero\Mdp\MarkovDecisionProcess;

$a = arange(12)->reshape(3, 4);
$a['1:, ::2']->toArray();                  // [[4, 6], [8, 10]]
$a->sum(axis: 0)->toList();                // [12, 15, 18, 21]

$P = [[[0.9, 0.1], [0.0, 1.0]], [[1.0, 0.0], [1.0, 0.0]]];   // run / repair
$R = [[10, -5], [0, -5]];
MarkovDecisionProcess::fromDense($P, $R)->policyIteration(0.95)->policy->toList();   // [0, 1]
```

Production setup (PHP-FPM preload, memory budget, threads, Octane, Docker)
is covered in the operations manual.

## Documentation

The documentation lives in `docs/` and builds with MkDocs Material
(`pip install mkdocs-material && mkdocs serve`):

- **Getting started**: installation, quickstart, choosing a backend
- **User guide**: arrays, indexing, maths, linear algebra, FFT, random, sparse, optimisation, LP/MILP, MDP, I/O, Laravel
- **Operations manual**: deployment, native extension, Docker, configuration reference, memory and threads, monitoring, performance tuning, troubleshooting, runbooks, security, upgrading
- **Architecture**: overview, C kernel, memory model, FFI binding, native extension, numerical methods, solvers, testing, decision records
- **Reference**: generated API pages, C ABI, error codes

## Correctness

**Measured against real code.** In the library code of 18 widely used scientific Python packages (scikit-learn,
pandas, statsmodels, scikit-image, astropy, MNE, ...), 99.2 % of the in-scope calls to NumPy and SciPy go to
functions that Tessero has verified against NumPy and SciPy on both backends, and 99.2 % on the FFI backend.
For NumPy alone the figures are 99.7 % and 99.7 %. Counting every public symbol equally instead, 87.7 % of
the in-scope NumPy core and 70.5 % of SciPy are verified on FFI. Both figures are generated from the code and
the recorded test runs: see [coverage of real-world usage](docs/project/usage-weighted-coverage.md) and the
[coverage page](docs/project/numpy-scipy-coverage.md).

| Suite | Size |
|---|---|
| C unit tests (`make -C csrc test`, `make sanitize`) | 270 checks, clean under ASan + UBSan |
| NumPy 2.4.4 / SciPy 1.17.1 fixtures generated by NumPy and SciPy (`tools/parity/run-tests.php`) | 654 kernel-registry functions pass on both backends; 225 464 values compared per backend |
| Parity against the older NumPy/SciPy fixtures (`tests/Parity`) | 258 cases |
| Slicing/broadcast fuzz from NumPy | 300 cases |
| Extension: phpt + the same fixtures + ufunc fixtures | 13 files + 505 cases, ASan-clean |
| Cross-backend parity (same random inputs, identical bytes required) | > 3 000 comparisons |
| Property tests (identities, round trips; random shapes, dtypes, layouts) | 5 properties × 40 cases (×2000 nightly) |
| Coverage-guided fuzzing (`make -C csrc fuzz`): slice parser, `.npy` header, memmap planning, JSON writer | 4 targets; ~13 million inputs so far, 2 bugs found and fixed |
| Soak (`tools/soak.php`): native memory, mappings, heap, RSS, descriptors must not drift | minutes locally; 5.5 h nightly; 24–72 h before releases |
| Unit, Laravel bridge, documentation examples, FPM preload e2e | all pass |

All PHP suites pass with the JIT off, `function` and `tracing`. CI
(`.github/workflows/ci.yml`) is configured for 4 OSes × PHP 8.2–8.5 × 3 JIT
modes, 6 native builds, sanitizers, a real Laravel application, the Docker
image and the docs. The verification actually performed for this release is
recorded in `CHANGELOG.md`.

## Performance

Median ms, 2-core x86-64 VM with AVX-512, PHP 8.4, same session as NumPy 2.4.
These runs are noisy (±40 %).

| Case | Tessero | NumPy |
|---|---:|---:|
| add 1M float64 | 0.9 | 0.9 |
| sum 1M | 0.4 | 0.3 |
| matmul 600×600 (OpenBLAS) | 3.4 | 3.4 |
| solve 600×600 | 5.8 | 5.4 |
| normal() 1M | 9.3 | 14.1 |
| exp 1M | 1.2 | 0.8 |
| log 1M | 1.7 | 1.0 |
| sort 1M | 30 | 7.5 |
| fft 2^20 (complex) | 73 | 35 |
| rfft 2^20 (real input) | 36–55 | 14 |
| linprog 200×200 dense | 3.1 | — |
| MDP 20 000 states × 4 actions, modified PI | 91 | — |

`exp`/`log` use Tessero's own vectorised kernels (< 1 ulp). Before them, `exp`
took 9 ms. The remaining gaps are sort (NumPy uses an AVX-512 quicksort) and
the complex FFT; both are on the roadmap.

## Layout

```
csrc/        libtessero: include/tessero.h (the whole ABI), src/*.c, tests/, Makefile
lib/         prebuilt binaries per platform
src/         FFI package (Tessero\…)
ext/         native extension (Tessero\Ext\…), with its synced copy of the kernel
laravel/     Laravel package (Tessero\Laravel\…)
resources/   preload.php for PHP-FPM
docker/      reference production image
bench/       benchmarks with regression gates (+ NumPy counterpart)
tests/       Unit (incl. properties, files, Math), Parity, Ext (incl. cross-backend), docs examples, e2e preload
csrc/fuzz/   libFuzzer targets and seed corpora
tools/       soak.php, sync-ext.sh, gen-api-docs.php
docs/        documentation site (MkDocs)
```

## Contributing and governance

See `CONTRIBUTING.md`, `GOVERNANCE.md`, `CODE_OF_CONDUCT.md` and `SECURITY.md`.

## Licence

BSD-3-Clause (`LICENSE`). The ziggurat tables in `csrc/src/ziggurat.h` are
NumPy's (BSD-3-Clause).
