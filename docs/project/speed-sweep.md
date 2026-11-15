# Speed sweep, 2026-11-15

First run on 2026-11-06; every tier re-timed on 2026-11-15 at commit 56102533 (see Method); the known gaps and the cases changed since were timed again at commit d2d4e2e6.

Every function family timed against NumPy 2.4.4 / SciPy 1.17.1, on the same inputs, single-threaded on both sides
(`OPENBLAS_NUM_THREADS=1`), on the benchmark machine of [benchmarks](benchmarks.md) (Intel Core i7-4790, Haswell,
AVX2; PHP 8.5.12, FFI backend unless the native extension is named): four alternating rounds of each side, each
case keeping its best time, while the host carried other work (load average 5 to 26 on 8 hardware threads). A ratio
is Tessero time / reference time: below 1, Tessero is faster.

| tier | what is timed | timed on both sides | geomean ratio | Tessero faster |
|---|---|---|---|---|
| 1 | all `scipy.special` element-wise functions, 100,000 values each (the parity fixture inputs, tiled) | 212 functions | **0.94** | 103 |
| 2 | 88 heavier calls at realistic sizes (linalg, FFT, signal, ndimage, interpolate, integrate, optimize, stats, sparse, csgraph, spatial, cluster) | 88 cases | **0.70** | 64 |
| 3 | per-call latency of every parity fixture call (small inputs) | 1,022 functions, 15,184 cases | **0.30** | 754 |

Reading it: on small inputs (tier 3) the fixed cost of a call dominates, and Tessero's compiled kernels behind one
FFI call beat SciPy's Python-level wrappers (stats 0.03, csgraph 0.03, signal 0.15); NumPy's own core functions are
the exception through FFI (np 1.56), as they are C all the way down. Through the native extension, which calls the
same kernel without the FFI layer, tier 3 takes 0.07 of the NumPy/SciPy time and is faster on 966 of the 1,022
functions, NumPy's core included (np 0.29; see "Per call through the native extension"). On element-wise math at
100,000 values (tier 1) the two are on a par. On heavy calls (tier 2) Tessero is ahead on optimizers, ODEs and
splines: clearly faster (by more than 10%) on 49 cases, within 10% either way on 24, and clearly slower on 15,
listed with what is known of their causes under the gaps below. Timed again after the fixes since the sweep,
`rotate` has moved ahead of SciPy (see "Known gaps").

## Usage-weighted mix

Tier 3 counts every function once; real code does not call them equally. Here each function's per-call time is
weighted by how often the library code of 18 widely used scientific Python packages calls it (the corpus of the
[usage-weighted coverage](usage-weighted-coverage.md): 40,188 calls to in-scope NumPy and SciPy functions; calls to
excluded symbols are left out). The time of a mix is the sum over its functions of the per-call time times the
function's corpus calls, divided by those calls. The functions timed (tier 3, plus `bench/usage_bench.php` and
`bench/usage_bench.py` for the most-called functions tier 3 does not time) carry 36,696 of the 40,188 calls. The
inputs are small (the fixture and usage-bench inputs), so the fixed cost of a call dominates.

| mix | weighted over | NumPy/SciPy us | extension us | FFI us | extension faster on | FFI faster on |
|---|---|---|---|---|---|---|
| real-usage mix | 36,696 calls | 7.20 | 0.75 | 4.41 | 92% | 19% |
| ML operations | 20 functions | 1.45 | 0.40 | 3.70 | 95% | 20% |

ML operations: `numpy.dot`, `numpy.exp`, `numpy.log`, `numpy.sum`, `numpy.mean`, `numpy.max`, `numpy.min`,
`numpy.argmax`, `numpy.argmin`, `numpy.where`, `numpy.sqrt`, `numpy.clip`, `numpy.maximum`, `numpy.zeros`,
`numpy.asarray`, `numpy.concatenate`, `numpy.argsort`, `numpy.unique`, `numpy.linalg.norm`, `numpy.absolute`.

"Faster on" is the share of the mix's weighted calls that go to functions where Tessero's per-call time is below
NumPy/SciPy's. Through FFI the real-usage mix as a whole takes less time than in NumPy/SciPy, yet is faster on only
a fifth of its calls. NumPy's core functions make 96% of the calls (`array`, `asarray`, `zeros`, `dot`, `sqrt` lead)
and cost NumPy 2.7 us per call on average, Tessero 4.1 us through FFI. The SciPy functions make 4% of the calls but
about two thirds of the NumPy/SciPy time, 117 us per call on average in SciPy's Python-level wrappers against 12.8 us
in Tessero. The native extension removes the FFI layer's cost: 0.56 us per call on NumPy's core functions, 5.4 us on
SciPy's.

The inputs here are small. Large matrix products run in OpenBLAS on both sides and are on a par: a 600 x 600
`matmul` takes 9.1 ms against NumPy's 9.9 ms in this sweep's run of the [benchmarks](benchmarks.md) bench.

## Classical machine learning

The functions classical machine learning code calls, scikit-learn style: array creation and validation, shape
manipulation, matrix products, reductions, indexing and decisions, element-wise math, random sampling,
optimization, sparse matrices, linear algebra, distances and neighbours, special functions and distributions. Each
function's per-call time on small inputs (the tier 3 rows, `bench/usage_bench.php` and `bench/ml_bench.php` for the
rest, against their Python counterparts) is weighted by its static call count: in the library code of the 18
corpus packages, and in scikit-learn's alone (scikit-learn 1.9.1, 2,044 calls to these functions; the per-package
counts are in `tools/parity/usage.json`).

### By category

| category | weighting | weighted over | NumPy/SciPy us | extension us | FFI us | extension faster on | FFI faster on |
|---|---|---|---|---|---|---|---|
| creation | 18 packages | 9,218 calls | 0.49 | 0.21 | 2.66 | 100% | 0% |
| creation | scikit-learn | 895 calls | 0.44 | 0.20 | 2.49 | 100% | 0% |
| shape | 18 packages | 1,647 calls | 1.35 | 1.06 | 5.64 | 50% | 0% |
| shape | scikit-learn | 197 calls | 1.45 | 0.96 | 5.20 | 77% | 0% |
| matmul | 18 packages | 1,649 calls | 0.93 | 0.43 | 5.52 | 100% | 0% |
| matmul | scikit-learn | 213 calls | 0.92 | 0.43 | 5.51 | 100% | 0% |
| reductions | 18 packages | 1,930 calls | 3.30 | 0.34 | 1.84 | 100% | 94% |
| reductions | scikit-learn | 288 calls | 3.21 | 0.37 | 2.02 | 100% | 88% |
| indexing | 18 packages | 1,277 calls | 1.10 | 0.36 | 4.86 | 100% | 0% |
| indexing | scikit-learn | 129 calls | 1.23 | 0.34 | 3.53 | 100% | 0% |
| element-wise | 18 packages | 3,149 calls | 0.63 | 0.32 | 4.04 | 100% | 0% |
| element-wise | scikit-learn | 232 calls | 0.65 | 0.32 | 4.10 | 100% | 0% |
| random | unweighted | 4 functions | 3.83 | 0.52 | 13.2 | 100% | 0% |
| optimize | 18 packages | 34 calls | 873 | 29.9 | 48.8 | 100% | 100% |
| optimize | scikit-learn | 10 calls | 979 | 32.7 | 34.2 | 100% | 100% |
| sparse | 18 packages | 45 calls | 70.5 | 1.49 | 10.4 | 100% | 100% |
| sparse | scikit-learn | 17 calls | 83.5 | 1.42 | 11.1 | 100% | 100% |
| linalg | 18 packages | 90 calls | 68.7 | 3.87 | 11.1 | 100% | 100% |
| linalg | scikit-learn | 45 calls | 63.5 | 4.04 | 11.4 | 100% | 100% |
| distance | 18 packages | 65 calls | 10.1 | 1.16 | 5.97 | 100% | 60% |
| distance | scikit-learn | 11 calls | 9.93 | 0.96 | 4.96 | 100% | 55% |
| special | 18 packages | 32 calls | 74.5 | 0.91 | 4.52 | 69% | 69% |
| special | scikit-learn | 7 calls | 15.8 | 0.65 | 4.42 | 14% | 14% |
| distributions | unweighted | 3 functions | 49.9 | 1.10 | 9.11 | 100% | 100% |
| all | 18 packages | 19,136 calls | 3.14 | 0.42 | 3.61 | 96% | 11% |
| all | scikit-learn | 2,044 calls | 8.02 | 0.60 | 3.69 | 97% | 16% |

### By function

| function | category | 18-package calls | scikit-learn calls | NumPy/SciPy us | extension us | FFI us |
|---|---|---|---|---|---|---|
| numpy.asarray | creation | 2,335 | 212 | 0.56 | 0.24 | 2.96 |
| numpy.array | creation | 3,265 | 186 | 0.57 | 0.23 | 2.87 |
| numpy.zeros | creation | 2,094 | 276 | 0.25 | 0.17 | 1.87 |
| numpy.ones | creation | 724 | 73 | 0.93 | 0.18 | 3.97 |
| numpy.empty | creation | 800 | 148 | 0.25 | 0.17 | 1.80 |
| numpy.reshape | shape | 155 | 24 | 0.83 | 0.24 | 1.31 |
| numpy.ravel | shape | 151 | 27 | 0.47 | 0.21 | 1.24 |
| numpy.expand_dims | shape | 62 | 6 | 1.78 | 0.85 | 6.63 |
| numpy.concatenate | shape | 823 | 46 | 1.23 | 1.31 | 6.72 |
| numpy.vstack | shape | 233 | 48 | 2.13 | 1.19 | 6.60 |
| numpy.hstack | shape | 223 | 46 | 1.83 | 1.20 | 6.39 |
| numpy.dot | matmul | 1,530 | 201 | 0.86 | 0.36 | 5.43 |
| numpy.matmul | matmul | 36 | 2 | 1.20 | 0.36 | 5.29 |
| numpy.einsum | matmul | 83 | 10 | 2.09 | 1.74 | 7.33 |
| numpy.sum | reductions | 1,202 | 172 | 2.31 | 0.27 | 1.71 |
| numpy.mean | reductions | 452 | 60 | 3.84 | 0.38 | 1.40 |
| numpy.std | reductions | 92 | 10 | 10.9 | 0.42 | 1.54 |
| numpy.var | reductions | 60 | 12 | 10.1 | 0.42 | 1.55 |
| numpy.bincount | reductions | 124 | 34 | 1.95 | 0.79 | 5.00 |
| numpy.argmax | indexing | 148 | 42 | 1.23 | 0.28 | 1.81 |
| numpy.argmin | indexing | 117 | 12 | 1.18 | 0.28 | 1.83 |
| numpy.argsort | indexing | 261 | 40 | 1.51 | 0.39 | 3.26 |
| numpy.where | indexing | 751 | 35 | 0.92 | 0.37 | 6.50 |
| numpy.exp | element-wise | 670 | 36 | 0.46 | 0.35 | 3.79 |
| numpy.log | element-wise | 815 | 81 | 0.46 | 0.36 | 3.69 |
| numpy.sqrt | element-wise | 1,362 | 90 | 0.43 | 0.29 | 3.41 |
| numpy.clip | element-wise | 302 | 25 | 2.35 | 0.25 | 8.39 |
| Generator.normal | random | -- | -- | 1.02 | 0.80 | 17.7 |
| Generator.uniform | random | -- | -- | 1.94 | 0.75 | 17.5 |
| Generator.choice | random | -- | -- | 10.6 | 0.31 | 13.6 |
| Generator.permutation | random | -- | -- | 1.75 | 0.25 | 3.96 |
| scipy.optimize.minimize | optimize | 28 | 10 | 979 | 32.7 | 34.2 |
| scipy.optimize.least_squares | optimize | 6 | 0 | 381 | 17.2 | 117 |
| scipy.sparse.csr_matrix | sparse | 19 | 3 | 58.0 | 1.56 | 9.60 |
| scipy.sparse.csc_matrix | sparse | 9 | 1 | 57.5 | 1.55 | 9.69 |
| scipy.sparse.hstack | sparse | 17 | 13 | 91.4 | 1.38 | 11.5 |
| scipy.linalg.svd | linalg | 21 | 15 | 29.4 | 6.00 | 15.8 |
| scipy.linalg.eigh | linalg | 24 | 11 | 34.4 | 3.11 | 9.96 |
| scipy.linalg.cholesky | linalg | 11 | 4 | 9.43 | 0.79 | 4.89 |
| scipy.linalg.solve | linalg | 17 | 9 | 40.6 | 1.29 | 5.80 |
| scipy.linalg.lstsq | linalg | 10 | 3 | 44.6 | 5.69 | 16.2 |
| scipy.sparse.linalg.svds | linalg | 7 | 3 | 500 | 8.62 | 15.5 |
| scipy.spatial.distance.cdist | distance | 26 | 5 | 4.08 | 1.15 | 5.48 |
| scipy.spatial.distance.pdist | distance | 25 | 6 | 14.8 | 0.81 | 4.53 |
| scipy.spatial.cKDTree | distance | 14 | 0 | 12.9 | 1.80 | 9.44 |
| scipy.special.expit | special | 10 | 6 | 0.42 | 0.59 | 4.40 |
| scipy.special.softmax | special | 0 | 0 | 10.8 | 1.08 | 4.98 |
| scipy.special.logsumexp | special | 22 | 1 | 108 | 1.06 | 4.58 |
| scipy.stats.norm.pdf | distributions | -- | -- | 57.2 | 1.11 | 5.03 |
| scipy.stats.norm.logpdf | distributions | -- | -- | 60.5 | 1.01 | 4.81 |
| scipy.stats.norm.rvs | distributions | -- | -- | 32.0 | 1.19 | 17.5 |

### By model type

| model type | weighting | functions | NumPy/SciPy us | extension us | FFI us | extension faster on | FFI faster on |
|---|---|---|---|---|---|---|---|
| linear and logistic | 18 packages | dot, matmul, mean, clip, log, minimize, lstsq, expit, csr_matrix | 10.3 | 0.66 | 5.00 | 100% | 16% |
| linear and logistic | scikit-learn | dot, matmul, mean, clip, log, minimize, lstsq, expit, csr_matrix | 27.1 | 1.24 | 5.47 | 98% | 19% |
| tree ensembles | 18 packages | argsort, bincount, where, argmax, csc_matrix | 1.57 | 0.41 | 5.18 | 100% | 1% |
| tree ensembles | scikit-learn | argsort, bincount, where, argmax, csc_matrix | 1.76 | 0.45 | 4.03 | 100% | 1% |
| clustering and neighbours | 18 packages | argmin, mean, std, sum, cdist, cKDTree | 3.12 | 0.33 | 1.75 | 100% | 92% |
| clustering and neighbours | scikit-learn | argmin, mean, std, sum, cdist, cKDTree | 2.98 | 0.32 | 1.71 | 100% | 93% |
| dimensionality reduction | 18 packages | dot, mean, var, svd, eigh, svds | 4.11 | 0.48 | 4.63 | 100% | 27% |
| dimensionality reduction | scikit-learn | dot, mean, var, svd, eigh, svds | 9.42 | 0.83 | 5.25 | 100% | 33% |

### At realistic sizes

| case | category | NumPy/SciPy ms | extension ms | FFI ms |
|---|---|---|---|---|
| numpy.matmul 2000x2000 | matmul | 327 | 325 | 321 |
| scipy.linalg.svd 1000x1000 (full) | linalg | 503 | 608 | 555 |
| scipy.linalg.eigh 1000x1000 | linalg | 262 | 246 | 258 |
| scipy.linalg.solve 1000x1000 | linalg | 31.5 | 27.7 | 26.6 |
| scipy.linalg.cholesky 1000x1000 | linalg | 13.8 | 14.4 | 14.6 |
| scipy.linalg.lstsq 10,000 x 50 | linalg | 13.7 | 13.8 | 14.4 |
| scipy.sparse.linalg.svds k=10, 2,000 x 500 (dense input) | linalg | 137 | 88.1 | 87.2 |
| scipy.spatial.distance.pdist 2,000 x 50 (euclidean) | distance | 45.4 | 69.4 | 67.0 |
| scipy.spatial.distance.cdist 1,000 x 1,000 x 50 (euclidean) | distance | 21.7 | 32.8 | 32.7 |
| scipy.spatial.cKDTree build 100,000 x 3 | distance | 27.5 | 28.4 | 30.8 |
| scipy.spatial.cKDTree query 1,000 points, k=5 | distance | 2.02 | 2.55 | 3.28 |
| logistic regression, L-BFGS-B, 10,000 x 50 | optimize | 21.2 | 59.2 | 81.7 |
| logistic regression, Newton-CG, 10,000 x 50 | optimize | 3544 | 3307 | 3614 |
| logistic regression, TNC, 10,000 x 50 | optimize | 60.6 | 141 | 156 |
| scipy.special.expit 1M | special | 7.64 | 9.87 | 10.5 |
| scipy.special.logsumexp 1M | special | 16.9 | 8.70 | 8.37 |
| scipy.stats.norm.logpdf 1M | distributions | 24.1 | 18.5 | 18.8 |

The weights are static call sites in library code, not run time, and say nothing about how much time a program
spends in each function. Method calls (`x.sum()`, `rng.normal()`) cannot be attributed statically, so the random
Generator methods and the distribution methods (`norm.pdf`, `norm.logpdf`, `norm.rvs`) have no weight: their rows
are unweighted, the mean over their functions, and "all" covers the weighted categories only. `Generator.choice` is
available on both backends. "Faster on" is the share of the weighted calls that go to functions where Tessero's
per-call time is below NumPy/SciPy's.

In the scikit-learn `special` row one function carries most of the time: `logsumexp`, 1 of the 7 calls, takes 108 us
in SciPy against 1.06 us through the extension. The other 6 calls go to `expit`, which is slower in Tessero (0.42 us
in SciPy, 0.59 us through the extension, 4.40 us through FFI), so the row's weighted time is lower while Tessero is
faster on only 14% of its calls.

The per-call `optimize` rows solve small problems end to end (two parameters); `minimize` and `least_squares` are
Tessero's PHP code, the same on both backends. The `cKDTree` row builds a tree over 8 points (Tessero's `KDTree`).
In the model-type mixes, each model type's functions are those of the usual scikit-learn estimators of that kind.

At realistic sizes the inputs come from the same PCG64 streams on both sides, and `svds` takes the dense matrix on
both. Large dense linear algebra runs in OpenBLAS and LAPACK on both sides and is on a par, except the full `svd`
(1.1 to 1.2x). `pdist` and `cdist` take about 1.5x SciPy's time. The logistic regression minimizes the mean log-loss
with its gradient, both evaluated with array operations each iteration. With L-BFGS-B both sides stop by their
convergence tests (25 iterations in Tessero at a loss of 6.2e-5, 26 in SciPy at 1.9e-5); Tessero's extra time is in
evaluating the objective and the gradient from PHP. Newton-CG and TNC now run SciPy's algorithms step for step
(Newton-CG as SciPy's code, with its line search; TNC as a port of the C code SciPy wraps): TNC stops after 12
iterations on both sides at a loss of 8.0e-5, Newton-CG after 17 iterations in Tessero (9.9e-5) and 18 in SciPy
(6.1e-5). On this separable problem Newton-CG's end point depends on the last bits of the gradient: SciPy itself
stops after 15 iterations at 2.3e-4 when its gradient multiplies by a transposed copy of the matrix instead of the
transposed view. On objectives that round the same way in PHP and Python, both methods return exactly SciPy's
iterates and results. TNC takes 2.3 to 2.6x SciPy's time, for the same reason as L-BFGS-B; Newton-CG is on a par.

Conditions: timed on 2026-11-15 (08:39 to 08:47 UTC) at commit 56102533, after the sweep: four alternating rounds of
FFI, the extension and NumPy/SciPy for `bench/ml_bench.php` and `bench/ml_bench.py`, single-threaded
(`OPENBLAS_NUM_THREADS=1`), at a load average of 5 to 13; each case keeps its best time on each side. The other
per-call rows are the sweep's. The Newton-CG and TNC rows were timed again after their ports (14:40 to 14:42 UTC,
the same data and library, three alternating rounds, load 7 to 8).

## Fixed during the sweep

| function | problem | before | after |
|---|---|---|---|
| `scipy.fft` dct / idct / dst / idst (and the n-D forms) | direct O(N^2) sums | 1M points: about 5 hours (estimated from 8,000 points: 1.1 s) | 50 ms (SciPy 17-30 ms); agrees with SciPy to 1e-16 of the output scale |
| `interpolate.CubicSpline`, `make_interp_spline` (k=3) | the tridiagonal system solved as a dense n x n matrix | 10k knots: 8-17 s and 800 MB | 76 ms, O(n) memory (LAPACK `dgbsv`, as SciPy) |
| `sparse.linalg.lsmr` | `maxiter` read from the `conlim` argument: 100 million iterations per default call | 15 s on a 10 x 6 system | 16 us |
| `signal.butter`, `cheby1`, `cheby2`, `ellip`, `bessel`, `iirfilter` | `output='sos'` / `'zpk'` silently answered with (b, a) | wrong format, no error | refused with a clear error (convert with `tf2sos`) |
| `np.modf`, `npoly` polymulx and the derivatives, `csgraph.laplacian`, `slinalg.invpascal` | the sign of an exact zero differed from NumPy / SciPy | -- | match; the parity comparator now checks the sign of real zeros |

## Fixed after the sweep

The changes of 2026-11-11 to 2026-11-15, by this re-timing: before = the 2026-11-10 tables of this page (tier 2 at
its last timing; tier 3 the 2026-11-10 run behind them), after and NumPy/SciPy = this sweep. Results are unchanged
byte for byte unless the row says otherwise.

| case | before | after | NumPy/SciPy | how |
|---|---|---|---|---|
| `integrate.quad` sin(x) exp(-x) on [0, 10] | 0.0187 ms | 0.0051 ms | 0.0055 ms | QUADPACK's DQAGSE ported step for step from SciPy's C translation (21-point Gauss-Kronrod, extrapolation) instead of a 15-point rule without extrapolation: value, error estimate and number of calls now match SciPy's (also `dblquad`, `tplquad`, `nquad`) |
| `optimize.curve_fit`, 1000 points | 5.03 ms | 2.25 ms; 1.36 ms with a model that takes the whole array | 0.58 ms | MINPACK's `lmdif`, as SciPy runs it, instead of Tessero's own Levenberg-Marquardt: the same parameters, covariance and number of calls as SciPy, bit for bit on models built from + - * / alone |
| `sparse.linalg.eigsh` k=6, 100 x 100 grid Poisson (CSR) | 743 ms | 443 ms | 436 ms | the search for a missed repeated eigenvalue starts from the next Ritz vector of the main run instead of from scratch (it was 40% of the time); the same accuracy, the last bits may differ |
| `linalg.cholesky` 1000x1000 | 20.6 ms | 14.2 ms | 13.8 ms | LAPACK on one column-major copy of the triangle it reads, without LAPACKE's row-major copies in and out; the factor's bits are the same |
| `numpy.linalg.slogdet` 600x600 | 7.27 ms | 5.54 ms | 5.90 ms | the same for the LU factorization (also `det`, `inv`, `cho_factor`, `cho_solve`, `lu_solve`; `lu_factor` and `cho_factor` return column-order factors, as SciPy does, which the solves read in place) |
| `spatial.KDTree` build, 100k points | 58.1 ms | 55.4 ms from PHP rows; 30.8 ms from an NDArray | 30.7 ms | an NDArray is used in place (as SciPy takes an ndarray), and through the extension PHP rows are converted in one C call; the same tree |
| `spatial.KDTree`, 100 queries (k=5) in 100k | 0.363 ms | 0.268 ms; 0.254 ms from an NDArray | 0.207 ms | each child takes the box bound its parent computed instead of computing it again; the same search and results |
| `spatial.ConvexHull`, 10k (2-D) | 1.69 ms | 1.52 ms from PHP rows; 0.64 ms from an NDArray | 1.41 ms | the same NDArray input; the same vertices, area and perimeter |
| `special.kelvin_zeros`, `ber_zeros`, `jn_zeros` (tier 3: their fixture calls, per call through the extension) | 15,595; 1,826; 520 us | 98; 12.3; 124 us | 98; 12.5; 132 us | SciPy's own zero routines (`klvnzo`, `jyzo`, from the vendored xsf code, with an iteration cap) instead of scans in steps of 0.02 or 0.05 with refinement; the zeros now follow SciPy's values |
| small calls through the native extension (tier 3, per call) | `bitwise_and` 2.17; `ldexp` 2.33; `linalg.inv` 4.42; `linalg.slogdet` 2.35; `concatenate` 2.22 us | 0.83; 0.85; 1.66; 1.13; 1.31 us | 0.65; 0.64; 6.02; 4.20; 1.23 us | the extension no longer zeroes its 28 KB of argument buffers on every call (a hardening compiler flag did it), finds a method's routine by an index instead of by name, and checks argument sizes without 64-bit divisions; element-wise integer and float routines read a C-contiguous input in place instead of copying it. Over all 1,022 functions the extension's per-call time fell to 0.58 of the 2026-11-10 run's (the NumPy/SciPy side to 0.96) |

One change made a case slower: `optimize.least_squares` (method 'lm', the Rosenbrock residuals) now runs MINPACK's
`lmder` with SciPy's 2-point differences, as SciPy does, and returns SciPy's results; it takes 0.76 ms
instead of 0.31 ms, against SciPy's 5.8 ms.

After the classical machine learning timing, `minimize` with method 'TNC' became a port of the TNC code SciPy wraps
(it was Tessero's own truncated Newton, which on the logistic regression ran 800 iterations without converging), and
method 'Newton-CG' follows SciPy's code step for step, its line search included (MINPACK's `dcsrch`, then
`scalar_search_wolfe2`); `fmin_tnc` and `fmin_ncg` with them. Both now return SciPy's results, which changes their
answers. The logistic regression at 10,000 x 50 (see the classical machine learning section): TNC 5,934 ms through
the extension and 5,423 ms through FFI before, 141 and 156 ms after, against SciPy's 61 ms; Newton-CG 4,382 and
5,318 ms before, 3,307 and 3,614 ms after, against SciPy's 3,544 ms.

Then `ndimage.map_coordinates`, `shift`, `zoom`, `affine_transform` and `rotate` were corrected. At the B-spline
orders (2 to 5) they had matched SciPy only in the modes 'mirror' and 'reflect': in 'constant' (SciPy's default,
the mode these cases time), 'nearest' and 'wrap' their results were off by up to 0.2 to 0.4 on data in [0, 1], and
`prefilter=False` was ignored. They now follow SciPy's code (`NI_GeometricTransform`, `NI_ZoomShift` and the
spline prefilter, with its padding at the edges) and give SciPy's answers, in the input's dtype, on 872 of 882
calls checked against SciPy, on both backends (the other 10 are bool results with a cval outside [0, 1], with the
same truth values). The sweep's times for these cases, and the earlier `zoom` and `shift` rows of this section, were
of the old code. Re-timed alone on 2026-11-15 (18:42 to 18:47 UTC, the installed library at commit d2d4e2e6, eight
alternating rounds of FFI and SciPy, at a load average of 8 to 14; `shift` three rounds): `rotate` 512^2 at 30 deg
49.7 ms in the sweep, now 29.6 ms against SciPy's 36.1 ms; `map_coordinates` 1000^2 at 100k points 54.8 ms, now
41.7 ms against 36.0 ms; `zoom` 500^2 x2 21 ms, now 23.6 ms against 74.1 ms; `shift` 1000^2 at order 3 42.9 ms
against 89.6 ms.

Earlier changes, 2026-11-06 to 2026-11-13 (each row's before and after timed together; the first rows are the quick
gaps, closed the day of the first sweep; results unchanged, checked by parity):

| case | before | after | NumPy/SciPy | how |
|---|---|---|---|---|
| `numpy.trapezoid` 1M | 353 ms | 1.6 ms | 3.2 ms | 1-D float64 fast path: the same per-element operations into one buffer instead of three temporaries |
| `numpy.einsum` ij,jk->ik 300^2 | 267 ms | 15 ms | 6.7 ms | offsets kept incrementally, the last label a strided (vectorizable) inner loop; same summation order |
| `ndimage.maximum_filter` 5x5, 1000^2 | 822 ms | 25 ms | 19 ms | separable 1-D passes, as SciPy computes a box min/max |
| `numpy.median` 1M | 144 ms | 13 ms | 8.4 ms | introselect instead of a full sort |
| `numpy.percentile` 1M (one q) | 119 ms | 18 ms | 9.1 ms | introselect (at most two order statistics) |
| `numpy.unique` 1M | 97 ms | 15 ms | 8.7 ms | values sorted alone when no index is wanted; the stable argsort is the radix one |
| `numpy.cov` 50 x 100k | 235 ms | 45 ms | 28 ms | one BLAS dgemm, as NumPy |
| `spatial.KDTree` query, 10k points (k=5) in 100k | hours (exhaustive PHP search) | 55 ms, tree build included | 49 ms (build 31 + query 18) | a k-d tree in the kernel (`distance.kdtree_query`), the same neighbours, distances and order |
| `spatial.Delaunay`, 10k points (2-D) | 57 s | 29 ms | 46 ms | the sweep-hull algorithm in the kernel (`distance.delaunay_2d`, O(n log n), the hull exact by construction) instead of a Bowyer-Watson that scanned every triangle in PHP; also behind `griddata`, `LinearNDInterpolator`, `CloughTocher2DInterpolator` |
| `cluster.hierarchy.linkage` ward, 2000 | 1.6 s | 70 ms | 92 ms | each cluster caches its nearest later neighbour, so a merge rescans only the rows it affects (about n^2 instead of n^3); the same merges, distances and tie order, byte for byte |
| `spatial.ConvexHull`, 10k (2-D) | 43 ms | 4.1 ms | 1.5 ms | the monotone chain in the kernel (`distance.convex_hull_2d`), the same vertices |
| `linalg.solve_triangular` 1000 | 4.2 ms | 0.88 ms | 1.0 ms | the column-major `dtrtrs` on the row-major matrix (triangle and transpose flipped) instead of LAPACKE's row-major copies, the matrix read in place |
| `numpy.percentile` 1M (one q) | 18 ms | 5.9 ms | 12 ms | a two-pointer (Hoare) introselect, and the second order statistic of an interpolated quantile as the smallest of the rest (one scan) |
| `numpy.median` 1M | 13 ms | 12 ms | 11 ms | the same introselect |
| `numpy.diff` 1M | 127 ms | 1.4 ms | 1.2 ms | operands already in the loop dtype read in place instead of through a one-element conversion call each, and a float64 difference row by row (the same subtraction, whose operands keep their order); also behind `ediff1d` and the `poly*` routines; bytes unchanged, NaN bits included |
| `numpy.outer` 1000 x 1000 | 128 ms | 10 ms | 1.3 ms | the same in-place operand read (also `inner`, `kron`, `trapezoid`, `cumulative_trapezoid`, `geomspace`, `pad`) |
| `optimize.curve_fit`, 1000 points | 10.6 ms | 5.0 ms | 0.61 ms | the forward-difference Jacobian and the normal equations J^T J, J^T r of each Levenberg-Marquardt iteration in one kernel call (`linalg.fd_normal`, the same sums in the same order), J^T J kept for the covariance, and the model called with its parameters unpacked; the same parameters and covariance, bit for bit (before and after timed on the same host state) |
| `sparse.linalg.eigsh` k=6, 100 x 100 grid Poisson (CSR) | 2.3 s | 0.74 s | 0.50 s | the Lanczos basis orthogonalized by BLAS matrix-vector products (classical Gram-Schmidt, twice) and the Ritz vectors formed by one matrix product, instead of scalar loops; a restart keeps the wanted Ritz vectors plus the converged ones, at most half the free slots (ARPACK's rule), instead of a fixed 13 of 20: 1,539 products with the matrix instead of 2,470 (ARPACK: 1,562). The same eigenvalues to about 1e-14 relative (checked against dense LAPACK on 140 cases) |
| `spatial.ConvexHull`, 10k (2-D) | 4.3 ms | 2.0 ms | 1.5 ms | the points sorted by one stable radix argsort by x, each run of equal x then by y, instead of `qsort` with a comparison function (the kernel call 2.1 -> 1.0 ms), and passed to the kernel as one flat list cast in one loop instead of a validated copy then a nested-array conversion; the same vertices, area and volume |
| `spatial.KDTree`, 100k points: build | 91 ms | 58 ms | 33 ms | the points cast into one flat list in one loop instead of a copy of rows then a nested-array conversion, and not kept in PHP (the class keeps the count) |
| `spatial.KDTree`, 100 queries (k=5) in 100k | 0.70 ms | 0.36 ms | 0.22 ms | a tree handed back by the caller is checked node by node as the search reaches it (and each index and point it reads) instead of in a pass over the whole tree and data on every call: the per-call cost no longer grows with the tree (4.1 ms at 1M points, now 0.05 ms); a damaged tree is still never read out of bounds |
| `signal.correlate` 10k * 101 | 3.7 ms | 0.25 ms | 0.29 ms | the direct N-D sum walks each in2 row as one contiguous multiply-add (in2 reversed once, not indexed per term), four in1 values at a time so each cell is loaded and stored once, in AVX2 / AVX-512 clones with contraction into FMA off; every cell receives the same terms in the same order, bit for bit; also behind `convolve`, `fftconvolve`, `oaconvolve`, `correlate2d` |
| `signal.resample_poly` 1M, 3/2 | 380 ms | 37 ms | 49 ms | the taps of each output walked as one run, the input index stepped down by one per tap instead of a range test and a division per tap, and away from the edges four outputs of the same phase summed side by side (independent sums over the same taps, each in the order it had alone); the same output, bit for bit (404 cases); also `upfirdn` |
| `signal.medfilt` 100k, k=5 | 12.9 ms | 2.8 ms | 3.0 ms | a window of up to 64 values without NaN sorted by insertion instead of `qsort` with a comparison function (the same order), and in 1-D the window copied as one slice; the same output, bit for bit, NaN included (87,600 cases); also `medfilt2d` and `order_filter` |
| `linalg.pinv` 300x300 | 136 ms | 36 ms | 29 ms | V diag(1/s) U^T as one BLAS `dgemm` on U^T with its rows scaled by the kept 1/s, as NumPy forms it, instead of a scalar triple loop; as close to NumPy and SciPy as before |
| `ndimage.sobel` 1000^2 | 46 ms | 24 ms | 13 ms | each line extended into one buffer, then the filter added tap by tap across the whole line (a vectorizable loop; every output receives the same terms in the same order), and up to eight lines copied out of the image and back together; the same output, bit for bit (1,005 cases); also `correlate1d`, `convolve1d`, `gaussian_filter`, `uniform_filter` and the other separable filters |
| `ndimage.laplace` 1000^2 | 49 ms | 26 ms | 18 ms | the same |
| `ndimage.maximum_filter` 5x5, 1000^2 | 47 ms | 33 ms | 25 ms | eight lines copied at a time, as above |
| `ndimage.label` 1000^2 | 23 ms | 6.0 ms | 6.8 ms | two passes with union-find (each pixel joins the labels of its earlier neighbours, then takes its component's number) instead of a flood fill with a division per pixel visited, and a boolean input read in place instead of converted to float64; a component's first pixel in row-major order opens its smallest label, so the components keep the flood fill's numbers: the same labels and count, checked against the old code on 2,016 cases (every structure family; float64 with NaN, bool, uint8, int64, float32, a transposed view and a PHP list) and against a flood fill in the C tests; a one-sided structure keeps the flood fill |
| `numpy.fft.fft2` 1024^2 | 77 ms | 53 ms | 31 ms | the transform's levels run as passes over the whole row instead of a recursion (one gather of the inputs in digit-reversed order, then each level's butterflies on every block: the same operations on the same values, without hundreds of calls and one-iteration loops per row), and complex values transposed in tiles two words at a time instead of a `memcpy` call per value; every transform the same, bit for bit (1,057 transforms checked against the old code: every length from 1 to 130, powers of 2, 3, 5, primes, Bluestein lengths, NaN, inf and -0, 2-D) |
| `numpy.fft.rfft2` 1024^2 | 41 ms | 26 ms | 12 ms | the same, and the real FFT's split of the half-length spectrum without two integer divisions per bin, in place in the output (bins k and m-k from the same pair) instead of through a separate buffer; `irfft` builds its spectrum in the output likewise |
| `scipy.fft.dct` 1M | 53 ms | 38 ms | 23 ms | the same transforms |
| `signal.hilbert` 1M | 121 ms | 91 ms | 65 ms | the same |
| `signal.resample` 100k -> 50k | 2.6 ms | 1.6 ms | 1.7 ms | the same |
| `signal.welch` 1M, nperseg 1024 | 36 ms | 19 ms | 101 ms | the same |
| `numpy.fft.fft2` 1024^2 | 71 ms | 44 ms | 33 ms | the rows above, and along a leading axis the kernel gathers the columns a chunk at a time into rows, transforms them and scatters them back (`scipyfft.fft_axis`) instead of a transposed copy, the rows and a copy back: the same transform for every column, bit for bit (573 more cases against the old code: every axis of 3-D arrays, padded and cut lengths, all three norms, strided input; and a C test against the transposed form); also `ifft2`, `fftn`, `ifftn`, `rfftn`, `irfftn` and `fft` / `ifft` along any axis but the last |
| `numpy.fft.rfft2` 1024^2 | 43 ms | 21.5 ms | 13.7 ms | the same |
| `signal.savgol_filter` 1M (31, 3) | 27 ms | 6.2 ms | 20 ms | the outputs whose window lies inside the signal (all but the edges) summed tap by tap across blocks of outputs, in SIMD clones with contraction into FMA off, instead of a branch and a range test per tap: every output receives the same terms in the same order (1,140 cases, every mode, bit for bit) |
| `signal.sosfilt` order 8, 1M | 17.7 ms | 8.6 ms | 8.8 ms | four sections per pass, each sample through all of them, instead of a pass per section: the sections of neighbouring samples overlap in the pipeline, and each section sees the same inputs in the same order |
| `signal.filtfilt` order 4, 1M | 27 ms | 19.8 ms | 15.2 ms | the filter's coefficients and state in registers (a copy of the loop for each filter length up to 9) instead of memory that every output store might alias, and the backward pass run backwards through the buffer instead of on a reversed copy; the same values |
| `numpy.clip` 1M | 2.9 ms | 1.85 ms | 1.38 ms | the upper bound applied in place to the lower bound's new array (when the dtype stays) instead of into a third array |
| `sparse.csgraph.shortest_path` 500 nodes | 121 ms | 64 ms | 74 ms | the Floyd-Warshall row update (every method computes the distance matrix this way) as a select over the row in SIMD clones instead of two branches per entry: the same test and update for each entry, so the same distances, bit for bit (1,188 cases: every method, directed and not, negative weights and cycles, NaN, inf and -inf weights) |
| `ndimage.median_filter` 512^2, 3 | 69 ms | 21.7 ms | 34.5 ms | a point whose window lies inside the array reads it through precomputed offsets, in the same order, instead of mapping every element through the boundary mode; windows of up to 64 values without NaN sorted by insertion instead of `qsort` with a comparison function (the same order, -0 and 0 included); also `rank_filter` and `percentile_filter`, bit for bit (1,440 cases, every mode, origins, 1-D to 3-D) |
| `signal.convolve2d` 500x500 * 5x5 | 39 ms | 8.8 ms | 16 ms | the same direct sum |
| `numpy.convolve` 100k * 101 | 6.4 ms | 2.6 ms | 2.7 ms | each output a BLAS `ddot`, as NumPy computes it (its `DOUBLE_dot` calls `cblas_ddot`): now bit-identical to NumPy on 613,596 of 613,596 test values, where the plain summation loop matched 22% exactly; also `numpy.correlate` |
| `stats.rankdata` 1M | 171 ms | 94 ms | 148 ms | the stable order from the kernel radix argsort (`tsr_argsort`: NaN last in input order, -0 equal to 0, the order `std::stable_sort` gave) instead of a comparison sort; and that argsort, above 4,096 values, sorts (key, index) pairs: one most-significant-digit pass over the highest bits where the keys differ, then least-significant-digit passes inside each bucket, skipping digits every key shares. A stable order is unique, so the same ranks, bit for bit (1,866 rank, correlation and slope calls against the old code: every method and NaN policy, ties, -0, axes; and the argsort on arrays of 4,095 to 1M values in six dtypes, with NaN, inf, -0 and subnormals); also `spearmanr`, `kendalltau` and every stable `argsort` |
| `numpy.argsort` 1M | 85 ms | 44 ms | 57 ms | the same argsort |
| `stats.spearmanr` 100k | 18.5 ms | 8.6 ms | 20.8 ms | the same ranks, and a contiguous float64 argument read in place instead of copied |
| `stats.kendalltau` 10k | 2.19 ms | 1.52 ms | 1.72 ms | the same argsort |
| `stats.linregress` 1M | 19.2 ms | 10.8 ms | 11.6 ms | the centred sums of squares and products in one pass, the three multiply-add chains side by side (each in the order it had alone) instead of two centred copies and three dot products, the minimum and maximum in one loop, and a contiguous float64 argument read in place; the same result, bit for bit |
| `stats.ks_2samp` 100k vs 100k | 91 ms | 67 ms | 72 ms | each sample sorted by the kernel sort where the sorted array is unique (no NaN, not both signs of zero; otherwise `std::sort` as before), and the two empirical CDFs at every point counted by a forward merge instead of two binary searches per point; the same statistic, location and p-value, bit for bit (3,104 calls against the old code, also `kstest`, `ks_1samp`, `shapiro`, `cramervonmises`, `anderson`, `anderson_ksamp` and `epps_singleton_2samp`, which share the sort). Most of what remains is the p-value: the same `smirnov` sum SciPy evaluates |
| `numpy.unique` 1M (1000 values) | 15.2 ms | 3.9 ms | 9.7 ms | up to 4,096 distinct values without NaN: each value's first index and count gathered in a hash table, then only the distinct values sorted (values grouped by ==, the first zero of the input kept, as the stable sort keeps it); more distinct values or a NaN take the sort, now in place in the input copy instead of spread into (value, index) pairs; the same values, indices, inverse and counts, bit for bit (4,875 calls, the set routines included); `return_inverse` with counts on the same data 92 -> 12 ms |
| `stats.binned_statistic` 1M, 100 bins | 93 ms | 38 ms | 57 ms | with increasing edges, each sample's bin guessed from its place between the first and last edge and moved to where `x < edge` turns (at most a few edges; the binary search when not confirmed): the edges are sorted, so that index is the binary search's; the values read in place, the bin numbers kept as int64 instead of a float64 copy, and in one dimension the bin index used as the flat index directly; the same statistic, edges and bin numbers, bit for bit (3,840 calls, every statistic, even and uneven edges, infinite edges, 2-D); also `binned_statistic_2d` and `_dd` |
| `signal.find_peaks` 1M | 8.4 ms | 4.2 ms | 4.4 ms | a contiguous float64 signal read in place instead of copied, and the plateau edges of each peak stored only when `plateau_size` (the one filter that reads them) is given; the same peaks and properties, bit for bit (420 calls, every filter, plateaus, NaN, inf, int and float32 input, strided views) |
| `ndimage.zoom` 500^2 x2 (order 3) | 132 ms | 18 ms | 74 ms | the coordinate along each axis depends on that axis's output index alone, so the cval cases, the folded coordinate, the spline weights and each tap's offset are computed once per index and axis instead of once per output point; each point then sums its taps as before, the same products in the same order; the same output, bit for bit (7,096 calls: orders 0 to 5, every mode, zoom factors above and below 1 and per axis, 1-D to 3-D, NaN and inf, int input); also `shift` (1000^2, order 3: 175 -> 50 ms, SciPy 96 ms; 9,400 calls) |
| `numpy.cumsum` 1M | 2.6 ms | 1.19 ms | 2.40 ms | along a single run (a 1-D array, or the last axis of a C-order block) the running value kept in a register instead of read back from the element just stored, which put a store-to-load forward on every step; the same additions in the same order, bit for bit (1,672 calls, every dtype and axis, NaN, inf, -0, integer wrap); also `cumprod` |
| `numpy.cov` 50 x 100k | 47.9 ms | 40.5 ms | 29.8 ms | a contiguous float64 input read in place into the centred matrix instead of first copied whole; the same values, bit for bit (636 calls, also `corrcoef`). NumPy forms X X^T with `dsyrk`, half the products of the `dgemm` used here; switching would change the rounding of the results |
| `stats.gaussian_kde` 1000 points, 1000 evaluations | 16.7 ms | 11.4 ms | 12.3 ms | for one-dimensional data the kernel sum runs on plain doubles instead of through the general d-dimensional loop and its small vectors; the same divisions, products, sums and `exp` calls in the same order, bit for bit (300 calls: one to three dimensions, 2 to 1000 points, ties, NaN, float32). The time left is the million `exp` calls |
| `ndimage.sobel` 1000^2 | 33.9 ms | 3.18 ms | 8.83 ms | a contiguous float64 input read in place instead of copied first: with the copy and the result two 8 MB blocks were in use, and on this host allocating and filling two such blocks costs about 7 ms per call (the memory goes back to the system when they are freed and its pages fault in again), most of the 8 ms a 3-tap pass took. Along an axis other than the last, whole rows of the image are filtered at once instead of one gathered line at a time, and a line's in-range samples are copied, not mapped one by one; every output receives the same terms in the same order, bit for bit (39,900 calls on each backend: 1-D to 4-D, every mode, NaN, inf, -0, origins); also `correlate1d`, `convolve1d`, `uniform_filter1d`, `gaussian_filter1d` and `prewitt`. A later byte-level check showed one exception: where two NaNs meet in a sum, the NaN result can carry the other sign bit (every other bit is the same; the old code was itself not consistent there) |
| `ndimage.laplace` 1000^2 | 36.6 ms | 6.63 ms | 12.7 ms | the same |
| `ndimage.maximum_filter` 5x5, 1000^2 | 33.5 ms | 6.10 ms | 20.4 ms | the same input and row handling (with every bit the same, NaN included: a maximum takes one of its inputs), and the passes alternate between the result and one scratch array instead of copying back after each; the same comparisons in the same order; also `minimum_filter` and the 1-D filters |
| `numpy.clip` 1M | 2.70 ms | 1.31 ms | 1.11 ms | both bounds applied in one pass (`tsr_clip`, a new C entry point) instead of a maximum and then a minimum pass, with the same comparisons in the same order, bit for bit (1,904 calls on each backend: every dtype, NaN and -0 bounds, crossed bounds, out-of-range integers, strided and reversed views). What remains is the write into the fresh result array, which costs NumPy the same |
| `signal.filtfilt` order 4, 1M | 26.4 ms | 15.8 ms | 15.6 ms | a contiguous float64 signal read in place instead of copied first (one 8 MB block less in use, and the page faults it brought); the same values, bit for bit (497 calls with `hilbert` on each backend: lengths from 1 to 65,536, NaN, inf, -0, integer and strided input) |
| `signal.hilbert` 1M | 141 ms | 88.6 ms | 76.1 ms | the input read in place and transformed as real values (`tsr_fft_real` reads x as (x, 0.0), the values the zero-filled complex copy held) instead of first spread into a 16 MB complex buffer; the same output, bit for bit. The time left is the FFT itself. A later byte-level check showed one exception: where two NaNs meet in a sum, the NaN result can carry the other sign bit (every other bit is the same; the old code was itself not consistent there) |
| `fft.dct` 1M | 58.8 ms | 26.5 ms | 28.2 ms | the cos and sin of pi k / 2N (two million libm calls per transform) kept in the FFT plan cache per length instead of computed on every call, from the same expressions, so the same values; the input read in place instead of copied first; the same output, bit for bit, with the cache and without it (816 calls on each backend: `dct`, `idct`, `dst`, `idst`, types 2 and 3, the direct sums up to 128 points and the FFT path above, NaN, inf, -0, integer and reversed input). A later byte-level check showed one exception: where two NaNs meet in a sum, the NaN result can carry the other sign bit (every other bit is the same; the old code was itself not consistent there) |
| `numpy.fft.fft2` 1024^2 | 64.5 ms | 29.7 ms | 32.2 ms | large-block reuse: freed blocks of 1 MiB or more are kept by libtessero (64 MiB per process in the CLI, 16 MiB in a web SAPI) and handed out again, instead of going back to the operating system and page-faulting in again on the next call; the rows step and the axis-0 step each made a fresh 16 MB array. Also `rfft2` 40.0 -> 15.7 ms (NumPy 12.2), and smaller gains wherever large temporaries repeat (`linkage`, `Delaunay`, `dct`, `cov` 40.5 -> 34.3 ms against NumPy's 32.2). The same values: every equivalence check of this page (13 suites, about 74,000 calls) gives the old code's results with reuse on |
| `numpy.einsum` ij,jk->ik 300^2 | 16.7 ms | 6.17 ms | 7.93 ms | in a product of matrices a whole run of the summed label adds rows of b to one output row; they are now added four at a time per element instead of one row per pass, each element still receiving its products one by one in label order: the same sums, bit for bit, with a quarter of the loads and stores (336 calls on each backend: products with inner lengths 1 to 41, batched, ellipsis and broadcast forms, transposed and reduced outputs, three operands, NaN, inf, -0, integers). A later byte-level check showed one exception: where two NaNs meet in a sum, the NaN result can carry the other sign bit (every other bit is the same; the old code was itself not consistent there) |
| `scipy.special` element-wise, 100,000 values | `huber` 0.88, `radian` 0.53, `entr` 1.13, `xlog1py` 1.69 ms | 0.73, 0.38, 0.94, 1.45 ms | -- | a float64 loop over the whole run for each of the 241 functions (generated with the registry, in a translation unit of its own) instead of the engine's generic per-element loop: each element still runs the very same compiled element function, so the results are the same to the last bit, NaN sign bits included (every function on contiguous, scalar-broadcast, strided, float32 and scalar inputs). Cheap functions gain 10 to 30%; tier 1 above was re-timed after it (2026-11-09, `bench/tier1_bench.php`) |
| fixed cost per call, FFI backend | scalar `special.ndtr` 2.7 us; `numpy.diff` on 8 values 10.9 us | 1.3 us; 7.2 us | 0.22 us; 3.3 us | the registry binding prepares each routine once (label, result type, sequence and in-place arguments), fills the kernel's array descriptors in place instead of building and copying one per argument, keeps short string arguments and the C types it needs, and reuses the operand buffers of scalar element-wise calls; also `unique`, `interp`, `trapezoid` on 8 values 1.4-1.6x faster. Tier 3 above was re-timed after it (2026-11-09). Results unchanged, byte for byte |
| `scipy.special` and distribution methods on small arrays | `ndtr` on 8 values 9.8 us; `jv(1.0, x)` 21 us; `jv` 2x1 by 4 24 us; `norm().cdf` on 8 values 24 us | 5.0; 9.2; 12.6; 6.0 us | 0.52; 2.6; 4.3; 41 us | PHP scalar arguments passed as one value with zero strides instead of a new 0-d array each, broadcast strides computed instead of views built, the pointer, shape and stride arrays reused, the output array built directly, and each function's metadata looked up once; in the kernel, a loop over several rows on one thread no longer enters an OpenMP parallel region (also the NumPy ufunc engine on broadcast or strided operands). Best of three alternating rounds at a load average of about 11; results unchanged, byte for byte (47,869 fixture results and 84 calls with scalar, float32, complex, NaN-payload and broadcast operands) |
| array arithmetic and axis functions on small arrays (8 values) | `x * 2.0` 11.8 us; `x + x` 8.9 us; `copy` 5.0 us; `numpy.partition` 18.7 us; `numpy.searchsorted` 17.3 us | 4.8; 5.2; 3.1; 11.1; 11.8 us | 0.68; 0.46; 0.23; 1.35; 1.30 us | binary and unary operations, copies, casts and the loop ufuncs fill one reused set of shape and stride arrays instead of four new ones, pass the buffers' cached pointers, compute broadcast strides once without building views, and pass a PHP scalar's bytes directly instead of a new 0-d array; the generalised ufuncs parse their C types once, write the array descriptors in place, and pass a scalar input after the first through a slot of its own. Best of three alternating rounds at a load average of about 11; results unchanged, byte for byte (47,869 fixture results and 1,727 calls: seven dtypes, a strided view, a transpose and a 0-d array against ten kinds of PHP scalar, NaN payloads, broadcasts and `out=`) |
| routine calls, small sorts and `numpy.fft` on small inputs (8 values) | `argsort` 18.8 us; `lexsort` (2 x 5) 17.6 us; `sort` 9.8 us; `fft` 10.5 us; `unique` with all outputs 21.0 us; `NDArray::array` of 4 floats 4.0 us | 3.7; 4.6; 4.1; 4.7; 12.6; 2.9 us | 1.36; 1.10; 0.92; 5.26; 21.8; 0.45 us | the FFI binding parses each C type once (a type given by name was parsed on every cast), arrays write their own kernel descriptor, and routine results are wrapped without repeating the constructor's checks and keep the block's pointer for freeing; in the kernel, a stable argsort of 32 keys or fewer runs an insertion sort instead of the radix passes, whose 96 KB of count tables were zeroed and scanned on every call (the stable order is unique, so the permutation is the same); `argsort`, `sort` and `numpy.fft` along the last axis skip the axis round trip. Best of three alternating rounds; results unchanged, byte for byte (47,869 fixture results and 2,271 calls: sorts of 1 to 40 values with ties, NaN payloads and -0.0 in six dtypes, FFTs along every axis with every norm) |
| PHP lists as arguments, and generalised ufuncs with a scalar result (8 values) | `median` of a list 14.1 us; `percentile` of a list 16.9 us; `nansum` of 4 integers 15.7 us; `searchsorted` of one value 10.5 us; `broadcast_shapes` of three lists 20.2 us; `special.ndtr` of a list 7.8 us | 7.8; 9.8; 8.6; 7.2; 14.0; 6.6 us | 19.9; 55.7; 4.1; 1.25; 2.2; 1.0 us | a list of up to 4,096 values goes to the kernel as a block holding the values, shape and dtype `NDArray::array()` gives it, instead of becoming an array whose buffer is allocated and freed around the call; a generalised ufunc's 0-d result is read back from a slot as the number `item()` returns, instead of through an array. Best of three alternating rounds; results unchanged, byte for byte (47,869 fixture results and 353 calls with list operands, ragged, non-numeric, empty and too deep lists and their exceptions included). Tier 3 above was re-timed after both rows (2026-11-10) |
| generalised ufuncs and routine arguments on small inputs (8 values) | `median` 5.6 us; `percentile` 6.7 us; `partition` 9.8 us; `nansum` of 4 integers 5.7 us; `empty_like` 4.4 us; `swapaxes` (2 x 2) 5.8 us | 4.6; 6.0; 8.5; 4.9; 3.7; 4.4 us | 19.7; 55.6; 1.35; 3.2; 0.31; 0.57 us | a generalised ufunc keeps the C blocks of its last call (axis masks, array descriptors, parameters, output shapes) and reuses them when no call is using them, instead of allocating six per call; a routine writes `null` and integer arguments into their slots without a call each; and a pointer is read back as an integer through a union instead of two casts, which every new array block and adopted result goes through (`NDArray::empty` 2.0 -> 1.8 us). Best of three alternating rounds; results unchanged, byte for byte (47,869 fixture results and the 4,375 calls of the earlier rows' checks). Tier 3 above was re-timed after it (2026-11-10) |
| `special.ellip_harm_2`, `ellip_normal` (h2 = 5, k2 = 8, n = 3, p = 2), and the Kelvin-function zeros | `ellip_harm_2` 18.3 ms; `ellip_normal` 18.6 ms; `kelvin_zeros(10)` 36.4 ms; `ber_zeros(50)` 15.9 ms; `kei_zeros(20)` 7.7 ms | 12.4 us; 21.3 us; 25.8 ms; 11.0 ms; 5.3 ms | 44.1 us; 145 us; 0.20 ms; 0.078 ms; 0.039 ms | the 256-point Gauss-Legendre rule of the two quadratures (a 256 x 256 tridiagonal eigenproblem, nearly all of a call) is computed once per process instead of on every call; the Kelvin zeros' bisection carries f at the bracket's lower end instead of evaluating it again on every step, as the Bessel and Airy zeros already did. Best of 15 calls in each of three alternating rounds (2026-11-13, at a load average of about 10); results unchanged, byte for byte (47,869 fixture results). In tier 3 above since the 2026-11-15 re-timing; the Kelvin zeros have since moved to SciPy's own routine (see the first table) |

The sorts behind `median`, `quantile`, `unique` and the set routines were libc `qsort` with a comparison function;
they now use `sortSimd` (and `tsr_argsort` for stable orders). `einsum` also stopped skipping a product once a
factor was zero, so a NaN in a later operand propagates as in NumPy.

## Known gaps, largest first

The tier-2 cases clearly slower than SciPy in this sweep, largest gap first; the other five (`median`, the
`shortest_path` Dijkstra case, `clip`, `diff`, `medfilt`, 1.12 to 1.24) are in the tier 2 table below. Since the
2026-11-10 tables, `eigsh`, `ConvexHull`, the Kelvin and Bessel zeros and `quad` have left this list (see "Fixed
after the sweep"), and since this sweep `rotate`, now faster than SciPy after its correction (29.6 ms against
36.1 ms, see the same section). The times below were taken again on 2026-11-15, after the fixes since the sweep:
each case alone, the installed library at commit d2d4e2e6, eight alternating rounds of FFI and SciPy (18:42 to
18:47 UTC, at a load average of 8 to 14), each side's best; the `KDTree` build keeps the sweep's figures. The
sweep's figures for each case are in the tier 2 table below.

| case | Tessero | SciPy | cause | remedy |
|---|---|---|---|---|
| `optimize.curve_fit`, 1000 points | 2.29 ms; 1.16 ms with a model that takes the whole array | 0.60 ms | a per-point PHP model is called once per point and evaluation. A whole-array model is called once per evaluation, as SciPy calls it; since this sweep the Jacobian's evaluations go to it in one call laid out along the data axis, and lmdif takes them as one array, with the same results. Through the extension, with the extension's arrays, the fit takes 0.44 ms against SciPy's 0.56 ms (best of 200 fits, three runs each side). The time left through FFI is its cost per array operation (about 5 us, some 60 operations per fit) | the native extension |
| `spatial.KDTree` build, 100k points from PHP rows | 55.4 ms | 30.7 ms | casting the PHP point rows into one block through FFI; from an NDArray the build takes 30.8 ms, on a par | pass an NDArray |
| `numpy.fft.rfft2` 1024^2 | 15.4 ms | 11.9 ms | the transforms themselves: timed alone after the sweep, the real FFT of the rows took 10.3 ms against 3.6 ms and the complex FFT down the columns 10.0 ms against 5.4 ms (at a load average near 12) | -- |
| `ndimage.binary_erosion` 1000^2 | 12.3 ms | 9.7 ms | not yet profiled | -- |
| `spatial.KDTree`, 100 queries (k=5) in 100k | 0.250 ms; 0.232 ms from an NDArray | 0.207 ms; 0.202 ms | not yet profiled | -- |
| `ndimage.map_coordinates` 1000^2, 100k points | 41.7 ms | 36.0 ms | the interpolation of the points, now SciPy's own operations (see "Fixed after the sweep"; 54.8 ms in the sweep, with the old code); not yet profiled again | -- |
| `signal.hilbert` 1M | 73.5 ms | 66.4 ms | the FFT itself, after the input is read in place; in the sweep 86.8 against 58.3 ms, most of which was load | -- |

## Where SciPy fails and Tessero completes

On the tier-1 inputs SciPy crashed (segmentation fault), aborted or ran for hours where Tessero returned results:
`mathieu_cem`, `mathieu_sem`, `mathieu_modcem1`, `mathieu_modcem2`, `mathieu_modsem1`, `mathieu_modsem2`,
`pro_ang1`, `pro_ang1_cv`, `pro_cv`, `pro_rad1`, `pro_rad1_cv`, `pro_rad2`, `pro_rad2_cv`, `obl_ang1`,
`obl_ang1_cv`, `obl_cv`, `obl_rad1`, `obl_rad1_cv`, `obl_rad2`, `obl_rad2_cv` and `nctdtrinc` (stopped after
6 hours in the first sweep, at the 300-second limit per function in this one); `pbdv` and `pbvv` raised a memory-allocation error. `spherical_jn/yn/in/kn` are not comparable here: the
fixtures mix derivative=0 and 1 rows, and SciPy takes one derivative flag per call.

## Per call through the native extension

Tier 3 again with the native extension (`ext/`) instead of FFI, re-timed on 2026-11-15 at commit 56102533: every
parity fixture call that passed its check, the same cases, timed through `Tessero\Ext\*`. The extension calls the
same kernel without the FFI layer, which costs a few microseconds per call; where the work itself dominates, the two
backends take the same time (`stats.tukey_hsd`: 37 ms on both, SciPy 45 ms). Against NumPy/SciPy the extension is
about 13x faster per call on average (geomean 0.074) and faster on 966 of 1,022 functions, NumPy's own core
functions included (0.29; through FFI they are about 1.6x slower). Of the 56 functions still slower, 36 are within
1.3x, and 33 take NumPy/SciPy under a microsecond per call (`shape`, `ndim`, `promote_types`); some return the input
or a view in NumPy where Tessero returns a copy (`ascontiguousarray`, `asfortranarray`, `atleast_1d`).

Best of four runs on each side, alternating with the FFI runs, at a load average of 5 to 26 (on 2026-11-10, at
commit c5f434b0, the same method gave 0.12 and 887; at commit ad837126 0.12 and 890, at commit 655992c6 0.12 and
882; on 2026-11-06, at commit 2184ba75 and at times above 40, three runs had given 0.11 and 893):

| module | functions | geomean ratio | faster |
|---|---|---|---|
| cluster | 4 | 0.03 | 4 |
| csgraph | 14 | 0.00 | 14 |
| distance | 26 | 0.11 | 26 |
| fft | 18 | 0.38 | 17 |
| hierarchy | 26 | 0.01 | 26 |
| interpolate | 6 | 0.01 | 6 |
| linalg | 30 | 0.23 | 30 |
| ndimage | 66 | 0.09 | 66 |
| np | 193 | 0.29 | 159 |
| npoly | 155 | 0.06 | 155 |
| scipyfft | 11 | 0.58 | 3 |
| signal | 124 | 0.04 | 121 |
| slinalg | 87 | 0.06 | 87 |
| sparse | 23 | 0.03 | 23 |
| special | 89 | 0.28 | 79 |
| splinalg | 14 | 0.01 | 14 |
| stats | 111 | 0.01 | 111 |
| windows | 25 | 0.06 | 25 |
| **all** | 1022 | **0.07** | 966 |

## Sorting against NumPy

`NDArray::sortSimd()` (an AVX2 quicksort with in-register sorting networks) uses the threads set with
`Tessero::setThreads()`; NumPy's `np.sort` uses one. With four threads it sorts a million random values about 2x
faster than NumPy (float64 1.8x, float32 1.9x, int64 1.9x); on one thread it runs level with NumPy, whose default
sort on this machine is Intel's x86-simd-sort vector quicksort (int64 equal, float types up to 1.2x behind). On sorted
or reversed input it is about 5x faster than `np.sort`'s default; NumPy's `kind='stable'` (Timsort) is faster still
there, and on values appended to a sorted array. All of Tessero's sorts return the same result as `np.sort`.

Milliseconds, 1,000,000 values, best of 7 calls in each of 5 alternating rounds; each call returns a new sorted array
on both sides (PHP app vs Python app). Fastest in bold.

| data | np.sort | np.sort stable | sortSimd | sortSimd 4 threads | sortRuns | radix |
|---|---|---|---|---|---|---|
| float64 random | 13.7 | 76.3 | 17.1 | **7.7** | 86.1 | 37.1 |
| float64 10 distinct values | **3.8** | 32.2 | 5.7 | 4.0 | 49.0 | 23.8 |
| float64 sorted | 13.8 | **1.7** | 2.5 | 2.4 | 4.2 | 35.1 |
| float64 reversed | 14.0 | **2.2** | 3.0 | 2.9 | 4.8 | 34.9 |
| float64 sorted, 1% of values moved | 14.2 | 15.9 | 15.6 | **7.1** | 32.9 | 34.9 |
| float64 sorted, 1% random values appended | 14.2 | **3.2** | 16.8 | 7.8 | 26.1 | 34.8 |
| float32 random | 7.4 | 74.1 | 8.9 | **3.9** | 74.7 | 33.8 |
| int64 random | 18.5 | 69.7 | 18.4 | **9.6** | 84.4 | 30.2 |

`sortRuns()` (a natural merge sort) is meant for mostly sorted data on machines without AVX2, where `sortSimd()` falls
back to the radix sort; it does not beat NumPy's Timsort. `sort()` is now the AVX2 sort as well (its time is the
`sortSimd` column); the `radix` column is the radix sort, the path of CPUs without AVX2. Scripts: `bench/sort_bench.php` and `bench/sort_bench.py`; measured on commit 8ce79ab3 under the
host load described in Method, so single-thread results within about 20% vary between runs (the multi-thread
advantage held in every run). How the sorts work and their tests: [sorting design](sort-design.md).

## Method

- Tier 1: the `scipy.special` element-wise blocks of the parity fixtures (`tests/fixtures/parity/special.json`),
  each argument column tiled to 100,000 values; median of repeated runs (`bench/tier1_bench.php` and
  `bench/tier1_bench.py`, the first float64 block of each function; SciPy runs each function in a child process, so
  one that crashes or hangs on the fixture rows is reported as failed).
- Tier 2: `bench/tier2_bench.php` and `bench/tier2_bench.py`, the same calls on the same inputs (seeded generators:
  Tessero's PCG64 streams match NumPy's); median of repeated runs, fast calls in an inner loop.
- Tier 3: `tools/parity/run-tests.php --time` times every fixture call that passed its check (same arguments, 3-200
  repetitions, median), `tools/parity/time_calls.py` times the same cases in NumPy/SciPy, and
  `tools/parity/time_report.py` pairs them case for case. Stateful calls (seeded generators, frozen distributions)
  and in-place calls are left out; so are 87 functions that could not be resolved in the Python namespaces and 988
  cases that raised in SciPy.
- The run behind the tables: every tier timed on 2026-11-15 at commit 56102533 (the build deployed that day): four
  rounds, each running in turn the gate bench of [benchmarks](benchmarks.md) (`bench/run.php`, then
  `bench/numpy_bench.py`), tier 1, tier 2, and tier 3 through FFI, through the extension and in NumPy/SciPy,
  single-threaded (`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`), at a load average of 5 to 26 on 8 hardware threads;
  then four rounds of `bench/usage_bench.php` (FFI, extension) and `bench/usage_bench.py` for the usage-weighted mix.
  Each case keeps its best time on each side, as the host carries unrelated load. Ratios within 10% are read as on
  par.
- Timed again after the sweep: the known gaps and the cases changed since, on 2026-11-15 at commit d2d4e2e6 (the
  installed library), with `bench/tier2_bench.php` and `bench/tier2_bench.py`: each case alone, eight alternating
  rounds of FFI, the extension and NumPy/SciPy (18:42 to 18:47 UTC, at a load average of 8 to 14), each side's best.
  The `KDTree` build keeps the sweep's figure. These are the times in "Known gaps" and in the re-timed rows of
  "Fixed after the sweep".
- Results unchanged: since 2026-11-09 every speed change is also checked against the parity suite's recorded
  results (`tools/parity/run-tests.php --hash`), 47,869 results, on the code before and after the change: none
  changed.
- `bench/sweep_report.py` compares any two `{case: ms}` result files.

## Tier 2, all cases

| case | Tessero ms | NumPy/SciPy ms | ratio |
|---|---|---|---|
| scipy.interpolate make_interp_spline k=3, 1M points | 78.7 | 1.55e+03 | 0.05 |
| scipy.sparse.csgraph connected_components 500 nodes | 0.4 | 4.37 | 0.09 |
| scipy.integrate solve_ivp Lotka-Volterra RK45 | 0.727 | 6.22 | 0.12 |
| scipy.optimize least_squares Rosenbrock residuals | 0.764 | 5.81 | 0.13 |
| scipy.optimize minimize Nelder-Mead rosen n=5 | 0.884 | 6.39 | 0.14 |
| scipy.signal welch 1M, nperseg 1024 | 17.6 | 116 | 0.15 |
| scipy.signal detrend 1M | 9.98 | 43.3 | 0.23 |
| scipy.ndimage zoom 500x500 x2 (old code; see Fixed after the sweep) | 21 | 76.6 | 0.27 |
| scipy.optimize minimize L-BFGS-B rosen n=50 | 25.3 | 88.2 | 0.29 |
| scipy.sparse.linalg spsolve tridiagonal 100k | 12 | 39.1 | 0.31 |
| scipy.ndimage maximum_filter 1000x1000, 5 | 6.18 | 19.7 | 0.31 |
| scipy.stats spearmanr 100k | 8.5 | 22.2 | 0.38 |
| numpy unique 1M (1000 values) | 3.62 | 9.11 | 0.40 |
| numpy trapezoid 1M | 1.43 | 3.42 | 0.42 |
| scipy.ndimage sobel 1000x1000 | 3.93 | 8.87 | 0.44 |
| scipy.stats binned_statistic 1M, 100 bins | 25.5 | 54.8 | 0.46 |
| scipy.spatial ConvexHull 10k (2-D), array input | 0.639 | 1.34 | 0.48 |
| scipy.signal savgol_filter 1M (31, 3) | 8.08 | 16.7 | 0.48 |
| scipy.optimize root (2 equations) | 0.0185 | 0.0372 | 0.50 |
| scipy.stats rankdata 1M | 77 | 146 | 0.53 |
| numpy cumsum 1M | 1.36 | 2.51 | 0.54 |
| scipy.linalg solve_triangular 1000x1000 | 0.368 | 0.667 | 0.55 |
| numpy polyfit deg 5, 100k | 7.95 | 14.2 | 0.56 |
| numpy histogram 1M, 100 bins | 7.15 | 12.6 | 0.57 |
| numpy percentile 1M | 5.57 | 9.66 | 0.58 |
| scipy.signal convolve2d 500x500 * 5x5 | 8.92 | 15.2 | 0.59 |
| scipy.signal decimate 1M by 10 | 12.8 | 21 | 0.61 |
| scipy.ndimage distance_transform_edt 512x512 | 11.3 | 17.4 | 0.65 |
| numpy argsort 1M | 45.4 | 68.1 | 0.67 |
| scipy.ndimage median_filter 512x512, 3 | 23 | 33.8 | 0.68 |
| scipy.ndimage laplace 1000x1000 | 7.68 | 11.3 | 0.68 |
| scipy.stats skew 1M | 9.58 | 14 | 0.69 |
| scipy.signal resample 100k -> 50k | 1.68 | 2.37 | 0.71 |
| scipy.stats kurtosis 1M | 9.17 | 12.7 | 0.72 |
| scipy.signal filtfilt order 4, 1M | 8.64 | 11.8 | 0.74 |
| scipy.signal correlate 10k * 101 | 0.245 | 0.33 | 0.74 |
| scipy.signal resample_poly 1M, 3/2 | 30.2 | 40.2 | 0.75 |
| numpy searchsorted 1M into 100k | 109 | 141 | 0.77 |
| scipy.interpolate PchipInterpolator 10k knots, 1M points | 75.4 | 92.7 | 0.81 |
| scipy.stats mannwhitneyu 100k vs 100k | 20.1 | 24.7 | 0.81 |
| numpy einsum ij,jk->ik 300x300 | 6.24 | 7.38 | 0.85 |
| scipy.ndimage label 1000x1000 | 5.53 | 6.48 | 0.85 |
| numpy fft2 1024x1024 | 27.3 | 31.7 | 0.86 |
| scipy.interpolate CubicSpline 10k knots, 1M points | 78.4 | 91.1 | 0.86 |
| scipy.signal sosfilt order 8, 1M | 8.49 | 9.75 | 0.87 |
| scipy.sparse.csgraph minimum_spanning_tree 500 nodes | 5.5 | 6.3 | 0.87 |
| scipy.stats gaussian_kde 1000 points, 1000 evaluations | 9.75 | 11.1 | 0.88 |
| scipy.spatial Delaunay 10k (2-D) | 27.6 | 31.2 | 0.88 |
| scipy.stats kendalltau 10k | 1.53 | 1.72 | 0.89 |
| scipy.signal find_peaks 1M | 3.92 | 4.34 | 0.90 |
| scipy.stats linregress 1M | 10.9 | 11.9 | 0.91 |
| scipy.integrate quad sin(x) exp(-x) on [0, 10] | 0.00511 | 0.00553 | 0.92 |
| scipy.sparse.linalg cg 2-D Poisson 100x100 | 20.1 | 21.8 | 0.92 |
| scipy.stats ks_2samp 100k vs 100k | 65.3 | 70.3 | 0.93 |
| numpy linalg slogdet 600x600 | 5.54 | 5.9 | 0.94 |
| scipy.interpolate RBFInterpolator 500 -> 10k (2-D) | 70.6 | 73.6 | 0.96 |
| scipy.linalg svdvals 300x300 | 8.88 | 9.25 | 0.96 |
| scipy.linalg lstsq 1000x100 | 5.4 | 5.58 | 0.97 |
| numpy cov 50 x 100k | 32.4 | 33.5 | 0.97 |
| scipy.linalg pinv 300x300 | 23.9 | 24.6 | 0.97 |
| numpy convolve 100k * 101 | 2.46 | 2.53 | 0.97 |
| scipy.linalg qr 500x500 | 16.2 | 16.4 | 0.99 |
| scipy.cluster.hierarchy linkage ward 2000x3 | 62 | 62.4 | 0.99 |
| scipy.linalg schur 300x300 | 56 | 56.1 | 1.00 |
| scipy.fft dct 1M | 22.2 | 22 | 1.01 |
| scipy.spatial KDTree build 100k (3-D), array input | 30.8 | 30.4 | 1.01 |
| scipy.sparse.linalg eigsh k=6, 2-D Poisson 100x100 | 443 | 436 | 1.02 |
| numpy interp 1M into 1000 | 59.3 | 58.2 | 1.02 |
| scipy.linalg eig 300x300 (general) | 61.6 | 60.4 | 1.02 |
| scipy.linalg cholesky 1000x1000 | 14.2 | 13.8 | 1.03 |
| scipy.stats norm.ppf 1M | 45 | 43.5 | 1.04 |
| scipy.cluster.vq kmeans2 10k x 3, k=8 | 4.22 | 3.99 | 1.06 |
| scipy.spatial ConvexHull 10k (2-D) | 1.52 | 1.41 | 1.08 |
| scipy.signal medfilt 100k, k=5 | 2.63 | 2.34 | 1.12 |
| numpy diff 1M | 1.38 | 1.22 | 1.14 |
| numpy clip 1M | 1.43 | 1.25 | 1.14 |
| scipy.sparse.csgraph shortest_path 500 nodes (Dijkstra) | 70 | 60.3 | 1.16 |
| numpy median 1M | 10.7 | 8.63 | 1.24 |
| scipy.spatial KDTree query 100 points, k=5, array input | 0.254 | 0.202 | 1.25 |
| scipy.spatial KDTree query 100 points, k=5 | 0.268 | 0.207 | 1.29 |
| scipy.ndimage binary_erosion 1000x1000 | 13.8 | 9.79 | 1.41 |
| scipy.ndimage rotate 512x512, 30 deg (old code; see Fixed after the sweep) | 49.7 | 35.3 | 1.41 |
| scipy.signal hilbert 1M | 86.8 | 58.3 | 1.49 |
| scipy.ndimage map_coordinates 1000x1000, 100k points (old code; see Fixed after the sweep) | 54.8 | 36.4 | 1.50 |
| numpy rfft2 1024x1024 | 15.9 | 9.86 | 1.61 |
| scipy.spatial KDTree build 100k (3-D) | 55.4 | 30.7 | 1.80 |
| scipy.optimize curve_fit exp decay, 1000 points, array model | 1.36 | 0.59 | 2.31 |
| scipy.optimize curve_fit exp decay, 1000 points | 2.25 | 0.584 | 3.85 |

## Tier 3, by module

| module | functions | geomean ratio | faster |
|---|---|---|---|
| cluster | 4 | 0.12 | 4 |
| csgraph | 14 | 0.03 | 14 |
| distance | 26 | 0.39 | 25 |
| fft | 18 | 1.26 | 5 |
| hierarchy | 26 | 0.04 | 26 |
| interpolate | 6 | 0.03 | 6 |
| linalg | 30 | 0.82 | 18 |
| ndimage | 66 | 0.40 | 52 |
| np | 193 | 1.56 | 65 |
| npoly | 155 | 0.34 | 139 |
| scipyfft | 11 | 0.95 | 3 |
| signal | 124 | 0.15 | 109 |
| slinalg | 87 | 0.23 | 85 |
| sparse | 23 | 0.19 | 21 |
| special | 89 | 0.69 | 39 |
| splinalg | 14 | 0.04 | 14 |
| stats | 111 | 0.03 | 108 |
| windows | 25 | 0.38 | 21 |
| **all** | 1022 | **0.30** | 754 |

## Tier 1, all functions

| scipy.special function | Tessero ms | SciPy ms | ratio |
|---|---|---|---|
| bdtr | 3 | 47.5 | 0.06 |
| bdtrc | 3.3 | 47.6 | 0.07 |
| smirnov | 0.719 | 6.13 | 0.12 |
| comb | 0.937 | 4.51 | 0.21 |
| diric | 2.67 | 5.79 | 0.46 |
| ellipk | 0.414 | 0.887 | 0.47 |
| nbdtrc | 7.01 | 14.5 | 0.48 |
| ellipe | 0.408 | 0.839 | 0.49 |
| bdtri | 41.4 | 84.9 | 0.49 |
| nbdtr | 8.48 | 15.9 | 0.53 |
| yn | 6.79 | 11.4 | 0.60 |
| sindg | 2.08 | 3.12 | 0.67 |
| factorial | 1.36 | 1.98 | 0.68 |
| log_wright_bessel | 70.2 | 101 | 0.69 |
| ndtri_exp | 1.41 | 2.01 | 0.70 |
| boxcox | 2.18 | 2.97 | 0.73 |
| wright_bessel | 64.8 | 84.8 | 0.76 |
| yvp | 201 | 255 | 0.79 |
| perm | 0.863 | 1.08 | 0.80 |
| logit | 1.04 | 1.28 | 0.81 |
| kn | 25.5 | 31.2 | 0.82 |
| expn | 16.6 | 20.3 | 0.82 |
| it2j0y0 | 33.9 | 40.7 | 0.83 |
| rgamma | 5.4 | 6.47 | 0.83 |
| shichi | 14.7 | 17.4 | 0.84 |
| wrightomega | 7.29 | 8.61 | 0.85 |
| spence | 2.03 | 2.4 | 0.85 |
| chdtr | 4.74 | 5.59 | 0.85 |
| btdtrib | 27.4 | 32.2 | 0.85 |
| it2i0k0 | 28.5 | 32.9 | 0.87 |
| sici | 3.65 | 4.18 | 0.87 |
| k1 | 3.31 | 3.78 | 0.88 |
| k1e | 3.31 | 3.78 | 0.88 |
| stdtrit | 78.5 | 88.6 | 0.89 |
| i1e | 5.1 | 5.73 | 0.89 |
| i0e | 5.34 | 5.99 | 0.89 |
| y1 | 1.95 | 2.18 | 0.90 |
| i1 | 6.35 | 7.08 | 0.90 |
| tklmbda | 96.3 | 107 | 0.90 |
| nbdtri | 163 | 182 | 0.90 |
| log1p | 0.843 | 0.938 | 0.90 |
| y0 | 2.4 | 2.67 | 0.90 |
| psi | 10.8 | 12 | 0.90 |
| chdtrc | 20.4 | 22.4 | 0.91 |
| nbdtrik | 173 | 190 | 0.91 |
| yve | 76 | 83.6 | 0.91 |
| nbdtrin | 256 | 281 | 0.91 |
| smirnovi | 959 | 1.05e+03 | 0.91 |
| betainc | 2.86 | 3.13 | 0.91 |
| stdtridf | 87.9 | 96.2 | 0.91 |
| k0 | 4.36 | 4.78 | 0.91 |
| stdtr | 28.7 | 31.4 | 0.91 |
| it2struve0 | 38 | 41.4 | 0.92 |
| ivp | 78.8 | 85.8 | 0.92 |
| nctdtridf | 416 | 453 | 0.92 |
| chndtridf | 731 | 792 | 0.92 |
| chdtri | 71.8 | 77.7 | 0.92 |
| polygamma | 28.8 | 31.2 | 0.93 |
| yv | 83.7 | 90.5 | 0.93 |
| kvp | 56.7 | 61.2 | 0.93 |
| k0e | 3.74 | 4.03 | 0.93 |
| chndtr | 19.4 | 20.8 | 0.93 |
| itairy | 21.7 | 23.4 | 0.93 |
| kolmogi | 21.2 | 22.7 | 0.93 |
| bdtrin | 210 | 223 | 0.94 |
| nrdtrisd | 115 | 122 | 0.94 |
| kve | 20.5 | 21.8 | 0.94 |
| chdtriv | 160 | 169 | 0.95 |
| iti0k0 | 29.1 | 30.5 | 0.95 |
| kv | 21.7 | 22.8 | 0.95 |
| jn | 56 | 58.7 | 0.95 |
| sinc | 1.55 | 1.63 | 0.95 |
| owens_t | 11.1 | 11.6 | 0.96 |
| i0 | 6.47 | 6.73 | 0.96 |
| btdtria | 191 | 199 | 0.96 |
| chndtrinc | 1.51e+03 | 1.57e+03 | 0.96 |
| voigt_profile | 10.5 | 10.8 | 0.97 |
| ive | 29.1 | 30 | 0.97 |
| nctdtr | 314 | 324 | 0.97 |
| hyperu | 2.34e+04 | 2.4e+04 | 0.97 |
| beip | 73.9 | 75.9 | 0.97 |
| ellipeinc | 2.4 | 2.47 | 0.97 |
| eval_chebyt | 38.5 | 39.5 | 0.97 |
| kei | 41.2 | 42.3 | 0.98 |
| iv | 47.7 | 48.7 | 0.98 |
| jv | 62.8 | 63.9 | 0.98 |
| ncfdtri | 2.57e+03 | 2.61e+03 | 0.98 |
| ber | 76.6 | 77.9 | 0.98 |
| bei | 78.6 | 79.9 | 0.98 |
| lpmv | 271 | 275 | 0.98 |
| struve | 387 | 392 | 0.99 |
| airy | 174 | 176 | 0.99 |
| ncfdtrinc | 610 | 616 | 0.99 |
| eval_chebys | 57.4 | 57.9 | 0.99 |
| eval_sh_chebyu | 109 | 110 | 0.99 |
| ncfdtr | 44.9 | 45.3 | 0.99 |
| pdtri | 55 | 55.4 | 0.99 |
| chndtrix | 509 | 513 | 0.99 |
| kerp | 34.6 | 34.8 | 0.99 |
| eval_laguerre | 43.5 | 43.7 | 0.99 |
| betaincc | 10.4 | 10.4 | 1.00 |
| cbrt | 1.95 | 1.96 | 1.00 |
| betaincinv | 76.7 | 76.8 | 1.00 |
| eval_legendre | 45.8 | 45.8 | 1.00 |
| j0 | 3.01 | 3.01 | 1.00 |
| ncfdtridfn | 458 | 456 | 1.00 |
| cotdg | 2.75 | 2.73 | 1.01 |
| gammaincc | 14.3 | 14.2 | 1.01 |
| boxcox1p | 2.61 | 2.59 | 1.01 |
| ker | 51.1 | 50.6 | 1.01 |
| pdtrik | 351 | 347 | 1.01 |
| nctdtrit | 3.56e+04 | 3.51e+04 | 1.01 |
| keip | 40.8 | 40.2 | 1.01 |
| zeta | 33.7 | 33.3 | 1.01 |
| exp1 | 26.6 | 26.2 | 1.02 |
| cosm1 | 1.35 | 1.33 | 1.02 |
| eval_sh_chebyt | 38.8 | 38.1 | 1.02 |
| bdtrik | 94.8 | 93 | 1.02 |
| eval_chebyu | 44.7 | 43.9 | 1.02 |
| elliprj | 18.3 | 17.9 | 1.02 |
| eval_jacobi | 164 | 161 | 1.02 |
| gdtr | 4.34 | 4.25 | 1.02 |
| fdtridfd | 161 | 158 | 1.02 |
| pseudo_huber | 3.06 | 2.99 | 1.02 |
| pdtrc | 9.11 | 8.91 | 1.02 |
| ellipj | 2.06 | 2.01 | 1.02 |
| besselpoly | 120 | 118 | 1.02 |
| assoc_laguerre | 13.3 | 12.9 | 1.03 |
| itstruve0 | 24.7 | 24.1 | 1.03 |
| betaln | 8.59 | 8.37 | 1.03 |
| tandg | 1.95 | 1.9 | 1.03 |
| fdtri | 124 | 121 | 1.03 |
| digamma | 6.02 | 5.85 | 1.03 |
| ellipkinc | 1.83 | 1.78 | 1.03 |
| berp | 79 | 76.6 | 1.03 |
| gammaincinv | 54.2 | 52.5 | 1.03 |
| eval_genlaguerre | 206 | 199 | 1.03 |
| eval_chebyc | 39 | 37.7 | 1.03 |
| nrdtrimn | 111 | 108 | 1.04 |
| gammainccinv | 69.4 | 66.9 | 1.04 |
| pdtr | 9.1 | 8.76 | 1.04 |
| airye | 258 | 249 | 1.04 |
| eval_sh_jacobi | 94 | 90.3 | 1.04 |
| jve | 58.8 | 56.4 | 1.04 |
| gammainc | 6.13 | 5.87 | 1.04 |
| log_expit | 1.68 | 1.6 | 1.05 |
| gdtrc | 11.7 | 11.2 | 1.05 |
| elliprd | 12.6 | 12 | 1.05 |
| hyp0f1 | 39.1 | 37.3 | 1.05 |
| jvp | 87.5 | 83.4 | 1.05 |
| ncfdtridfd | 360 | 342 | 1.05 |
| xlogy | 0.872 | 0.829 | 1.05 |
| mathieu_b | 66.7 | 63.4 | 1.05 |
| modstruve | 346 | 328 | 1.06 |
| fdtrc | 25.8 | 24.2 | 1.07 |
| eval_sh_legendre | 35.5 | 33.3 | 1.07 |
| fdtr | 25.2 | 23.6 | 1.07 |
| zetac | 12.9 | 12.1 | 1.07 |
| gdtrix | 55.2 | 51.5 | 1.07 |
| mathieu_a | 84 | 78.4 | 1.07 |
| erfi | 2.02 | 1.89 | 1.07 |
| inv_boxcox | 1.98 | 1.84 | 1.07 |
| expi | 36.2 | 33.6 | 1.08 |
| betainccinv | 81.3 | 75.4 | 1.08 |
| elliprf | 9.1 | 8.41 | 1.08 |
| binom | 7.11 | 6.56 | 1.08 |
| powm1 | 5.06 | 4.66 | 1.09 |
| pbwa | 101 | 92.7 | 1.09 |
| j1 | 3.39 | 3.11 | 1.09 |
| gdtrib | 255 | 234 | 1.09 |
| gdtria | 52.9 | 48.5 | 1.09 |
| elliprc | 8.72 | 7.98 | 1.09 |
| hyp1f1 | 1.54e+03 | 1.39e+03 | 1.10 |
| inv_boxcox1p | 2.22 | 2 | 1.11 |
| round | 0.417 | 0.376 | 1.11 |
| dawsn | 0.926 | 0.827 | 1.12 |
| log_ndtr | 3.19 | 2.85 | 1.12 |
| itmodstruve0 | 19.5 | 17.4 | 1.12 |
| eval_gegenbauer | 251 | 223 | 1.13 |
| gamma | 4.76 | 4.19 | 1.13 |
| elliprg | 36.4 | 31.9 | 1.14 |
| gammaln | 4.52 | 3.96 | 1.14 |
| poch | 3.31 | 2.9 | 1.14 |
| beta | 8.88 | 7.77 | 1.14 |
| exp10 | 2.94 | 2.56 | 1.15 |
| itj0y0 | 42.1 | 36.6 | 1.15 |
| loggamma | 1.17 | 1.01 | 1.15 |
| expit | 0.979 | 0.841 | 1.16 |
| erf | 1.76 | 1.5 | 1.17 |
| exp2 | 2.4 | 2.05 | 1.17 |
| kolmogorov | 4.59 | 3.92 | 1.17 |
| rel_entr | 0.872 | 0.744 | 1.17 |
| erfinv | 15.6 | 13.3 | 1.17 |
| cosdg | 2.27 | 1.91 | 1.19 |
| gammasgn | 0.38 | 0.32 | 1.19 |
| hyp2f1 | 900 | 756 | 1.19 |
| exprel | 1.79 | 1.5 | 1.19 |
| erfc | 1.74 | 1.45 | 1.20 |
| expm1 | 0.884 | 0.734 | 1.20 |
| radian | 0.365 | 0.302 | 1.21 |
| entr | 0.606 | 0.501 | 1.21 |
| kl_div | 0.803 | 0.654 | 1.23 |
| softplus | 2.05 | 1.63 | 1.26 |
| erfcx | 1.06 | 0.842 | 1.26 |
| xlog1py | 1.16 | 0.89 | 1.31 |
| fresnel | 8.2 | 6.24 | 1.31 |
| ellipkm1 | 2.13 | 1.62 | 1.32 |
| agm | 2.51 | 1.9 | 1.32 |
| ndtri | 2.54 | 1.85 | 1.37 |
| huber | 0.315 | 0.23 | 1.37 |
| erfcinv | 2.41 | 1.74 | 1.39 |
| ndtr | 2.33 | 1.66 | 1.41 |

## Timing history, 2026-11-06 to 2026-11-10

The timings before the 2026-11-15 run, kept for the record: they are where the "before" columns of "Fixed during
the sweep" and "Fixed after the sweep" come from. Every table above comes from the 2026-11-15 run (see Method).

- Tier 2, first run (2026-11-06): each side ran three times, alternating, and each case kept its best time, as the
  host carries unrelated load (load average 16 to 22 on 8 cores during the last run).
- Tier 2 cases re-timed after their fixes, from 2026-11-07: `sort()` as the AVX2 sort, the `KDTree` built once (in
  the constructor, as SciPy does; before, the build row timed no tree at all and every query rebuilt it), the sparse
  solvers on CSR input (the three sparse cases had failed: the solvers took dense matrices only), and `linkage`,
  `Delaunay`, `solve_triangular`, `median`, `percentile`, `curve_fit`, `eigsh`, `ConvexHull` and `KDTree` (the last
  four re-timed on their own, at a load average of 11 to 19; the spatial cases over twelve alternating rounds, as
  their times moved by up to half between rounds); `resample_poly`, `medfilt`, `pinv`, `sobel`, `laplace` and
  `maximum_filter` (five alternating rounds of the new code, the old code and SciPy, at a load average of 26 to 41,
  which slowed both sides: their before and after times come from the same rounds, so they sit above the sweep's
  times for the same code); `label` (the same, at a load average of 16 to 18); the FFT cases (two runs of five
  alternating rounds, at a load average of 10 to 21; each side's best of the ten; `fft2` and `rfft2` once more after
  the leading-axis change, five rounds at a load average of 12 to 14); `savgol_filter`, `sosfilt`, `clip` and
  `filtfilt` (five alternating rounds each, at a load average of 12 to 17); `shortest_path` and `median_filter` (four
  alternating rounds each, at a load average of 7 to 15); `rankdata`, `argsort`, `spearmanr`, `kendalltau` and
  `linregress` (ten alternating rounds, at a load average of 7 to 17); `unique` (six alternating rounds, at a load
  average of 6 to 10); `ks_2samp` (twelve alternating rounds, at a load average of up to 38); `binned_statistic` (six
  alternating rounds, at a load average of 5 to 10); `find_peaks` (six alternating rounds, at a load average of 18);
  `zoom` (six alternating rounds, at a load average of 6 to 9); `cov` and `cumsum` (six alternating rounds, at a load
  average of 9 to 10); `gaussian_kde` (six alternating rounds, at a load average of 8 to 9); `clip`, `sobel`,
  `laplace` and `maximum_filter` (six alternating rounds, at a load average of 6 to 21, `clip` twelve more);
  `filtfilt` and `hilbert` (six alternating rounds, at a load average of 10 to 14); `dct` (six alternating rounds, at
  a load average of 10 to 17); after the large-block reuse, `fft2`, `rfft2`, `dct`, `linkage`, `kmeans2`, `sosfilt`,
  `connected_components`, `ConvexHull` and `Delaunay` (six alternating rounds, at a load average of 7 to 9);
  `einsum` (six alternating rounds, at a load average of 9 to 10); and, again after the large-block reuse, `cov`,
  `clip`, `hilbert`, `pinv`, `svdvals` and `cholesky` (six alternating rounds, at a load average of 9 to 11).
- Tiers 1 and 3 re-timed on 2026-11-09, after the lower per-call cost of the FFI path (commit c6e18549): two
  alternating rounds of each side (`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`, at a load average of 9 to 18), each
  case keeping its best time on each side.
- Tier 3 re-timed on 2026-11-10, after the small-array changes (commits c5ae8bd3 and 655992c6): four alternating
  rounds of each side (at a load average of 8 to 15), each case keeping its best time on each side. The geomean rose
  from 0.37 to 0.42 although Tessero's own times fell (to 0.90 of the 2026-11-09 times on `np`, 0.79 on `special`,
  0.68 on `signal`): the NumPy/SciPy side ran faster on this host than in the two 2026-11-09 rounds (to 0.54 of
  those times on `npoly`, 0.57 on `scipyfft`, 0.62 on `slinalg`), which had flattered the earlier figure. Four rounds
  per side give the steadier reference.
- Tier 3 re-timed again on 2026-11-10, at commit ad837126 (after the changes of commits 5960de74 and ad837126):
  four rounds of FFI, the extension and NumPy/SciPy in turn (at a load average of 8 to 17), each case keeping its
  best time on each side. Tessero's FFI times fell to 0.79 of those of the earlier 2026-11-10 run (`fft` 0.66, `np`
  0.78, `special` 0.87) and the NumPy/SciPy side's to 0.98; through the extension, to 0.96 (the NumPy/SciPy side to
  0.95 of that run's).
- Tier 3 re-timed a third time on 2026-11-10, at commit c5f434b0 (after the generalised-ufunc change): four rounds of
  FFI, the extension and NumPy/SciPy in turn (at a load average of 5 to 21), each case keeping its best time on each
  side. Tessero's FFI times fell to 0.975 of those of the ad837126 run (`stats` 0.90, `sparse` 0.94, `np` 0.97) and
  the NumPy/SciPy side's stayed at 1.005; through the extension, whose code did not change, 1.006. `csgraph` and
  `cluster`, the first fixtures of each round, read 1.34 and 1.28 on Tessero's side in this run, while alternating
  runs of those fixtures on the code before and after give 1.04 and 0.97: the host's other jobs, not the change.
  Leaving those two modules out of both runs, the geomean fell from 0.350 to 0.338.
- Tier 1 re-timed on 2026-11-10 (the code of commit c880558f): four alternating rounds of each side (at a load
  average of 8 to 26), each function keeping its best time on each side. Tessero's times fell to 0.89 of the
  2026-11-09 times and SciPy's to 0.98; the one kernel change to these loops in between is that an element-wise loop
  on one thread no longer runs inside an OpenMP parallel region (commit c5ae8bd3).
- The 2026-11-15 run against the 2026-11-10 tables: tier 2 gained four cases (`curve_fit` with a model that takes
  the whole array, and the `KDTree` build and query and `ConvexHull` from an NDArray). Tessero's own times fell to
  0.956 of the earlier ones in tier 1, 0.889 in tier 2 (the 84 shared cases) and 0.89 in tier 3 through FFI (0.58
  through the extension); the NumPy/SciPy side's fell to 0.965, 0.863 and 0.96, so the ratios moved less than
  Tessero's times.
