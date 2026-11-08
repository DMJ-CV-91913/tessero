# Speed sweep, 2026-11-06

Every function family timed against NumPy 2.4.4 / SciPy 1.17.1, on the same inputs, single-threaded on both sides
(`OPENBLAS_NUM_THREADS=1`), on the benchmark machine of [benchmarks](benchmarks.md) (Intel Core i7-4790, Haswell,
AVX2; PHP 8.5, FFI backend). A ratio is Tessero time / reference time: below 1, Tessero is faster.

| tier | what is timed | timed on both sides | geomean ratio | Tessero faster |
|---|---|---|---|---|
| 1 | all `scipy.special` element-wise functions, 100,000 values each (the parity fixture inputs, tiled) | 212 functions | **1.05** | 31 |
| 2 | 84 heavier calls at realistic sizes (linalg, FFT, signal, ndimage, interpolate, integrate, optimize, stats, sparse, csgraph, spatial, cluster), re-timed after the fixes below | 84 cases | **0.83** | 48 |
| 3 | per-call latency of every parity fixture call (small inputs) | 1,015 functions, 15,117 cases | **0.62** | 615 |

Reading it: on small inputs (tier 3) the fixed cost of a call dominates, and Tessero's compiled kernels behind one
FFI call beat SciPy's Python-level wrappers (stats 0.06, csgraph 0.04, signal 0.28); NumPy's own core functions are
the exception (np 3.20), as they are C all the way down. On element-wise math at 100,000 values (tier 1) the two are
on a par. On heavy calls (tier 2) Tessero is ahead on optimizers, ODEs and splines: clearly faster (by more than 10%)
on 37 cases, within 10% either way on 13, and clearly slower on 34, mostly where an algorithm is not yet the best one
available (the gaps below).

## Fixed during the sweep

| function | problem | before | after |
|---|---|---|---|
| `scipy.fft` dct / idct / dst / idst (and the n-D forms) | direct O(N^2) sums | 1M points: about 5 hours (estimated from 8,000 points: 1.1 s) | 50 ms (SciPy 17-30 ms); agrees with SciPy to 1e-16 of the output scale |
| `interpolate.CubicSpline`, `make_interp_spline` (k=3) | the tridiagonal system solved as a dense n x n matrix | 10k knots: 8-17 s and 800 MB | 76 ms, O(n) memory (LAPACK `dgbsv`, as SciPy) |
| `sparse.linalg.lsmr` | `maxiter` read from the `conlim` argument: 100 million iterations per default call | 15 s on a 10 x 6 system | 16 us |
| `signal.butter`, `cheby1`, `cheby2`, `ellip`, `bessel`, `iirfilter` | `output='sos'` / `'zpk'` silently answered with (b, a) | wrong format, no error | refused with a clear error (convert with `tf2sos`) |
| `np.modf`, `npoly` polymulx and the derivatives, `csgraph.laplacian`, `slinalg.invpascal` | the sign of an exact zero differed from NumPy / SciPy | -- | match; the parity comparator now checks the sign of real zeros |

## Fixed after the sweep

The quick gaps, closed the same day (tier-2 cases, single-threaded; results unchanged, checked by parity):

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
| `signal.convolve2d` 500x500 * 5x5 | 39 ms | 8.8 ms | 16 ms | the same direct sum |
| `numpy.convolve` 100k * 101 | 6.4 ms | 2.6 ms | 2.7 ms | each output a BLAS `ddot`, as NumPy computes it (its `DOUBLE_dot` calls `cblas_ddot`): now bit-identical to NumPy on 613,596 of 613,596 test values, where the plain summation loop matched 22% exactly; also `numpy.correlate` |

The sorts behind `median`, `quantile`, `unique` and the set routines were libc `qsort` with a comparison function;
they now use `sortSimd` (and `tsr_argsort` for stable orders). `einsum` also stopped skipping a product once a
factor was zero, so a NaN in a later operand propagates as in NumPy.

## Known gaps, largest first

The gaps profiled so far; every case slower than SciPy is in the tier 2 table below (`quad`, `shortest_path`
and `median_filter` are the largest not yet profiled).

| case | Tessero | SciPy | cause | remedy |
|---|---|---|---|---|
| `optimize.curve_fit`, 1000 points | 5.0 ms | 0.61 ms | the model is a PHP function called once per point: 36 evaluations of 1000 points, about 3 ms, where SciPy calls it once per evaluation on the whole array; the iterations' arithmetic runs in the kernel | -- |
| `spatial.KDTree`, 100k points: build / 100 queries | 58 / 0.36 ms | 33 / 0.22 ms | the kernel build is about 35 ms; the rest is the constructor casting the PHP point arrays into one flat list (SciPy receives an ndarray) | -- |
| `sparse.linalg.eigsh` k=6, 100 x 100 grid Poisson (CSR) | 0.74 s | 0.50 s | after the Lanczos run, a second one on the complement of the converged vectors checks that no repeated eigenvalue was missed (this spectrum has many); it takes about a third of the time | -- |
| `spatial.ConvexHull`, 10k (2-D) | 2.0 ms | 1.5 ms | the kernel call is about 1.0 ms (the sort and the chain); the rest is casting the PHP point arrays (about 0.5 ms) and the call | -- |
| `special.ellip_harm_2`, Kelvin-function zeros (tier 3) | 100-600x | -- | slow per-call algorithms | -- |

## Where SciPy fails and Tessero completes

On the tier-1 inputs SciPy crashed (segmentation fault), aborted or ran for hours where Tessero returned results:
`mathieu_cem`, `mathieu_sem`, `mathieu_modcem1`, `mathieu_modcem2`, `mathieu_modsem1`, `mathieu_modsem2`,
`pro_ang1`, `pro_ang1_cv`, `pro_cv`, `pro_rad1`, `pro_rad1_cv`, `pro_rad2`, `pro_rad2_cv`, `obl_ang1`,
`obl_ang1_cv`, `obl_cv`, `obl_rad1`, `obl_rad1_cv`, `obl_rad2`, `obl_rad2_cv` and `nctdtrinc` (stopped after
6 hours); `pbdv` and `pbvv` raised a memory-allocation error. `spherical_jn/yn/in/kn` are not comparable here: the
fixtures mix derivative=0 and 1 rows, and SciPy takes one derivative flag per call.

## Per call through the native extension

Tier 3 again with the native extension (`ext/`) instead of FFI, on commit 2184ba75: every parity fixture call
that passed its check, the same cases, timed through `Tessero\Ext\*`. The extension calls the same kernel without
the FFI layer, which costs tens of microseconds per call; where the work itself dominates, the two backends take the
same time (`stats.tukey_hsd`: 54 ms on both). Against NumPy/SciPy the extension is about 9x faster per call on
average and faster on 893 of 1,022 functions, NumPy's own core functions included (0.47; through FFI they were about
3x slower).

Best of three runs on each side, alternating (the host carried unrelated load, at times above 40 on 8 cores; a
single run of each gave 0.08, flattering the extension, which had run at the lower load):

| module | functions | geomean ratio | faster |
|---|---|---|---|
| cluster | 4 | 0.04 | 4 |
| csgraph | 14 | 0.01 | 14 |
| distance | 26 | 0.14 | 26 |
| fft | 18 | 0.33 | 17 |
| hierarchy | 26 | 0.01 | 26 |
| interpolate | 6 | 0.01 | 6 |
| linalg | 30 | 0.31 | 26 |
| ndimage | 66 | 0.13 | 66 |
| np | 193 | 0.47 | 133 |
| npoly | 155 | 0.11 | 148 |
| scipyfft | 11 | 0.65 | 4 |
| signal | 124 | 0.05 | 116 |
| slinalg | 87 | 0.06 | 87 |
| sparse | 23 | 0.03 | 23 |
| special | 89 | 0.92 | 49 |
| splinalg | 14 | 0.02 | 14 |
| stats | 111 | 0.01 | 109 |
| windows | 25 | 0.07 | 25 |
| **all** | 1022 | **0.11** | 893 |

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
  each argument column tiled to 100,000 values; median of repeated runs.
- Tier 2: `bench/tier2_bench.php` and `bench/tier2_bench.py`, the same calls on the same inputs (seeded generators:
  Tessero's PCG64 streams match NumPy's); median of repeated runs, fast calls in an inner loop. Re-timed after the
  fixes above (last on 2026-11-07): `sort()` as the AVX2 sort, the `KDTree` built once (in the constructor, as SciPy
  does; before, the build row timed no tree at all and every query rebuilt it), the sparse solvers on CSR input (the
  three sparse cases had failed: the solvers took dense matrices only), and `linkage`, `Delaunay`, `ConvexHull`,
  `solve_triangular`, `median`, `percentile`, `curve_fit`, `eigsh`, `ConvexHull` and `KDTree` as listed above (the
  last four re-timed on their own, at a load average of 11 to 19; the spatial cases over twelve alternating rounds,
  as their times moved by up to half between rounds), `resample_poly`, `medfilt`, `pinv`, `sobel`, `laplace` and
  `maximum_filter` (five alternating rounds of the new code, the old code and SciPy, at a load average of 26 to 41,
  which slowed both sides: their before and after times come from the same rounds, so they sit above the sweep's
  times for the same code), `label` (the same, at a load average of 16 to 18) and the FFT cases (two runs of five
  alternating rounds, at a load average of 10 to 21; each side's best of the ten; `fft2` and `rfft2` once more
  after the leading-axis change, five rounds at a load average of 12 to 14), and `savgol_filter`, `sosfilt`, `clip` and
  `filtfilt` (five alternating rounds each, at a load average of 12 to 17). Each side ran three times, alternating,
  and each case keeps its best time, as the host carries unrelated load (load average 16 to 22 on 8 cores during the
  last run). Ratios within 10% are read as on par.
- Tier 3: `tools/parity/run-tests.php --time` times every fixture call that passed its check (same arguments, 3-200
  repetitions, median), `tools/parity/time_calls.py` times the same cases in NumPy/SciPy, and
  `tools/parity/time_report.py` pairs them case for case. Stateful calls (seeded generators, frozen distributions)
  and in-place calls are left out; so are 86 functions that could not be resolved in the Python namespaces and 988
  cases that raised in SciPy. Tier 3 ran before the `lsmr` fix.
- `bench/sweep_report.py` compares any two `{case: ms}` result files.

## Tier 2, all cases

| case | Tessero ms | NumPy/SciPy ms | ratio |
|---|---|---|---|
| scipy.optimize least_squares Rosenbrock residuals | 0.307 | 8.16 | 0.04 |
| scipy.interpolate make_interp_spline k=3, 1M points | 88 | 2.06e+03 | 0.04 |
| scipy.sparse.csgraph connected_components 500 nodes | 0.256 | 5.36 | 0.05 |
| scipy.optimize minimize Nelder-Mead rosen n=5 | 0.983 | 12.3 | 0.08 |
| scipy.integrate solve_ivp Lotka-Volterra RK45 | 1.57 | 8.35 | 0.19 |
| scipy.signal welch 1M, nperseg 1024 | 18.8 | 101 | 0.19 |
| scipy.optimize minimize L-BFGS-B rosen n=50 | 30 | 143 | 0.21 |
| scipy.signal detrend 1M | 9.86 | 46 | 0.21 |
| scipy.signal savgol_filter 1M (31, 3) | 6.19 | 20.1 | 0.31 |
| scipy.sparse.linalg spsolve tridiagonal 100k | 18.2 | 50.2 | 0.36 |
| numpy histogram 1M, 100 bins | 7.7 | 18.8 | 0.41 |
| scipy.sparse.csgraph minimum_spanning_tree 500 nodes | 4.04 | 8.85 | 0.46 |
| numpy percentile 1M | 5.86 | 12.4 | 0.47 |
| numpy trapezoid 1M | 2.1 | 3.99 | 0.53 |
| scipy.optimize root (2 equations) | 0.0392 | 0.0721 | 0.54 |
| scipy.signal convolve2d 500x500 * 5x5 | 8.81 | 16.1 | 0.55 |
| scipy.ndimage distance_transform_edt 512x512 | 11.7 | 20 | 0.59 |
| numpy polyfit deg 5, 100k | 11 | 17.6 | 0.62 |
| scipy.spatial Delaunay 10k (2-D) | 28.8 | 46.3 | 0.62 |
| numpy searchsorted 1M into 100k | 117 | 167 | 0.70 |
| scipy.cluster.vq kmeans2 10k x 3, k=8 | 4.16 | 5.91 | 0.70 |
| scipy.interpolate PchipInterpolator 10k knots, 1M points | 80.7 | 114 | 0.71 |
| scipy.stats skew 1M | 10.5 | 14.8 | 0.71 |
| scipy.stats kurtosis 1M | 10.1 | 13.8 | 0.74 |
| scipy.signal resample_poly 1M, 3/2 | 37.1 | 49.2 | 0.75 |
| scipy.cluster.hierarchy linkage ward 2000x3 | 69.7 | 91.9 | 0.76 |
| scipy.linalg eig 300x300 (general) | 60.2 | 77.8 | 0.77 |
| scipy.stats mannwhitneyu 100k vs 100k | 29.6 | 37.7 | 0.78 |
| scipy.stats spearmanr 100k | 20.6 | 26 | 0.79 |
| scipy.signal decimate 1M by 10 | 19.5 | 24.3 | 0.80 |
| scipy.stats norm.ppf 1M | 48.7 | 59.2 | 0.82 |
| scipy.interpolate RBFInterpolator 500 -> 10k (2-D) | 79.6 | 95.1 | 0.84 |
| scipy.ndimage rotate 512x512, 30 deg | 51.7 | 60.9 | 0.85 |
| scipy.linalg solve_triangular 1000x1000 | 0.88 | 1.04 | 0.85 |
| scipy.signal correlate 10k * 101 | 0.245 | 0.289 | 0.85 |
| scipy.interpolate CubicSpline 10k knots, 1M points | 89.8 | 102 | 0.88 |
| scipy.ndimage label 1000x1000 | 5.99 | 6.8 | 0.88 |
| scipy.signal medfilt 100k, k=5 | 2.78 | 3.05 | 0.91 |
| scipy.ndimage binary_erosion 1000x1000 | 13.3 | 14.4 | 0.93 |
| scipy.linalg schur 300x300 | 57.9 | 61.4 | 0.94 |
| scipy.linalg qr 500x500 | 16 | 16.8 | 0.96 |
| scipy.linalg lstsq 1000x100 | 5.38 | 5.56 | 0.97 |
| numpy convolve 100k * 101 | 2.63 | 2.72 | 0.97 |
| scipy.signal resample 100k -> 50k | 1.62 | 1.67 | 0.97 |
| scipy.signal sosfilt order 8, 1M | 8.56 | 8.8 | 0.97 |
| scipy.sparse.linalg cg 2-D Poisson 100x100 | 28.7 | 29.3 | 0.98 |
| scipy.ndimage map_coordinates 1000x1000, 100k points | 69.2 | 70.6 | 0.98 |
| numpy interp 1M into 1000 | 63.3 | 64.4 | 0.98 |
| numpy median 1M | 11.9 | 11.4 | 1.04 |
| numpy diff 1M | 1.52 | 1.38 | 1.10 |
| scipy.linalg svdvals 300x300 | 12.5 | 11.2 | 1.12 |
| scipy.linalg cholesky 1000x1000 | 16.1 | 14.5 | 1.12 |
| scipy.stats gaussian_kde 1000 points, 1000 evaluations | 21.1 | 17.7 | 1.19 |
| numpy cumsum 1M | 3.15 | 2.64 | 1.19 |
| numpy argsort 1M | 95.3 | 79.6 | 1.20 |
| scipy.linalg pinv 300x300 | 35.5 | 29.4 | 1.21 |
| numpy linalg slogdet 600x600 | 7.27 | 5.73 | 1.27 |
| scipy.signal filtfilt order 4, 1M | 19.8 | 15.2 | 1.30 |
| scipy.stats kendalltau 10k | 2.68 | 2.04 | 1.32 |
| scipy.ndimage maximum_filter 1000x1000, 5 | 32.9 | 24.6 | 1.33 |
| numpy fft2 1024x1024 | 44 | 33.1 | 1.33 |
| numpy clip 1M | 1.85 | 1.38 | 1.34 |
| scipy.spatial ConvexHull 10k (2-D) | 2.01 | 1.46 | 1.38 |
| scipy.signal find_peaks 1M | 6.25 | 4.51 | 1.39 |
| numpy cov 50 x 100k | 47.8 | 33.9 | 1.41 |
| scipy.signal hilbert 1M | 91.1 | 64.8 | 1.41 |
| scipy.stats ks_2samp 100k vs 100k | 111 | 76.6 | 1.45 |
| numpy unique 1M (1000 values) | 16.9 | 11.7 | 1.45 |
| scipy.stats binned_statistic 1M, 100 bins | 83.2 | 57 | 1.46 |
| scipy.ndimage laplace 1000x1000 | 26.1 | 17.9 | 1.46 |
| scipy.ndimage zoom 500x500 x2 | 155 | 106 | 1.47 |
| scipy.sparse.linalg eigsh k=6, 2-D Poisson 100x100 | 743 | 498 | 1.49 |
| numpy einsum ij,jk->ik 300x300 | 17.4 | 11.3 | 1.53 |
| numpy rfft2 1024x1024 | 21.5 | 13.7 | 1.57 |
| scipy.stats rankdata 1M | 235 | 149 | 1.58 |
| scipy.spatial KDTree query 100 points, k=5 | 0.363 | 0.218 | 1.67 |
| scipy.fft dct 1M | 38.4 | 23 | 1.67 |
| scipy.stats linregress 1M | 24.4 | 14.1 | 1.74 |
| scipy.spatial KDTree build 100k (3-D) | 58.1 | 32.6 | 1.78 |
| scipy.ndimage median_filter 512x512, 3 | 66.1 | 35.7 | 1.85 |
| scipy.ndimage sobel 1000x1000 | 23.6 | 12.8 | 1.85 |
| scipy.sparse.csgraph shortest_path 500 nodes (Dijkstra) | 148 | 59.5 | 2.49 |
| scipy.integrate quad sin(x) exp(-x) on [0, 10] | 0.0187 | 0.0057 | 3.27 |
| scipy.optimize curve_fit exp decay, 1000 points | 5.03 | 0.608 | 8.27 |

## Tier 3, by module

| fixture module | functions | geomean ratio | Tessero faster |
|---|---|---|---|
| cluster | 4 | 0.23 | 4 |
| csgraph | 13 | 0.04 | 13 |
| distance | 26 | 0.95 | 15 |
| fft | 18 | 2.73 | 0 |
| hierarchy | 26 | 0.08 | 26 |
| interpolate | 6 | 0.06 | 6 |
| linalg | 30 | 1.50 | 8 |
| ndimage | 66 | 0.82 | 43 |
| np | 193 | 3.20 | 42 |
| npoly | 155 | 0.69 | 107 |
| scipyfft | 11 | 1.44 | 2 |
| signal | 122 | 0.28 | 98 |
| slinalg | 83 | 0.40 | 69 |
| sparse | 23 | 0.32 | 20 |
| special | 89 | 2.06 | 26 |
| splinalg | 14 | 0.21 | 13 |
| stats | 111 | 0.06 | 106 |
| windows | 25 | 0.58 | 17 |
| **all** | 1015 | **0.62** | 615 |

## Tier 1, all functions

| scipy.special function | Tessero ms | SciPy ms | ratio |
|---|---|---|---|
| bdtr | 1.92 | 52.3 | 0.04 |
| bdtrc | 2.13 | 52.5 | 0.04 |
| comb | 1.06 | 4.31 | 0.25 |
| bdtri | 19.2 | 69.4 | 0.28 |
| diric | 2.57 | 5.47 | 0.47 |
| ellipe | 0.548 | 0.774 | 0.71 |
| ellipk | 0.66 | 0.855 | 0.77 |
| wright_bessel | 63 | 78.6 | 0.80 |
| factorial | 1.64 | 1.99 | 0.82 |
| perm | 0.892 | 1.08 | 0.83 |
| log_wright_bessel | 69.6 | 83 | 0.84 |
| boxcox | 2.37 | 2.77 | 0.85 |
| ndtri_exp | 1.58 | 1.84 | 0.86 |
| ncfdtridfd | 333 | 375 | 0.89 |
| nbdtri | 87.1 | 96.3 | 0.90 |
| polygamma | 27.2 | 30 | 0.91 |
| it2j0y0 | 35.5 | 39 | 0.91 |
| hyperu | 2.03e+04 | 2.15e+04 | 0.94 |
| nrdtrimn | 98.6 | 103 | 0.96 |
| nrdtrisd | 103 | 107 | 0.96 |
| kve | 21.3 | 21.8 | 0.98 |
| kv | 22.5 | 23 | 0.98 |
| nctdtrit | 3.28e+04 | 3.35e+04 | 0.98 |
| tklmbda | 93 | 94.5 | 0.98 |
| it2i0k0 | 30.8 | 31.3 | 0.99 |
| voigt_profile | 10.4 | 10.5 | 0.99 |
| expn | 15.9 | 16 | 0.99 |
| yv | 83.1 | 83.3 | 1.00 |
| stdtridf | 87.1 | 87.3 | 1.00 |
| it2struve0 | 39.6 | 39.7 | 1.00 |
| kvp | 55.2 | 55.3 | 1.00 |
| struve | 340 | 340 | 1.00 |
| nctdtridf | 404 | 404 | 1.00 |
| yve | 74.4 | 74.1 | 1.00 |
| nbdtrc | 2.71 | 2.69 | 1.00 |
| zeta | 31.5 | 31.3 | 1.01 |
| nbdtr | 3.03 | 3.01 | 1.01 |
| gammaincc | 13.5 | 13.4 | 1.01 |
| k1e | 3.71 | 3.67 | 1.01 |
| lpmv | 254 | 251 | 1.01 |
| pdtri | 51.6 | 51.1 | 1.01 |
| ncfdtri | 2.39e+03 | 2.36e+03 | 1.01 |
| ncfdtridfn | 439 | 434 | 1.01 |
| pdtrik | 337 | 333 | 1.01 |
| nbdtrik | 167 | 165 | 1.01 |
| bdtrik | 90.7 | 89.6 | 1.01 |
| bei | 76.4 | 75.3 | 1.01 |
| modstruve | 322 | 317 | 1.02 |
| gammainc | 5.81 | 5.71 | 1.02 |
| eval_jacobi | 201 | 198 | 1.02 |
| chndtrinc | 1.48e+03 | 1.45e+03 | 1.02 |
| mathieu_a | 75.9 | 74.5 | 1.02 |
| betaincc | 10.4 | 10.2 | 1.02 |
| chndtrix | 494 | 485 | 1.02 |
| i1e | 5.66 | 5.54 | 1.02 |
| gammainccinv | 65.7 | 64.4 | 1.02 |
| chndtridf | 713 | 697 | 1.02 |
| betainc | 3.19 | 3.11 | 1.02 |
| ncfdtrinc | 601 | 587 | 1.02 |
| airye | 250 | 244 | 1.02 |
| fdtridfd | 155 | 151 | 1.02 |
| mathieu_b | 62.7 | 61.1 | 1.02 |
| pbwa | 91.7 | 89.5 | 1.03 |
| btdtrib | 28.2 | 27.5 | 1.03 |
| berp | 77.6 | 75.6 | 1.03 |
| stdtrit | 76.2 | 74.3 | 1.03 |
| k1 | 3.75 | 3.66 | 1.03 |
| btdtria | 195 | 190 | 1.03 |
| eval_sh_jacobi | 72.4 | 70.5 | 1.03 |
| chdtr | 4.97 | 4.84 | 1.03 |
| nbdtrin | 243 | 236 | 1.03 |
| beip | 73.2 | 71.3 | 1.03 |
| y1 | 2.04 | 1.99 | 1.03 |
| eval_genlaguerre | 148 | 144 | 1.03 |
| gdtrib | 237 | 231 | 1.03 |
| besselpoly | 118 | 115 | 1.03 |
| k0e | 3.96 | 3.85 | 1.03 |
| itairy | 23.2 | 22.5 | 1.03 |
| betaincinv | 76.8 | 74.5 | 1.03 |
| bdtrin | 210 | 204 | 1.03 |
| jvp | 80.9 | 78.5 | 1.03 |
| ivp | 76.9 | 74.6 | 1.03 |
| eval_sh_legendre | 48.8 | 47.4 | 1.03 |
| assoc_laguerre | 12.3 | 11.9 | 1.03 |
| eval_sh_chebyu | 66.4 | 64.3 | 1.03 |
| chdtrc | 21 | 20.3 | 1.03 |
| chdtri | 70.7 | 68.4 | 1.03 |
| airy | 168 | 163 | 1.03 |
| elliprj | 18.5 | 17.9 | 1.03 |
| eval_chebyc | 37.1 | 35.9 | 1.03 |
| eval_chebyu | 53.1 | 51.2 | 1.04 |
| eval_laguerre | 32.5 | 31.3 | 1.04 |
| smirnovi | 992 | 957 | 1.04 |
| gdtrc | 11.2 | 10.8 | 1.04 |
| shichi | 14.7 | 14.2 | 1.04 |
| eval_chebys | 66 | 63.6 | 1.04 |
| ber | 76 | 73.2 | 1.04 |
| betainccinv | 77.2 | 74.3 | 1.04 |
| hyp0f1 | 37 | 35.6 | 1.04 |
| i1 | 7.14 | 6.87 | 1.04 |
| fdtri | 120 | 115 | 1.04 |
| eval_chebyt | 37 | 35.6 | 1.04 |
| chdtriv | 158 | 152 | 1.04 |
| owens_t | 10.9 | 10.4 | 1.04 |
| nctdtr | 306 | 293 | 1.04 |
| expi | 34.3 | 32.8 | 1.04 |
| gdtria | 48.6 | 46.5 | 1.04 |
| jv | 62.6 | 60 | 1.04 |
| jve | 57.8 | 55.3 | 1.05 |
| pdtr | 8.64 | 8.25 | 1.05 |
| iti0k0 | 31 | 29.6 | 1.05 |
| ncfdtr | 41.8 | 39.8 | 1.05 |
| wrightomega | 7.4 | 7.05 | 1.05 |
| hyp2f1 | 775 | 738 | 1.05 |
| exp1 | 26.5 | 25.2 | 1.05 |
| yn | 7.07 | 6.72 | 1.05 |
| y0 | 2.49 | 2.36 | 1.05 |
| gdtr | 4.33 | 4.1 | 1.06 |
| ker | 49 | 46.3 | 1.06 |
| pdtrc | 8.84 | 8.36 | 1.06 |
| kolmogi | 21.9 | 20.7 | 1.06 |
| fdtrc | 24.8 | 23.4 | 1.06 |
| keip | 39.1 | 36.9 | 1.06 |
| sinc | 1.64 | 1.54 | 1.06 |
| hyp1f1 | 1.43e+03 | 1.35e+03 | 1.06 |
| stdtr | 28.6 | 27 | 1.06 |
| kei | 41.5 | 39.1 | 1.06 |
| chndtr | 19.5 | 18.4 | 1.06 |
| fdtr | 24.5 | 23 | 1.06 |
| psi | 10.6 | 9.94 | 1.06 |
| gammaincinv | 53.6 | 50.3 | 1.07 |
| jn | 58.5 | 54.7 | 1.07 |
| kn | 27.2 | 25.4 | 1.07 |
| erfinv | 13.6 | 12.7 | 1.07 |
| powm1 | 4.72 | 4.38 | 1.08 |
| logit | 1.19 | 1.1 | 1.08 |
| i0e | 5.88 | 5.45 | 1.08 |
| kerp | 34.4 | 31.8 | 1.08 |
| gdtrix | 52.9 | 48.9 | 1.08 |
| zetac | 12.4 | 11.5 | 1.08 |
| ive | 30.7 | 28.3 | 1.09 |
| eval_gegenbauer | 739 | 681 | 1.09 |
| eval_sh_chebyt | 66.2 | 61 | 1.09 |
| spence | 2.19 | 2.01 | 1.09 |
| k0 | 4.89 | 4.48 | 1.09 |
| elliprf | 8.92 | 8.1 | 1.10 |
| itstruve0 | 25 | 22.4 | 1.12 |
| elliprd | 12.5 | 11.2 | 1.12 |
| elliprc | 8.35 | 7.42 | 1.13 |
| betaln | 9.04 | 7.99 | 1.13 |
| digamma | 6.13 | 5.39 | 1.14 |
| i0 | 7.15 | 6.26 | 1.14 |
| pseudo_huber | 3.19 | 2.78 | 1.15 |
| rgamma | 5.57 | 4.83 | 1.15 |
| elliprg | 35.7 | 30.9 | 1.16 |
| beta | 8.94 | 7.74 | 1.16 |
| ellipeinc | 2.59 | 2.23 | 1.16 |
| j1 | 3.54 | 3.06 | 1.16 |
| j0 | 3.34 | 2.87 | 1.16 |
| exp10 | 2.9 | 2.49 | 1.16 |
| gamma | 4.63 | 3.97 | 1.17 |
| poch | 3.22 | 2.75 | 1.17 |
| binom | 7.23 | 6.16 | 1.17 |
| erfi | 2.12 | 1.8 | 1.17 |
| boxcox1p | 2.86 | 2.43 | 1.18 |
| cotdg | 2.92 | 2.47 | 1.18 |
| exp2 | 2.42 | 2.03 | 1.19 |
| gammaln | 4.58 | 3.84 | 1.19 |
| cosm1 | 1.48 | 1.22 | 1.21 |
| itmodstruve0 | 20.2 | 16.5 | 1.22 |
| yvp | 238 | 194 | 1.23 |
| itj0y0 | 44.3 | 36 | 1.23 |
| ellipkinc | 2.05 | 1.67 | 1.23 |
| iv | 56.8 | 45.8 | 1.24 |
| cbrt | 2.16 | 1.74 | 1.24 |
| tandg | 2.14 | 1.7 | 1.26 |
| log_expit | 1.89 | 1.49 | 1.27 |
| erfc | 1.8 | 1.4 | 1.29 |
| eval_legendre | 61.1 | 47 | 1.30 |
| sindg | 2.24 | 1.72 | 1.30 |
| log_ndtr | 3.47 | 2.65 | 1.31 |
| kolmogorov | 4.74 | 3.6 | 1.32 |
| erf | 1.92 | 1.45 | 1.32 |
| exprel | 1.95 | 1.46 | 1.34 |
| log1p | 1.21 | 0.903 | 1.34 |
| ndtri | 2.38 | 1.77 | 1.35 |
| fresnel | 7.96 | 5.87 | 1.36 |
| sici | 4.34 | 3.2 | 1.36 |
| inv_boxcox | 2.37 | 1.74 | 1.36 |
| erfcx | 1.25 | 0.916 | 1.37 |
| inv_boxcox1p | 2.59 | 1.89 | 1.37 |
| cosdg | 2.39 | 1.74 | 1.38 |
| expit | 1.09 | 0.785 | 1.39 |
| ellipkm1 | 2.14 | 1.53 | 1.39 |
| xlogy | 1.03 | 0.729 | 1.41 |
| ndtr | 2.24 | 1.59 | 1.41 |
| agm | 2.56 | 1.81 | 1.42 |
| dawsn | 1.12 | 0.781 | 1.44 |
| softplus | 2.2 | 1.52 | 1.45 |
| ellipj | 2.75 | 1.89 | 1.46 |
| loggamma | 1.36 | 0.92 | 1.47 |
| erfcinv | 2.59 | 1.7 | 1.52 |
| expm1 | 1.07 | 0.691 | 1.56 |
| rel_entr | 1.01 | 0.631 | 1.60 |
| kl_div | 1.02 | 0.617 | 1.65 |
| entr | 0.798 | 0.469 | 1.70 |
| gammasgn | 0.535 | 0.299 | 1.79 |
| smirnov | 0.842 | 0.436 | 1.93 |
| round | 0.577 | 0.297 | 1.94 |
| xlog1py | 1.57 | 0.802 | 1.95 |
| radian | 0.503 | 0.246 | 2.05 |
| huber | 0.562 | 0.218 | 2.58 |
