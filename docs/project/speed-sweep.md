# Speed sweep, 2026-11-06

Every function family timed against NumPy 2.4.4 / SciPy 1.17.1, on the same inputs, single-threaded on both sides
(`OPENBLAS_NUM_THREADS=1`), on the benchmark machine of [benchmarks](benchmarks.md) (Intel Core i7-4790, Haswell,
AVX2; PHP 8.5, FFI backend). A ratio is Tessero time / reference time: below 1, Tessero is faster.

| tier | what is timed | timed on both sides | geomean ratio | Tessero faster |
|---|---|---|---|---|
| 1 | all `scipy.special` element-wise functions, 100,000 values each (the parity fixture inputs, tiled) | 212 functions | **1.05** | 31 |
| 2 | 83 heavier calls at realistic sizes (linalg, FFT, signal, ndimage, interpolate, integrate, optimize, stats, sparse, csgraph, spatial, cluster) | 80 cases | **2.05** | 26 |
| 3 | per-call latency of every parity fixture call (small inputs) | 1,015 functions, 15,117 cases | **0.62** | 615 |

Reading it: on small inputs (tier 3) the fixed cost of a call dominates, and Tessero's compiled kernels behind one
FFI call beat SciPy's Python-level wrappers (stats 0.06, csgraph 0.04, signal 0.28); NumPy's own core functions are
the exception (np 3.20), as they are C all the way down. On element-wise math at 100,000 values (tier 1) the two are
on a par. On heavy calls (tier 2) Tessero is ahead on optimizers, ODEs and splines, and behind where an algorithm is
not yet the best one available (the gaps below).

## Fixed during the sweep

| function | problem | before | after |
|---|---|---|---|
| `scipy.fft` dct / idct / dst / idst (and the n-D forms) | direct O(N^2) sums | 1M points: about 5 hours (estimated from 8,000 points: 1.1 s) | 50 ms (SciPy 17-30 ms); agrees with SciPy to 1e-16 of the output scale |
| `interpolate.CubicSpline`, `make_interp_spline` (k=3) | the tridiagonal system solved as a dense n x n matrix | 10k knots: 8-17 s and 800 MB | 76 ms, O(n) memory (LAPACK `dgbsv`, as SciPy) |
| `sparse.linalg.lsmr` | `maxiter` read from the `conlim` argument: 100 million iterations per default call | 15 s on a 10 x 6 system | 16 us |
| `signal.butter`, `cheby1`, `cheby2`, `ellip`, `bessel`, `iirfilter` | `output='sos'` / `'zpk'` silently answered with (b, a) | wrong format, no error | refused with a clear error (convert with `tf2sos`) |
| `np.modf`, `npoly` polymulx and the derivatives, `csgraph.laplacian`, `slinalg.invpascal` | the sign of an exact zero differed from NumPy / SciPy | -- | match; the parity comparator now checks the sign of real zeros |

## Known gaps, largest first

| case | Tessero | SciPy | cause | remedy |
|---|---|---|---|---|
| `spatial.KDTree` query, 100 points in 100k | 23 s | 0.2 ms | exhaustive search in PHP (no tree) | a kd-tree in the kernel |
| `spatial.Delaunay`, 10k points | 35 s | 29 ms | Bowyer-Watson in PHP | a kernel implementation |
| `numpy.trapezoid` 1M | 353 ms | 3.2 ms | to be profiled | -- |
| `ndimage.maximum_filter` 5x5, 1000^2 | 822 ms | 19 ms | full window per pixel | separable 1-D passes (van Herk / Gil-Werman) |
| `numpy.einsum` ij,jk->ik 300^2 | 267 ms | 6.7 ms | loops | matrix-product patterns to BLAS |
| `cluster.hierarchy.linkage` ward, 2000 | 1.3 s | 57 ms | O(n^3) | nearest-neighbour chain, O(n^2) |
| `spatial.ConvexHull`, 10k (2-D) | 28 ms | 1.3 ms | monotone chain in PHP | kernel implementation |
| `numpy.median` / `percentile` 1M | 144 / 119 ms | 8 / 9 ms | full sort | selection (introselect) |
| `optimize.curve_fit`, 1000 points | 9.9 ms | 0.6 ms | to be profiled | -- |
| `numpy.unique` 1M | 97 ms | 8.7 ms | to be profiled | -- |
| `numpy.cov` 50 x 100k | 235 ms | 28 ms | no BLAS | one `dsyrk` / `dgemm` |
| `linalg.solve_triangular` 1000 | 4.2 ms | 0.6 ms | to be profiled | LAPACK `dtrtrs` |
| `sparse.linalg` spsolve / cg / eigsh on sparse input | -- | 21-453 ms | the solvers take dense matrices only | sparse (CSR) operands |
| `special.ellip_harm_2`, Kelvin-function zeros (tier 3) | 100-600x | -- | slow per-call algorithms | -- |

## Where SciPy fails and Tessero completes

On the tier-1 inputs SciPy crashed (segmentation fault), aborted or ran for hours where Tessero returned results:
`mathieu_cem`, `mathieu_sem`, `mathieu_modcem1`, `mathieu_modcem2`, `mathieu_modsem1`, `mathieu_modsem2`,
`pro_ang1`, `pro_ang1_cv`, `pro_cv`, `pro_rad1`, `pro_rad1_cv`, `pro_rad2`, `pro_rad2_cv`, `obl_ang1`,
`obl_ang1_cv`, `obl_cv`, `obl_rad1`, `obl_rad1_cv`, `obl_rad2`, `obl_rad2_cv` and `nctdtrinc` (stopped after
6 hours); `pbdv` and `pbvv` raised a memory-allocation error. `spherical_jn/yn/in/kn` are not comparable here: the
fixtures mix derivative=0 and 1 rows, and SciPy takes one derivative flag per call.

## Method

- Tier 1: the `scipy.special` element-wise blocks of the parity fixtures (`tests/fixtures/parity/special.json`),
  each argument column tiled to 100,000 values; median of repeated runs.
- Tier 2: `bench/tier2_bench.php` and `bench/tier2_bench.py`, the same calls on the same inputs (seeded generators:
  Tessero's PCG64 streams match NumPy's); median of repeated runs, fast calls in an inner loop.
- Tier 3: `tools/parity/run-tests.php --time` times every fixture call that passed its check (same arguments, 3-200
  repetitions, median), `tools/parity/time_calls.py` times the same cases in NumPy/SciPy, and
  `tools/parity/time_report.py` pairs them case for case. Stateful calls (seeded generators, frozen distributions)
  and in-place calls are left out; so are 86 functions that could not be resolved in the Python namespaces and 988
  cases that raised in SciPy. Tier 3 ran before the `lsmr` fix.
- `bench/sweep_report.py` compares any two `{case: ms}` result files.

## Tier 2, all cases

| case | Tessero ms | NumPy/SciPy ms | ratio |
|---|---|---|---|
| scipy.optimize least_squares Rosenbrock residuals | 0.145 | 5.73 | 0.03 |
| scipy.interpolate make_interp_spline k=3, 1M points | 77.7 | 1.43e+03 | 0.05 |
| scipy.integrate solve_ivp Lotka-Volterra RK45 | 0.735 | 5.94 | 0.12 |
| scipy.optimize minimize Nelder-Mead rosen n=5 | 0.821 | 5.97 | 0.14 |
| scipy.sparse.csgraph connected_components 500 nodes | 0.655 | 4.15 | 0.16 |
| scipy.signal detrend 1M | 10.1 | 37.4 | 0.27 |
| scipy.optimize minimize L-BFGS-B rosen n=50 | 25.5 | 84.8 | 0.30 |
| scipy.signal welch 1M, nperseg 1024 | 33.6 | 94.9 | 0.35 |
| scipy.optimize root (2 equations) | 0.0184 | 0.0332 | 0.55 |
| scipy.ndimage distance_transform_edt 512x512 | 10.2 | 17 | 0.60 |
| numpy polyfit deg 5, 100k | 8.51 | 13.2 | 0.65 |
| numpy histogram 1M, 100 bins | 8.45 | 12 | 0.70 |
| scipy.spatial KDTree build 100k (3-D) | 20 | 28.1 | 0.71 |
| scipy.sparse.csgraph minimum_spanning_tree 500 nodes | 4.08 | 5.6 | 0.73 |
| scipy.stats skew 1M | 8.99 | 12.1 | 0.74 |
| scipy.stats kurtosis 1M | 8.43 | 11.2 | 0.75 |
| numpy searchsorted 1M into 100k | 105 | 132 | 0.80 |
| scipy.interpolate PchipInterpolator 10k knots, 1M points | 76.7 | 88.1 | 0.87 |
| scipy.interpolate CubicSpline 10k knots, 1M points | 77.4 | 88.3 | 0.88 |
| scipy.stats mannwhitneyu 100k vs 100k | 20.7 | 23.5 | 0.88 |
| scipy.stats spearmanr 100k | 18.5 | 19.6 | 0.94 |
| scipy.cluster.vq kmeans2 10k x 3, k=8 | 3.97 | 4.13 | 0.96 |
| scipy.linalg svdvals 300x300 | 8.87 | 9.04 | 0.98 |
| scipy.signal decimate 1M by 10 | 18.5 | 18.8 | 0.99 |
| scipy.linalg schur 300x300 | 54.7 | 55.1 | 0.99 |
| scipy.linalg qr 500x500 | 16 | 16 | 1.00 |
| scipy.linalg lstsq 1000x100 | 5.55 | 5.52 | 1.01 |
| scipy.interpolate RBFInterpolator 500 -> 10k (2-D) | 70.9 | 69.9 | 1.01 |
| scipy.linalg eig 300x300 (general) | 60.3 | 59.5 | 1.01 |
| numpy interp 1M into 1000 | 58.2 | 54.2 | 1.07 |
| scipy.stats norm.ppf 1M | 44 | 40.5 | 1.09 |
| numpy diff 1M | 1.24 | 1.13 | 1.10 |
| scipy.stats rankdata 1M | 148 | 133 | 1.11 |
| numpy cumsum 1M | 2.59 | 2.3 | 1.13 |
| scipy.ndimage map_coordinates 1000x1000, 100k points | 53.9 | 43.8 | 1.23 |
| numpy linalg slogdet 600x600 | 6.62 | 5.38 | 1.23 |
| scipy.ndimage binary_erosion 1000x1000 | 11.6 | 9.41 | 1.23 |
| scipy.linalg cholesky 1000x1000 | 16.9 | 13.4 | 1.25 |
| scipy.ndimage rotate 512x512, 30 deg | 47.5 | 36.8 | 1.29 |
| scipy.stats gaussian_kde 1000 points, 1000 evaluations | 13.8 | 10.7 | 1.30 |
| scipy.stats kendalltau 10k | 2.21 | 1.69 | 1.30 |
| scipy.stats ks_2samp 100k vs 100k | 86.4 | 66 | 1.31 |
| scipy.stats binned_statistic 1M, 100 bins | 66.3 | 49.7 | 1.33 |
| scipy.signal find_peaks 1M | 5.82 | 4.04 | 1.44 |
| numpy argsort 1M | 83.5 | 54.1 | 1.54 |
| scipy.signal filtfilt order 4, 1M | 16.5 | 10.4 | 1.58 |
| scipy.signal resample 100k -> 50k | 2.48 | 1.43 | 1.73 |
| scipy.ndimage zoom 500x500 x2 | 130 | 73.4 | 1.77 |
| scipy.integrate quad sin(x) exp(-x) on [0, 10] | 0.00914 | 0.00511 | 1.79 |
| scipy.ndimage median_filter 512x512, 3 | 60.7 | 31.8 | 1.91 |
| scipy.stats linregress 1M | 21.3 | 11 | 1.93 |
| numpy fft2 1024x1024 | 54.5 | 28 | 1.95 |
| scipy.signal savgol_filter 1M (31, 3) | 27 | 13.2 | 2.04 |
| scipy.sparse.csgraph shortest_path 500 nodes (Dijkstra) | 112 | 54.9 | 2.05 |
| scipy.signal hilbert 1M | 102 | 47.5 | 2.14 |
| numpy convolve 100k * 101 | 6.1 | 2.72 | 2.25 |
| numpy clip 1M | 2.54 | 1.04 | 2.43 |
| scipy.ndimage laplace 1000x1000 | 29.1 | 11.6 | 2.52 |
| scipy.signal convolve2d 500x500 * 5x5 | 34.9 | 13.7 | 2.55 |
| scipy.ndimage label 1000x1000 | 18.5 | 6.13 | 3.01 |
| scipy.fft dct 1M | 49.9 | 16.5 | 3.02 |
| scipy.ndimage sobel 1000x1000 | 26.9 | 8.03 | 3.35 |
| numpy rfft2 1024x1024 | 29.5 | 8.64 | 3.42 |
| scipy.signal medfilt 100k, k=5 | 8.24 | 2.12 | 3.89 |
| scipy.linalg pinv 300x300 | 120 | 23.9 | 5.02 |
| scipy.linalg solve_triangular 1000x1000 | 4.16 | 0.612 | 6.80 |
| scipy.signal resample_poly 1M, 3/2 | 260 | 34.8 | 7.47 |
| numpy cov 50 x 100k | 235 | 28.3 | 8.30 |
| numpy unique 1M (1000 values) | 97.2 | 8.7 | 11.17 |
| scipy.signal correlate 10k * 101 | 3.4 | 0.282 | 12.06 |
| numpy percentile 1M | 119 | 9.1 | 13.03 |
| numpy median 1M | 144 | 8.45 | 17.01 |
| scipy.optimize curve_fit exp decay, 1000 points | 9.89 | 0.565 | 17.48 |
| scipy.spatial ConvexHull 10k (2-D) | 27.6 | 1.28 | 21.65 |
| scipy.cluster.hierarchy linkage ward 2000x3 | 1.31e+03 | 56.8 | 23.07 |
| numpy einsum ij,jk->ik 300x300 | 267 | 6.66 | 40.11 |
| scipy.ndimage maximum_filter 1000x1000, 5 | 822 | 19.1 | 43.13 |
| numpy trapezoid 1M | 353 | 3.21 | 110.17 |
| scipy.spatial Delaunay 10k (2-D) | 3.51e+04 | 29.3 | 1199.83 |
| scipy.spatial KDTree query 100 points, k=5 (Tessero: exhaustive search) | 2.3e+04 | 0.201 | 114681.98 |

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
