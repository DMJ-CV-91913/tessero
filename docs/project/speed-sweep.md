# Speed sweep, 2026-11-10

First run on 2026-11-06; tier 3 re-timed on 2026-11-10, tier 1 on 2026-11-09, tier 2 last on 2026-11-07 (see Method).

Every function family timed against NumPy 2.4.4 / SciPy 1.17.1, on the same inputs, single-threaded on both sides
(`OPENBLAS_NUM_THREADS=1`), on the benchmark machine of [benchmarks](benchmarks.md) (Intel Core i7-4790, Haswell,
AVX2; PHP 8.5, FFI backend). A ratio is Tessero time / reference time: below 1, Tessero is faster.

| tier | what is timed | timed on both sides | geomean ratio | Tessero faster |
|---|---|---|---|---|
| 1 | all `scipy.special` element-wise functions, 100,000 values each (the parity fixture inputs, tiled) | 212 functions | **1.05** | 60 |
| 2 | 84 heavier calls at realistic sizes (linalg, FFT, signal, ndimage, interpolate, integrate, optimize, stats, sparse, csgraph, spatial, cluster), re-timed after the fixes below | 84 cases | **0.67** | 68 |
| 3 | per-call latency of every parity fixture call (small inputs) | 1,022 functions, 15,184 cases | **0.42** | 700 |

Reading it: on small inputs (tier 3) the fixed cost of a call dominates, and Tessero's compiled kernels behind one
FFI call beat SciPy's Python-level wrappers (stats 0.05, csgraph 0.04, signal 0.18); NumPy's own core functions are
the exception (np 2.05), as they are C all the way down. On element-wise math at 100,000 values (tier 1) the two are
on a par. On heavy calls (tier 2) Tessero is ahead on optimizers, ODEs and splines: clearly faster (by more than 10%)
on 49 cases, within 10% either way on 26, and clearly slower on 9, mostly where an algorithm is not yet the best one
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

The sorts behind `median`, `quantile`, `unique` and the set routines were libc `qsort` with a comparison function;
they now use `sortSimd` (and `tsr_argsort` for stable orders). `einsum` also stopped skipping a product once a
factor was zero, so a NaN in a later operand propagates as in NumPy.

## Known gaps, largest first

The gaps profiled so far; every case slower than SciPy is in the tier 2 table below (`quad` is the
largest not yet profiled).

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

Tier 3 again with the native extension (`ext/`) instead of FFI, re-timed on 2026-11-10 at commit 655992c6: every parity fixture call
that passed its check, the same cases, timed through `Tessero\Ext\*`. The extension calls the same kernel without
the FFI layer, which costs several microseconds per call; where the work itself dominates, the two backends take the
same time (`stats.tukey_hsd`: 54 ms on both). Against NumPy/SciPy the extension is about 8x faster per call on
average and faster on 882 of 1,022 functions, NumPy's own core functions included (0.51; through FFI they are about
2x slower).

Best of four runs on each side, alternating, at a load average of 6 to 12 (on 2026-11-06, at commit 2184ba75 and
at times above 40, three runs had given 0.11 and 893):

| module | functions | geomean ratio | faster |
|---|---|---|---|
| cluster | 4 | 0.05 | 4 |
| csgraph | 14 | 0.01 | 14 |
| distance | 26 | 0.21 | 26 |
| fft | 18 | 0.31 | 17 |
| hierarchy | 26 | 0.01 | 26 |
| interpolate | 6 | 0.01 | 6 |
| linalg | 30 | 0.37 | 26 |
| ndimage | 66 | 0.14 | 66 |
| np | 193 | 0.51 | 120 |
| npoly | 155 | 0.12 | 148 |
| scipyfft | 11 | 0.56 | 6 |
| signal | 124 | 0.05 | 119 |
| slinalg | 87 | 0.08 | 87 |
| sparse | 23 | 0.04 | 23 |
| special | 89 | 0.92 | 46 |
| splinalg | 14 | 0.02 | 14 |
| stats | 111 | 0.01 | 109 |
| windows | 25 | 0.12 | 25 |
| **all** | 1022 | **0.12** | 882 |

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
  `filtfilt` (five alternating rounds each, at a load average of 12 to 17), and `shortest_path` and `median_filter`
  (four alternating rounds each, at a load average of 7 to 15), and `rankdata`, `argsort`, `spearmanr`, `kendalltau`
  and `linregress` (ten alternating rounds, at a load average of 7 to 17), `unique` (six alternating rounds, at
  a load average of 6 to 10), `ks_2samp` (twelve alternating rounds, at a load average of up to 38),
  `binned_statistic` (six alternating rounds, at a load average of 5 to 10), `find_peaks` (six alternating
  rounds, at a load average of 18), `zoom` (six alternating rounds, at a load average of 6 to 9), `cov` and
  `cumsum` (six alternating rounds, at a load average of 9 to 10) `gaussian_kde` (six alternating rounds, at a
  load average of 8 to 9) and `clip`, `sobel`, `laplace` and `maximum_filter` (six alternating rounds, at a load
  average of 6 to 21, `clip` twelve more) `filtfilt` and `hilbert` (six alternating rounds, at a load average of
  10 to 14) `dct` (six alternating rounds, at a load average of 10 to 17) and, after
  the large-block reuse, `fft2`, `rfft2`, `dct`, `linkage`, `kmeans2`, `sosfilt`, `connected_components`,
  `ConvexHull` and `Delaunay` (six alternating rounds, at a load average of 7 to 9), `einsum` (six alternating
  rounds, at a load average of 9 to 10) and, again after the large-block reuse, `cov`, `clip`, `hilbert`, `pinv`,
  `svdvals` and `cholesky` (six alternating rounds, at a load average of 9 to 11). Each side ran three times, alternating,
  and each case keeps its best time, as the host carries unrelated load (load average 16 to 22 on 8 cores during the
  last run). Ratios within 10% are read as on par.
- Tier 3: `tools/parity/run-tests.php --time` times every fixture call that passed its check (same arguments, 3-200
  repetitions, median), `tools/parity/time_calls.py` times the same cases in NumPy/SciPy, and
  `tools/parity/time_report.py` pairs them case for case. Stateful calls (seeded generators, frozen distributions)
  and in-place calls are left out; so are 87 functions that could not be resolved in the Python namespaces and 988
  cases that raised in SciPy.
- Tiers 1 and 3 re-timed on 2026-11-09, after the lower per-call cost of the FFI path (commit c6e18549): two
  alternating rounds of each side (`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`, at a load average of 9 to 18), each
  case keeping its best time on each side.
- Tier 3 re-timed on 2026-11-10, after the small-array changes (commits c5ae8bd3 and 655992c6): four alternating
  rounds of each side (at a load average of 8 to 15), each case keeping its best time on each side. The geomean rose
  from 0.37 to 0.42 although Tessero's own times fell (to 0.90 of the 2026-11-09 times on `np`, 0.79 on `special`,
  0.68 on `signal`): the NumPy/SciPy side ran faster on this host than in the two 2026-11-09 rounds (to 0.54 of
  those times on `npoly`, 0.57 on `scipyfft`, 0.62 on `slinalg`), which had flattered the earlier figure. Four rounds
  per side give the steadier reference.
- Results unchanged: since 2026-11-09 every speed change is also checked byte for byte with
  `tools/parity/run-tests.php --hash`, which records a hash of the raw bytes of every fixture call's result (arrays by
  dtype, shape and bytes, so NaN sign and payload bits count; floats by their bits; exceptions by class and message),
  47,869 results, on the code before and after the change: every hash is the same.
- `bench/sweep_report.py` compares any two `{case: ms}` result files.

## Tier 2, all cases

| case | Tessero ms | NumPy/SciPy ms | ratio |
|---|---|---|---|
| scipy.optimize least_squares Rosenbrock residuals | 0.307 | 8.16 | 0.04 |
| scipy.interpolate make_interp_spline k=3, 1M points | 88 | 2.06e+03 | 0.04 |
| scipy.sparse.csgraph connected_components 500 nodes | 0.309 | 4.57 | 0.07 |
| scipy.optimize minimize Nelder-Mead rosen n=5 | 0.983 | 12.3 | 0.08 |
| scipy.integrate solve_ivp Lotka-Volterra RK45 | 1.57 | 8.35 | 0.19 |
| scipy.signal welch 1M, nperseg 1024 | 18.8 | 101 | 0.19 |
| scipy.optimize minimize L-BFGS-B rosen n=50 | 30 | 143 | 0.21 |
| scipy.signal detrend 1M | 9.86 | 46 | 0.21 |
| scipy.ndimage zoom 500x500 x2 | 18.1 | 74.3 | 0.24 |
| scipy.ndimage maximum_filter 1000x1000, 5 | 6.1 | 20.4 | 0.30 |
| scipy.signal savgol_filter 1M (31, 3) | 6.19 | 20.1 | 0.31 |
| scipy.sparse.linalg spsolve tridiagonal 100k | 18.2 | 50.2 | 0.36 |
| scipy.ndimage sobel 1000x1000 | 3.18 | 8.83 | 0.36 |
| numpy unique 1M (1000 values) | 3.86 | 9.66 | 0.40 |
| numpy histogram 1M, 100 bins | 7.7 | 18.8 | 0.41 |
| scipy.stats spearmanr 100k | 8.58 | 20.8 | 0.41 |
| scipy.sparse.csgraph minimum_spanning_tree 500 nodes | 4.04 | 8.85 | 0.46 |
| numpy percentile 1M | 5.86 | 12.4 | 0.47 |
| numpy cumsum 1M | 1.19 | 2.4 | 0.50 |
| scipy.ndimage laplace 1000x1000 | 6.63 | 12.7 | 0.52 |
| numpy trapezoid 1M | 2.1 | 3.99 | 0.53 |
| scipy.optimize root (2 equations) | 0.0392 | 0.0721 | 0.54 |
| scipy.signal convolve2d 500x500 * 5x5 | 8.81 | 16.1 | 0.55 |
| scipy.ndimage distance_transform_edt 512x512 | 11.7 | 20 | 0.59 |
| numpy polyfit deg 5, 100k | 11 | 17.6 | 0.62 |
| scipy.ndimage median_filter 512x512, 3 | 21.7 | 34.5 | 0.63 |
| scipy.stats rankdata 1M | 94.2 | 148 | 0.64 |
| scipy.stats binned_statistic 1M, 100 bins | 37.5 | 56.8 | 0.66 |
| numpy searchsorted 1M into 100k | 117 | 167 | 0.70 |
| scipy.spatial Delaunay 10k (2-D) | 27 | 38.6 | 0.70 |
| scipy.interpolate PchipInterpolator 10k knots, 1M points | 80.7 | 114 | 0.71 |
| scipy.stats skew 1M | 10.5 | 14.8 | 0.71 |
| scipy.stats kurtosis 1M | 10.1 | 13.8 | 0.74 |
| scipy.signal resample_poly 1M, 3/2 | 37.1 | 49.2 | 0.75 |
| scipy.cluster.hierarchy linkage ward 2000x3 | 61.5 | 80.9 | 0.76 |
| scipy.linalg eig 300x300 (general) | 60.2 | 77.8 | 0.77 |
| scipy.stats mannwhitneyu 100k vs 100k | 29.6 | 37.7 | 0.78 |
| numpy argsort 1M | 44.4 | 57.2 | 0.78 |
| numpy einsum ij,jk->ik 300x300 | 6.17 | 7.93 | 0.78 |
| scipy.signal decimate 1M by 10 | 19.5 | 24.3 | 0.80 |
| scipy.stats norm.ppf 1M | 48.7 | 59.2 | 0.82 |
| scipy.interpolate RBFInterpolator 500 -> 10k (2-D) | 79.6 | 95.1 | 0.84 |
| scipy.ndimage rotate 512x512, 30 deg | 51.7 | 60.9 | 0.85 |
| scipy.linalg solve_triangular 1000x1000 | 0.88 | 1.04 | 0.85 |
| scipy.signal correlate 10k * 101 | 0.245 | 0.289 | 0.85 |
| scipy.sparse.csgraph shortest_path 500 nodes (Dijkstra) | 64.1 | 73.5 | 0.87 |
| scipy.interpolate CubicSpline 10k knots, 1M points | 89.8 | 102 | 0.88 |
| scipy.ndimage label 1000x1000 | 5.99 | 6.8 | 0.88 |
| scipy.stats kendalltau 10k | 1.52 | 1.72 | 0.88 |
| scipy.signal medfilt 100k, k=5 | 2.78 | 3.05 | 0.91 |
| numpy fft2 1024x1024 | 29.7 | 32.2 | 0.92 |
| scipy.ndimage binary_erosion 1000x1000 | 13.3 | 14.4 | 0.93 |
| scipy.stats linregress 1M | 10.8 | 11.6 | 0.93 |
| scipy.stats gaussian_kde 1000 points, 1000 evaluations | 11.4 | 12.3 | 0.93 |
| scipy.linalg schur 300x300 | 57.9 | 61.4 | 0.94 |
| scipy.stats ks_2samp 100k vs 100k | 67.3 | 71.9 | 0.94 |
| scipy.fft dct 1M | 23.5 | 24.9 | 0.94 |
| scipy.linalg pinv 300x300 | 27.2 | 28.9 | 0.94 |
| scipy.signal find_peaks 1M | 4.19 | 4.4 | 0.95 |
| scipy.linalg qr 500x500 | 16 | 16.8 | 0.96 |
| scipy.linalg svdvals 300x300 | 9.43 | 9.78 | 0.96 |
| scipy.linalg lstsq 1000x100 | 5.38 | 5.56 | 0.97 |
| numpy convolve 100k * 101 | 2.63 | 2.72 | 0.97 |
| scipy.signal resample 100k -> 50k | 1.62 | 1.67 | 0.97 |
| scipy.sparse.linalg cg 2-D Poisson 100x100 | 28.7 | 29.3 | 0.98 |
| scipy.ndimage map_coordinates 1000x1000, 100k points | 69.2 | 70.6 | 0.98 |
| numpy interp 1M into 1000 | 63.3 | 64.4 | 0.98 |
| scipy.cluster.vq kmeans2 10k x 3, k=8 | 4.35 | 4.45 | 0.98 |
| scipy.signal sosfilt order 8, 1M | 8.88 | 8.82 | 1.01 |
| numpy clip 1M | 1.46 | 1.44 | 1.01 |
| scipy.signal filtfilt order 4, 1M | 15.8 | 15.6 | 1.02 |
| scipy.signal hilbert 1M | 81.5 | 79.3 | 1.03 |
| numpy median 1M | 11.9 | 11.4 | 1.04 |
| numpy cov 50 x 100k | 34.3 | 32.2 | 1.06 |
| numpy diff 1M | 1.52 | 1.38 | 1.10 |
| scipy.spatial ConvexHull 10k (2-D) | 1.69 | 1.44 | 1.17 |
| scipy.linalg cholesky 1000x1000 | 20.6 | 17.1 | 1.20 |
| numpy linalg slogdet 600x600 | 7.27 | 5.73 | 1.27 |
| numpy rfft2 1024x1024 | 15.7 | 12.2 | 1.29 |
| scipy.sparse.linalg eigsh k=6, 2-D Poisson 100x100 | 743 | 498 | 1.49 |
| scipy.spatial KDTree query 100 points, k=5 | 0.363 | 0.218 | 1.67 |
| scipy.spatial KDTree build 100k (3-D) | 58.1 | 32.6 | 1.78 |
| scipy.integrate quad sin(x) exp(-x) on [0, 10] | 0.0187 | 0.0057 | 3.27 |
| scipy.optimize curve_fit exp decay, 1000 points | 5.03 | 0.608 | 8.27 |

## Tier 3, by module

| module | functions | geomean ratio | faster |
|---|---|---|---|
| cluster | 4 | 0.16 | 4 |
| csgraph | 14 | 0.04 | 14 |
| distance | 26 | 0.51 | 21 |
| fft | 18 | 2.05 | 0 |
| hierarchy | 26 | 0.05 | 26 |
| interpolate | 6 | 0.04 | 6 |
| linalg | 30 | 1.02 | 13 |
| ndimage | 66 | 0.48 | 51 |
| np | 193 | 2.05 | 54 |
| npoly | 155 | 0.43 | 123 |
| scipyfft | 11 | 0.98 | 3 |
| signal | 124 | 0.18 | 106 |
| slinalg | 87 | 0.27 | 86 |
| sparse | 23 | 0.25 | 21 |
| special | 89 | 2.02 | 30 |
| splinalg | 14 | 0.05 | 14 |
| stats | 111 | 0.05 | 107 |
| windows | 25 | 0.42 | 21 |
| **all** | 1022 | **0.42** | 700 |

## Tier 1, all functions

| scipy.special function | Tessero ms | SciPy ms | ratio |
|---|---|---|---|
| bdtr | 3.22 | 64.4 | 0.05 |
| bdtrc | 3.55 | 61 | 0.06 |
| smirnov | 0.848 | 4.14 | 0.20 |
| comb | 1.2 | 5.3 | 0.23 |
| bdtri | 45.3 | 120 | 0.38 |
| nbdtr | 8.98 | 17.8 | 0.50 |
| nbdtrc | 8.06 | 15.7 | 0.51 |
| diric | 3.76 | 6.2 | 0.61 |
| ellipk | 0.596 | 0.981 | 0.61 |
| ndtri_exp | 1.57 | 2.33 | 0.67 |
| ellipe | 0.664 | 0.96 | 0.69 |
| boxcox | 2.41 | 3.28 | 0.74 |
| expn | 16.8 | 21.2 | 0.79 |
| yn | 10.1 | 12.7 | 0.79 |
| betainc | 3.27 | 3.98 | 0.82 |
| perm | 0.932 | 1.12 | 0.84 |
| chdtriv | 185 | 216 | 0.86 |
| ncfdtridfn | 470 | 544 | 0.87 |
| nrdtrimn | 119 | 135 | 0.88 |
| nctdtridf | 427 | 484 | 0.88 |
| airy | 193 | 217 | 0.89 |
| it2j0y0 | 36.3 | 40.7 | 0.89 |
| chndtr | 20.1 | 22.5 | 0.89 |
| ncfdtri | 2.69e+03 | 3e+03 | 0.90 |
| pdtri | 55.1 | 60.5 | 0.91 |
| chdtri | 79.1 | 86.8 | 0.91 |
| ndtr | 2.34 | 2.56 | 0.92 |
| k0e | 3.95 | 4.28 | 0.92 |
| hyperu | 2.54e+04 | 2.76e+04 | 0.92 |
| kvp | 57.6 | 62.3 | 0.92 |
| k1 | 3.62 | 3.92 | 0.92 |
| betaincinv | 83.3 | 89.5 | 0.93 |
| airye | 284 | 303 | 0.94 |
| k0 | 4.74 | 5.06 | 0.94 |
| gdtrix | 58.4 | 62.3 | 0.94 |
| bdtrin | 224 | 239 | 0.94 |
| gdtrib | 269 | 285 | 0.94 |
| factorial | 1.94 | 2.04 | 0.95 |
| it2i0k0 | 31.1 | 32.4 | 0.96 |
| eval_legendre | 49.2 | 51.3 | 0.96 |
| kn | 30 | 31.1 | 0.96 |
| chdtr | 5.28 | 5.47 | 0.97 |
| ivp | 78.9 | 81.7 | 0.97 |
| betaincc | 11.2 | 11.5 | 0.97 |
| eval_sh_jacobi | 95.8 | 98.8 | 0.97 |
| eval_sh_chebyu | 115 | 119 | 0.97 |
| gammainc | 6.08 | 6.26 | 0.97 |
| chdtrc | 22.4 | 23 | 0.97 |
| eval_laguerre | 50.2 | 51.3 | 0.98 |
| i0e | 5.86 | 5.97 | 0.98 |
| btdtrib | 30.6 | 31.2 | 0.98 |
| smirnovi | 1.04e+03 | 1.06e+03 | 0.99 |
| nbdtrin | 268 | 272 | 0.99 |
| i1e | 5.55 | 5.62 | 0.99 |
| hyp0f1 | 39.5 | 40 | 0.99 |
| eval_sh_chebyt | 41.5 | 41.9 | 0.99 |
| exp1 | 27.5 | 27.7 | 0.99 |
| it2struve0 | 40.5 | 40.8 | 0.99 |
| stdtrit | 83.3 | 83.7 | 0.99 |
| powm1 | 5.21 | 5.23 | 1.00 |
| itairy | 23.1 | 23 | 1.00 |
| ber | 88.1 | 87.8 | 1.00 |
| eval_jacobi | 205 | 204 | 1.01 |
| nbdtri | 189 | 188 | 1.01 |
| modstruve | 351 | 348 | 1.01 |
| bei | 85.5 | 84.7 | 1.01 |
| eval_sh_legendre | 35.5 | 35.1 | 1.01 |
| i1 | 6.95 | 6.86 | 1.01 |
| stdtridf | 100 | 98.8 | 1.02 |
| nctdtrit | 3.67e+04 | 3.61e+04 | 1.02 |
| log_wright_bessel | 89.5 | 88.1 | 1.02 |
| yvp | 217 | 213 | 1.02 |
| besselpoly | 127 | 125 | 1.02 |
| eval_genlaguerre | 225 | 220 | 1.02 |
| mathieu_b | 65.5 | 64.2 | 1.02 |
| i0 | 6.92 | 6.77 | 1.02 |
| chndtridf | 925 | 905 | 1.02 |
| ncfdtr | 50.5 | 49.4 | 1.02 |
| stdtr | 31.3 | 30.6 | 1.02 |
| logit | 1.18 | 1.15 | 1.03 |
| pdtrc | 9.58 | 9.3 | 1.03 |
| mathieu_a | 81.1 | 78.6 | 1.03 |
| nbdtrik | 179 | 173 | 1.03 |
| expi | 37 | 35.8 | 1.03 |
| bdtrik | 102 | 98.3 | 1.04 |
| fdtridfd | 168 | 162 | 1.04 |
| sinc | 1.7 | 1.64 | 1.04 |
| beip | 86.3 | 83.1 | 1.04 |
| pdtr | 9.3 | 8.96 | 1.04 |
| fdtrc | 29.1 | 27.9 | 1.04 |
| agm | 3.02 | 2.91 | 1.04 |
| betainccinv | 84.8 | 81.4 | 1.04 |
| iti0k0 | 32.4 | 31.1 | 1.04 |
| gammaincc | 14.4 | 13.9 | 1.04 |
| lpmv | 278 | 266 | 1.04 |
| ncfdtrinc | 655 | 624 | 1.05 |
| chndtrix | 627 | 595 | 1.05 |
| fdtri | 137 | 130 | 1.05 |
| struve | 387 | 366 | 1.06 |
| jve | 62.6 | 59.2 | 1.06 |
| berp | 88.1 | 83.1 | 1.06 |
| cbrt | 2.21 | 2.08 | 1.06 |
| spence | 2.42 | 2.28 | 1.06 |
| gammainccinv | 72.1 | 67.8 | 1.06 |
| btdtria | 237 | 222 | 1.07 |
| pdtrik | 410 | 383 | 1.07 |
| gdtr | 4.61 | 4.29 | 1.07 |
| eval_chebyc | 43.4 | 40.4 | 1.08 |
| inv_boxcox1p | 2.35 | 2.18 | 1.08 |
| k1e | 4.15 | 3.85 | 1.08 |
| eval_chebyt | 42.2 | 39.1 | 1.08 |
| owens_t | 12.3 | 11.4 | 1.08 |
| iv | 52.7 | 48.7 | 1.08 |
| pbwa | 101 | 92.7 | 1.08 |
| wright_bessel | 99.8 | 91.9 | 1.09 |
| fdtr | 28.7 | 26.4 | 1.09 |
| assoc_laguerre | 14.2 | 13 | 1.09 |
| jv | 69.2 | 63.6 | 1.09 |
| hyp1f1 | 1.6e+03 | 1.47e+03 | 1.09 |
| gamma | 4.79 | 4.37 | 1.09 |
| nctdtr | 347 | 316 | 1.10 |
| nrdtrisd | 147 | 134 | 1.10 |
| eval_chebyu | 48.9 | 44.3 | 1.11 |
| kolmogi | 24.3 | 21.8 | 1.11 |
| jn | 65.8 | 59.2 | 1.11 |
| exp2 | 2.48 | 2.23 | 1.11 |
| kei | 46.5 | 41.7 | 1.11 |
| itstruve0 | 26.9 | 24.1 | 1.11 |
| gammaincinv | 59.5 | 53 | 1.12 |
| gdtria | 53.7 | 47.7 | 1.13 |
| boxcox1p | 3 | 2.66 | 1.13 |
| inv_boxcox | 2.15 | 1.91 | 1.13 |
| exp10 | 3.08 | 2.73 | 1.13 |
| binom | 7.8 | 6.89 | 1.13 |
| kve | 24 | 21.2 | 1.14 |
| keip | 44.8 | 39.3 | 1.14 |
| chndtrinc | 1.9e+03 | 1.67e+03 | 1.14 |
| yv | 119 | 105 | 1.14 |
| ndtri | 2.54 | 2.22 | 1.14 |
| exprel | 1.88 | 1.64 | 1.14 |
| gdtrc | 13.2 | 11.5 | 1.14 |
| j0 | 3.53 | 3.06 | 1.15 |
| sindg | 2.31 | 2 | 1.16 |
| elliprj | 23.3 | 20.1 | 1.16 |
| zeta | 39.7 | 34.1 | 1.16 |
| beta | 10.2 | 8.71 | 1.17 |
| poch | 3.7 | 3.15 | 1.18 |
| eval_chebys | 69.4 | 59 | 1.18 |
| kerp | 40.5 | 34.4 | 1.18 |
| kv | 26.1 | 22.1 | 1.18 |
| ncfdtridfd | 426 | 360 | 1.18 |
| expit | 1.06 | 0.888 | 1.20 |
| erfi | 2.52 | 2.11 | 1.20 |
| elliprc | 9.92 | 8.25 | 1.20 |
| gammaln | 4.94 | 4.09 | 1.21 |
| polygamma | 39.9 | 32.9 | 1.21 |
| psi | 16.9 | 13.9 | 1.21 |
| yve | 102 | 83.7 | 1.21 |
| jvp | 106 | 87 | 1.22 |
| shichi | 19.5 | 16 | 1.22 |
| betaln | 10.8 | 8.89 | 1.22 |
| zetac | 18 | 14.5 | 1.24 |
| gammasgn | 0.401 | 0.322 | 1.24 |
| itmodstruve0 | 21.8 | 17.5 | 1.24 |
| elliprd | 15.6 | 12.5 | 1.25 |
| hyp2f1 | 1.1e+03 | 877 | 1.26 |
| ellipkinc | 2.41 | 1.92 | 1.26 |
| itj0y0 | 47.3 | 37.6 | 1.26 |
| rgamma | 7.58 | 5.93 | 1.28 |
| tandg | 2.58 | 2.01 | 1.28 |
| ker | 63.8 | 49.8 | 1.28 |
| tklmbda | 134 | 104 | 1.28 |
| expm1 | 0.989 | 0.768 | 1.29 |
| log_ndtr | 3.56 | 2.74 | 1.30 |
| erf | 2.09 | 1.6 | 1.30 |
| erfinv | 18.5 | 14.1 | 1.31 |
| fresnel | 8.59 | 6.39 | 1.35 |
| digamma | 8.47 | 6.24 | 1.36 |
| voigt_profile | 16.1 | 11.9 | 1.36 |
| cotdg | 4.32 | 3.18 | 1.36 |
| eval_gegenbauer | 323 | 234 | 1.38 |
| sici | 5.57 | 4.02 | 1.38 |
| j1 | 4.43 | 3.19 | 1.39 |
| ellipeinc | 3.57 | 2.54 | 1.40 |
| cosm1 | 2.18 | 1.55 | 1.41 |
| entr | 0.721 | 0.511 | 1.41 |
| log1p | 1.32 | 0.926 | 1.42 |
| pseudo_huber | 4.75 | 3.32 | 1.43 |
| elliprf | 13.1 | 9.11 | 1.43 |
| elliprg | 49.6 | 34.5 | 1.44 |
| round | 0.519 | 0.358 | 1.45 |
| radian | 0.457 | 0.314 | 1.46 |
| wrightomega | 11.6 | 7.93 | 1.46 |
| ive | 44.5 | 30.2 | 1.47 |
| erfc | 2.39 | 1.59 | 1.50 |
| loggamma | 1.51 | 0.995 | 1.52 |
| cosdg | 3.24 | 2.13 | 1.52 |
| kl_div | 1.01 | 0.657 | 1.54 |
| y1 | 3.44 | 2.22 | 1.55 |
| huber | 0.386 | 0.244 | 1.58 |
| erfcx | 1.46 | 0.914 | 1.59 |
| kolmogorov | 6.28 | 3.84 | 1.64 |
| log_expit | 2.58 | 1.57 | 1.64 |
| dawsn | 1.71 | 1.04 | 1.64 |
| ellipj | 3.73 | 2.23 | 1.67 |
| ellipkm1 | 2.96 | 1.77 | 1.67 |
| y0 | 4.35 | 2.59 | 1.68 |
| erfcinv | 3.34 | 1.97 | 1.69 |
| softplus | 3.01 | 1.67 | 1.80 |
| rel_entr | 1.48 | 0.755 | 1.96 |
| xlogy | 1.71 | 0.84 | 2.03 |
| xlog1py | 2.09 | 0.916 | 2.28 |
