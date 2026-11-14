# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org) (breaking changes are allowed in 0.x
minor releases and are always listed; see `docs/operations/upgrading.md`).

## [Unreleased]

### Changed

- **SciPy's own algorithms for `quad` and the special-function zeros** (their values now follow SciPy's, from the
  same steps):
  - `integrate.quad` is QUADPACK's DQAGSE, ported step for step from SciPy's C translation: the 10-21 Gauss-Kronrod
    rule, bisection of the largest error and the epsilon-algorithm extrapolation past endpoint singularities. It was
    a 7-15 rule without extrapolation. The value, the error estimate and the number of integrand calls are SciPy's:
    21 calls instead of 45 for `sin(x) exp(-x)` on [0, 10], 2 to 5 times fewer on endpoint singularities (`sqrt`,
    `log`, `1/sqrt` at 0), 21 instead of 15 where one 15-point panel was enough. Equal limits return
    `[0.0, 0.0]` without a call and reversed limits are integrated as SciPy does (over `[b, a]`, negated).
    `dblquad`, `tplquad` and `nquad` use it.
  - The zeros of the Kelvin functions (`kelvin_zeros`, `ber_zeros` ... `keip_zeros`), of `J_n`, `J_n'`, `Y_n`, `Y_n'`
    (`jn_zeros`, `jnp_zeros`, `yn_zeros`, `ynp_zeros`, `jnyn_zeros`), of `Y0`, `Y1`, `Y1'` (`y0_zeros`, `y1_zeros`,
    `y1p_zeros`) and of the Airy functions (`ai_zeros`, `bi_zeros`) come from SciPy's specfun routines (KLVNZO, JYZO,
    CYZO, AIRYZO) in the vendored xsf: asymptotic starts and Newton steps, where Tessero scanned in steps of 0.02 or
    0.05 and refined each sign change (4 to 180 times slower than SciPy). Unlike xsf, a zero that does not converge
    ends with an error instead of iterating forever (`ber_zeros(5000)` reaches past the functions' range). The Bessel
    and Airy limits rise from 1,000 and 2,000 zeros to a bound on the work.
  - `pro_cv_seq` and `obl_cv_seq` take every mode's characteristic value from one SEGV call, as SciPy does, instead
    of one `pro_cv` / `obl_cv` call per mode.
  - `optimize.curve_fit` (`LeastSquares::curveFit`) runs MINPACK's `lmdif`, the Levenberg-Marquardt method with a
    QR-factored step that SciPy's `curve_fit` uses without bounds (SciPy 1.17.1's C translation, in the kernel as
    `linalg.lmdif`), with leastsq's defaults (`maxfev` 200 (n + 1), `ftol` = `xtol` = 1.49012e-8, `gtol` 0). It ran
    Tessero's own Levenberg-Marquardt on the normal equations. On models whose arithmetic is the same in PHP and
    NumPy the parameters, residuals, number of evaluations and status are SciPy's bit for bit (11 test problems, both
    backends), and the covariance (leastsq's `cov_x`, by a fixed-order triangular inverse) is within 1 ulp of
    SciPy's, which its BLAS forms. A failed fit raises `ConvergenceError` with leastsq's message; a covariance that
    cannot be estimated is all `INF`, as in SciPy. New: `fullOutput: true` returns
    `[popt, pcov, infodict (nfev, fvec), mesg, ier]`, and xdata given as an NDArray (either backend) calls the model on
    the whole array, as SciPy does (`f(NDArray $x, float ...$p)`), the data staying in the kernel; such a model gets a
    Jacobian's n evaluations in one call (x as a column, the parameters as rows) when that gives the same bits as n
    calls, checked once per fit. The 1,000-point exponential fit: 5.0 ms before, about 0.6 ms with an array model
    through the extension (SciPy 0.61 ms), 1.4 ms through FFI, 1.8 to 2.6 ms with a per-point model.
- **`signal.fftconvolve` and `oaconvolve` use the FFT on large inputs**, as SciPy does (both inputs padded to a
  length of 2, 3 and 5 factors, real transforms along the last axis): where the direct sum's multiply-adds exceed the
  FFT's work about 20 times over. Those results move by rounding, as SciPy's do; small inputs keep the direct sum and
  its exact values.
- **Faster, with results unchanged byte for byte**: `ndimage.gaussian_filter`, `uniform_filter` (and the other
  separable filters) read the input in place and write the last pass straight into the result, dropping three full
  copies of the image; `signal.lfilter` runs its main loop without per-term index tests and, when `a[0] == 1`,
  without the division by it on the recurrence's critical path; `stats.page_trend_test` keeps the exact null
  distribution's table per column count between calls, as SciPy's module-level state does; the `spatial.KDTree`
  constructor and `query` concatenate rows that are lists in one call.
- **`Np::finfo` and `Np::iinfo` of a dtype name are computed once** (FFI package): their numbers depend only on the
  dtype, so the kernel is asked on the first call and the result is kept (returned as a copy); about 0.2 us a call
  instead of about 4 (NumPy 0.5). `finfo` is among the 15 most-called NumPy functions in the usage corpus.
- **Lower per-call cost for the most-called NumPy helpers through FFI**, with results unchanged byte for byte:
  - Sequence arguments (`concatenate`, `vstack`, `hstack`, `column_stack`, `stack`, ...) reuse their argument-array
    type and write NDArray items directly; `concatenate` of two small arrays drops from about 16 to about 7 us.
  - `zeros_like` / `empty_like` of an NDArray with a dimension, and `atleast_1d` of one such array, are answered in
    PHP with the same new array the kernel returns, without the routine call (about 2 us saved each).
- **`NDArray::matmul` / `dot` of two small matrices about 2x faster** (FFI package, results unchanged byte for byte):
  two C-contiguous float matrices of one dtype go straight to the same BLAS gemm call, and gemm takes its operands'
  addresses directly instead of through libtessero pointers and casts. `dot` of 3 x 3 matrices about 14 -> 7 us;
  `numpy.dot` is the fourth most-called NumPy function in the usage corpus.
- `tools/parity/usage_speed.php`: per-call speed weighted by real-world usage. Each timed function's per-call time
  is weighted by how often the corpus of the usage-weighted coverage calls it, giving the time a million and a
  billion calls of that mix take on each side.

- **`special.ellip_harm_2` and `special.ellip_normal` about 1,000x faster** (18 ms to 12 and 21 us a call; SciPy
  takes 44 and 145 us): the 256-point Gauss-Legendre rule of their quadratures is computed once per process instead
  of on every call. The Kelvin-function zeros (`kelvin_zeros`, `ber_zeros` ... `keip_zeros`) are about 1.4x faster:
  their bisection no longer evaluates the function at the bracket's lower end twice. Results unchanged, byte for byte.

- **A default native memory budget in the CLI**: half of the machine's physical memory, on both backends, where it
  was unlimited. The CLI also runs queue workers and artisan commands, and a single oversized call there (a
  `numpy.meshgrid` of nine 9-value vectors asks for 26 GiB) is now refused up front with `MemoryError` /
  `Tessero\Ext\MemoryException` instead of exhausting a shared host. `TESSERO_MEMORY_BUDGET` now also applies in the
  CLI (FFI package); `TESSERO_MEMORY_BUDGET=0` or `tessero.memory_budget = -1` opts out; an explicit budget is kept.
  Web SAPIs keep max(256 MiB, `memory_limit`). The kernel exports `tsr_physical_bytes()` for it (a compatible ABI
  addition).
- **Lower fixed cost per call on the FFI backend**: the registry binding prepares each routine once (its label, result
  type, sequence and in-place arguments), fills the kernel's array descriptors in place instead of building and copying
  a `tsr_array` per argument, keeps short string arguments and the C types it needs, and reuses the operand buffers of
  scalar element-wise calls. `NDArray::meta()` and the C-order strides of new arrays are cheaper too. A scalar
  `Special::ndtr(0.3)` 2.7 -> 1.3 us; small routines (`Np::diff`, `unique`, `interp`, `trapezoid` on 8 values) 1.4-1.6x
  faster. Results are unchanged, byte for byte.
- **Element-wise registry functions on small arrays** (`scipy.special`, distribution methods such as
  `Stats::norm()->cdf($x)`): PHP scalar arguments next to arrays are passed as one value with zero strides instead of
  as a new 0-d array each, broadcasting computes the strides instead of building views, the pointer, shape and stride
  arrays are reused, the output goes straight to the constructor and keeps the `void*` it was allocated with, and each
  function's metadata is looked up once. In the kernel, an element-wise loop over several rows runs without an OpenMP
  parallel region when it uses one thread (also the NumPy ufunc engine behind `Math::hypot`, `arctan2` and the other
  loop ufuncs on broadcast or strided operands). `Special::ndtr` on 8 values 9.8 -> 5.0 us, `jv(1.0, $x)` 21 -> 9.2 us,
  a 2x1 by 4 broadcast `jv` 24 -> 12.6 us, `Stats::norm()->cdf` on 8 values 24 -> 6.0 us (SciPy 41 us). Results are
  unchanged, byte for byte (NaN sign and payload bits included).
- **Array arithmetic, copies and axis functions on small arrays**: `NDArray` binary and unary operations, `copy()`,
  `astype()` and the loop ufuncs of `Math` fill a reused set of shape and stride arrays instead of four new ones per
  call, pass the buffers' cached `void*`, compute broadcast strides once (without building views), build the result
  through the constructor, and pass a PHP scalar operand's bytes directly instead of a new 0-d array (when no cast
  is needed). The generalised ufuncs (`partition`, `searchsorted`, `percentile`, `median`, ...) parse their C types
  once, write the kernel's array descriptors in place, and pass a scalar input after the first (`kth`, a value to
  search for) through a slot of its own. On 8 values: `$x->mul(2.0)` 11.8 -> 4.8 us, `$x->add($x)` 8.9 -> 5.2 us,
  `copy()` 5.0 -> 3.1 us, `Np::partition` 18.7 -> 11.1 us, `Np::searchsorted` 17.3 -> 11.8 us. Results are
  unchanged, byte for byte.
- **Routine calls, list arguments, small sorts and `numpy.fft` on small inputs**: the FFI binding parses each C type
  it casts to once (a cast by name parsed it on every call, in `Library::address()` and `Buffer::ptr()` among others),
  an `NDArray` argument writes its own descriptor into the kernel's slot (`NDArray::writeMeta()`), number results are
  read without a call, a routine's array result keeps the block's `void*` for freeing and is wrapped without the
  constructor repeating the checks the binding has just made, and variadic routines prepare their signature once.
  `NDArray::array()`, `fromFlat()`, `empty()`, `zeros()`, `copy()` and `astype()` check a shape once. In the kernel, a
  stable argsort of 32 keys or fewer (`argsort`, `lexsort`, `unique` with indices, and others) runs an insertion sort
  instead of the radix passes, whose 96 KB of count tables were zeroed and scanned on every call; the stable order is
  unique, so the permutation is the same. `argsort()` along the last axis of a C-contiguous array reads it in place,
  and `sort()` and `numpy.fft` along the last axis skip the axis round trip. On 8 values: `argsort()` 18.8 -> 3.7 us,
  `Np::lexsort` (2 x 5) 17.6 -> 4.6 us, `sort()` 9.8 -> 4.1 us, `Fft::fft` 10.5 -> 4.7 us, `Fft::irfft` 15.4 ->
  8.2 us, `Np::unique` with all outputs 21.0 -> 12.6 us, `NDArray::array` of 4 floats 4.0 -> 2.9 us, `Np::ndim`
  2.2 -> 1.6 us, `Np::flipud` 5.3 -> 4.4 us. Results are unchanged, byte for byte (NaN sign and payload bits
  included).
- **PHP lists as arguments, and generalised ufuncs with a scalar result**: a list of up to 4,096 values passed to a
  registry routine, an element-wise function or a generalised ufunc is packed into a block of its own, with the
  values, shape and dtype `NDArray::array()` gives it (`NDArray::listParts()` and `flatBytes()`, the code `array()`
  itself now runs), instead of becoming an array with a buffer of its own that is freed after the call. A
  generalised ufunc's 0-d result (`median`, `percentile` with one q, `nansum`, `searchsorted` of one value, ...) is
  written to a slot of its own and read back as the number `item()` returns, instead of through an array, and its
  other outputs are wrapped without the constructor repeating the checks made on the kernel's shape. On 8 values:
  `Np::median` of a list 14.1 -> 7.8 us (of an array 8.2 -> 5.8 us), `Np::percentile` of a list 16.9 -> 9.8 us,
  `Np::nansum` of 4 integers 15.7 -> 8.6 us, `Np::searchsorted` of one value 10.5 -> 7.2 us,
  `Np::broadcastShapes` of three lists 20.2 -> 14.0 us, `Special::ndtr` of a list 7.8 -> 6.6 us. Results are
  unchanged, byte for byte (NaN sign and payload bits included), and so are the exceptions for ragged, non-numeric,
  empty and too deep lists.
- **Generalised ufuncs, routine arguments and new blocks on the FFI backend**: a generalised ufunc keeps the C blocks
  of its last call (axis masks, array descriptors, parameters, output shapes) and reuses them when no call is using
  them, instead of allocating six per call; a routine writes `null` and integer arguments (`axis`, `dtype=None`, ...)
  into their slots without a call each; and `Library::address()` reads a pointer back as an integer through a union
  instead of two casts, which every new array block and every adopted routine result goes through. On 8 values:
  `Np::median` 5.6 -> 4.6 us (of a list 7.6 -> 6.3 us), `Np::percentile` 6.7 -> 6.0 us, `Np::partition` 9.8 -> 8.5
  us, `Np::nansum` of 4 integers 5.7 -> 4.9 us, `Np::emptyLike` 4.4 -> 3.7 us, `Np::swapaxes` 5.8 -> 4.4 us,
  `NDArray::empty` 2.0 -> 1.8 us. Results are unchanged, byte for byte (NaN sign and payload bits included).
- **`np.diff`, `outer`, `inner`, `kron`, `ediff1d`, `trapezoid`, `cumulativeTrapezoid`, `geomspace`, `pad` (odd
  reflection) and the `poly*` routines**: their shared element-wise step read every operand through a one-element
  conversion call, also when no conversion was needed; operands already in the loop dtype are now read in place, and a
  float64 difference runs row by row. `np.diff` on 1M values 127 -> 1.4 ms (NumPy 1.2), `np.outer` 1000 x 1000
  128 -> 10 ms (NumPy 1.3).
  Results are unchanged, byte for byte (NaN sign and payload bits included).

- **Large-block reuse**: blocks of 1 MiB or more freed with `tsr_free` are kept per process and handed out again by
  `tsr_alloc` for the next array of about their size, instead of going back to the operating system and page-faulting
  in again on the next call (a 1024 x 1024 `fft2` in a fresh process 64.5 -> 29.7 ms, NumPy 32.2; `rfft2` 40.0 -> 15.7 ms). The blocks stay
  libtessero's and are released only by it. Limit per process: 64 MiB in the CLI, 16 MiB in a web SAPI; set with
  `tsr_set_alloc_cache_limit` / `tsr_alloc_cache_limit` / `tsr_alloc_cache_bytes`, `Tessero::setAllocCacheLimit()` /
  `allocCacheLimit()` / `allocCacheBytes()` and the same on `Tessero\Ext\Engine`, or `TESSERO_ALLOC_CACHE_BYTES`
  (`0` turns it off). Kept blocks are outside the memory budget. Results are unchanged. Off in the ownership-check and AddressSanitizer builds.
- **`sort()` (C `tsr_sort`) is now the AVX2 quicksort of `sortSimd()` where the CPU has AVX2**, the radix sort
  elsewhere. The result is unchanged (the sorts return the same bytes); the time is 2-25x lower below a few thousand
  values and about 2x lower on 1M random values (one thread: 17 ms against NumPy's 14 ms, was 31 ms). `sortSimd()`
  stays as an explicit name. The radix sort stays tested on AVX2 machines through a test-only switch.

### Added

- **`tools/parity/run-tests.php --hash=OUT.txt`**: one line per fixture call with a hash of the result's raw bytes
  (arrays by dtype, shape and bytes, floats by their bits, exceptions by class and message), so two builds can be
  compared byte for byte (`diff` the files). **`bench/tier1_bench.php` / `bench/tier1_bench.py`**: the element-wise
  tier of the speed sweep (every `scipy.special` function on the fixture rows tiled to 100,000 values), paired by
  `bench/sweep_report.py`.
- **`SparseLinalg` solvers take sparse matrices** (`spsolve`, `eigsh`, `cg`, `bicgstab`, `bicg`, `cgs`, `minres`, `gmres`,
  `lgmres`, `gcrotmk`, `qmr`, `tfqmr`; both backends): pass the CSR array that `Sparse::diags` / `Sparse::csrArray` return
  (or a `CsrMatrix` on FFI). The bindings hand it to the kernel as the group data, indices, indptr, shape, checked before
  use. Iterative solvers use sparse products (the same results as on the dense matrix); `spsolve` a banded LU sized by
  the matrix's bandwidth (dense fallback when the band is wide and the matrix fits, else an error); `eigsh` thick-restart
  Lanczos with a deflation search so repeated eigenvalues are found (`SM` dense up to n = 4096; `sigma` unsupported).
  Before, these solvers took dense matrices only: a 100k tridiagonal `spsolve` (80 GB dense) now takes 15 ms (SciPy
  51 ms); a 100 x 100 grid Poisson system: `spsolve` 25 ms (SciPy 31), `cg` 21 ms (SciPy 21).
- **`Generator::multivariateNormal(..., method:)` takes `'cholesky'` and `'eigh'`** besides `'svd'`, on both
  backends, each factorisation computed as NumPy computes it (C: `tsr_pcg64_multivariate_normal_method`). A cov that is
  not positive definite fails `'cholesky'` with an error, as NumPy raises.
- **Typed C entry points `tsr_cholesky`, `tsr_solve`, `tsr_slogdet`** (requested by a native consumer): row-major float64,
  running the registry routines of the same name, so the results equal the registry's byte for byte, without packing
  `tsr_arg` / `tsr_result`.
- **`tsr_unary_find` / `tsr_binary_find` and `tsr_unary_name` / `tsr_binary_name`**: the core element-wise op codes by
  NumPy ufunc name (`exp`, `tanh`, `add`, ...), which were documented only in an internal header. New page
  `docs/reference/c-function-lookup.md`: the four ways to reach a function from C, and why registry ids may be kept
  for a process but not across releases.
- **Two more sorts with the same result as `sort()`**: `NDArray::sortSimd()` (C `tsr_sort_simd`, an AVX2 quicksort:
  faster on random data, many duplicates and sorted input; the radix sort where AVX2 is missing) and
  `NDArray::sortRuns()` (C `tsr_sort_runs`, a natural merge sort: fastest on mostly sorted data, slowest on random
  data), on both backends. All three sort the same order-preserving keys by comparison only, so their output is
  byte-identical on every CPU, thread count and input alignment; `sort()` (radix) stays the default. When to use
  which, with measurements: `docs/project/benchmarks.md`.

### Performance

- **`Spatial\Delaunay` (2-D) in the kernel**: the sweep-hull algorithm (`distance.delaunay_2d`, O(n log n), the hull exact
  by construction) replaces a Bowyer-Watson that scanned every triangle in PHP: 10k points 57 s -> 29 ms (SciPy 46 ms).
  The same simplex set (parity against SciPy; PHPUnit against the old algorithm); `griddata`, `LinearNDInterpolator`
  and `CloughTocher2DInterpolator` build on it. The PHP Bowyer-Watson remains for input the sweep cannot triangulate.
- **`Spatial\ConvexHull` (2-D)**: the monotone chain in the kernel (`distance.convex_hull_2d`), the same vertices:
  10k points 43 -> 4.1 ms (SciPy 1.5).
- **`hierarchy.linkage`**: each cluster caches its nearest later neighbour, so a merge rescans only the rows it affects
  (about n^2 instead of n^3); the same Z byte for byte, ties included (C test against the old loop, all 7 methods).
  Ward on 2000 points 1.6 s -> 70 ms (SciPy 92).
- **`linalg.solve_triangular`**: the column-major `dtrtrs` on the row-major matrix (triangle and transpose flipped)
  instead of LAPACKE's row-major copies, a contiguous matrix read in place: 1000 x 1000 4.2 -> 0.88 ms (SciPy 1.0).
- **`median`, `percentile`, `quantile`**: a two-pointer (Hoare) introselect, and the second order statistic of an
  interpolated quantile taken as the smallest of the rest: percentile 1M 18 -> 5.9 ms (NumPy 12), median 13 -> 12 ms
  (NumPy 11). The same values (the k-th value of the total order has one bit pattern).
- **`Spatial\KDTree` builds its tree once, in the constructor, and every query searches it** (new registry routine
  `distance.kdtree_build`; `distance.kdtree_query` takes its idx/nodes/box). 100 queries in 100k points: 0.57 ms,
  was 37 ms (SciPy 0.20 ms); the query reads the data and the tree in place. The constructor now pays the build
  (88 ms with the conversion of the PHP point arrays; SciPy 34 ms from an ndarray), where the old timing built nothing.
  A malformed tree is refused before use. `kdtree_build` has no SciPy counterpart, so it has no fixture; the C tests
  and PHPUnit pin it (queries on its tree byte-identical to a fresh build and to an exhaustive search).
- Routine arguments that are already contiguous float64 are copied with one `memcpy` instead of element by element
  (`fn_arg_doubles`): the same bytes, less time per call in every routine that reads arrays.
- **`Spatial\KDTree` is a k-d tree**: the queries ran an exhaustive search in PHP (10,000 queries in 100,000 points: hours);
  they now run in the kernel (`distance.kdtree_query`, both backends: median splits, branch-and-bound, a k-best heap),
  55 ms for that case with the tree build included (SciPy: 49 ms). Same neighbours, distances and order (nearest
  first, ties by index); non-finite points are refused, as SciPy refuses them.
- **Faster `median`, `percentile`/`quantile`, `unique`, `einsum`, `cov`, `trapezoid`, `maximum_filter`/`minimum_filter`**
  (1M values / 1000^2 images, single-threaded): median 144 -> 13 ms, percentile 119 -> 18 ms (introselect), unique
  97 -> 15 ms, the set routines likewise (their libc `qsort` replaced by `sortSimd` / the radix `tsr_argsort`), einsum
  ij,jk->ik 300^2 267 -> 15 ms (incremental offsets, strided inner loop, same summation order), cov 50 x 100k
  235 -> 45 ms (one BLAS dgemm, as NumPy), trapezoid 353 -> 1.6 ms, maximum_filter 5x5 822 -> 25 ms (separable,
  as SciPy). Values unchanged.
- **`sortSimd()` sorts in place at the data's own width**: 8 float32/int32 keys per AVX2 vector instead of 4 widened
  64-bit keys, integers without any conversion, floats mapped to order-preserving integers in place (the map is its
  own inverse), and no scratch buffer. 1M random values: float32 24.8 -> 18.0 ms, int32 24.0 -> 17.0 ms, int64
  25.3 -> 22.0 ms, float64 25.2 -> 23.3 ms; sorted input about twice as fast. Output unchanged, byte for byte.
- **`sortSimd()` leaves go to in-register AVX2 sorting networks** (up to 64 keys for 64-bit data, 128 for 32-bit;
  generated straight-line code, `tools/gen-sort-net.php`), and float64 is sorted as doubles (one-instruction
  compares and min/max; the zeros' signs are restored afterwards, and denormals-are-zero mode falls back to integer
  keys). 1M random values: float64 23.3 -> 15.9 ms, float32 18.0 -> 7.8 ms, int64 22.0 -> 16.7 ms (NumPy 17.1),
  int32 17.0 -> 7.4 ms. Now the fastest of the three sorts on almost every input. Output unchanged, byte for byte.
- **`sortSimd()` is multithreaded** from 131,072 values, using the threads set with `Tessero::setThreads()`
  (OpenMP tasks for the top partitions; the pivots do not depend on the thread count, so the output is the
  one-thread output, byte for byte). 1M random values with 4 threads: float64 5.7 ms, float32 3.4, int64 6.5,
  int32 2.8 -- 2.2-2.6x faster than NumPy's single-threaded sort (14.1 / 7.5 / 17.1 / 6.8 ms).

### Fixed

- **`Special::hyp1f1(a, b, INF)` no longer hangs.** For x = +inf Boost's evaluation can return a scaling exponent
  near 2^63, and the loop that removed it 707 at a time ran about 10^16 times (SciPy hangs on the same inputs, e.g.
  `hyp1f1(0.5, 1.5, inf)`). The loop's step count is now computed and the factor applied only while it still
  changes the result: the same result as the original loop, bit for bit (NaN for most such inputs), at once. Every
  other input gives the same bytes as before (`THIRD_PARTY_NOTICES.md`, local modifications).

- **`sort()` (the radix sort) crashed on small thread stacks**: it kept up to 49 KB of count tables per recursion level
  on the stack, which overflowed musl's 128 KB default for threads and OpenMP workers (Alpine; the CI musl job failed
  from the sort commits on) and glibc threads whose stack is shared with large static TLS. The tables now live in the
  heap block of the scratch buffer; results unchanged, speed equal or up to 8% faster at 10M values. Regression test
  on a 128 KB thread at the full recursion depth.
- **`Generator::defaultRng([...])` refused list elements of 2^32 and above**; NumPy's `SeedSequence` splits every element
  into little-endian 32-bit words (`[2**40 + 7, 5]` seeds as `[7, 256, 5]`), and both backends now do the same.
- **`einsum` hid a NaN behind a zero**: once a running product reached 0 the remaining operands were skipped, so
  `einsum('i,i', [0], [nan])` gave 0; NumPy gives nan.
- **`scipy.fft` dct / idct / dst / idst were O(N^2)** (direct sums; 1M points took hours): above 128 points they now run
  through one real FFT (Makhoul), 1M points in about 50 ms, agreeing with SciPy to 1e-16 of the output scale; small
  sizes keep the exact direct sums. New fixture cases above the switch (129, 257, 1000 points; 3x130 and 130x3).
- **`interpolate.CubicSpline` / `make_interp_spline` (k=3) solved their tridiagonal system densely** (n x n: 800 MB
  and 8-17 s at 10,000 knots): now LAPACK `dgbsv` on the band, as SciPy does; 76 ms, O(n) memory.
- **`sparse.linalg.lsmr` read `maxiter` from the `conlim` argument**, so a default call ran 100 million iterations
  (15 s on a 10 x 6 system; the results were right): now 16 us. `lsmr` still ignores `damp`, `atol`, `btol` and
  `conlim` and runs 4n iterations unless `maxiter` is given.
- **`signal.butter`, `cheby1`, `cheby2`, `ellip`, `bessel`, `iirfilter` silently ignored `output=`**: they return
  (b, a) only and now refuse `output='sos'` / `'zpk'` with an error instead of answering in the wrong format.
- **Signed zeros now as NumPy / SciPy**: `np.modf(-0.0)` gives a `-0.0` fraction; `polymulx` and the polynomial
  derivatives give `c[0]*0` (NumPy); `csgraph.laplacian` follows SciPy's operation order (off-diagonal zeros
  `-0.0`); `slinalg.invpascal` triangular zeros are `+0.0` (SciPy's default exact=True result is integer).
- **`sortSimd()` with threads under per-thread FP modes**: OpenMP worker threads keep their own MXCSR, so a worker
  left in denormals-are-zero mode compared subnormal float64 values as zero while the caller (not in DAZ) had
  chosen the double-compare path. Every thread of the sort now runs with the caller's MXCSR and restores its own.
- **Registry input gate**: no signed overflow building the message for an element offset of INT64_MIN (found by
  the libFuzzer CI job).

### Testing

- **Speed sweep tooling**: `tools/parity/run-tests.php --time=OUT.json` times every passing fixture call,
  `tools/parity/time_calls.py` times the same cases in NumPy/SciPy and `tools/parity/time_report.py` pairs them;
  `bench/tier2_bench.{php,py}` (83 heavier calls) and `bench/sweep_report.py`. Results: `docs/project/speed-sweep.md`.
- **The parity comparator checks the sign of exact zeros** in real results (complex parts and four rotation-based
  decompositions exempt, each with its reason).
- **`make -C csrc coverage`** (gcov line coverage of the C tests) and a nightly job that runs it.
- **Sort risk register and hardening tests** (`docs/project/sort-design.md`): an independent qsort oracle over
  the documented order at every size 0..1100, the 0-1 principle exhaustively for the leaf networks, 15
  adversarial / structured / float-edge patterns at serial and parallel sizes, forced heapsort fallback, DAZ in
  worker threads, concurrent sorts from application OpenMP threads and pthreads, and a new fuzz target `sort`
  (CI and nightly). A mutation check (planted bugs) confirmed the suite catches each one.

## [0.3.0] - 2026-11-05

Everything since 0.2.0: the NumPy/SciPy parity program, the security hardening, and the speed work below. The C ABI
gains functions only, so it stays at generation 2 (`libtessero.so.2`); see `docs/operations/upgrading.md` for the
behaviour changes. The library and the PHP package check that their versions match, so update both together
(`make -C csrc` after updating the package).

### Added

- **`make -C csrc install`** (`PREFIX`, default `/usr/local`; `DESTDIR`, `INCLUDEDIR`, `LIBDIR`, `BINDIR`). It installs
  the library and the public headers (`tessero.h`, `tessero_version.h`, `tessero_native.h`, `tessero_repro_f32.h`), so
  a native consumer builds against `--with-tessero=$PREFIX` and can compare a vendored `tessero_repro_f32.h` with the
  installed one. `make uninstall` removes them again.
- **`tsr_build_commit()`**: the git commit the library was built from (`-dirty` when `csrc/` had uncommitted
  changes, `unknown` outside a checkout unless `make TSR_BUILD_COMMIT=<sha>`). `Tessero::info()`, `vendor/bin/tessero
  info` and `tessero:doctor --json` report it as `commit`, so a deployment can check that its library matches the
  commit in `composer.lock`. `tsr_version()` alone is the same for every commit of a release. Additive, so the ABI stays 2.
- **Windows: `make -C csrc` copies the MinGW runtime DLLs next to `tessero.dll`** (OpenBLAS, gfortran, quadmath,
  winpthread, ...; `BUNDLE_DLLS=0` turns it off). The FFI loader already searches that directory while it loads the
  library, so a fresh build now loads without `$MINGW_PREFIX/bin` on `PATH`. Before, only CI did the copy.

- **Opt-in GPU-matching floating-point profile** (`csrc/src/fpenv.c`, declared in `tessero.h`; feature
  `fp_flush_profile`). NVIDIA GPUs flush subnormal values to zero and expose no preserve mode, and NaN results can
  carry different bit patterns on different processors. `tsr_fpenv_flush_enter` / `tsr_fpenv_flush_leave` switch
  the calling thread to flush-to-zero for a scope and restore its previous state exactly; `tsr_flush_subnormal_f32`
  and `tsr_canon_nan_f32` normalise arrays in place with integer bit operations; `tsr_nan_first_f32` finds the
  first NaN, with f64 twins of the three array helpers. The flush scope is for f32 work only: NVIDIA GPUs keep f64
  subnormals, so f64 inside a scope would diverge. Additive only: no existing function, default, golden or self-check changes, and nothing runs unless
  called.
- **Reproducible-f32 transcendental primitives** (`tsr_repro_rsqrt_f32`, `tsr_repro_exp2_f32`,
  `tsr_repro_exp_f32`, `tsr_repro_erf_f32`; `csrc/src/repro_f32.c`, declared in
  `csrc/include/tessero.h`). A **new, original contribution, NOT part of the ported NumPy/SciPy
  architecture.** NumPy/SciPy transcendentals are accuracy-first and defer to the system
  `libm`/vendor math, which is not bit-reproducible across hardware, drivers, or SIMD width.
  These are reproducible **by construction** - built only from correctly-rounded f32
  operations, giving one defined f32 bit-value on every SIMD width, thread count,
  and CPU ISA (and, via a mirrored GPU shader, the GPU). Proven bit-identical across
  scalar/AVX2/AVX-512/NEON/SVE, 1 vs 4 threads, and x86 == ARM (`csrc/repro/ARCH_MATRIX.md`);
  four frozen goldens + canonical spec in `csrc/repro/GOLDENS.md`. Minimax coefficients are
  from Cephes (public domain) and Abramowitz & Stegun 7.1.26; the reproducible construction
  and the resulting bit-identity are the contribution. The ~1,845 parity-gated NumPy/SciPy
  functions are unaffected.
- **Reproducible `recip` and `tanh` primitives** (`tsr_repro_recip_f32`, `tsr_repro_tanh_f32`), with goldens
  `f7f9ec19` and `5b43f3d6` in `csrc/repro/GOLDENS.md`.
- **Reproducible composites** (`tsr_repro_{sigmoid,gelu_tanh,softmax,rmsnorm,layernorm}_f32`): one definition of
  each composition, built on the primitives. Goldens confirmed on two independent hosts, each scalar == AVX2 ==
  4-thread.
- **Reproducible backward composites** (gradients of `gelu_tanh`, `erf`, `sigmoid`, `tanh`, `rmsnorm`,
  `layernorm`, and the norm `gamma` gradient) in the single header `csrc/include/tessero_repro_f32.h`. Goldens
  recorded in `csrc/repro/GOLDENS.md`, approved on two hosts.
- Talario's GPU kernels mirror these definitions and match them bit for bit on twelve NVIDIA GPU models across four
  generations, Ampere to Blackwell (attested at 1024 x 1024 by Talario); other GPUs are not yet attested.
- **NumPy/SciPy parity program** (`prompts/numpy-scipy-parity-program.md`, ADR 0011, ADR 0012). Coverage is
  measured, not asserted. `tools/parity/census.py` lists every public symbol of NumPy 2.4.4 and SciPy 1.17.1,
  and `tools/parity/report.py` classifies each one from the code and the recorded fixture runs
  (`docs/project/numpy-scipy-coverage.md`).
  - **Kernel function registry** (ADR 0011). One C table of functions (ufunc, gufunc, distribution, routine,
    random) interpreted at run time by both bindings. The extension registers `Tessero\Ext\Special`,
    `Stats`, `Np` and `Random\Generator` from it at MINIT. The FFI façades `Tessero\Special`, `Stats`, `Np`
    and the `Generator` methods are generated from it (`tools/parity/gen-facades.php`).
  - **scipy.special**: every real-valued ufunc, from the code SciPy itself runs (xsf, Boost.Math and SciPy's
    glue, vendored with their licences; see `THIRD_PARTY_NOTICES.md`).
  - **scipy.stats**: the continuous and discrete distributions, on a port of `rv_continuous`/`rv_discrete`
    (`csrc/cxx/stats_dist.*`: the public-method semantics and the generic QUADPACK/brentq/bisection
    fallbacks, with SciPy's QUADPACK and brentq vendored). Also descriptive statistics, correlation and
    hypothesis tests. A new class `Tessero\Stats\Distribution` (and `Tessero\Ext\Distribution`) provides
    `pdf`...`isf`, `stats`, `moment`, `entropy`, `interval` and `rvs`.
  - **NumPy statistics, sets and searching** (`Tessero\Np`, `Tessero\Ext\Np`): median, quantile/percentile
    (all methods), nan-reductions, histogram/bincount/digitize, cov/corrcoef, unique and set operations,
    partition, lexsort, searchsorted, correlate/convolve.
  - **NumPy array construction and manipulation** (`Tessero\Np`, `Tessero\Ext\Np`, ADR 0013): the split and
    stack families, `concatenate`, `tile`/`repeat`/`resize`, `flip`/`roll`/`rot90`, axis moves, broadcasting,
    `append`/`insert`/`delete`, `diag`/`tri*`/`*_indices`, `*_like`, `take`/`compress`/`nonzero`, index
    conversions, `meshgrid`, `select`/`choose`, `pad` (all modes). Also the numeric routines: `outer`/`inner`/
    `kron`/`cross`/`tensordot`, `diff`/`gradient`/`trapezoid`/`unwrap`, `interp`, `logspace`/`geomspace`/
    `vander`, the `poly*` functions, the window functions, `i0`/`sinc`, `nan_to_num`, and the in-place writers
    `put`, `putmask`, `place`, `copyto`, `fill_diagonal` and `put_along_axis`. That is 118 functions, verified
    by NumPy fixtures on both backends. Integer results are exact above 2^53, including `unique`, the set
    operations and `quantile`'s discrete methods, which now keep the input's integer or bool dtype as NumPy
    does.
  - **numpy.random.Generator distributions** from NumPy's own `distributions.c` on the kernel PCG64. Streams
    are bit-identical to NumPy on both backends. The extension gains a stateful
    `Tessero\Ext\Random\Generator` (`defaultRng`, `integers`, `permutation`, `shuffle`,
    `getState`/`setState`).
  - **Fixtures generated by NumPy/SciPy for every registry function**, replayed on both backends by
    `tools/parity/run-tests.php`. They are generated in a pinned numerical environment
    (`tools/parity/gen-fixtures.sh`, ADR 0012), and every file records that environment.
  - **Deviation rules** (`docs/project/reference-deviations.md`, with counts generated from the fixtures). These
    cover inputs where SciPy crashes or never returns, rows where SciPy's own value is undefined (found with
    AddressSanitizer or by history dependence), SciPy exceptions on invalid parameters where its contract says
    NaN, and a vectorised result that disagrees with SciPy's element-by-element one.
  - `tools/parity/claims-check.php` fails the build on coverage language or percentages that the generated
    pages do not back. CI runs the fixtures on both backends.
  - **Coverage of real-world usage** (`docs/project/usage-weighted-coverage.md`). `tools/parity/usage.py`
    counts how often the library code of 18 widely used scientific Python packages calls each NumPy and SciPy
    function, and `report.py` weights the verified coverage by those counts and lists the most-called functions
    that are not yet verified on both backends.

- **Beta tracking.**
  - The NumPy/SciPy coverage inventory
    (`docs/project/numpy-scipy-coverage.md`, 169 rows) is generated from the
    public API by `tools/api-dump.php` and `tools/gap_inventory.py`, so it
    cannot claim a function that does not exist.
  - A gap analysis interprets the inventory.
  - An open-issue list: 50 issues with acceptance criteria, kept in
    `.github/beta-issues.json`. `tools/beta-issues.php` renders it and
    imports it into GitHub with `gh`.
  - A live-testing checklist for Windows, macOS, arm64/musl, FPM, Laravel and
    long soaks.
- **Backend parity** (ADR 0010, `docs/architecture/adr/0010-shared-kernels-and-vector-math.md`):
  - The loop ufuncs moved from the extension into libtessero (`csrc/src/ufunc.c`,
    `tsr_ufunc_*`). The FFI package gains **`Tessero\Math`**, with the same 74
    functions, signatures, `out:` semantics and results as `Tessero\Ext\Math`.
  - Memory mapping and `.npy` headers moved into libtessero (`csrc/src/mmap.c`:
    `tsr_mmap_plan/open/flush/close`, `tsr_npy_header`, `tsr_npy_write_header`).
  - The FFI package gains `NDArray::memmap()`, `openMemmap()`,
    `load($path, $mmapMode)`, `save()`, `flush()`, `isMemmap()`,
    `isReadonly()`, `isWritable()`, `filename()`, and
    `Tessero::info()['memory_mapped']`. It checks open_basedir and refuses
    stream wrappers, like the extension.
  - The extension gains `.npy` I/O: `NDArray::load($path, ?$mmapMode)`,
    `$a->save($path)` (byte-identical to `numpy.save`) and
    `NDArray::openMemmap()`, with Fortran-order files mapped with Fortran
    strides.
  - The extension reduces over a **list of axes** with **`keepdims`**: `sum
    prod min max any all mean var std`, and `argmin`/`argmax` with `keepdims`.
    Results are bit-identical to the FFI package.
  - Laravel: `Tessero::memmap()`, `load()` and `openMemmap()` on either backend.
- **Hardening**:
  - libFuzzer targets for the slice parser, the `.npy` header parser,
    memory-map planning and the JSON writer (`csrc/fuzz/`, `make -C csrc fuzz`),
    with seed corpora and oracles.
  - A standalone mutation driver for machines without libFuzzer
    (`make fuzz-standalone`).
  - Property-based tests (`tests/Unit/PropertyTest.php`).
  - A cross-backend byte-parity test (`tests/Ext/BackendParityTest.php`).
  - A soak harness (`tools/soak.php`).
  - A nightly workflow: 30 min of fuzzing per target, 5.5 h soaks per backend,
    property tests ×50.
  - `docs/operations/soak-testing.md`, the memory-safety review, a
    *Safeguards* table in the memory model, and crash-debugging steps (core
    dumps, gdb, ASan builds of both bindings) in troubleshooting.
  - `tests/Unit/MemorySafetyTest.php` and `ext/tests/014-memory-safety.phpt`.
- **Performance**:
  - Vectorised `exp`/`log` for float64 and float32 (`csrc/src/vmath.h`:
    branch-free, fdlibm algorithms, < 1 ulp). `exp` on 1M values went from
    9.3 ms to 1.2 ms.
  - Real-input FFT (`tsr_rfft`, `tsr_irfft`: half-length transform); `Fft::rfft`
    and `irfft` use it.
  - `sort` uses an MSD + LSD hybrid radix sort with parallel buckets: 1M float64
    went from 53 ms to 30 ms.
- Bench cases `log 1M` and `rfft 2^20 real`. `.npy` fixtures written by NumPy
  (`tests/fixtures/generate_npy.py`).
- Error code -9 `TSR_EIO` for operating-system I/O errors.

### Performance

Measured as a native PHP app (FFI backend) against a native Python app (NumPy 2.4.4) on the same machine, one
thread, `bench/run.php` vs `bench/numpy_bench.py`. Every change below returns the same bits as before: 1,001 cases
compared byte for byte against the previous library, plus parity (1845 functions on both backends), the
AddressSanitizer + UBSan C tests and a ThreadSanitizer run of the FFT cache (the second round below: 1,135 cases,
same checks). Across the 21 shared bench cases the geometric mean went from 1.21x to 1.10x NumPy's time after the
first round.

- **Random variates with scalar parameters** (`normal`, `exponential`, `gamma`, ... on both backends): the draw loop
  validated the same scalar parameters once per element, parsing argument names each time, and recomputed every
  operand's address per element. Scalars are now checked once and drawn in a tight loop into a contiguous output, in
  the same order. `normal(size: 1M)` 35 -> 10.8 ms (NumPy 11.4), `exponential` 19.8 -> 5.4 ms, `random` 7.9 -> 3.8 ms.
- **FFT plan cache**: plans (twiddle tables), Bluestein chirps and real-FFT twiddles are kept by length and
  direction, up to 64 MiB in all, least recently used out first, read-only and reference-counted so threads share
  them. Rebuilding the plan was more than half the time of a 2^20 transform. A plan of only radix 2-5 factors also
  drops its full twiddle table, halving its memory. Cached memory is outside the `tsr_alloc` budget (it is bounded by
  the cap); a plan too large for the cache is built and freed per call, as before. `fft 2^20` 83 -> 66 ms,
  `rfft 2^20` 64 -> 37 ms; `scipy.signal` routines that transform segment by segment reuse one plan.
- **`mean` no longer computes the variance's second pass** (`tsr_moments` accepts NULL `m2`; both backends): 1M
  values 2.02 -> 0.39 ms (NumPy 0.34).
- **`min` / `max` of contiguous float64 use AVX2** when the CPU has it, with exact fix-ups so the first NaN and the
  first of tied -0.0 / 0.0 are returned as before: 1M values 1.13 -> 0.33 ms. Other CPUs keep the scalar loop.
- **FFT without copies around the transform**: `tsr_fft` stages its input through a buffer only when input and
  output overlap, and the new `tsr_fft_real` transforms real input directly (the plan's first step reads `x` as
  `(x, 0.0)`, so the conversion pass to complex128 goes away). `Fft::fft` / `ifft` call them out of place, `rfft` /
  `irfft` no longer copy their input or the inner result, and `Tessero\Ext` `Engine::fft` transforms out of place.
  `fft` 2^20 of real input 66 -> 49 ms (NumPy 52), complex 67 -> 51 ms (NumPy 47), `rfft` 37 -> 34 ms.
- **`var` / `std` without a row buffer**: the squared deviations are summed in 128-value blocks with the same split
  and rounding as the buffered sum, so the result is unchanged. 1M values 2.0 -> 1.1 ms (NumPy 2.9).
- **Reductions to a scalar** (`sum`, `prod`, `min`, `max`, `argmin`, `argmax`, `any`, `all`, `mean`, `var`, `std`
  with no axis, FFI backend): the same kernel call writes into a reused result slot instead of a new 0-d array with
  its own shape and stride descriptors, which was most of the cost of a small reduction. 10 values: `sum` 7.0 ->
  1.9 us (NumPy 1.0), `mean` 7.1 -> 1.6 us (NumPy 2.7), `var` 21.5 -> 1.7 us (NumPy 9.1).
- **FFT plan cache limit is a setting**: `tsr_set_fft_cache_limit` / `tsr_fft_cache_limit` / `tsr_fft_cache_bytes`,
  `Tessero::setFftCacheLimit()` / `fftCacheLimit()` / `fftCacheBytes()` and the same on `Tessero\Ext\Engine`. The
  library default rises to 256 MiB, enough to keep a Bluestein plan of about 1M points (a repeated 1,000,003-point
  `fft` 490 -> about 220 ms, NumPy 260). Web SAPIs get 32 MiB, since each php-fpm worker holds its own cache outside
  the memory budget (power-of-two transforms up to 2^20 points still fit); `TESSERO_FFT_CACHE_BYTES` overrides both,
  `0` turns caching off. See docs/operations/memory-and-threads.md.
- **`solve` / `cholesky` without LAPACKE's row-major copies**: `Tessero\Linalg\Linalg` and the kernel's
  `linalg.solve` / `linalg.cholesky` give LAPACK the column-major copy LAPACKE's row-major wrapper would build, made
  once with a tiled transpose, and skip its copy back. LAPACK sees the identical matrix, so results are unchanged.
  `solve` 600x600 6.1 -> 5.2 ms (NumPy 5.7), 2000x2000 197 -> 140 ms (NumPy 143); `cholesky` 2000x2000 111 -> 101 ms.

### Earlier in this cycle

- **Universal functions** `Tessero\Ext\Math`: 74 element-wise functions with
  NumPy broadcasting, type resolution and `out:` arrays.
  - Kernel family: bit-identical to the `NDArray` methods.
  - Loop family: `cbrt exp2 trunc arcsinh arccosh arctanh degrees radians erf erfc gamma lgamma signbit fmax fmin copysign nextafter logaddexp logaddexp2 fmod`.
  - Checked against NumPy/SciPy (64 cases).
- **Memory-mapped arrays** `NDArray::memmap($file, $mode, $shape, $dtype, $offset)`:
  the modes of numpy.memmap, shape inference, `flush()` / `flush(sync: true)`,
  and the mapping released after the last view.
- ADR-0009, guide pages *Universal functions* and *Memory-mapped arrays*.

### Security

From the memory-safety review (`docs/project/memory-safety-review.md`):

- **Element-count overflow (both backends, and C reshape).** A shape whose
  element count wraps around 64 bits could describe a small allocation as a
  huge array. Such shapes could come from `fromBytes`, from `unserialize()` of
  a cached or queued payload, from `zeros`/`empty`, or from `reshape`.
  Negative dimensions with a positive product were also accepted when
  unserializing. All sizes are now overflow-checked before allocating.
- **FFI `Generator::shuffle` on a read-only memory map** wrote through
  `PROT_READ` pages and crashed the process. It now throws.
- **Preload header in the shared temp directory** was written under a
  predictable name and reused if present. It is now created exclusively
  (0600) and deleted after loading.
- **Library version check.** The FFI package refuses a libtessero binary from
  a different version instead of risking mismatched layouts.
- **Released memory is never passed to C.** This covers `flush()` after a
  mapping is closed and LAPACK pointers during PHP's shutdown destructor pass.

### Fixed

- **Windows: the DLL exports only the C ABI.** MinGW exported every non-static function (1621 symbols, including
  vendored and internal ones); a `.def` file generated from `tessero.h` limits it to the 141 declared functions,
  and `make abi-check` compares the DLL's export table with the header (platform-debt PD-12).
- **Windows: BLAS thread control works.** `tsr_set_blas_threads` / `Tessero::setBlasThreads()` were a no-op on
  Windows (the lookup used `dlsym`); they now find OpenBLAS / MKL with `GetProcAddress`, and the `openblas_threads`
  feature is reported (PD-11).
- **`scipy.linalg.pascal` is O(n^2)**: every entry was a separate O(n) binomial product, so a large `n` ran for
  minutes (a CI fuzzing timeout). Entries now come from one binomial row per row or anti-diagonal, the same
  function `invpascal` uses, with identical values: `pascal(2000)` 12.3 s -> 0.19 s.
- **API reference and `tools/parity/api.json` regenerated**: they predated several C ABI additions and the I/O
  readers' size-limit parameters, which kept CI's "API reference is up to date" check failing.
- **Windows FFI loader: no `Calling FFI::new() statically is deprecated` notice** on every load
  (`Library::cdefBesideItsDlls` built its buffer with a static `FFI::new()`; it now uses the kernel32 FFI instance).
- **Registry routine errors now use Tessero's exception classes.** On both backends, an index error from a
  registry routine is `IndexError` (`IndexException` in the extension), and a dtype or type error is
  `DTypeError` (`DTypeException`); these used to be PHP's `ValueError`. FFI messages name the method in
  camelCase, as the extension's do (`Np::putAlongAxis`). A float given where an integer is required is a
  type error, as in NumPy.
- **`Np::unique`, the set operations, `quantile` and 0-d integer results lost precision above 2^53.** They are
  exact now.
- **`Np::searchsorted` accepted a sorted array that is not 1-D.** It now raises, as NumPy does.
- **`Np::bincount` rejected a PHP list of floats.** It now accepts one and converts it with `int()`, as NumPy
  does for a Python list.
- **Reference values depended on the generating machine.** On AVX-512 CPUs, NumPy's float64
  `exp`/`log`/`power`/`expm1`/`log1p`/`sinh`/`arcsin`/`cbrt` differ from the C library by an ulp on some
  inputs. All fixtures, including the older `parity.json` and `ufunc.json`, are now generated with those
  dispatch targets disabled (ADR 0012). Hand-set tolerances that only absorbed the difference were removed:
  random draws are compared exactly, and `cov`/`corrcoef`/`correlate`/`convolve` use the default tolerance.
- **`Math::cbrt` returns the C library's value**, as NumPy does outside AVX-512
  (`cbrt(27) = 3.0000000000000004`). The Newton correction that matched AVX-512 NumPy was removed.
- **Memory safety of the new numerical code** (an independent verification and a sanitizer fuzz pass over all
  539 registry functions, about 370 000 calls under AddressSanitizer and UBSan):
  - Negative trimming proportions read out of bounds, and could crash PHP (`trim_mean`, `trimboth`, `trim1`,
    and `levene`/`fligner` with `center='trimmed'`).
  - `lmoment` with a huge order read out of bounds.
  - `cumulative_sum`/`cumulative_prod` with `include_initial` other than 0/1 returned uninitialised memory.
  - `bincount` with ±inf indexed memory wildly.
  - C++ exceptions (allocation failures, oversized requests) terminated the PHP process instead of raising.
    Every C++ entry point is now guarded, including callbacks from the vendored QUADPACK and brentq.
  - Parameters that size an array now raise `MemoryError` beyond the memory budget, or the machine's RAM when
    none is set.
  - NaN integer parameters, oversized axis and bin values, and integer overflow in `fisher_exact` and
    `cramervonmises_2samp` are now handled; every float→int conversion is defined.
  - `tests/Unit/AdversarialInputTest.php` pins every case on both backends.
- **Where SciPy itself crashes, Tessero returns NaN; where SciPy raises, Tessero raises.** Mathieu functions for
  |q| > 2·10⁷ or m > 3·10⁸; `hyp2f1` for |c| ≥ 2⁵³ on its recurrence; `hyp1f1` with |b| ≥ 2⁵² on its integer-b
  path (these three crash SciPy). `levy_stable` with α = 1 far in the tail raises (SciPy: `RecursionError`). A
  Boost overflow raises (SciPy: `OverflowError`), for example in `ncf.pdf` and `ncf.isf`.
- **Integer inputs follow NumPy's int64 arithmetic.** `ptp`, `nansum`, `nanprod`, `nanmin`, `nanmax`, the
  cumulative functions, `partition`, `correlate` and `convolve` are exact above 2⁵³ and wrap on overflow as
  NumPy does. `correlate`/`convolve` return NumPy's integer dtype.
- **NumPy input validation:** `bincount` rejects float arrays and a non-integer `minlength`, `ptp` rejects
  booleans, and `average` checks weight shapes (`_weights_are_valid`).
- **Vendored SciPy code on inputs where SciPy itself fails** (`THIRD_PARTY_NOTICES.md`). The spheroidal wave
  functions' work arrays are sized for their loops (SciPy overruns them). `qstar` no longer reads before its
  array for m = 0. cdflib's non-central t sum (`nctdtridf`, `nctdtrinc`) stops only when it provably cannot
  terminate, instead of looping forever.


- **`sort()`/`argsort()` ignored kernel errors (both bindings).** When the
  memory budget ran out while the kernel allocated sort scratch space,
  the result came back unsorted (or with zeroed indices) instead of throwing.
  Both bindings now check every return code and throw `MemoryError` or
  `MemoryException`. The FFI sparse constructors and products
  (`tsr_dense_to_csr`, `tsr_csr_matvec`, `tsr_csr_matmat`) are now checked
  too. Regression tests are in `MemorySafetyTest` and
  `014-memory-safety.phpt` (finding F-9 in the memory-safety review).
- **Slice parser, found by fuzzing:**
  - A blank item (`"0, ,1"`) read past its comma, an out-of-bounds read in C.
    Blanks are now bounded, and `"1 , 2"` (blanks around commas) parses as it
    should; it was wrongly rejected before.
  - Steps near 2^62 overflowed the element-count and stride arithmetic, in C and
    in the FFI package's PHP slicer.
- FFI `NDArray` binary and unary operations with `out:` wrote straight into
  `out` even when `out` partially overlapped an input (a shifted view of the
  same buffer), so they read already-overwritten values. They now go through a
  temporary, as the extension does.
- `clone` of an FFI array now copies the data (it shared the buffer), matching
  the extension. Writing a clone no longer changes the original or a memory map.
- Multi-axis `mean`/`var`/`std` in the FFI package now merge the axes in
  ascending order whatever order they are listed in, so results do not depend
  on the listing order.
- `mean`, `var` and `std` compute exactly as NumPy does (two-pass pairwise) on
  both backends; `cbrt` corrects glibc's 1-ulp error on exact cubes.

### Verification record (reference machine: Linux x86-64, 2 cores, AVX-512, PHP 8.4.21, GCC 13, NumPy 2.4.4)

- C: 270 checks, optimised, ASan + UBSan, Clang without OpenMP.
- Fuzzing: about 13 million mutated inputs across the four targets under ASan +
  UBSan (standalone driver), clean after the two slice-parser fixes. libFuzzer
  itself could not be installed here (package downloads blocked); CI runs it.
- FFI suites: 706 tests in all three JIT modes (JIT off, function, tracing).
- Extension: 14 phpt files and 505 parity/ufunc cases, also under ASan +
  UBSan. Unit, extension and backend-parity suites together: 1 214 pass, 26
  FFI-only skipped. Preload end-to-end passes.
- Laravel bridge: 5 tests on both backends. Documentation examples: 59 checks
  with both backends.
- Soak (`tools/soak.php`): 30 minutes per backend, running concurrently.
  - Extension, JIT off: 785 608 iterations.
  - FFI package, tracing JIT: 475 522 iterations.
  - Result: native and mapped bytes back at 0 after every iteration; heap slope 0; RSS slope +13.7 / -5.2 KiB/min (thresholds 16 / 64); no descriptor growth. PASS.
  - An earlier 11-minute run of both backends was also clean.
- Not run: the Windows memory-map path (`csrc/src/mmap.c`, Win32 API, never
  compiled), the CI matrix, 24-72 h soaks.

## [0.2.0] - 2026-09-26

### Added

- **Native extension `ext-tessero`** (`Tessero\Ext\NDArray`, `Engine`, `Operand`):
  - Arrays: operators `+ - * / ** %`, slicing via `$a['…']`, boolean masks and
    integer-list indexing, `foreach` (native iterator), `count()`, `clone`, `==`.
  - Serialisation: `json_encode`/`toJson()` (byte-identical to `json_encode`
    with `JSON_PRESERVE_ZERO_FRACTION`), `serialize()`.
  - Maths: reductions along one axis, `matmul`, `sort`, `where`.
  - `Engine`: `linprog`, `milp`, MDP solvers, FFT and NumPy-identical `random`/`normal`/`integers`.
  - INI settings `tessero.threads`, `tessero.memory_budget`, `tessero.epsilon`,
    re-applied at every request start.
  - Kernel symbols are kept private (visibility + linker version script), so
    the extension coexists with the FFI library.
- **Linear and mixed-integer programming** in C: `Optimize\LinearProgramming::linprog`/`milp`
  with SciPy conventions (bounds, status codes, marginals), `LinprogResult`,
  `MilpResult::gap()`.
- **Markov decision processes** in C: `Mdp\MarkovDecisionProcess` with
  `fromDense`, `fromSparse`, `fromTransitions`, `valueIteration`,
  `policyIteration` (exact or modified), `evaluate`, `qValues`,
  `finiteHorizon`, `simulate`.
- **OpenMP threads** for large element-wise kernels and MDP sweeps
  (`Tessero::setThreads`, `Engine::setMaxThreads`). Results are identical for
  any thread count.
- `tsr_array` metadata in the ABI with a C slice parser, broadcasting helpers
  and a JSON writer. `NDArray::sliceNative()`, `toJson()`, `meta()`,
  `__serialize`/`__unserialize`.
- `tsr_reduce_mid` (fast reductions along non-last axes), `tsr_cumulative`,
  `tsr_pcg64_shuffle`, error codes -6, -7, -8.
- **Laravel package `tessero/laravel`**: service provider, `Tessero` facade,
  `AsNDArray`/`TensorCast` Eloquent casts (JSON or exact binary),
  `NumericArray` validation rule, `response()->ndarray()`, `tessero:doctor`,
  settings re-applied per Octane request/task and queued job.
- `tessero info --json`. `tessero doctor` reports the kernel, SIMD, OpenMP and
  the extension.
- Documentation site (MkDocs): user guide, operations manual (deployment,
  configuration, memory and threads, monitoring, troubleshooting, runbooks,
  security, upgrading, Docker), architecture (kernel, memory model, bindings,
  numerical methods, solvers, testing, 8 ADRs), generated API reference, C ABI
  and error codes. `tests/docs/examples.php` checks the numbers in the guides.
- Project files: `CONTRIBUTING.md`, `GOVERNANCE.md`, `CODE_OF_CONDUCT.md`,
  `SECURITY.md`, `RELEASING.md`, `MAINTAINERS.md`.
- Reference Docker image (`docker/Dockerfile`).
- Benchmarks for `linprog` and MDP solvers.

### Changed

- `NDArray` extends `Tessero\Internal\OperandBase`, which is `Tessero\Ext\Operand`
  when the extension is loaded, so FFI arrays gain operators.
- `NDArray` constructor-promoted properties are no longer `readonly` (needed by `__unserialize`).
- LP/MILP workspaces are allocated with `tsr_alloc` and count against the
  memory budget.
- MILP reports status 1 (limit) instead of pruning a subtree whose relaxation
  hit the iteration limit or numerical trouble, and stops with an error on
  out-of-memory instead of pruning.

### Fixed

- Signed integer overflow in int cumulative products, sums, products and
  int64 matmul is now defined two's-complement wrap-around (was UB, flagged by UBSan).
- Nelder-Mead produced wrong iterates under `opcache.jit=function` (PHP JIT
  miscompile; the loop counter now lives in a property; see
  `docs/project/upstream-bugs.md`, issue 4).
- Stacked `Linalg::det`/`slogdet` could return a wrong value under
  `opcache.jit=tracing` after a long warm-up (PHP JIT defect, not yet
  isolated). LAPACK pivots are now copied out of `CData` in one call (issue 5).

### Security

- LP/MILP reject NaN/±INF in `c`, `A`, `b` (and NaN bounds) instead of
  returning undefined results.
- MDP constructors and extension solvers reject non-finite rewards and
  probabilities and malformed CSR input (`indptr` not starting at 0,
  decreasing, or pointing past the data), which could previously read out of bounds
  in the extension.

### Removed

- The operator-only extension `tessero_ops` (replaced by `ext-tessero`).

### Verification record

Run on the reference machine (Linux x86-64, 2 cores, AVX-512, GCC, PHP 8.4.21
NTS, NumPy 2.4.4, SciPy 1.17.1):

- C: 173 checks, optimised and under ASan + UBSan.
- FFI package: unit, numerics, solvers, parity (258 cases) and fuzz (300)
  suites, with the JIT off, `function` and `tracing`.
- Extension: 10 phpt files and the extension parity suite (439 run, 26
  FFI-only skipped), also under ASan + UBSan.
- Laravel bridge tests on both backends. Documentation examples. FPM preload e2e.
- Not run here (no network access to GitHub/Packagist, no container daemon):
  the CI matrix on other platforms and PHP versions, the service provider
  inside a real Laravel application, and the Docker image build.

## [0.1.0] - 2026-09-26

### Added

- `libtessero` C kernel: allocator with budget, element-wise ops for 7
  dtypes with SIMD dispatch, pairwise reductions, radix sort, mixed-radix +
  Bluestein FFT, PCG64/SeedSequence/ziggurat RNG (NumPy-identical), CSR with
  CG/BiCGSTAB.
- FFI package: `NDArray` (views, slicing, broadcasting, NEP 50 promotion),
  `Linalg` (LAPACKE), `Fft`, `Random\Generator`, `Sparse\CsrMatrix`,
  `Optimize` (BFGS, L-BFGS-B, Nelder-Mead, brentq, newton, root, least
  squares, curve_fit), `Io\Npy`.
- FPM preload support, `tessero doctor`, parity and fuzz suites, benchmarks,
  CI configuration.
- Optional `tessero_ops` extension for operators.
