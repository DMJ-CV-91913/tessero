# Fuzzing

Fuzz targets for the kernel's untrusted-input entry points. Each target is an `LLVMFuzzerTestOneInput`
(`fuzz/fuzz_<name>.c`) that CI builds under clang libFuzzer (`make -C csrc fuzz`) and that builds everywhere
under the standalone ASan+UBSan mutation driver (`make -C csrc fuzz-standalone`, `fuzz/driver.c`). Corpora live
in `fuzz/corpus/<name>/` and are replayed first so a crash in a committed seed is caught deterministically.

## Targets

`FUZZ_TARGETS` (no function registry, built with `-DTSR_NO_REGISTRY`):
`slice npy mmap_plan json npy_write`.

`FUZZ_REG_TARGETS` (drive public routines through `tsr_fn_routine`; link the whole kernel, C + C++ + LAPACK):
`einsum hist textio dtype routines`.

| target | entry points |
|--------|--------------|
| `npy_write` | `tsr_npy_write_header` + round-trip through `tsr_npy_header` |
| `einsum` | `np.einsum` (subscripts + 1-3 small operands) |
| `hist` | `np.histogram`, `np.histogram_bin_edges`, `np.histogram2d`, `np.histogramdd` |
| `textio` | `np.loadtxt`, `np.genfromtxt`, `np.fromfile` (content in a private temp file) |
| `dtype` | `.npy` descr parser, `io_dtype`, `np.promote_types`/`can_cast`/`result_type` dtype tables |
| `routines` | every registry routine (slot from the first bytes), fuzzed arg kinds; file-opening routines denied |

The shared harness helpers are in `fuzz/fuzz_common.h`: a byte reader (edge-value doubles: NaN, Inf, huge,
subnormal, size-like), `tsr_arg` builders backed by exact-size heap blocks (so ASan sees overreads), and the
oracle -- every result is well formed, freed once, and `tsr_allocated()` returns to where it started (no leaked
kernel allocation, including on error paths). A 256 MiB `tsr_set_budget` makes size-like arguments fail cleanly.

## Determinism and reproducers

The standalone driver is deterministic: corpus files load in sorted order, mutation comes only from the seeded
generator, so `-seed=N -runs=M` over a fixed corpus replays the same inputs. It prints the seed, replayed-corpus
count, run count, elapsed time and execs/s. `-timeout=SEC` caps a single input; `-artifact_prefix=P` writes the
crashing or timing-out input to `P{crash,timeout}-<run>` before the process dies, and `./fuzz_<t> <file>` replays
exactly that input. CI should pass the same `-seed` it records.

`TSR_FUZZ_ONLY=npoly.,signal.` restricts the `routines` target to routines whose names start with one of the
comma-separated prefixes (slot numbers then index the reduced table, so a saved input replays with the same setting).

`FUZZ_DEFS=-DFUZZ_SKIP_KNOWN` steers the registry targets around the open findings below (`FUZZ_SKIP_KNOWN`
blocks in the harnesses plus `fuzz/known_ubsan.supp`) so a long run keeps exploring past them. The default build
(and CI without `FUZZ_DEFS`) reports them. Each entry is removed once the maintainer's fix lands.

Throughput measured on the shared host (gcc 13, ASan+UBSan, nice -19, 1 thread): `npy_write` ~120k execs/s,
`dtype` ~100k, `einsum`/`hist` 3-15k, `textio` ~6k (temp-file I/O per input), `routines` ~20k.

## Open findings

Bytes are hex; run `printf '<hex>' | xxd -r -p > in && ./fuzzsa_routines in` (default routine table, no
`TSR_FUZZ_ONLY`). Found by the batch A campaigns in files outside batch A, not fixed there:

| id | routine | site | reproducer |
|----|---------|------|------------|
| A-N1 | `interpolate.splantider` | `r_splantider` scipy_interpolate.c:409 (heap read past the coefficients; use-after-free in another input) | `c2020107bd010480313a418055422dc24ee135b9` |
| A-N2 | `hierarchy.fcluster` | `incon_collect` scipy_cluster_hier.c:228 (heap read out of bounds) | `bc030107ea010480418042804380440000000000` |

The float->int cast class (F11) is closed in the batch A and batch B files; `fuzz/known_ubsan.supp` keeps it
suppressed under `FUZZ_SKIP_KNOWN` per source file for every other file (it no longer has a blanket
`float-cast-overflow:*`), so a cast in a fixed file is fatal in every run.

The `routines` harness remains productive: these are the classes found so far, not an exhaustive audit of the
registry routines. Most are one systemic gap -- a public routine trusts an argument (a size, an order, a
coefficient-array length, a rank, a NaN/Inf number) without validating it before using it to index or allocate.

## Fixed findings

Batch B (`np_polynomial.c`, `np_poly.c`, `scipy_signal.c`). Each reproducer is committed as a `routines` corpus file
(`fuzz/corpus/routines/<id>-*`) and now replays clean; its `FUZZ_SKIP_KNOWN` entry is gone. The systemic
float->int cast of numeric arguments (F11's class) is closed in these three files: `scipy_signal.c` converts every
integer argument through `sig_i64` / `sig_i32` (NaN, the infinities and out-of-range values become a sentinel below
every lower bound the routines check) and bounds sizes taken from arguments with `sig_alloc_ok` before they reach an
allocation; the polynomial modules check orders and degrees with `npoly_int` (an integer, NumPy's TypeError
otherwise). `fuzz/known_ubsan.supp` keeps its `float-cast-overflow:*` line for the files of the class that are still
open; the batch B campaigns ran with a suppression list that leaves these three files out, so a cast there is fatal.

| id | routine(s) | was | now |
|----|-----------|-----|-----|
| F16 F17 F19 F25 F28 F29 F32 | `npoly.*` (chebint, chebmulx, lag2poly, chebdiv, chebmul, hermeint, herm2poly, ...) | an empty coefficient series, a NaN / huge order or degree, mismatched 2-D/3-D points indexed or sized buffers out of bounds | NumPy's errors ("Coefficient array is empty", IndexError, "... must be an integer"); orders whose quadratic work or memory is out of reach are refused |
| F12 F14 F22 F37 | `signal.lp2bp_zpk` / `lp2hp_zpk` / `lp2bs_zpk` / `bilinear_zpk` (and `lp2lp_zpk`) | more zeros than poles: negative degree, complex stores past the zero buffer | SciPy's "Improper transfer function. Must have at least as many poles as zeros." |
| F30 | `signal.zpk2sos` | N-d root arrays walked with the first stride only | read in C order through every stride (`read_carr`); unpairable roots refused |
| F21 | `signal.lp2lp` (and lp2hp/lp2bp/lp2bs, normalize, tf2zpk, tf2sos, freqs, bode, group_delay, freqz) | an empty `b` or `a` read `a[0]` / sized `calloc(-1)` | refused |
| F35 | `signal.iirdesign` | `wp`/`ws` read as raw doubles at `data[0]` | read through dtype and strides; SciPy's one-or-two-elements, same-shape rule |
| F38 | `signal.dbode` | a 0-d or empty `num`/`den` indexed `shape[0] - 1` | refused |
| F39 | `signal.welch`, `csd`, `coherence`, `spectrogram`, `stft` | a negative / NaN `nperseg` reached `calloc` | SciPy's segment checks (`nperseg` >= 1, `noverlap` < `nperseg`, `nfft` >= `nperseg`) and a size bound |
| F26 | `signal.peak_widths`, `peak_prominences` | a peak index outside `x` dereferenced | SciPy's "peak ... is not a valid index for `x`"; `rel_height` < 0 refused |
| F3  | `signal.stft` | the input copy (a `tsr_alloc` block) released with `free()`: 64 bytes leaked per call | released with `fn_free_doubles` |
| F24 | `windows.tukey` | the alpha >= 1 (Hann) leak, already fixed by the bugfix stream; the reproducer is the same path | committed as corpus; NaN `alpha` and a NaN / huge `M` now refused |
| F27 | `windows.chebwin` | O(M^2) direct cosine sum (minutes at M = 65536) | the FFT form SciPy uses above M = 4096 (the direct sum, unchanged, below) |
| F11 (part) | `signal.firwin_2d` | NaN `hsize` cast to int64 | checked conversion and a size bound |

Found by the batch B campaigns themselves (`TSR_FUZZ_ONLY` restricted to these modules, no skip entries), fixed in
the same change and committed as `fuzz/corpus/routines/N*` (slot bytes rewritten to the default table):

| id | routine | was | now |
|----|---------|-----|-----|
| N1  | `signal.cspline1d` (and `symiirorder1`) | an empty signal wrote `out[-1]` | empty result / SciPy's "did not converge" |
| N2  | `signal.symiirorder2` | fewer than two samples read `x[1]` | SciPy's "did not converge" |
| N3  | `signal.envelope` | `n_out` > len(z) read the spectrum past its end | zero-padded as `irfft(Z, n=n_out)`; band and `n_out` checks |
| N4  | `signal.findfreqs` | a 1-slot pole buffer leaked when there were no poles (LeakSanitizer at exit: no input saved, so a C test only) | released |
| N5  | `windows.taylor` | `nbar` cast unchecked (9.3e18); quadratic work | checked; work bound |
| N6  | `signal.medfilt` / `medfilt2d` | kernel sizes cast unchecked, `prod(kernel)` bytes allocated | checked; the zero-padded box is never materialised |
| N7  | `signal.remez` | band edges outside [0, fs/2] sized the grid; a one-point grid overflowed the candidate list | edges validated; list sized for both endpoints |
| N8  | `signal.find_peaks` | a uint8/bool condition array read 8 bytes per element | read by dtype |
| N9  | `windows.dpss` | M = 4096 ran a full O(M^3) eigensolve past the timeout | only the Kmax eigenpairs above 1024 points |
| N10 | `signal.filtfilt` / `lfilter_zi` | an empty `b` or `a` indexed `b[0]` | refused |
| N11 | `signal.convolve` | an empty `v` in 'same' mode read past the result | SciPy's "v cannot be empty" |
| N12 | `signal.correlate` (N-d convolve/correlate) | an empty `in2` in 'same' mode read past the result | SciPy's "second array argument cannot be empty" |
| N13 | `windows.dpss` | M = 4096 with Kmax = 4096 ran inverse iteration for every vector past the timeout | refused above Kmax * M^2 = 2^32 (M > 1024) |

Batch A (`np_numeric.c`, `np_io.c`, `np_sets.c`, `scipy_ndimage.c`, `scipy_sparse.c`, `scipy_fft.c`,
`spatial_distance.c`, `mmap.c`, `stats_fn_desc.cpp`, `stats_fn_tests.cpp`). Regression checks in
`tests/test_main.c` (`test_fuzz_batch_a`); reproducers committed as `fuzz/corpus/<target>/regress_F*` (crafted
anew where the recorded input no longer reached the bug with the current harness) and replaying clean; their
`FUZZ_SKIP_KNOWN` blocks (incl. the special.* integer clamp and the ndimage 0-d skip) and suppressions are gone.
Integer-valued arguments go through the checked conversions in `src/fn.h` (`fn_arg_i64` with an error,
`fn_dbl_i64` / `fn_num_i64` where the caller range-checks): no float->int cast of an unchecked double remains in
these files.

| id | routine(s) | was | status: behaviour now (as SciPy / NumPy) | corpus |
|----|-----------|-----|------------------------------------------|--------|
| F1  | `np.fromfile` text, integer dtype | NaN/huge value cast to the integer type | fixed: integers parsed as integers (`300` -> 44 in uint8), `2.7` / `nan` raise "could not be read to its end due to unmatched data" | textio/regress_F1_fromfile_int_nan |
| F2  | `special.ellip_harm` / `_2` / `ellip_normal` | `p < 1` indexed `z[-1]` | fixed: NaN unless `n >= 0` and `1 <= p <= 2n+1` | routines/regress_F2_* |
| F4  | `tsr_npy_write_header` | wrote headers the reader refuses | fixed: refuses (TSR_ESHAPE) a dimension >= 9223372036854775800 or a byte count beyond int64; the npy_write oracle now requires every written header to parse | npy_write/regress_F4_int64max_dim |
| F5  | `np.einsum` | a 0-length label bound to a longer axis read out of bounds | fixed: sizes 0 and n (n != 1) are a broadcast error; 0 against 1 gives 0 | einsum/regress_F5_zero_axis |
| F6  | `np.loadtxt` / `np.genfromtxt` | `memcpy` from NULL for an empty file | fixed: no copy of 0 values | textio/regress_F6_genfromtxt_empty |
| F7  | `np.histogram` / `_bin_edges` | `(bins+1)*8` wrapped | fixed: "array is too big" | hist/regress_F7_bins_overflow |
| F8  | `np.einsum` | non-contiguous dots misparsed (`olab` unset); a short output `.` read past the string | fixed: NumPy's "'.' that is not part of an ellipsis" error | einsum/regress_F8_* |
| F9  | `np.histogram` estimator bins | `ceil((last-first)/width)` cast unchecked | fixed: NaN / infinity raise as `int()` does, beyond int64 "array is too big" | hist/regress_F9_estimator_quotient |
| F10 | `np.fromfile` | offset/count NaN cast; `size - offset` overflow | fixed: integral offset/count required, negative offset "Invalid argument", past the end "negative dimensions are not allowed" | textio/regress_F10_* |
| F11 (part) | `ndimage.find_objects` | NaN label cast to int64 | fixed: NaN / out-of-range labels are background; max_label bounded | routines/regress_F11_find_objects_nan |
| F13 | `stats.boxcox_normplot` / `yeojohnson_normplot` / `probplot` | empty sample offset a NULL pointer | fixed: empty results (`probplot`: r = 0.0); `N` integral and >= 0, `lb > la`, positive data for Box-Cox | routines/regress_F13_* |
| F15 | `ndimage` rank/morphology filters | 0-d input indexed `stride[-1]` | fixed: a 0-d input is one element (filters return it, morphological_laplace 0); window volume and origin checked | routines/regress_F15_* |
| F18 | `stats.ecdf` | a NaN never advanced the tie loop (minutes) | fixed: 1-D only, NaN refused | routines/regress_F18_ecdf |
| F20 | `special.pbvv_seq` / `pbdv_seq` | one recurrence per order (quadratic); `(int)v` of NaN/1e300 | fixed: one xsf recurrence as SciPy's `_specfun` call; NaN / infinity / too large refused | routines/regress_F20_* |
| F23 | `special.mathieu_odd_coef` / `_even_coef` | an order ~1e18 indexed `fc[256]` | fixed: at most 251 coefficients, `m` integral and in int range, NaN/inf predictor raises | routines/regress_F23_mathieu_odd_coef_1e18 |
| F31 F33 | `sparse.eye` / `eye_array` / `identity` / `diags` / `spdiags` | dense m*n intermediate, product overflow | fixed: CSR built directly (no dense matrix), dimensions within int32 indices, offsets/shape validated, offsets count must match | routines/regress_F31_*, regress_F33_* |
| F34 | `stats.chisquare` / `power_divergence` | length 1 against an empty side read the empty side | fixed: broadcasts to length 0 (statistic 0, p NaN) | routines/regress_F34_chisquare |
| F36 | `distance.minkowski_distance` | 0-d input indexed `shape[-1]` | fixed: 0-d and length-1 axes broadcast; mismatched rows refused (they were read past y) | routines/regress_F36_minkowski_distance_0d |
| F40 | `scipyfft.fht` | 0-d input indexed `shape[-1]` | fixed: refused (SciPy raises IndexError) | routines/regress_F40_fht_0d |
| CI | `linalg.svd` / `slinalg.svd` / `linalg.lstsq` (np_linalg.c) | dgesdd / dgelsd looped on a matrix with NaN or infinity (CI libFuzzer hang) | fixed: the SVD drivers check the input and report NumPy's "SVD did not converge" (also procrustes) | routines/regress_svd_nonfinite_hang |

Found by the batch A campaigns themselves in the batch A files (routine slot restricted to the batch A routines,
`FUZZ_SKIP_KNOWN`), fixed in the same change and covered by the campaign corpus replay:

| routine(s) | was | now |
|-----------|-----|-----|
| `np.copyto` / `put` / `putmask` / `place` / `fill_diagonal` / `put_along_axis` | a number as the destination: stored through a NULL `arr.data` | "must be an array" |
| `special.erf_zeros` / `fresnel*_zeros`, `pbdn_seq` | fixed `[260]` / `[300]` stack buffers overflowed | heap buffers; nt <= 1000 (deflation is at least quadratic) |
| `special.bernoulli` / `euler` | quadratic recurrence (minutes at n = 65536) | SciPy's specfun bernob / eulerb (linear) |
| `special.mathieu_*_coef` | specfun's int `m * m` overflowed (m > 46340); a huge q made fcoef index `fc[km - 1]` wildly | refused / fcoef's 251 NaNs |
| `special.*_zeros` (Bessel, Kelvin, Airy) | scans from x = 0 for large orders; full scans for impossible counts | start below the first zero, value-only scan, nt bounds |
| `special.multigammaln`, `np.newton_cotes`, `stats.lmoment` | unbounded loops on a large d / rn / order | SciPy's a > (d-1)/2 check and bounds; lmoment skips the NaN-overwritten moments |
| `np.round` (negative decimals), `np.unpackbits`, `special.factorial2`, int sums | float->int casts of NaN / 2^63 | checked |
| `np.gradient` (edge_order < 1), `np.polyint` (2-D k), `integrate.cumulative_simpson` (empty axis, mismatched x) | read past the input | NumPy / SciPy errors |
| `stats.binomtest`, `cumfreq` / `relfreq`, `multivariate_normal` / `_t`, `dirichlet`, `multinomial`, `matrix_normal`, `wishart`, ... | k / n beyond int64 cast; NaN bin index; argument shapes never compared (read past x) | checked |
| `ndimage` labels / masks / origins / sizes, geometric transforms, EDT, watershed, find_objects | short labels and masks read past their end; origins drove index overflow; NaN coordinates cast; an empty input sampled `in[0]`; sampling 0 underflowed the EDT envelope; negative watershed input indexed the cost queue | validated (SciPy's "invalid origin", "input and labels must have the same shape", ...); mirror/reflect folded in O(1) |

