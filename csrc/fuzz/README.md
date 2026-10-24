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

All found by this workstream; reproducers are NOT committed (they still crash). Bytes are hex; run
`printf '<hex>' | xxd -r -p > in && ./fuzzsa_<target> in`. "cast/overflow" items are UBSan reports (integer
overflow or float->int out of range on an unchecked argument cast); the others are ASan memory errors.

The `routines` harness remains productive: these are the classes found so far, not an exhaustive audit of the
1095 registry routines. Most are one systemic gap -- a public routine trusts an argument (a size, an order, a
coefficient-array length, a rank, a NaN/Inf number) without validating it before using it to index or allocate.
Each `FUZZ_SKIP_KNOWN` entry is removed as its fix lands, so the next long run surfaces the next routine.

ASan memory errors (read/write out of bounds; the highest severity):

| id | routine(s) | site | trigger |
|----|-----------|------|---------|
| F2  | `special.ellip_harm` / `_2` / `ellip_normal` | `ellip_lame` np_numeric.c:4460 | `p < 1` makes `tp-1` index `z[-1]` |
| F5  | `np.einsum` | `r_einsum` np_numeric.c:3687 | a label on a 0-length axis and a longer one is read out of bounds |
| F8  | `np.einsum` | `r_einsum` np_numeric.c:3630 | dots that are not one contiguous `...` per term misparse, leaving `olab` uninitialized (nondeterministic) |
| F11 | `ndimage.find_objects` | np_numeric | NaN/Inf argument cast to a size (UBSan, but drives OOB) |
| F15 | `ndimage.*` rank/morphology filters | `windowed_rank_buf` scipy_ndimage.c:398 | a 0-d input indexes `stride[-1]` |
| F36 | `distance.minkowski_distance` | spatial_distance.c:380 | a 0-d input indexes `shape[-1]` (same class as F15) |
| F23 | `special.mathieu_odd_coef` / `_even_coef` | `mathieu_coef_run` np_numeric.c:4318/4326 | an order ~4.6e18 indexes a fixed `[256]` buffer |

Out-of-bounds writes to a too-small result block:

| id | routine | site |
|----|---------|------|
| F13 | `stats.boxcox_normplot` / `yeojohnson_normplot` / `probplot` | `stat_osm` stats_fn_desc.cpp:2096 (offset on a NULL pointer) |
| F34 | `stats.chisquare` / `power_divergence` | `stats_fn_tests.cpp:920` (NULL `f_exp` dereference) |

UBSan integer/float-cast reports (an argument cast to int/int64 without a range check; the standalone driver treats
these as fatal, suppressed under `FUZZ_SKIP_KNOWN` by `fuzz/known_ubsan.supp`):

| id | routine | site |
|----|---------|------|
| F1  | `np.fromfile` (text, integer dtype) | `io_store` np_io.c:107 |
| F7  | `np.histogram` / `_bin_edges` | `hist_edges` np_sets.c:885 (`(bins+1)*8` wraps) |
| F9  | `np.histogram` with an estimator bins | `hist_edges` np_sets.c:859 (`(last-first)/width` -> int) |
| F10 | `np.fromfile` | `r_fromfile` np_io.c:127/134 (offset/count: NaN cast and `size - offset` overflow) |
| F31 F33 | `sparse.diags` / `eye_array` / `eye` / `spdiags` | scipy_sparse.c:349/472 (dimension product overflows) |

Leaks and unbounded work:

| id | routine | note |
|----|---------|------|
| F18 | `stats.ecdf` | a 2-D input runs for minutes (no rank check) |
| F20 | `special.pbvv_seq` | a large `n` runs past the timeout (no cap) |

Not a memory bug (noted for completeness):

| id | routine | note |
|----|---------|------|
| F6  | `np.loadtxt` / `np.genfromtxt` | `memcpy` from a NULL buffer when the file yields no values (np_io.c:246; UBSan nonnull) |
| F4  | `tsr_npy_write_header` | writes a header for a shape whose byte count overflows int64, which `tsr_npy_header` then rejects (writer/reader asymmetry, no overflow) |

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

