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
| F11 | `ndimage.find_objects`, `signal.firwin_2d` | np_numeric/scipy_signal | NaN/Inf argument cast to a size (UBSan, but drives OOB) |
| F15 | `ndimage.*` rank/morphology filters | `windowed_rank_buf` scipy_ndimage.c:398 | a 0-d input indexes `stride[-1]` |
| F36 | `distance.minkowski_distance` | spatial_distance.c:380 | a 0-d input indexes `shape[-1]` (same class as F15) |
| F16 F17 F19 F25 F28 F29 F32 | the whole `npoly.*` module | np_polynomial.c (cheb_integ:99, cheb_mulx:63, rb_div:684, r_lag2poly:1320, herm2poly, ...) | a coefficient-array length or integer order drives an allocation or index with no bound (one class) |
| F21 | `signal.lp2lp` | `sig_normalize` scipy_signal.c:1962 | over-long coefficient array |
| F23 | `special.mathieu_odd_coef` / `_even_coef` | `mathieu_coef_run` np_numeric.c:4318/4326 | an order ~4.6e18 indexes a fixed `[256]` buffer |
| F26 | `signal.peak_widths`, `signal.peak_prominences` | `peak_prom_core` scipy_signal.c:6579 | SEGV on a crafted peak index |
| F30 | `signal.zpk2sos` | scipy_signal.c | heap overflow |
| F35 | `signal.iirdesign` | `iid_iirdesign` scipy_signal.c:7351 | heap overflow in the filter-design pipeline |
| F38 | `signal.dbode` | scipy_signal.c | heap overflow |
| F39 | `signal.welch`, `signal.csd` | scipy_signal.c | `calloc` count*size overflow from a crafted segment length |

Out-of-bounds writes to a too-small result block (complex128 stored past the end):

| id | routine | site |
|----|---------|------|
| F12 | `signal.lp2bp_zpk` | scipy_signal.c:5480 |
| F14 | `signal.lp2hp_zpk` | scipy_signal.c:4293 |
| F22 | `signal.lp2bs_zpk` | scipy_signal.c:5507 |
| F37 | `signal.bilinear_zpk` | scipy_signal.c:4372 |
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
| F3  | `signal.stft` | leaks a 64-byte `tsr_alloc` block on the success path |
| F24 | `windows.tukey` | leaks a 64-byte `tsr_alloc` block |
| F18 | `stats.ecdf` | a 2-D input runs for minutes (no rank check) |
| F20 | `special.pbvv_seq` | a large `n` runs past the timeout (no cap) |
| F27 | `windows.chebwin` | a large length runs past the timeout |

Not a memory bug (noted for completeness):

| id | routine | note |
|----|---------|------|
| F6  | `np.loadtxt` / `np.genfromtxt` | `memcpy` from a NULL buffer when the file yields no values (np_io.c:246; UBSan nonnull) |
| F4  | `tsr_npy_write_header` | writes a header for a shape whose byte count overflows int64, which `tsr_npy_header` then rejects (writer/reader asymmetry, no overflow) |
