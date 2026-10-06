# Third-party code in Tessero

Tessero's own code is under the BSD 3-Clause License (`LICENSE`). The kernel (`csrc/`, copied into `ext/libtessero/` by
`tools/sync-ext.sh`) includes the code below, all under permissive licences. The licence texts ship next to
the code; binaries built from this repository must carry this file and those texts.

| Component | Version | Licence | Location | Used for |
|---|---|---|---|---|
| SciPy special-function glue: Boost wrappers, cdflib, `_cosine.c`, `wright.cc`, `xsf_wrappers.cpp` (real part), ellint_carlson | SciPy 1.17.1 | BSD-3-Clause (`csrc/third_party/scipy/LICENSE.txt`) | `csrc/third_party/scipy/` | `scipy.special` |
| SciPy QUADPACK (`scipy/integrate/__quadpack.c`) | SciPy 1.17.1 | BSD-3-Clause | `csrc/third_party/scipy/quadpack/` | `scipy.stats` generic moments, entropy, cdf |
| SciPy `brentq` (`scipy/optimize/Zeros/brentq.c`) | SciPy 1.17.1 | BSD-3-Clause | `csrc/third_party/scipy/zeros/` | `scipy.stats` generic ppf |
| xsf (SciPy's special-function library) | commit 0d0a593 | BSD-3-Clause (`csrc/third_party/xsf/LICENSE`) | `csrc/third_party/xsf/` | `scipy.special` |
| Boost.Math (the 178 headers used, standalone mode) | commit 5e088ff | Boost Software License 1.0 (`csrc/third_party/boost/LICENSE_1_0.txt`) | `csrc/third_party/boost/` | `scipy.special` (the functions SciPy takes from Boost) |
| NumPy random distributions (`numpy/random/src/distributions/*`) | NumPy 2.4.4 | BSD-3-Clause (`csrc/third_party/numpy/LICENSE.txt`, `csrc/third_party/numpy/random/LICENSE.md`) | `csrc/third_party/numpy/random/` | `numpy.random.Generator` methods, `rvs` of `scipy.stats` distributions |

Code written for Tessero that follows SciPy's or NumPy's Python/Cython code line by line (and therefore is a
derived work under their BSD-3-Clause licence) says so in its header: `csrc/third_party/scipy/cython_ports.hpp`
(the Cython inline functions behind `scipy.special` ufuncs), `csrc/cxx/stats_dist.*` and `csrc/cxx/dist_*.cpp`
(`rv_continuous`/`rv_discrete` and the distributions of `scipy/stats/_continuous_distns.py`,
`_discrete_distns.py`, `_levy_stable`), `csrc/cxx/stats_fn_*.cpp` (the `scipy.stats` functions),
`csrc/src/np_*.c` (NumPy functions and `Generator` methods).

## Local modifications

Every change to vendored code is marked in the source with `Tessero` (`grep -rn "Tessero" csrc/third_party`).

**Build glue (no change in results).**

- `numpy/include/numpy/npy_shim.h` (new) replaces `Python.h`, `npy_common.h` and `npy_math.h` for the random
  distributions; `numpy/random/distributions.h` includes it.
- `scipy/sf_error.h` (new) stands in for SciPy's error reporting, which raises Python warnings: errors are not
  reported (Tessero returns the same values SciPy returns with warnings ignored).
- `scipy/boost_special_functions.h`: no Python API; Boost evaluation errors return the value SciPy returns.
- `scipy/quadpack/quadpack.h` (new): the prototypes of `__quadpack.h` without the Python glue.
- `scipy/wright.cc`: the unused `Python.h` include removed. `scipy/gen_harmonic.h`: `<cinttypes>` added.
- `scipy/xsf_wrappers.h` (new): the subset of `xsf_wrappers.h` that `_cosine.c` needs.

**Changes on inputs where SciPy's own code is undefined or never finishes.** Outside those inputs the results
are unchanged; the parity fixtures verify that.

- `xsf/xsf/specfun/specfun.h`, spheroidal wave functions (`aswfa`, `cbk`, `qstar`, `rmn1`, `rmn2l`, `rmn2so`,
  `rmn2sp`, `rswfp`, `rswfo`, `segv`): the fixed work arrays (100 to 300 elements) are sized for the loop bounds
  (`TSR_SPH_BUF`); the originals overrun them for large `c` (heap corruption in SciPy). Define
  `TSR_XSF_ORIGINAL_BUFFERS` for the original sizes (used by `tools/parity/ub_scan.py` under AddressSanitizer).
  The registry additionally returns NaN for |m|, |n| > 1000 or finite |c| > 1000 (`guard:` in
  `spec/special.yaml`).
- `xsf/xsf/specfun/specfun.h`, `qstar`: for m = 0 the original reads `ap[-1]`; Tessero returns NaN.
- `xsf/xsf/specfun/specfun.h`, the spheroidal work arrays (`cbk`, `rmn2l`, `rmn2so`, `rmn2sp`, `rswfp`, `rswfo`,
  `segv`, ...): every buffer is value-initialised (`new T[n]()`, and `calloc(1, size)` for the `malloc(size)` of
  `sphd_wave.h` and `par_cyl.h`, with the original size expression). The originals leave some uninitialised, and
  for a few inputs a result then depends on whatever the allocator hands back (found in `obl_rad2_cv(10, 50,
  -1.5, -2.5, 2.5)`: -inf, or NaN under `MALLOC_PERTURB_=165`). With zero-filled buffers Tessero's result is
  deterministic; rows where SciPy's own result depends on the garbage are undefined in the fixtures
  (`docs/project/reference-deviations.md`).
- `xsf/xsf/cephes/hyp2f1.h`: when |c| ≥ 2^53, `c + aid == c` and the recurrence on c calls `hyp2f1` with the same
  arguments forever (SciPy overflows the stack and crashes); Tessero returns NaN on exactly that path.
- `boost/math/special_functions/hypergeometric_1F1.hpp`: for |b| ≥ 2^52 (and b = −∞) every `float_next(b)` step is
  still a negative integer, so the recursion never ends (SciPy crashes); Tessero returns NaN on exactly that path.
- `scipy/boost_special_functions.h`, `user_overflow_error`: SciPy sets Python's `OverflowError` and returns 0;
  Tessero records the same message as the call's error, and the registry raises it after the element loop.
- `scipy/cdflib.c`, `cumtnc` (non-central t, used by `nctdtridf` and `nctdtrinc`): SciPy's forward sum never ends
  when the partial sum stays negative or is NaN. The sum is stopped only when it provably cannot end (the rest of
  the series is bounded by 2(|d| + |e|)/(1 − λ/i) and the partial sum lies below minus that bound), with 10⁹
  terms as a backstop; a thread-local flag (`tsr_cdflib_runaway`) makes the rest of that evaluation return at
  once, and the wrappers in `cython_ports.hpp` return NaN. Marked `capped:` in `spec/special.yaml`; the inputs
  are listed in `tools/parity/crashes-special.json`.
