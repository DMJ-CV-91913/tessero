# 0011. One function registry in the kernel, interpreted by both bindings

- Status: Accepted
- Date: 2026-09-28

## Context

Closing the NumPy/SciPy gap means adding several hundred functions: the `scipy.special` ufuncs, NumPy's
statistics and set routines, the `Generator` methods, and the `scipy.stats` distributions and tests. Written by
hand in two bindings, as the earlier functions were, each function would cost a C implementation, an FFI
wrapper and a Zend wrapper. Each copy would also be a place for the two backends to disagree.

## Decision

1. **The kernel holds a table of callable functions** (`csrc/src/fn.h`, `fn_core.c`, `fn_tables.c`). Each entry
   names the function (`special.gammainc`, `stats.levene`, `random.normal`), its kind, its arguments with
   defaults and enums, its outputs and its documentation. The kinds are:
   - `FN_UFUNC`: element-wise, with broadcasting.
   - `FN_GUFUNC`: a reduction or other core operation over axes, with NumPy's `axis`, `keepdims` and
     `nan_policy`.
   - `FN_DIST`: a `scipy.stats` distribution, whose methods run on SciPy's `rv_continuous`/`rv_discrete`
     semantics in `csrc/cxx/stats_dist.*`.
   - `FN_ROUTINE`: arbitrary arguments and several results.
   - `FN_RANDOM`: a `Generator` method on the caller's PCG64 stream.
2. **Both bindings interpret the table at run time.** The kernel describes itself as JSON. The extension
   registers `Tessero\Ext\Special`, `Stats`, `Np` and `Random\Generator` methods with arginfo from that
   description at MINIT. The FFI package calls through `Tessero\Native\Registry`. Its façade classes
   (`Tessero\Special`, ...) are generated from the same description by `tools/parity/gen-facades.php`, so IDEs
   and reflection see real methods; a freshness check runs in the gate.
3. **Special functions come from the code SciPy calls** (xsf, Boost.Math and SciPy's glue, vendored), and the
   element loops are generated from `spec/special.yaml` by `tools/parity/gen_registry.py`. The spec carries
   only what cannot be read from the code: the reference function, domains for the fixtures, `guard:`/
   `capped:` for inputs where SciPy fails, and tolerances with reasons.
4. **Verification is per registry entry.** A function counts as covered on a backend only when its generated
   NumPy/SciPy fixture passes there (`tools/parity/run-tests.php`, `report.py`).

## Consequences

- Adding a function means one kernel entry. Both backends get it with identical results, because the same code
  runs.
- The extension's methods are not in a stub file; `tools/api-dump.php` reads them by reflection.
- The kernel needs a C++17 compiler (xsf and Boost are C++): the FFI binary links libstdc++ statically, and the
  extension uses `PHP_REQUIRE_CXX`.

## Alternatives considered

- *Generate C bindings for both backends from the spec*: more generated code to review, with the same
  guarantee.
- *Keep writing functions by hand*: at this scale, two implementations would drift.
