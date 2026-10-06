# Claude Code prompt: the Tessero NumPy/SciPy parity program

> **How to run it**
>
> 1. Clone the repository on a Linux or macOS machine or VM. You need PHP
>    8.2+ with `ffi`, the PHP headers (`phpize`), gcc or clang, Python 3.11+,
>    OpenBLAS/LAPACKE, Composer and `mkdocs-material`.
> 2. Create a branch: `git switch -c parity-program`.
> 3. Start Claude Code in the repository root. Paste everything below the line
>    as the first message. For an unattended run, use a disposable container
>    or VM and grant edit and shell permissions up front (for example
>    `claude --permission-mode acceptEdits` with Bash allowed in
>    `.claude/settings.json`).
> 4. The program is long: expect several days of agent time and many context
>    compactions. It is built to resume. If a session ends, start a new one
>    with: *"Resume the parity program: read CLAUDE.md and .work/ledger.md,
>    then continue."*

---

You are the lead engineer of **Tessero**, a numerical computing platform for
PHP. It has a C11 kernel (`csrc/`, `libtessero`), an FFI package (`src/`,
namespace `Tessero\`), a Zend extension (`ext/`, namespace `Tessero\Ext\`)
and a Laravel bridge (`laravel/`). NumPy and SciPy are the specification
(ADR 0002).

Your mission is to take Tessero from its current coverage to a defensible,
measured, near-complete implementation of the in-scope NumPy and SciPy
surface, on **both** backends. The results must be verified against NumPy
and SciPy by generated fixtures. The coverage figures must come from the
code and the tests, never from prose.

Work **methodically and without stopping** until the definition of done in
section 9 is met. Do not ask the user questions: decide, record the
decision, and continue.

## 0. Where you start (measured on 0.2.0)

- **Curated inventory** (`docs/project/numpy-scipy-coverage.md`, generated
  by `tools/api-dump.php` and `tools/gap_inventory.py`): 169 rows. The FFI
  package covers 79 (47 %) and the extension 48 (28 %). SciPy rows are 27 %
  (FFI) and 8 % (extension).
- **Symbol census** (NumPy 2.4.4 / SciPy 1.17.1 public names):
  - `numpy`: 497 top-level names, plus `linalg` 32, `fft` 18 and
    `random.Generator` 45 methods.
  - `scipy.stats`: 110 continuous and 21 discrete distributions, plus 183
    other names.
  - Other SciPy modules: `special` 359, `linalg` 112, `optimize` 82,
    `sparse` 69, `sparse.linalg` 43, `sparse.csgraph` 26, `interpolate` 63,
    `signal` 158, `integrate` 38, `spatial` 24, `ndimage` 75, `fft` 41.
- **The extension lags the FFI package.** It has no linalg, full FFT,
  stateful Generator, sparse, optimize or `.npz`, and it lacks about 20
  NDArray methods.
- **Existing assets:**
  - the kernel ufunc registry (`csrc/src/ufunc.c`, `tsr_ufunc_*`);
  - shared mmap and `.npy` code (ADR 0010);
  - fixture generators (`tests/fixtures/generate_*.py`);
  - the test suites `BackendParityTest`, `PropertyTest` and
    `MemorySafetyTest`;
  - the fuzz targets (`csrc/fuzz`), the soak harness (`tools/soak.php`) and
    the beta issue list (`.github/beta-issues.json`).

Read these before writing code:

- `docs/architecture/overview.md`, `c-kernel.md`, `memory-model.md`,
  `native-extension.md` and `ffi-binding.md`;
- ADRs 0002, 0003, 0004, 0009 and 0010;
- `docs/project/numpy-scipy-gap-analysis.md`, `open-issues.md` and
  `memory-safety-review.md`;
- `csrc/include/tessero.h`, `csrc/src/ufunc.c`, `src/Math.php` and
  `ext/tessero_ufunc.c`.

Match the existing style, error-code conventions (negative `TSR_E*` mapped to
typed exceptions), memory-budget rules and documentation voice.

## 1. Operating rules

1. **Autonomy.**
   - Never pause for confirmation. When a design choice is open, pick the
     option that is most NumPy/SciPy-compatible and simplest to verify.
   - Record the choice in `.work/decisions.md` (one line each). Write an ADR
     in `docs/architecture/adr/` when the choice is architectural.
2. **Integrity rules.** Breaking any of these invalidates the program:
   - A function counts as covered only when a generated fixture test
     compares it with NumPy/SciPy and passes on **both** backends.
   - Never loosen a tolerance, skip a test, delete a fixture or edit
     generated files to make something pass. If the accuracy target is
     wrong, change it in the registry with a written justification and the
     measured error.
   - Never mark a symbol out of scope to raise a percentage. Exclusions come
     only from the fixed categories in section 3.3, and each needs an entry in
     `tools/parity/scope.yaml`.
   - Never state a coverage number in the docs or README by hand. Numbers
     are generated into `docs/project/_generated/metrics.md` and included.
3. **Stuck policy.**
   - An item that still fails after three focused attempts is marked
     `BLOCKED` in the ledger with the error, what you tried and the next
     idea, and an entry is added to `.github/beta-issues.json`. Then move to
     the next item.
   - Return to blocked items at the end of each phase.
4. **No regressions.** The full gate (section 8) must be green at the end of
   every phase. Performance regressions over 15 % against `bench/` baselines
   need a fix or a recorded decision.
5. **Licensing.**
   - You may port or adapt permissively licensed algorithms, with
     attribution in `THIRD_PARTY_NOTICES.md` and in the source header:
     - Cephes, which SciPy distributes under BSD;
     - Boost.Math (BSL-1.0);
     - the netlib public-domain codes: QUADPACK, MINPACK, FITPACK, ODEPACK
       and ARPACK (BSD);
     - pocketfft (BSD);
     - HiGHS (MIT);
     - PROPACK (BSD);
     - Shewchuk's predicates (public domain);
     - the published papers.
   - **Never** copy GPL or LGPL code (R's `nmath`, GSL, GNU Octave) or
     Numerical Recipes code. Reimplement from the papers instead.
6. **Commits.**
   - One commit per completed ledger item on the current branch, as
     `feat(stats): add ttest_ind (Welch) [PAR-3.41]` or similar.
   - Never push, rewrite history or touch `main`.
7. **Determinism.**
   - Results must not depend on the thread count. Keep a fixed reduction
     order, as the existing kernels do.
   - RNG-based methods must use `Tessero\Random\Generator` (PCG64) and
     reproduce NumPy's algorithms, so seeded draws are bit-identical to
     `numpy.random.default_rng(seed)`.
8. **Memory safety.**
   - Every allocation goes through `tsr_alloc` (budgeted). Check every
     kernel return code in both bindings; F-9 was this bug.
   - Size arithmetic uses `tsr_shape_size`/`checked_count`.
   - New parsers get a fuzz target.

## 2. Resumability: do this first, before any feature work

1. Append a section **"Parity program (active)"** to `CLAUDE.md`. It holds
   the operating rules above in five lines, the ledger location, the
   per-item loop from section 7, and the gate commands. Claude Code reloads
   `CLAUDE.md` after every compaction, so this is how you keep your place.
2. Create `.work/ledger.md`. It lists every item from section 6 with a
   stable ID (`PAR-<phase>.<n>`), a status (`TODO`, `WIP`, `DONE`,
   `BLOCKED`), the commit hash and a single-line result. Keep it terse,
   because it is reread constantly.
3. Create `.work/decisions.md` and `.work/metrics-history.csv` (date, phase,
   and the section 3.4 metrics).
4. After every item, update the ledger **before** starting the next. After a
   compaction or a restart, reread `CLAUDE.md`, then the ledger, then
   `git log -5`, and continue with the first item that is `WIP` or `TODO`.

## 3. Measurement before building: make the comparison deterministic

Build the measuring instrument first, so that every later change moves a
number the reader can reproduce.

### 3.1 Pin the reference

Create `tools/parity/requirements.txt` with `numpy==2.4.*` and
`scipy==1.17.*`, and install it in a venv at `.venv-parity`. Every fixture
records the NumPy and SciPy versions and the platform.

### 3.2 Symbol census

Write `tools/parity/census.py`. It enumerates the public API: `__all__`, or
`dir()` without underscore names.

- **NumPy modules:** `numpy`, `numpy.linalg`, `numpy.fft`,
  `numpy.polynomial` (classes and their methods),
  `numpy.random.Generator` methods, and the `ndarray` methods and
  properties.
- **SciPy modules:** `scipy.special`, `stats` (each distribution plus its
  standard methods), `linalg`, `optimize`, `sparse`, `sparse.linalg`,
  `sparse.csgraph`, `interpolate`, `signal`, `signal.windows`, `integrate`,
  `spatial`, `spatial.distance`, `cluster.vq`, `cluster.hierarchy`,
  `ndimage`, `fft` and `constants`.

The census writes `tools/parity/census.json`: one record per symbol, with
module, name, kind and an alias-of target when two names are the same
object. For example, `acos` is `arccos` and `concat` is `concatenate`.

### 3.3 Scope file

Write `tools/parity/scope.yaml`. It maps every census symbol to exactly one
of the following:

- `implemented: <Tessero symbol(s)>`, with a fixture test ID;
- `planned: PAR-x.y`;
- `excluded: <category>`, where the category must be one of:
  - **E1** object, string, bytes, structured and record dtypes, and
    `numpy.char`/`numpy.rec`;
  - **E2** `numpy.ma` masked arrays;
  - **E3** legacy or deprecated APIs: `numpy.matrix`, `RandomState` and the
    legacy `numpy.random.*` module functions, and names SciPy marks as
    deprecated;
  - **E4** Python-runtime specifics: `numpy.testing`, printing options,
    `ctypeslib`, `f2py`, typing, `show_config`, `get_include` and similar;
  - **E5** file formats other than `.npy`, `.npz` and text/CSV (for
    example the MATLAB and NetCDF readers in `scipy.io`);
  - **E6** things superseded in the reference itself (for example
    `interp1d`, which SciPy calls legacy, when `make_interp_spline` is
    covered);
  - **E7** interactive, plotting or dataset helpers (`scipy.datasets`,
    `scipy.misc`).

`census.py --check` fails if any symbol is unmapped, if an `implemented`
entry lacks a passing test, or if a `planned` ID is missing from the ledger.

### 3.4 Metrics

Rewrite `tools/gap_inventory.py` as `tools/parity/report.py`. It reads the
census, the scope file, `tools/api-dump.php` output for both backends, and
the JUnit XML from the fixture tests. It regenerates
`docs/project/numpy-scipy-coverage.md` and
`docs/project/_generated/metrics.md` with these metrics:

- **M1: curated-row coverage.** Keep the existing 169 rows, add rows for
  new areas, and report both backends.
- **M2: symbol coverage of in-scope symbols,** per module and per backend.
  This is the headline number.
- **M3: excluded share** per module, so a reader can see what was left out
  and why.
- **M4: verified share.** Covered symbols whose fixture test passed in the
  current run. It must equal M2; any gap is a bug.
- **M5: backend parity.** The share of FFI-covered symbols that are also
  covered by the extension. The target is 100 %.

Print a before/after table after every phase and append it to
`metrics-history.csv`.

### 3.5 Claims audit

Write `tools/parity/claims-check.php`. It scans `README.md`, `docs/**` and
the Laravel README for coverage language: "NumPy for PHP", "SciPy
equivalent", "complete", "drop-in", "portable NumPy", and percentages. It
fails when a statement is not backed by `metrics.md`, or when a percentage
differs from it.

Wire both `census.py --check` and `claims-check.php` into `ci.yml`.
The README states what exists, and the measured numbers come in through the
include.

## 4. Architecture: one registry, both backends by construction

The 0.2 extension/FFI gap exists because each function was bound twice by
hand. Remove that failure mode before adding hundreds of functions.

### 4.1 Function registry

Create `spec/` as the single source of truth, with one YAML file per module
(`spec/special.yaml`, `spec/stats/*.yaml`, `spec/numpy_stats.yaml`, and so
on). Each entry has these fields:

```yaml
- name: gammainc                  # Tessero name (camelCase generated for PHP)
  ref: scipy.special.gammainc     # the specification
  kind: ufunc                     # ufunc | gufunc | reduction | routine | distribution | class
  signature: "(a, x) -> y"        # gufunc core dims like "(n),(n)->()" where needed
  dtypes: [f8, f4]                # loops; promotion follows NumPy
  c: tsr_sf_gammainc              # kernel entry (loop or routine)
  accuracy: {rtol: 1e-13}         # or {ulp: 2}, or {atol: .., rtol: ..}; stochastic: {exact_stream: true}
  domain:                         # fixture generator: grids, random draws, edge cases
    a: [grid(1e-3, 1e3, 60, log), edge]
    x: [grid(0, 1e3, 60), edge]
  php: {class: Special, backends: [ffi, ext]}
  doc: "Regularized lower incomplete gamma function P(a, x)."
```

### 4.2 Code generation

Write `tools/codegen/` in PHP, since PHP is the one runtime every
contributor has. From the registry it generates:

- **Kernel dispatch tables** (`csrc/src/gen/*.c`) and the declarations
  block in `tessero.h`. Keep the header FFI-parseable: no macros and no
  function-like constructs.
- **FFI façade classes** (`src/Gen/…`, re-exported through thin
  hand-written classes such as `Tessero\Special` and `Tessero\Stats`).
- **Extension code:** a `.stub.php` for each class, processed by PHP's
  `build/gen_stub.php` into arginfo, plus C method bodies that marshal
  arguments through one uniform path.
- **Laravel facade methods.**
- **Docs:** reference pages under `docs/reference/api/` and a function list
  per guide page.
- **Fixtures:** one Python script per module
  (`tests/fixtures/gen/<module>.py`), writing `.npz` files so values are
  byte-exact.
- **Tests:** one PHPUnit class per module, parameterised over both backends
  and over the fixture records.
- **Scope entries:** the `implemented` entries in `scope.yaml`.

Generated files carry a header saying they are generated. `codegen --check`
in CI fails when the generated output is stale.

### 4.3 Uniform native call path

The ufuncs already dispatch by name through `tsr_ufunc`. Add two more entry
points:

- **`tsr_gufunc`**: generalised ufuncs with core dimensions (NumPy's
  `(n)->()` model). It handles broadcasting of the loop dimensions,
  `axis`/`axes`, `keepdims` and `out`, and it copies a core slice into
  contiguous scratch only when the slice is strided. Everything that works
  along an axis sits on this: median, percentile, sort-based statistics,
  rankdata, `skew`, `kurtosis`, `trim_mean`, `mode`, `cumulative_trapezoid`,
  `polyval`, window functions and filters.
- **`tsr_routine_call(id, const tsr_arg *in, int nin, tsr_arg *out, int nout)`**:
  a tagged union for arrays, f64, i64, bool, a short string, i64 lists and
  a callback. It serves routines that do not fit the ufunc model
  (`histogram`, `unique`, spline construction, filter design, `quad`,
  `solve_ivp`, `KDTree`). Both bindings marshal through this one function,
  so a new routine needs only C code and a registry entry.
- **Callbacks** (the integrand, ODE right-hand side or objective) are a
  `tsr_callback {fn, ctx}`:
  - FFI passes a PHP closure as an `FFI` callback;
  - the extension passes a C trampoline that calls `zend_call_function`;
  - an exception thrown in PHP sets an abort flag, the algorithm unwinds
    with `TSR_EABORT`, and the binding rethrows the original exception.

  Port the existing PHP optimisers (`src/Optimize`) onto this interface
  whenever a phase needs them in the extension. Keep their current results
  bit-identical, and test that.
- **Stateful objects** (frozen distributions, `CubicSpline`, `KDTree`,
  `BSpline`, filters in second-order sections) are kernel handles
  (`tsr_obj*`) with a destructor. They are freed exactly once: from
  `Buffer`-style owners in FFI, and from `free_obj` in the extension. Each
  gets a `MemorySafetyTest` case.

### 4.4 Distribution framework (the core of the statistics package)

Mirror SciPy's `rv_continuous`/`rv_discrete` design in C:

- **Required per distribution:** `pdf` (or `pmf`) and `cdf`, each with its
  shape parameters and `loc`/`scale`.
- **Optional closed forms:** `logpdf`, `sf`, `logsf`, `ppf`, `isf`, the
  moments, `entropy` and `rvs`.
- **Generic fallbacks** mirror SciPy's approach and have their own declared
  (looser) accuracy class:
  - `ppf` by bracketing plus Brent on the cdf;
  - moments and `entropy` by adaptive Gauss–Kronrod quadrature;
  - `rvs` by inversion.
- **Methods for every distribution:** `pdf`, `logpdf`, `cdf`, `logcdf`,
  `sf`, `logsf`, `ppf`, `isf`, `rvs`, `mean`, `var`, `std`, `stats` (with
  moments `'mvsk'`), `moment(n)`, `entropy`, `median`, `interval`,
  `expect`, `support` and `fit`.
  - `fit` uses MLE through the optimiser, with the closed-form estimators
    SciPy uses where it has them.
  - Discrete distributions add `pmf`, `logpmf` and the discrete `ppf`.
- **Methods are ufuncs over (x, shapes…, loc, scale).** Broadcasting over
  parameters works exactly as in SciPy, on both backends.
- **PHP API:**
  - `Stats::norm(loc: 0, scale: 1)` returns a frozen distribution, and
    `Stats::norm()->cdf($x)` evaluates it.
  - Unfrozen calls look like `Stats\Norm::cdf($x, loc:, scale:)`.
  - The names are SciPy's: `t`, `chi2`, `weibull_min` becomes `weibullMin`,
    and so on. Keep a name map in the registry.

### 4.5 Accuracy policy

- **Special functions:** declared `rtol` on the stated domain, typically
  1e-13 to 1e-15. Measure the worst case over the fixture set and publish
  it in the generated reference page.
- **Distributions:** closed forms inherit the accuracy of their special
  functions; generic fallbacks use 1e-9 relative.
- **Hypothesis tests:** the statistic and p-value to 1e-10 relative, and
  the p-value also to 1e-300 absolute for tiny values.
- **Linear algebra, FFT, integrate and interpolate:** tolerances scaled by
  the condition number, as in the existing linalg fixtures.
- **Stochastic output:**
  - where NumPy's algorithm is reproduced, the stream is bit-identical;
  - otherwise a statistical acceptance test: a fixed seed, a KS or chi²
    p-value above 1e-4, plus moment checks.
- **Edge cases:** every function's fixtures include `nan`, `±inf`, `±0`,
  subnormals, empty arrays, length-1 axes, strided and negative-stride
  views, and the huge or tiny parameters where SciPy switches algorithms.

## 5. Performance

- Ufunc loops get SIMD contiguous paths where the existing `vmath.h` style
  applies, and OpenMP above the existing thresholds.
- Each phase adds benchmarks to `bench/` against SciPy at 1e3 and 1e6
  elements. A function more than 10× slower than SciPy on 1e6 elements
  gets a performance issue in `.github/beta-issues.json`. Do not block on
  it unless it is more than 50× slower.
- Also finish the performance items already open: vectorised sin/cos/tan
  (TSR-201), a SIMD sort (TSR-202) and a split-radix FFT (TSR-203).
  Schedule them after Phase 3, because stats uses them heavily.

## 6. Phases and items

Each phase ends shippable, with the full gate green, metrics recorded and the
changelog updated. The IDs go into the ledger. The bullet lists are the
minimum; the census may add symbols to a phase.

### Phase 0: infrastructure (PAR-0.x)

- Resumability files (section 2).
- Add a `--compare` mode to `bench/run.php`. It reads SciPy timings recorded
  by `bench/scipy_baseline.py` and prints the ratios.
- The census, the scope file, the report and the claims check (section 3).
- The registry and codegen (sections 4.1 and 4.2). Migrate the 74 existing
  ufuncs onto the registry first as a zero-diff refactor: generated output
  must match the current behaviour byte for byte, and `BackendParityTest`
  must pass.
- The `tsr_gufunc` and `tsr_routine_call` entry points, the callback
  bridge and the kernel object handles, with fuzz targets for any argument
  parsing.
- **Close the existing extension gaps** through the new path:
  - TSR-101, error-type parity;
  - TSR-102, the missing NDArray methods: `empty`, `view`, `expandDims`,
    `squeeze`, `swapAxes`, `moveAxis`, `broadcastTo`, `concatenate`,
    `stack`, full `take`/`put`, `filter`/`setWhere`, `diff`, `round`,
    `isclose`/`allclose`, `flatten`, `flatNonzero` and `contiguous`;
  - TSR-103, linalg via LAPACKE;
  - TSR-104, the full FFT;
  - TSR-105, a stateful Generator;
  - TSR-106, axis sort;
  - TSR-107, `.npz`.
- **Exit:** M5 (backend parity) = 100 % for everything that existed in 0.2.

### Phase 1: scipy.special, the foundation of stats (PAR-1.x)

These are all ufuncs with f8 and f4 loops.

- **Gamma family:** `gamma`, `gammaln`, `loggamma` (real), `gammasgn`,
  `rgamma`, `digamma`/`psi`, `polygamma`, `multigammaln`, `beta`,
  `betaln`, `poch`.
- **Incomplete functions:** `gammainc`, `gammaincc`, `gammaincinv`,
  `gammainccinv`, `betainc`, `betaincc`, `betaincinv`, `betainccinv`.
- **Error and normal functions:** `erf`, `erfc`, `erfcx`, `erfi`,
  `erfinv`, `erfcinv`, `ndtr`, `log_ndtr`, `ndtri`, `ndtri_exp`, `dawsn`,
  `owens_t`.
- **Logistic and entropy:** `expit`, `logit`, `log_expit`, `xlogy`,
  `xlog1py`, `entr`, `rel_entr`, `kl_div`, `huber`, `pseudo_huber`,
  `boxcox`, `boxcox1p`, `inv_boxcox`, `inv_boxcox1p`.
- **Axis-aware reductions:** `logsumexp` (with `b=` and `return_sign`),
  `softmax`, `log_softmax`.
- **Combinatorics:** `comb`, `perm` and `factorial` (exact and float),
  `factorial2`, `stirling2`, `binom`.
- **Bessel functions:** `j0`, `j1`, `jv`, `jn`, `y0`, `y1`, `yv`, `yn`,
  `i0`, `i0e`, `i1`, `i1e`, `iv`, `ive`, `k0`, `k0e`, `k1`, `k1e`, `kv`,
  `kve`, `spherical_jn`, `spherical_yn`.
- **Exponential integrals and zeta:** `expn`, `exp1`, `expi`, `zeta`,
  `zetac`, `spence`, `sinc` (NumPy's).
- **Distribution-support functions:** `chdtr` and `chdtri`, `stdtr` and
  `stdtrit`, `fdtr` and `fdtri`, `bdtr`, `pdtr`, `nbdtr`, `kolmogorov`,
  `smirnov` and their inverses.
- **Orthogonal polynomials:** `eval_legendre`, `eval_hermite`,
  `eval_laguerre`, `eval_chebyt`, `eval_jacobi`, and the quadrature nodes
  `roots_legendre` and `roots_hermite`.
- **Exit:** every `scipy.special` symbol in scope is `implemented` or
  `planned` for a later phase with a reason. Complex-argument variants are
  `planned` for Phase 9 once complex64 and complex128 loops exist.

### Phase 2: NumPy statistics, sets and searching (PAR-2.x)

- **Order statistics:** `median`, `percentile` and `quantile` with all 13
  NumPy methods and `weights=` where NumPy supports it; `nanmedian`,
  `nanpercentile`, `nanquantile`.
- **Other reductions:** `average` (weighted, `returned=`), `ptp`,
  `count_nonzero`.
- **The nan-reductions:** `nansum`, `nanprod`, `nanmin`, `nanmax`,
  `nanargmin`, `nanargmax`, `nanmean`, `nanvar`, `nanstd`, `nancumsum`,
  `nancumprod`.
- **Histograms and binning:** `histogram` (int bins, explicit edges, and
  the `auto`, `fd`, `doane`, `scott`, `stone`, `rice`, `sturges` and `sqrt`
  estimators, plus `range`, `density` and `weights`), `histogram_bin_edges`,
  `histogram2d`, `histogramdd`, `bincount`, `digitize`.
- **Covariance:** `cov` (all arguments), `corrcoef`, `correlate`,
  `convolve` (all three modes).
- **Sets:** `unique` with `return_index`, `return_inverse`, `return_counts`
  and `axis`, plus `unique_values`, `unique_counts`, `unique_inverse` and
  `unique_all`; `isin`, `intersect1d`, `union1d`, `setdiff1d`, `setxor1d`.
- **Searching and partial sorts:** `searchsorted` (left and right,
  `sorter`), `partition`, `argpartition`, `lexsort`, `sort_complex`.
- **Exit:** the NumPy statistics, set and searching rows are at 100 % on
  both backends.

### Phase 3: scipy.stats (PAR-3.x)

This is the statistics package.

- **Framework** (section 4.4) with frozen objects, `fit`, `expect`,
  `interval`, and the extension and Laravel façades.
- **Continuous distributions: all 110 in the census.** Order of work:
  1. `norm`, `t`, `chi2`, `f`, `gamma`, `beta`, `lognorm`, `expon`,
     `uniform`, `cauchy`, `laplace`, `logistic`, `weibull_min`,
     `weibull_max`, `gumbel_r`, `gumbel_l`, `pareto`, `genextreme`,
     `genpareto`, `invgamma`, `invgauss`, `rayleigh`, `triang`,
     `truncnorm`, `skewnorm`, `nct`, `ncx2`, `ncf`, `loggamma`,
     `loglaplace`, `lomax`, `halfnorm`, `halfcauchy`, `foldnorm`,
     `vonmises`, `burr`, `burr12`, `fisk`, `johnsonsu`, `johnsonsb`,
     `powerlaw`, `rice`, `maxwell`, `erlang`, `gengamma`, `genlogistic`,
     `exponweib`, `exponnorm`, `studentized_range`, `levy`, `levy_stable`
     (last).
  2. Then the rest, alphabetically.

  Each distribution needs pdf and cdf with closed forms where SciPy has
  them, and generic fallbacks otherwise.
- **Discrete distributions: all 21.** `poisson`, `binom`, `nbinom`,
  `geom`, `hypergeom`, `bernoulli`, `randint`, `betabinom`, `logser`,
  `zipf`, `zipfian`, `planck`, `boltzmann`, `dlaplace`, `skellam`,
  `yulesimon`, `nhypergeom`, `nchypergeom_fisher`,
  `nchypergeom_wallenius`, `betanbinom`, `poisson_binom`, and the rest in
  the census.
- **Multivariate distributions:** `multivariate_normal`,
  `multivariate_t`, `dirichlet`, `multinomial`, `wishart`, `invwishart`,
  `matrix_normal` (pdf/logpdf, cdf where SciPy has it, rvs).
- **NumPy-identical sampling in `Generator`:** `gamma`, `standard_gamma`,
  `beta`, `poisson`, `binomial`, `negative_binomial`, `geometric`,
  `hypergeometric`, `lognormal`, `chisquare`, `noncentral_chisquare`, `f`,
  `noncentral_f`, `standard_t`, `standard_cauchy`, `vonmises`, `wald`,
  `weibull`, `triangular`, `laplace`, `logistic`, `gumbel`, `pareto`,
  `power`, `rayleigh`, `zipf`, `logseries`, `multinomial`,
  `multivariate_hypergeometric`, `dirichlet`, `multivariate_normal`,
  `permuted`, `bytes`, and ziggurat `exponential` (fixing the current
  inversion method).
- **Descriptive statistics:** `describe`, `zscore`, `zmap`, `gzscore`,
  `mode`, `skew`, `kurtosis`, `moment`, `sem`, `iqr` (all interpolation
  methods), `gmean`, `hmean`, `pmean`, `trim_mean`, `trimboth`,
  `trim1`, `variation`, `rankdata` (all tie methods), `tmean`, `tvar`,
  `tstd`, `tmin`, `tmax`, `tsem`, `entropy`, `differential_entropy`,
  `median_abs_deviation`, `sigmaclip`, `scoreatpercentile`,
  `percentileofscore`, `cumfreq`, `relfreq`, `binned_statistic`,
  `binned_statistic_2d`, `binned_statistic_dd`, `find_repeats`.
- **Correlation and regression:** `pearsonr` (with CI), `spearmanr`,
  `kendalltau` (tau-b and tau-c), `weightedtau`, `somersd`,
  `pointbiserialr`, `linregress`, `theilslopes`, `siegelslopes`,
  `chatterjeexi`.
- **Hypothesis tests:**
  - t-tests: `ttest_1samp`, `ttest_ind` (Welch and trimmed),
    `ttest_ind_from_stats`, `ttest_rel`;
  - analysis of variance: `f_oneway`, `alexandergovern`, `kruskal`,
    `friedmanchisquare`, `tukey_hsd`, `dunnett`;
  - rank tests: `mannwhitneyu` (exact and asymptotic), `wilcoxon`,
    `brunnermunzel`, `ranksums`, `ansari`, `mood`, `fligner`, `levene`,
    `bartlett`;
  - count and contingency tests: `chisquare`, `power_divergence`,
    `chi2_contingency`, `fisher_exact`, `barnard_exact`, `boschloo_exact`,
    `binomtest`, `poisson_means_test`;
  - distribution tests: `ks_1samp`, `ks_2samp`, `kstest`, `cramervonmises`,
    `cramervonmises_2samp`, `anderson`, `anderson_ksamp`, `epps_singleton_2samp`;
  - normality tests: `shapiro` (the Royston algorithm), `normaltest`,
    `skewtest`, `kurtosistest`, `jarque_bera`;
  - other tests: `page_trend_test`, `median_test`, `mood`;
  - combining p-values: `combine_pvalues`, `false_discovery_control`.
- **Resampling:** `bootstrap` (percentile, basic and BCa),
  `permutation_test`, `monte_carlo_test`. These use the Generator for
  reproducibility and take PHP callbacks for the statistic.
- **Kernel density:** `gaussian_kde` (`scott` and `silverman` bandwidths,
  weights, `evaluate`, `integrate_box_1d`, `resample`).
- **Qualified quantities:** `contingency.association`,
  `contingency.expected_freq`, `qmc` (`Sobol`, `Halton`, `LatinHypercube`,
  `discrepancy`), `sampling` (`NumericalInverseHermite`).
- **Exit:**
  - M2 for `scipy.stats` is at least 95 % of in-scope symbols on both
    backends;
  - every distribution method is verified;
  - an end-to-end guide page, "Statistics with Tessero", mirrors SciPy's
    stats tutorial with executed examples.

### Phase 4: array creation, manipulation and indexing (PAR-4.x)

- **Creation:** `zeros_like`, `ones_like`, `full_like`, `empty_like`;
  `logspace`, `geomspace`; `meshgrid` (both `indexing` modes and
  `sparse=`), `mgrid`, `ogrid` and `indices` (as functions); `diag`,
  `diagflat`, `diagonal`, `tri`, `tril`, `triu`, `vander`;
  `fromfunction` (PHP callable), `fromiter`; `identity`;
  `asfortranarray`, `ascontiguousarray`; `copyto`.
- **Joining and splitting:** `vstack`, `hstack`, `dstack`,
  `column_stack`, `block`; `atleast_1d`, `atleast_2d`, `atleast_3d`;
  `split`, `array_split`, `hsplit`, `vsplit`, `dsplit`.
- **Rearranging:** `tile`, `repeat`, `flip`, `fliplr`, `flipud`, `roll`,
  `rot90`, `pad` (all modes: constant, edge, linear_ramp, maximum, mean,
  median, minimum, reflect, symmetric, wrap, empty); `resize`, `append`,
  `insert`, `delete`; `trim_zeros`; `swapaxes`, `moveaxis`, `rollaxis`;
  `broadcast_arrays`.
- **Indexing:** `nonzero`, `argwhere`, `flatnonzero`, `extract`,
  `place`, `putmask`, `put_along_axis`, `take_along_axis`, `choose`,
  `compress`, `select`, `piecewise`; full multi-axis advanced indexing
  (`$a[[i], [j]]` and mixed basic/advanced, with NumPy's result-shape
  rules); `ix_`, `ravel_multi_index`, `unravel_index`, `diag_indices`,
  `tril_indices`, `triu_indices`, `mask_indices`.
- **Ufunc machinery:**
  - `where=` masks;
  - the `reduce`, `accumulate`, `reduceat` and `outer` methods on every
    binary ufunc (`Math::add->reduce(...)`, or `Math::reduce('add', ...)`
    in PHP);
  - `vectorize`, as a PHP-callable map that is slow and documented as such.
- **Missing ufuncs:** `bitwise_and`, `bitwise_or`, `bitwise_xor`,
  `bitwise_not`, `left_shift`, `right_shift`, `bitwise_count`; `gcd`,
  `lcm`; `heaviside`; `ldexp`, `frexp`, `modf`, `divmod`; `float_power`;
  `spacing`; `nan_to_num`; `positive`; `fabs`; `cbrt` exists; `logaddexp`
  exists.
- **Exit:** the NumPy creation, shape, indexing and ufunc rows are at 100 %
  on both backends.

### Phase 5: dtypes and I/O (PAR-5.x)

- **Dtypes:** add `int8`, `int16`, `uint16`, `uint32`, `uint64`,
  `float16` (storage type, computed in f32 as NumPy does), `complex64`,
  and `datetime64`/`timedelta64` (units D, h, m, s, ms, us, ns).
  - Follow NumPy 2's promotion table exactly.
  - Generate the ufunc loops for each type through codegen.
  - The `.npy`/`.npz` reader and writer must handle all of them.
  - Add `finfo`, `iinfo`, `can_cast`, `result_type`, `promote_types`.
- **Text I/O:** `loadtxt`, `savetxt` and `genfromtxt` (delimiters,
  comments, `skiprows`, `usecols`, `max_rows`, converters as PHP callables,
  missing values); `tofile` and `fromfile`. Each text parser gets a fuzz
  target.
- **Exit:** the dtype rows are at 100 % in scope (E1 is excluded).

### Phase 6: linear algebra, FFT and polynomials (PAR-6.x)

- **numpy.linalg:** `matrix_power`, `multi_dot` (optimal ordering),
  `tensorsolve`, `tensorinv`, `outer`, `inner`, `vdot`, `kron`, `cross`,
  `tensordot`, `einsum` (with `optimize` path selection), `einsum_path`,
  `vecdot`, `matrix_transpose`, `matrix_norm`, `vector_norm`,
  `svdvals`, `diagonal`, `trace`. All decompositions also run in
  complex128 and complex64 (`zgesv`, `zgeev`, `zgesdd` and so on).
- **scipy.linalg:** `lu_factor`, `lu_solve`, `cho_factor`, `cho_solve`,
  `ldl`, `polar`, `qz`, `ordqz`, `schur`, `rsf2csf`, `hessenberg`,
  `expm`, `logm`, `sqrtm`, `funm`, `fractional_matrix_power`, `cosm`,
  `sinm`, `solve_banded`, `solveh_banded`, `solve_toeplitz`,
  `solve_circulant`, `solve_sylvester`, `solve_continuous_lyapunov`,
  `solve_discrete_lyapunov`, `solve_continuous_are`, `solve_discrete_are`,
  `null_space`, `orth`, `subspace_angles`, `block_diag`, `toeplitz`,
  `circulant`, `hankel`, `hadamard`, `pascal`, `companion`, `eig_banded`,
  `eigh_tridiagonal`, `lstsq` (LAPACK driver choice), `pinvh`, `norm`,
  `svdvals`, `clarkson_woodruff_transform`.
- **FFT** (numpy.fft and scipy.fft): `fftn`, `ifftn`, `rfft2`, `irfft2`,
  `rfftn`, `irfftn`, `hfft`, `ihfft`, `hfftn`; `dct` and `idct` (types
  1–4), `dst` and `idst` (types 1–4) with `dctn`/`dstn`; `next_fast_len`;
  `fht`/`ifht`. Include the split-radix complex FFT (TSR-203).
- **Polynomials:**
  - the functions `polyfit` (weights, `cov`), `polyval`, `roots`, `poly`,
    `polyder`, `polyint`, `polyadd`, `polysub`, `polymul`, `polydiv`;
  - the `numpy.polynomial` classes `Polynomial`, `Chebyshev`, `Legendre`,
    `Hermite`, `HermiteE` and `Laguerre` (`fit`, `roots`, `deriv`,
    `integ`, `domain`/`window`, arithmetic).
- **Exit:** the linalg, FFT and polynomial rows are at 100 % on both
  backends.

### Phase 7: integrate and interpolate (PAR-7.x)

- **Integration:**
  - fixed samples: `trapezoid`, `cumulative_trapezoid`, `simpson`,
    `cumulative_simpson`, `romb`;
  - adaptive quadrature: `quad` (a QUADPACK QAGS/QAGI port with `points`,
    `weight=` for QAWO/QAWF/QAWC, and `full_output` info), `quad_vec`,
    `dblquad`, `tplquad`, `nquad`, `fixed_quad`, `newton_cotes`, `qmc_quad`;
  - ODEs: `solve_ivp` with RK23, RK45, DOP853, Radau, BDF and LSODA
    (events, `dense_output`, `t_eval`, `vectorized`), `odeint` (a LSODA
    wrapper), and `solve_bvp`.
- **Interpolation:**
  - `interp` (NumPy's);
  - splines: `make_interp_spline`, `BSpline`, `CubicSpline` (all `bc_type`
    values), `PchipInterpolator`, `Akima1DInterpolator` (with the `makima`
    method), `CubicHermiteSpline`, `make_smoothing_spline`,
    `make_lsq_spline`, `UnivariateSpline`, `InterpolatedUnivariateSpline`
    and `LSQUnivariateSpline` (a FITPACK curfit port), `splrep`, `splev`,
    `splint`, `sproot`, `splder`, `splantider`;
  - gridded and polynomial interpolators: `RegularGridInterpolator` (the
    linear, nearest, slinear, cubic and quintic methods), `interpn`,
    `RectBivariateSpline`, `BarycentricInterpolator`,
    `KroghInterpolator`, `PPoly`, `BPoly`, `pade`, `approximate_taylor_polynomial`;
  - scattered data: `griddata` (needs Phase 10's Delaunay, so it is
    scheduled there), `RBFInterpolator` (all kernels), `NearestNDInterpolator`,
    `LinearNDInterpolator` and `CloughTocher2DInterpolator` (also after
    Delaunay).
- **Exit:** M2 is at least 90 % for `integrate` and `interpolate`.

### Phase 8: signal (PAR-8.x)

- **Convolution:** `convolve`, `correlate`, `fftconvolve`,
  `oaconvolve`, `choose_conv_method`, `convolve2d`, `correlate2d`,
  `correlation_lags`, `deconvolve`.
- **Filter design:** `butter`, `cheby1`, `cheby2`, `ellip`, `bessel`,
  `iirfilter`, `iirdesign`, `buttord`, `cheb1ord`, `cheb2ord`, `ellipord`,
  `iirnotch`, `iirpeak`, `iircomb`, `firwin`, `firwin2`, `remez`, `firls`,
  `minimum_phase`, `kaiserord`, `kaiser_beta`, `kaiser_atten`.
- **Filter representations:** `zpk2tf`, `tf2zpk`, `zpk2sos`, `sos2zpk`,
  `tf2sos`, `sos2tf`, `lp2lp`, `lp2hp`, `lp2bp`, `lp2bs`, `bilinear`,
  `bilinear_zpk`, `normalize`.
- **Filtering:** `lfilter` (with `zi`), `lfilter_zi`, `lfiltic`,
  `filtfilt` (the `pad` and `gust` methods), `sosfilt`, `sosfilt_zi`,
  `sosfiltfilt`, `medfilt`, `medfilt2d`, `wiener`, `savgol_filter`,
  `savgol_coeffs`, `decimate`, `detrend`, `hilbert`, `hilbert2`,
  `resample`, `resample_poly`, `upfirdn`.
- **Frequency response:** `freqz`, `freqz_sos`, `sosfreqz`, `freqs`,
  `group_delay`.
- **Spectral analysis:** `periodogram`, `welch`, `csd`, `coherence`,
  `spectrogram`, `stft`, `istft`, `ShortTimeFFT`, `lombscargle`,
  `check_COLA`, `check_NOLA`.
- **Peaks:** `find_peaks` (all properties), `peak_prominences`,
  `peak_widths`, `find_peaks_cwt`, `argrelextrema`, `argrelmax`,
  `argrelmin`.
- **Windows:** all of `scipy.signal.windows` (`get_window` plus about 30
  window functions, including `kaiser`, `tukey`, `dpss`, `chebwin`,
  `taylor` and `general_cosine`).
- **LTI systems:** `lti`, `dlti`, `lsim`, `dlsim`, `step`, `dstep`,
  `impulse`, `dimpulse`, `bode`, `dbode`, `freqresp`.
- **Waveforms:** `chirp`, `gausspulse`, `sawtooth`, `square`,
  `sweep_poly`, `unit_impulse`, `max_len_seq`.
- **Exit:** M2 for `signal` is at least 90 %.

### Phase 9: optimize (PAR-9.x)

- **`minimize` methods:** CG, Newton-CG, Powell, TNC, COBYLA, COBYQA,
  SLSQP, trust-constr, dogleg, trust-ncg, trust-exact and trust-krylov.
  Include constraint objects (`LinearConstraint`, `NonlinearConstraint`,
  `Bounds`) and the `callback` argument. Wherever SciPy's algorithm is
  deterministic, test for the same iterates (as BFGS already does).
- **Scalar methods:** `minimize_scalar` (brent, bounded, golden);
  `bisect`, `ridder`, `toms748`, `brenth`, `root_scalar` (all methods),
  `fixed_point`, `newton` in vectorised form, `elementwise.find_root` and
  `elementwise.find_minimum`.
- **Systems of equations:** `root` with hybr and lm (MINPACK ports), and
  broyden1/2, anderson, krylov and df-sane; `fsolve`.
- **Least squares:** `least_squares` with trf and dogbox (bounds, loss
  functions, `f_scale`, sparse Jacobians); `curve_fit` with bounds and
  `method=`; `nnls`; `lsq_linear`.
- **Global optimisation:** `differential_evolution` (all strategies,
  constraints, seeded through the Generator), `basinhopping`,
  `dual_annealing`, `shgo`, `direct`, `brute`.
- **Other routines:** `linear_sum_assignment` (LAPJV),
  `quadratic_assignment`, `check_grad`, `approx_fprime`, `line_search`,
  `rosen` and its helpers.
- **Linear programming (TSR-110):** `linprog` and `milp` get a HiGHS
  backend, loaded at runtime, with `method='highs'`, `'highs-ds'` and
  `'highs-ipm'`. Keep the built-in simplex as the fallback. Implement
  `milp` constraints, `integrality` and the options exactly as SciPy does.
- **Exit:** M2 for `optimize` is at least 90 %.

### Phase 10: sparse, spatial, cluster and ndimage (PAR-10.x)

- **Sparse formats:** `csr_array` and `csc_array` (plus the `_matrix`
  aliases), `coo_array`, `dia_array`, `bsr_array`, `lil_array` and
  `dok_array` as build-only formats. Arithmetic: sparse ± sparse, sparse
  @ sparse, elementwise `multiply`, and scalar operations. Structure:
  `kron`, `kronsum`, `hstack`, `vstack`, `block_array`, `block_diag`,
  `diags_array`, `eye_array`, `random_array`, `tril`, `triu`, `find`.
  I/O: `save_npz` and `load_npz`, byte-compatible with SciPy.
- **sparse.linalg:**
  - direct solvers: `spsolve` and `splu` (a sparse LU with COLAMD,
    TSR-111), `spilu`, `factorized`, `spsolve_triangular`;
  - iterative solvers: `gmres`, `lgmres`, `minres`, `qmr`, `gcrotmk`,
    `tfqmr`, `lsqr`, `lsmr`, plus the existing `cg` and `bicgstab`;
  - eigen and singular values: `eigsh` and `eigs` (an implicitly restarted
    Lanczos/Arnoldi port of ARPACK), `lobpcg`, `svds` (ARPACK and PROPACK
    back-ends);
  - `expm`, `expm_multiply`, `norm`, `onenormest`, `LinearOperator`
    (callbacks) and `aslinearoperator`.
- **sparse.csgraph:** `shortest_path` (Dijkstra, Bellman-Ford, Johnson,
  Floyd-Warshall), `connected_components`, `minimum_spanning_tree`,
  `breadth_first_order`, `depth_first_order`, `breadth_first_tree`,
  `depth_first_tree`, `laplacian`, `maximum_flow`, `maximum_bipartite_matching`,
  `min_weight_full_bipartite_matching`, `reverse_cuthill_mckee`,
  `structural_rank`.
- **spatial:**
  - distances: `distance.cdist` and `distance.pdist` (all metrics),
    `squareform`, and the individual metric functions;
  - trees and geometry: `KDTree` (`query`, `query_ball_point`,
    `query_pairs`, `count_neighbors`, `sparse_distance_matrix`),
    `ConvexHull` (Quickhull in 2D, 3D and nD), `Delaunay` (robust
    predicates), `Voronoi`, `HalfspaceIntersection`, `procrustes`,
    `geometric_slerp`;
  - `transform.Rotation` (quaternion, matrix, Euler and rotvec forms,
    `apply`, `mean`, `Slerp`).
- **cluster:** `vq.kmeans`, `vq.kmeans2`, `vq.vq`, `vq.whiten`;
  `hierarchy.linkage` (every method), `fcluster`, `fclusterdata`,
  `cophenet`, `inconsistent`, `leaves_list`, `optimal_leaf_ordering`,
  `cut_tree`, `to_tree`, `is_valid_linkage`. Dendrogram plotting is E7;
  dendrogram data is in scope.
- **ndimage:**
  - filters: `gaussian_filter`, `gaussian_filter1d`, `uniform_filter`,
    `median_filter`, `minimum_filter`, `maximum_filter`, `rank_filter`,
    `percentile_filter`, `generic_filter` (callback), `convolve`,
    `correlate`, `sobel`, `prewitt`, `laplace`, `gaussian_laplace`,
    `gaussian_gradient_magnitude`;
  - morphology: `binary_erosion`, `binary_dilation`, `binary_opening`,
    `binary_closing`, `binary_fill_holes`, `grey_erosion`, `grey_dilation`,
    `distance_transform_edt`, `distance_transform_cdt`;
  - measurements: `label`, `find_objects`, `sum_labels`, `mean`,
    `variance`, `center_of_mass`, `extrema`, `histogram`;
  - geometric transforms: `zoom`, `shift`, `rotate`, `affine_transform`,
    `map_coordinates`, `spline_filter` (all orders and modes).
- **Also in this phase:** the scattered-data interpolators deferred from
  Phase 7.
- **Exit:** M2 is at least 85 % for `sparse`, `sparse.linalg`, `csgraph`,
  `spatial`, `cluster` and `ndimage`.

### Phase 11: closing pass (PAR-11.x)

- Revisit every `BLOCKED` item.
- Rerun the census against the pinned versions and close any stragglers.
- Rebuild all generated documentation and write guide pages per module.
  Each gets executed examples, a "coming from Python" mapping table
  generated from the registry, and the measured accuracy per function.
- Update `docs/project/numpy-scipy-gap-analysis.md` and `open-issues.md`,
  the roadmap, the changelog and the README. All numbers come through the
  include, and the claims check must pass.
- Run the full gate plus a 2-hour soak per backend with a workload that
  exercises the new modules. Add them to `tools/soak.php`.

## 7. The per-item loop

Follow this loop exactly for every ledger item:

1. Mark the item `WIP` in the ledger. Read the SciPy/NumPy documentation
   for the symbol through `.venv-parity`: use `help()` and inspect
   signatures, and read SciPy's source to learn its conventions and edge
   cases.
2. **Specify:** add or extend the registry entry: signature, NumPy
   defaults, accuracy class, domain and edge cases.
3. **Fixtures:** regenerate with
   `.venv-parity/bin/python tests/fixtures/gen/<module>.py`. Check that the
   fixture covers every argument combination the SciPy documentation
   lists.
4. **Tests:** run codegen. The generated tests now fail, and that is
   expected.
5. **Implement** in C, in `csrc/src/<module>/…`: numerically stable forms,
   budgeted allocation, checked sizes, OpenMP only above thresholds, and
   no global state.
6. **Build and test:**
   - `make -C csrc test`;
   - the module's tests on FFI and on the extension (rebuild `ext/` after
     `tools/sync-ext.sh`);
   - `BackendParityTest` for the module;
   - ASan/UBSan for the new C file (`make -C csrc sanitize`).
7. **Document:** the generated reference page, and a short executed example
   in the relevant guide page that `tests/docs/examples.php` runs.
8. **Measure:** run `tools/parity/report.py`. The item's symbols must now
   count as `implemented` with passing tests on both backends.
9. **Commit**, then mark the item `DONE` in the ledger with the commit hash
   and one line: the maximum measured error against the accuracy target,
   plus a performance note if one is relevant.

Batch closely related symbols into one item when they share a kernel (for
example the incomplete-gamma family, or all `nan*` reductions). Never batch
across modules.

## 8. Gates

**Per item:** section 7, step 6.

**Per phase, and the full gate** (script it as `tools/parity/gate.sh`):

```bash
bash tools/sync-ext.sh --check
php tools/codegen/run.php --check
.venv-parity/bin/python tools/parity/census.py --check
make -C csrc test && make -C csrc sanitize
(cd ext && phpize && ./configure --enable-tessero && make -j && NO_INTERACTION=1 REPORT_EXIT_STATUS=1 make test TESTS=tests)
php -d ffi.enable=1 vendor/bin/phpunit                                   # FFI, parity, unit
php -d ffi.enable=1 -d extension=$PWD/ext/modules/tessero.so vendor/bin/phpunit --testsuite ext,unit
php -d ffi.enable=1 -d opcache.enable_cli=1 -d opcache.jit=tracing vendor/bin/phpunit --testsuite unit
php -d ffi.enable=1 vendor/bin/phpunit --testsuite laravel
php -d ffi.enable=1 tests/docs/examples.php
bash tests/e2e/preload.sh
make -C csrc fuzz-standalone FUZZ_RUNS=200000        # plus libFuzzer targets if clang is available
php -d ffi.enable=1 tools/soak.php --backend=ffi --duration=600
php -d extension=$PWD/ext/modules/tessero.so tools/soak.php --backend=ext --duration=600
php tools/parity/claims-check.php
mkdocs build --strict
php -d ffi.enable=1 -d memory_limit=2G bench/run.php       # gates on; Phase 0 adds --compare (SciPy baselines)
```

At the end of each phase, also:

- Launch an **independent verification subagent** that has not seen your
  implementation work. Give it the phase's ledger items and have it:
  - pick 25 random covered symbols;
  - write fresh SciPy comparisons from scratch (random inputs with new
    seeds and adversarial edge cases);
  - run them on both backends and report any mismatch.

  Every mismatch becomes an item in the next phase with a regression test.
- Update `CHANGELOG.md` (`[Unreleased]`), `docs/project/roadmap.md`
  (section 3 items as they close) and `.github/beta-issues.json` (close
  TSR items, add new ones). Rerender `open-issues.md` with
  `php tools/beta-issues.php render`.

## 9. Definition of done

The program is complete only when all of these hold:

1. **M2 (in-scope symbol coverage), on both backends:**
   - NumPy core (`numpy`, `linalg`, `fft`, `polynomial`, `Generator`,
     `ndarray` methods): **≥ 95 %**;
   - `scipy.special` and `scipy.stats`: **≥ 95 %**;
   - `linalg`, `optimize`, `integrate`, `interpolate`, `signal` and `fft`:
     **≥ 90 %**;
   - `sparse` (all three submodules), `spatial`, `cluster` and `ndimage`:
     **≥ 85 %**.
2. **M4 = M2** (every covered symbol verified in the final run) and **M5 =
   100 %**.
3. **Excluded share (M3):** at most 25 % of the NumPy core census and at
   most 10 % of every SciPy module. Each exclusion carries a fixed category.
4. **Gates:** the full gate is green, and the independent verification of
   the last phase found no open mismatch.
5. **No silent failures:** every kernel return code is checked, and a
   `MemorySafetyTest` or phpt case covers every new stateful handle and
   every allocation-heavy routine under a tight budget.
6. **Documentation:**
   - the coverage page, gap analysis, open issues, roadmap, changelog and
     README are regenerated or updated;
   - the claims check passes;
   - every module has a guide page with executed examples and a "coming
     from Python" table.
7. **Records:** the ledger has no `TODO` or `WIP` items. Any `BLOCKED` item
   has a matching open issue with a concrete next step.

When done, write `docs/project/parity-report.md`. It contains:

- before/after metrics per module, from `metrics-history.csv`;
- the accuracy summary: the worst measured error per module;
- performance against SciPy;
- the list of exclusions by category, the blocked items and the
  verification-agent results.

Then print a five-line summary. Until then, keep working: after each item,
go straight to the next.
