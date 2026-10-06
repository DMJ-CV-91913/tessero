<!-- Hand-authored analysis (not generated). Companion to numpy-scipy-coverage.md (generated marks)
     and numpy-scipy-gap-analysis.md (narrative). Reviewed 2026-10-02. -->

# NumPy / SciPy gap register and closure plan

This page is the meticulous, per-item record of every **genuine remaining gap** against NumPy/SciPy:
what it would take to implement in Tessero, whether it is truly portable to this architecture (and if
not, why, plus the workaround), and whether the PHP ecosystem already covers it. It is the companion to
the generated [coverage inventory](numpy-scipy-coverage.md) (the ● / ◐ / — marks) and the
[gap analysis](numpy-scipy-gap-analysis.md) narrative.

## How gaps are classified

"Portable to PHP" is the wrong axis: Tessero does the math in a **compiled C kernel** (OpenMP + SIMD) that
already links **LAPACK/BLAS** (`liblapacke`) and ships its own FFT (incl. Bluestein for prime lengths). PHP
is only a thin façade over FFI and the native extension, so PHP's own speed/type limits never block an
algorithm. Gaps are therefore classified on three real axes:

- **Tier 1 — composition.** Expressible from primitives we already ship. Implement now, standard parity loop.
- **Tier 2 — algorithm.** Real numerical work, no new external dependency.
- **Tier 3 — external C dependency (or heavy native algorithm).** A *correct* implementation really wants a
  battle-tested library (QHull, ARPACK, SuiteSparse, HiGHS) or is a large native undertaking. Because Tessero
  already links native libraries, this is a *packaging* decision, not a wall.
- **Tier 4 — not applicable to a PHP numeric API.** Not a math gap at all — a NumPy *dtype/idiom* feature that
  has no equivalent in a static PHP array API. Documented individually below with its functional substitute.

A fourth concern, **callback friction**, cuts across Tiers 2–3: algorithms that must invoke a *user-supplied
PHP function* repeatedly from inside the C loop (arbitrary-objective optimizers, `solve_ivp`). `Integrate\Quad`
already does this via an FFI callback, so it is proven — but each call crosses the FFI boundary and cannot be
OpenMP-parallelised across the callback, which shapes the design (vectorise the user function over arrays
where possible; document the per-call cost).

---

## Tier 4 first — the genuinely not-portable items (each with its solution)

These are the only items that are *not* a matter of effort. They do not map to a numeric PHP API; none
represents lost numerical capability, because each already has a functional equivalent that ships.

| NumPy feature | Why it does not port to a PHP numeric API | Solution / what to use instead |
|---|---|---|
| structured / record / object / string arrays | NumPy's field/dtype system (named fields, mixed types, Python objects per cell) has no analogue in a float/int/bool/complex kernel. | Out of scope. Keep metadata in PHP arrays/objects alongside the NDArray; numeric columns stay NDArrays. |
| `numpy.ma` masked arrays | A whole parallel array type carrying a boolean mask through every op. | Use the shipped **nan-aware reductions** (`nansum/nanmean/nanstd/…`) and boolean-mask indexing — the idiom PHP users actually need. |
| `mgrid` / `ogrid` | Grid *objects* constructed from Python slice syntax (`np.mgrid[0:3, 0:3]`). | **`meshgrid`** (ships) is the functional equivalent; expose a small helper if the open-grid (`ogrid`) broadcast form is wanted. |
| `fromfunction` | Takes a Python lambda evaluated over an index grid. | `meshgrid` + a ufunc expression, or `fromiter` for the iterator case. No callback-into-C needed. |
| `float16` / `longdouble` / `complex64` | Niche storage dtypes; the kernel is `float64` / `float32` / `complex128`. | Cast through `astype`; document the supported precision set. `float16` could be added as a storage-only dtype if a real use-case appears. |
| ufunc `reduce` / `accumulate` / `outer` **method protocol** (`np.add.reduce`) | These are *method objects hanging off a ufunc* — a Python object idiom, not a function. | The capability is **already shipped as named functions** (`sum`, `cumsum`, `outer`, …). Only the `.reduce` attribute syntax is absent, which is not meaningful in PHP. |

---

## NumPy gaps

| Gap | Tier | Approach in Tessero | PHP ecosystem |
|---|---|---|---|
| nd-FFT: `fftn/ifftn/rfftn/irfftn/hfft`, `fft2` family | **1** | Successive 1-D FFTs over each axis (our `fft` exists); `rfftn` = `rfft` on last axis then `fft` on the rest; `hfft` = `irfft` of conjugate-symmetric input. Pure façade+loop; no new kernel math. | **NONE** — no PHP lib has even 2-D FFT (`webit/fft` is 1-D pure-PHP). |
| `multivariate_normal`, `dirichlet`, `multinomial` | **1** | `mvn = mean + L·z`, `L = cholesky(cov)` (both ship); `dirichlet` = normalised `standardGamma`; `multinomial` = sequential conditional binomials. | **COVERED** by MathPHP (`Probability\Distribution\Multivariate\{Normal,Dirichlet,Multinomial}`), pure-PHP. |
| ufunc `gcd lcm heaviside float_power ldexp frexp divmod` | **1** | Element-wise entries in `fn_core.c`; `gcd/lcm` integer Euclid, `float_power` = `pow` with float promotion, `ldexp/frexp/modf` via libm. | NONE as array ufuncs (scalar only, hand-rolled). |
| `Polynomial` OO class | **1** | Thin PHP wrapper over the shipped `polyfit/polyval/polyadd/polyder/polyint/roots`. | MathPHP has a `Polynomial` class (pure-PHP). |

## scipy.linalg gaps

| Gap | Tier | Approach in Tessero | PHP ecosystem |
|---|---|---|---|
| `expm` (matrix exponential) | **2** | Higham scaling-and-squaring with degree-13 Padé — pure BLAS + existing `solve`. The reference algorithm; ~1e-15 accuracy. | **NONE** anywhere in PHP. |
| `logm`, `sqrtm` | **3** | Schur–Parlett: real Schur (LAPACK `dgees`, already linked) → block recurrence. `sqrtm` also via Denman–Beavers iteration (BLAS-only) as a dependency-free alternative. | **NONE**. |
| `Schur`, `QZ`, `Hessenberg` | **3** (low) | Direct LAPACK calls: `dgees` (Schur), `dgges` (QZ), `dgehrd`+`dorghr` (Hessenberg). LAPACK is **already linked**, so this is wiring, not new math. | Hessenberg **COVERED** (MathPHP, pure-PHP); Schur/QZ **NONE**. |
| `lu_factor`/`lu_solve` | **2** (low) | Expose LAPACK `getrf`/`getrs` (we already use them for `solve`). `cho_factor`/`cho_solve` already ship via `ScipyLinalg::choFactor/choSolve`. | NONE (MathPHP has LU decomposition but not the factor/solve split). |
| `null_space`, `orth` | **2** (low) | From the SVD we already compute: columns of `U`/`V` past/within the rank tolerance. `block_diag` already ships. | NONE. |

## scipy.optimize gaps

| Gap | Tier | Approach in Tessero | PHP ecosystem |
|---|---|---|---|
| `minimize`: CG, Newton-CG, Powell, TNC, trust-region | **2** | Standard line-search (CG/Newton-CG), direction-set (Powell), trust-region (dogleg/CG-Steihaug). We already have BFGS/Nelder-Mead/L-BFGS-B, so the harness exists. **Callback friction** (PHP objective per iter). | **NONE** (Rubix ML has only internal NN optimisers). |
| constrained: SLSQP, COBYLA, trust-constr | **2–3** | SLSQP = SQP with a dense QP subproblem (our dense QP/`linprog` machinery helps); COBYLA = linear-approx trust region; trust-constr = interior-point. Substantial but pure. | **NONE**. |
| `minimize_scalar` (brent/bounded/golden) | **1** | Brent's method (we already have `brentq` for roots) + golden section. | NONE dedicated. |
| `root`/`fsolve` for systems (hybr/Powell dogleg) | **2** | Powell hybrid (dogleg + Broyden/Jacobian). We ship damped-Newton `root`; add trust-region globalisation + numerical Jacobian (`fdJacobian` exists). | **NONE** (MathPHP root-finding is single-variable only). |
| global: `differential_evolution`, `basinhopping`, `dual_annealing`, `shgo` | **2** | DE (population + mutation/crossover), basin-hopping (local min + perturb + Metropolis), dual annealing (CSA + local polish), SHGO (simplicial homology). Pure; stochastic ones need our RNG (ships). Callback friction. | **NONE** (only unmaintained GA hobby repos). |
| `linear_sum_assignment` (Hungarian) | **2** (low) | Jonker–Volgenant / Kuhn–Munkres, O(n³), pure. | **COVERED** (pure-PHP): `oizys/hungarian`, `elgigi/hungarian-algorithm`. Cite as fallback; native is better for NDArray integration. |
| `nnls` (non-negative least squares) | **2** (low) | Lawson–Hanson active-set using existing `lstsq`. | **NONE**. |
| large LP/MILP (HiGHS-class) | **3** | We ship a dense simplex + branch-and-bound (good to a few thousand rows). For scale: FFI to **HiGHS**. | **NONE** for HiGHS; small-LP partial via `kerigard/lpsolve` (lp_solve FFI), `uestla/simplex-calculator`. |

## scipy.sparse gaps

| Gap | Tier | Approach in Tessero | PHP ecosystem |
|---|---|---|---|
| formats: CSC, COO, LIL, DOK, BSR, DIA | **2** | CSC = transposed CSR (cheap); COO/LIL/DOK are construction formats → convert to CSR/CSC; BSR/DIA for structured matrices. We ship CSR. | **NONE** — no sparse library exists in PHP at all. |
| sparse@sparse, sparse+sparse, `kron`, `hstack/vstack` | **2** | SpGEMM (Gustavson), sorted-merge add, structural kron/stack. | **NONE**. |
| `diags`, `random` (sparse constructors) | **1** | Build CSR directly from diagonals / sampled coordinates. | **NONE**. |
| iterative: `gmres`, `minres`, `lsqr`, `lsmr` | **2** | Krylov methods on our existing sparse matvec; GMRES(m) with restart, MINRES for symmetric-indefinite, LSQR/LSMR for least-squares. We ship CG/BiCGSTAB. | **NONE**. |
| direct: `spsolve`, `splu`, `factorized` | **3** | Sparse LU with AMD/COLAMD ordering → FFI to **SuiteSparse/UMFPACK**, or a native left-looking sparse LU (moderate). | **NONE**. |
| sparse eigensolvers: `eigsh`, `eigs`, `svds` | **3** | Native **Lanczos** (symmetric) / **Arnoldi** (general) with restarts — implementable without ARPACK (recommended), or FFI to ARPACK. | **NONE** (no ARPACK binding in PHP). |
| `csgraph`: shortest paths, MST, etc. | **2** | Dijkstra/Bellman-Ford/Floyd-Warshall/Johnson, Prim/Kruskal on CSR adjacency. We ship `connectedComponents`. | **NONE**. |

## scipy.special, scipy.stats gaps

| Gap | Tier | Approach in Tessero | PHP ecosystem |
|---|---|---|---|
| `softmax`, `log_softmax` | **1** | `softmax` = shift-max → `exp` → normalise; `log_softmax` = `x - logsumexp(x)` (we ship `logsumexp`). | `softmax` **COVERED** (MathPHP `Special::softmax`); `log_softmax` **NONE**. |
| `perm` (permutations) | **1** | `gamma(n+1)/gamma(n-k+1)` (ship `gamma`); `comb`/`factorial` already ship. | **COVERED** (MathPHP `Combinatorics::permutations`). |
| `gaussian_kde` (multivariate) | **2** | Gaussian kernel sum; bandwidth via Scott/Silverman; multivariate through the covariance (our `cholesky`/`cdist` ship). | **PARTIAL** — MathPHP/RubixML do univariate/basic KDE; multivariate-with-covariance is **NONE**. |

## scipy.interpolate gaps

| Gap | Tier | Approach in Tessero | PHP ecosystem |
|---|---|---|---|
| `interp1d` (linear/nearest/cubic) | **1** | Linear/nearest trivial (`interp` ships); cubic via the spline below. | `interp1d` direct **NONE**; MathPHP has spline classes. |
| `CubicSpline`, `make_interp_spline` (B-splines) | **2** | Natural/clamped/not-a-knot cubic via tridiagonal solve (ships); general B-splines via de Boor. | `CubicSpline` **PARTIAL** (MathPHP `NaturalCubicSpline`/`ClampedCubicSpline`); B-splines **NONE**. |
| `PchipInterpolator`, `Akima1DInterpolator` | **2** | Monotone Fritsch–Carlson (Pchip) and Akima slope rules — pure. | **NONE**. |
| `UnivariateSpline` (smoothing) | **2** | Penalised least-squares smoothing spline (GCV/`s` parameter). | **NONE**. |
| `RegularGridInterpolator`, `griddata`, `RBFInterpolator` | **2** | Multilinear/cubic on tensor grids; `griddata` linear via Delaunay (needs Tier-3 geometry) or nearest; RBF via a dense solve (ships). | `RegularGridInterpolator` **PARTIAL** (MathPHP); `griddata`/RBF **NONE**. |

## scipy.signal gaps  *(entire area is NONE in PHP — highest unique value)*

| Gap | Tier | Approach in Tessero | PHP ecosystem |
|---|---|---|---|
| `fftconvolve`, `oaconvolve` | **1** | Pad → `fftn` → multiply → `ifftn`; overlap-add blocking for long signals. Builds on nd-FFT. | **NONE**. |
| `butter`, `cheby1/2`, `filtfilt`, `sosfilt` | **2** | Analog prototype → bilinear transform → second-order sections; `filtfilt` = forward/backward `sosfilt` with edge padding. `lfilter` ships. | **NONE**. |
| `welch`, `periodogram`, `spectrogram`, `stft` | **2** | Windowed/overlapped `rfft` framing (windows + `rfft` ship); Welch = averaged periodograms. | **NONE**. |
| `find_peaks`, `savgol_filter`, `detrend`, `resample` | **2** | Local-maximum scan with prominence/width; Savitzky–Golay = local polynomial lstsq; detrend = polynomial/linear removal; resample = FFT or polyphase. | **NONE**. |

## scipy.integrate gaps

| Gap | Tier | Approach in Tessero | PHP ecosystem |
|---|---|---|---|
| `simpson`, `cumulative_trapezoid` | **1** | Composite Simpson / running trapezoid over samples. `trapezoid` + adaptive `quad` ship. | `simpson` **COVERED** (MathPHP); `cumulative_trapezoid` **NONE**. |
| `dblquad` (2-D adaptive) | **2** | Nested adaptive quadrature over our 1-D `quad`. Callback friction (PHP integrand). | **NONE**. |
| `solve_ivp` (RK45/RK23/DOP853/Radau/BDF), `odeint` | **2** | Embedded Runge–Kutta (Dormand–Prince 5(4)) with step control for non-stiff; implicit Radau/BDF for stiff (needs our `solve`). Callback friction (PHP RHS per step). | **NONE** — no ODE solver exists in PHP. |

## scipy.spatial / cluster / ndimage gaps

| Gap | Tier | Approach in Tessero | PHP ecosystem |
|---|---|---|---|
| `distance.cdist/pdist`, `KDTree` | ships | Already provided (`Distance`, `Spatial\KDTree`). | `cdist/pdist` **NONE** as such; KDTree **NONE**. |
| `ConvexHull`, `Delaunay`, `Voronoi` | **3** | Robust computational geometry → FFI to **QHull** (`libqhull_r`). Native Quickhull/incremental Delaunay is possible for 2-D/3-D but robustness (degeneracies) is the hard part. | **NONE** (only ad-hoc 2-D gists). |
| `cluster.vq.kmeans2`, `hierarchy.linkage` | **2** | k-means (Lloyd + k-means++ seeding) using our `cdist`; hierarchical via Lance–Williams agglomeration. | k-means **COVERED** (RubixML); hierarchical/`linkage` **NONE**. |
| ndimage: `gaussian_filter`, `median_filter`, `uniform_filter`, `sobel`, grey morphology, `map_coordinates`, `affine_transform` | **2** | Separable convolutions (Gaussian/uniform/sobel), rank filters (median), grey morphology; geometric transforms via coordinate mapping + interpolation. We ship `label`, `maximumFilter`, binary morphology, `convolve`. | **NONE** as an array API (GD/Imagick only operate on raster images). |

---

## Prioritisation

1. **Tier 1 (do first — safe, fast, high count-impact, verifiable):** nd-FFT, `fftconvolve`; ufunc fills (`gcd/lcm/heaviside/float_power/ldexp/frexp/divmod`); `multivariate_normal/dirichlet/multinomial`; `softmax/log_softmax`, `perm`; `simpson/cumulative_trapezoid`; `minimize_scalar`; sparse `diags`; `Polynomial` wrapper.
2. **Tier 2, high unique value (PHP has nothing):** the whole **signal** suite, **ODE** `solve_ivp`, **interpolation** splines, **ndimage** filters, sparse iterative/`csgraph`, dense optimisers + `nnls` + `linear_sum_assignment`, `expm`.
3. **Tier 3 (decide on dependency vs native):** Schur/QZ + `logm/sqrtm` (cheap — LAPACK already linked); sparse eig (native Lanczos recommended); `spsolve` (SuiteSparse vs native LU); geometry (QHull FFI); HiGHS (FFI, scale only).
4. **Tier 4:** document as out-of-scope (above); no implementation.

## Process guardrail (so coverage never silently drifts again)

Every item lands through the standard loop — spec/table → fixtures in the pinned env → kernel → façades →
`run-tests.php` both backends → ledger — and the coverage page is regenerated (`api-dump.php` →
`gap_inventory.py`). Add a gate check that **every façade class is referenced by at least one inventory row**
and that **`api-dump.php` dumps every generated façade class** (today it omits `ScipyLinalg`, `Distance`,
`Signal`, `Ndimage`, `Csgraph`), so a shipped-but-unlisted capability fails CI instead of silently reading
as 0 %.
