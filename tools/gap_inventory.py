"""
NumPy / SciPy coverage inventory for Tessero.

Every row names a NumPy or SciPy function and the Tessero call(s) that
provide it in the FFI package and in the native extension. Every reference is
checked against the public API dumped from the code (tools/api-dump.php), so
the gap analysis cannot claim a function that does not exist.

    php -d ffi.enable=1 -d extension=tessero tools/api-dump.php > build/api.json
    python tools/gap_inventory.py build/api.json > docs/project/numpy-scipy-coverage.md

Reference syntax: "Class::method" (short class aliases below), several refs
separated by spaces; "syntax:..." for language-level support (checked by the
test suites, not by this script). A leading "~" on a note marks partial support in both packages, "~X " in the
extension only, "~F " in the FFI package only.
The row set is the commonly used public API, not every NumPy/SciPy symbol.
"""
import json
import sys
from collections import OrderedDict

ALIAS = {
    "N": "Tessero\\NDArray", "M": "Tessero\\Math", "T": "Tessero\\Tessero", "L": "Tessero\\Linalg\\Linalg",
    "F": "Tessero\\Fft\\Fft", "R": "Tessero\\Random\\Generator", "S": "Tessero\\Sparse\\CsrMatrix",
    "Min": "Tessero\\Optimize\\Minimize", "Root": "Tessero\\Optimize\\Root", "LS": "Tessero\\Optimize\\LeastSquares",
    "LP": "Tessero\\Optimize\\LinearProgramming", "ASG": "Tessero\\Optimize\\Assignment", "MDP": "Tessero\\Mdp\\MarkovDecisionProcess", "Npy": "Tessero\\Io\\Npy",
    "XN": "Tessero\\Ext\\NDArray", "XM": "Tessero\\Ext\\Math", "XE": "Tessero\\Ext\\Engine",
    "XR": "Tessero\\Ext\\Random\\Generator", "NP": "Tessero\\Np", "SP": "Tessero\\Special", "ST": "Tessero\\Stats",
    "XNP": "Tessero\\Ext\\Np", "XSP": "Tessero\\Ext\\Special", "XST": "Tessero\\Ext\\Stats",
    "SL": "Tessero\\ScipyLinalg", "DST": "Tessero\\Distance", "SIG": "Tessero\\Signal",
    "NDI": "Tessero\\Ndimage", "CSG": "Tessero\\Csgraph", "IP": "Tessero\\Interpolate", "CL": "Tessero\\Cluster", "NPY": "Tessero\\Polynomial", "HI": "Tessero\\Hierarchy", "SPL": "Tessero\\SparseLinalg", "Q": "Tessero\\Integrate\\Quad",
    "KD": "Tessero\\Spatial\\KDTree", "DT": "Tessero\\Datetime\\Datetime",
    "SW": "Tessero\\SignalWindows", "Const": "Tessero\\Constants",
    # native-extension registry facades (kernel-backed, so they mirror the FFI classes on the extension)
    "XL": "Tessero\\Ext\\Linalg", "XSL": "Tessero\\Ext\\ScipyLinalg", "XF": "Tessero\\Ext\\Fft\\Fft",
    "XSIG": "Tessero\\Ext\\Signal", "XSW": "Tessero\\Ext\\SignalWindows", "XDST": "Tessero\\Ext\\Distance",
    "XNDI": "Tessero\\Ext\\Ndimage", "XCSG": "Tessero\\Ext\\Csgraph", "XIP": "Tessero\\Ext\\Interpolate", "XCL": "Tessero\\Ext\\Cluster", "XNPY": "Tessero\\Ext\\Polynomial", "XHI": "Tessero\\Ext\\Hierarchy", "XSPL": "Tessero\\Ext\\SparseLinalg",
}

# (area, [(name, ffi refs, ext refs, note)])
ROWS = OrderedDict()


def area(name, rows):
    ROWS[name] = rows


area("numpy: array creation", [
    ("array / asarray", "N::array N::asArray", "XN::array", ""),
    ("zeros / ones / empty / full", "N::zeros N::ones N::empty N::full", "XN::zeros XN::ones XN::full", "extension has no empty()"),
    ("arange / linspace", "N::arange N::linspace", "XN::arange XN::linspace", ""),
    ("eye / identity", "N::eye", "XN::eye", ""),
    ("frombuffer", "N::fromBytes", "XN::fromBytes", ""),
    ("zeros_like / ones_like / full_like", "NP::zerosLike NP::onesLike NP::fullLike", "XNP::zerosLike XNP::onesLike XNP::fullLike", ""),
    ("logspace / geomspace", "NP::logspace NP::geomspace", "XNP::logspace XNP::geomspace", ""),
    ("meshgrid / mgrid / ogrid", "NP::meshgrid", "XNP::meshgrid", "~meshgrid only (no mgrid/ogrid)"),
    ("diag / diagflat / tri", "NP::diag NP::diagflat NP::tri", "XNP::diag XNP::diagflat XNP::tri", ""),
    ("fromfunction / fromiter", "NP::fromiter", "XNP::fromiter", "~fromiter only (fromfunction is a Python-lambda idiom)"),
])
area("numpy: shape and layout", [
    ("reshape / ravel / flatten", "N::reshape N::ravel N::flatten", "XN::reshape XN::ravel XN::flatten", ""),
    ("transpose / .T", "N::transpose N::t", "XN::transpose XN::t", ""),
    ("swapaxes / moveaxis", "N::swapAxes N::moveAxis", "XNP::swapaxes XNP::moveaxis", ""),
    ("expand_dims / squeeze", "N::expandDims N::squeeze", "XNP::expandDims XNP::squeeze", ""),
    ("broadcast_to / broadcast_shapes", "N::broadcastTo N::broadcastShapes", "XNP::broadcastTo XNP::broadcastShapes", ""),
    ("concatenate / stack", "N::concatenate N::stack", "XNP::concatenate XNP::stack", ""),
    ("vstack / hstack / dstack / column_stack", "NP::vstack NP::hstack NP::dstack NP::columnStack", "XNP::vstack XNP::hstack XNP::dstack XNP::columnStack", ""),
    ("split / array_split", "NP::split NP::arraySplit", "XNP::split XNP::arraySplit", ""),
    ("tile / repeat", "NP::tile NP::repeat", "XNP::tile XNP::repeat", ""),
    ("flip / fliplr / flipud", "N::slice", "XN::slice", "~via '::-1' slices"),
    ("roll / rot90", "NP::roll NP::rot90", "XNP::roll XNP::rot90", ""),
    ("pad", "NP::pad", "XNP::pad", ""),
    ("copy / astype / view", "N::copy N::astype N::view", "XN::copy XN::astype XN::view", ""),
    ("ascontiguousarray", "N::contiguous", "XNP::ascontiguousarray", ""),
])
area("numpy: indexing", [
    ("basic slicing, None, Ellipsis", "N::slice N::sliceNative", "XN::slice", "same C parser; $a['1:, ::-1']"),
    ("boolean masks (get and set)", "N::filter N::setWhere", "syntax:$a[$mask]", ""),
    ("integer arrays along one axis (take / put)", "N::take N::put", "XN::take XN::put", ""),
    ("multi-axis fancy indexing (a[i, j] with arrays)", "NP::takeAlongAxis NP::ix_", "XNP::takeAlongAxis XNP::ix_", "~via take_along_axis / ix_"),
    ("np.ix_ / advanced assignment with broadcasting", "NP::ix_ NP::putAlongAxis NP::putmask", "XNP::ix_ XNP::putAlongAxis XNP::putmask", "~ix_ and put_along_axis/putmask"),
    ("where (select)", "N::where", "XN::where", ""),
    ("nonzero / flatnonzero / argwhere", "NP::nonzero NP::flatnonzero NP::argwhere", "XNP::nonzero XNP::flatnonzero XNP::argwhere", ""),
    ("searchsorted", "NP::searchsorted", "XNP::searchsorted", ""),
    ("item / tolist", "N::item N::toList N::toArray", "XN::item XN::toList XN::toArray", ""),
])
area("numpy: element-wise maths (ufuncs)", [
    ("arithmetic: add subtract multiply divide power mod floor_divide negative", "M::add M::subtract M::multiply M::divide M::power M::mod M::floorDivide M::negative",
     "XM::add XM::subtract XM::multiply XM::divide XM::power XM::mod XM::floorDivide XM::negative", "operators + - * / ** % on the extension"),
    ("exp log log2 log10 expm1 log1p exp2 sqrt cbrt square reciprocal", "M::exp M::log M::log2 M::log10 M::expm1 M::log1p M::exp2 M::sqrt M::cbrt M::square M::reciprocal",
     "XM::exp XM::log XM::cbrt", "exp/log vectorised in-house (< 1 ulp)"),
    ("trig and inverse trig, hyperbolic and inverse", "M::sin M::cos M::tan M::arcsin M::arccos M::arctan M::arctan2 M::sinh M::cosh M::tanh M::arcsinh M::arccosh M::arctanh",
     "XM::sin XM::arctanh", "~scalar libm: sin/cos not vectorised yet"),
    ("rounding: floor ceil rint trunc round", "M::floor M::ceil M::rint M::trunc N::round", "XM::floor XM::trunc", ""),
    ("comparison and logic", "M::equal M::less M::logicalAnd M::logicalXor", "XM::equal XM::logicalXor", ""),
    ("maximum minimum fmax fmin", "M::maximum M::minimum M::fmax M::fmin", "XM::fmax", ""),
    ("abs sign copysign signbit nextafter hypot", "M::absolute M::sign M::copysign M::signbit M::nextafter M::hypot", "XM::copysign XM::hypot", ""),
    ("isnan isinf isfinite", "M::isnan M::isinf M::isfinite", "XM::isnan", ""),
    ("logaddexp logaddexp2 degrees radians fmod", "M::logaddexp M::logaddexp2 M::degrees M::radians M::fmod", "XM::logaddexp XM::fmod", ""),
    ("real imag conj angle", "M::real M::imag M::conj M::angle", "XM::real XM::angle", ""),
    ("bitwise_and/or/xor, invert, shifts", "M::invert NP::bitwiseAnd NP::bitwiseOr NP::bitwiseXor", "XM::invert XNP::bitwiseAnd XNP::bitwiseOr XNP::bitwiseXor", "~no shifts"),
    ("gcd lcm heaviside sinc", "M::gcd M::lcm M::heaviside NP::sinc", "XM::gcd XM::lcm XM::heaviside XNP::sinc", ""),
    ("ldexp frexp modf divmod float_power spacing", "M::floatPower NP::modf NP::spacing", "XM::floatPower XNP::modf XNP::spacing", "~float_power, modf, spacing (no ldexp/frexp/divmod)"),
    ("clip / nan_to_num", "N::clip NP::nanToNum", "XN::clip XNP::nanToNum", ""),
    ("isclose / allclose", "N::isclose N::allclose", "", ""),
    ("out= argument", "M::apply", "XM::apply", "every ufunc; same-kind casting"),
    ("where= mask, reduce/accumulate/outer ufunc methods", "", "", ""),
])
area("numpy: reductions and statistics", [
    ("sum prod min max", "N::sum N::prod N::min N::max", "XN::sum XN::prod XN::min XN::max", "any axis or list of axes, keepdims"),
    ("argmin argmax any all", "N::argmin N::argmax N::any N::all", "XN::argmin XN::argmax XN::any XN::all", ""),
    ("mean var std (ddof)", "N::mean N::var N::std", "XN::mean XN::var XN::std", "NumPy's two-pass algorithm"),
    ("cumsum cumprod", "N::cumsum N::cumprod", "XN::cumsum XN::cumprod", ""),
    ("diff", "N::diff", "XNP::diff", ""),
    ("median / percentile / quantile", "NP::median NP::percentile NP::quantile", "XNP::median XNP::percentile XNP::quantile", "all 13 quantile methods"),
    ("average (weighted) / ptp", "NP::average NP::ptp", "XNP::average XNP::ptp", ""),
    ("nan-aware reductions (nansum, nanmean, ...)", "NP::nansum NP::nanmean NP::nanstd NP::nanvar NP::nanmedian NP::nanmin NP::nanmax", "XNP::nansum XNP::nanmean XNP::nanstd XNP::nanvar XNP::nanmedian XNP::nanmin XNP::nanmax", ""),
    ("count_nonzero", "NP::countNonzero", "XNP::countNonzero", ""),
    ("histogram / bincount / digitize", "NP::histogram NP::bincount NP::digitize", "XNP::histogram XNP::bincount XNP::digitize", ""),
    ("cov / corrcoef", "NP::cov NP::corrcoef", "XNP::cov XNP::corrcoef", ""),
    ("unique / isin / set operations", "NP::unique NP::isin NP::union1d NP::intersect1d NP::setdiff1d", "XNP::unique XNP::isin XNP::union1d XNP::intersect1d XNP::setdiff1d", ""),
])
area("numpy: sorting and searching", [
    ("sort / argsort (any axis)", "N::sort N::argsort", "XN::sort XN::argsort", "~X extension: flattened only; both about 4x slower than NumPy's AVX-512 sort"),
    ("partition / argpartition", "NP::partition NP::argpartition", "XNP::partition XNP::argpartition", ""),
    ("lexsort / searchsorted", "NP::lexsort NP::searchsorted", "XNP::lexsort XNP::searchsorted", ""),
])
area("numpy: linear algebra (numpy.linalg + products)", [
    ("matmul / dot / @", "N::matmul N::dot", "XN::matmul XN::dot", "OpenBLAS on FFI; C kernel on the extension"),
    ("solve inv det slogdet", "L::solve L::inv L::det L::slogdet", "XL::solve XL::inv XL::det XL::slogdet", "stacked inputs"),
    ("cholesky qr svd", "L::cholesky L::qr L::svd", "XL::cholesky XL::qr XL::svd", ""),
    ("eig eigh eigvals eigvalsh", "L::eig L::eigh L::eigvals L::eigvalsh", "XL::eig XL::eigh XL::eigvals XL::eigvalsh", ""),
    ("lstsq pinv norm cond matrix_rank trace", "L::lstsq L::pinv L::norm L::cond L::matrixRank L::trace", "XL::lstsq XL::pinv XL::norm XL::cond XL::matrixRank", "~X extension lacks trace"),
    ("matrix_power multi_dot tensorsolve tensorinv", "", "", ""),
    ("outer inner vdot kron cross tensordot einsum", "NP::outer NP::inner NP::vdot NP::kron NP::cross NP::tensordot NP::einsum", "XNP::outer XNP::inner XNP::vdot XNP::kron XNP::cross XNP::tensordot XNP::einsum", ""),
    ("complex128 decompositions", "", "", "real float64 only"),
])
area("numpy: FFT", [
    ("fft ifft (any length, any axis, norms)", "F::fft F::ifft", "XF::fft XF::ifft", ""),
    ("rfft irfft (half-length real transform)", "F::rfft F::irfft", "XF::rfft XF::irfft", ""),
    ("fft2 ifft2", "F::fft2 F::ifft2", "XF::fft2 XF::ifft2", ""),
    ("fftn ifftn rfft2 rfftn irfftn hfft ihfft", "F::fftn F::ifftn F::rfft2 F::rfftn F::irfftn F::hfft F::ihfft", "XF::fftn XF::ifftn XF::rfft2 XF::rfftn XF::irfftn XF::hfft XF::ihfft", ""),
    ("fftfreq rfftfreq fftshift ifftshift", "F::fftfreq F::rfftfreq F::fftshift F::ifftshift", "XF::fftfreq XF::rfftfreq XF::fftshift XF::ifftshift", ""),
])
area("numpy: random (Generator)", [
    ("default_rng / PCG64 / SeedSequence (bit-identical streams)", "R::defaultRng T::rng", "XR::defaultRng", ""),
    ("random uniform normal standard_normal integers", "R::random R::uniform R::normal R::standardNormal R::integers", "XR::random XR::uniform XR::normal XR::standardNormal XR::integers", ""),
    ("choice permutation shuffle", "R::choice R::permutation R::shuffle", "XR::permutation XR::shuffle", "~X extension: no choice"),
    ("exponential", "R::exponential", "XR::exponential", ""),
    ("gamma beta poisson binomial lognormal chisquare", "R::gamma R::beta R::poisson R::binomial R::lognormal R::chisquare", "XR::gamma XR::beta XR::poisson XR::binomial XR::lognormal XR::chisquare", "NumPy's own distribution code"),
    ("multivariate_normal dirichlet multinomial", "", "", ""),
    ("student_t weibull triangular laplace logistic gumbel pareto geometric", "R::standardT R::weibull R::triangular R::laplace R::logistic R::gumbel R::pareto R::geometric", "XR::standardT XR::weibull XR::triangular XR::laplace XR::logistic XR::gumbel XR::pareto XR::geometric", ""),
    ("getState / setState (persist streams)", "R::getState R::setState", "XR::getState XR::setState", ""),
])
area("numpy: I/O and memory", [
    ("save / load (.npy)", "N::save N::load Npy::save Npy::load", "XN::save XN::load", "byte-identical to numpy.save"),
    ("savez / savez_compressed / load .npz", "Npy::saveZ Npy::loadZ", "", ""),
    ("memmap / lib.format.open_memmap / load(mmap_mode=)", "N::memmap N::openMemmap N::load", "XN::memmap XN::openMemmap XN::load", ""),
    ("tobytes / frombuffer", "N::toBytes N::fromBytes", "XN::toBytes XN::fromBytes", ""),
    ("loadtxt / savetxt / genfromtxt (CSV)", "NP::loadtxt", "XNP::loadtxt", "~loadtxt only (no savetxt/genfromtxt)"),
    ("tofile / fromfile", "NP::fromfile", "XNP::fromfile", "~fromfile only (tofile via save/load)"),
    ("JSON (tolist + json)", "N::toJson", "XN::toJson", "written in C"),
])
area("numpy: dtypes", [
    ("float64 float32 int64 int32 uint8 bool complex128", "N::astype", "XN::astype", "NumPy 2 promotion rules"),
    ("int8 int16 uint16 uint32 uint64", "", "", ""),
    ("float16 complex64 longdouble", "", "", ""),
    ("datetime64 / timedelta64", "DT::datetime64 DT::timedelta64", "", ""),
    ("str / bytes / object / structured / record arrays", "", "", "out of scope"),
    ("masked arrays (numpy.ma)", "", "", ""),
])
area("numpy: polynomials and signal basics", [
    ("polyfit / polyval / roots / Polynomial", "NP::polyfit NP::polyval NP::roots", "XNP::polyfit XNP::polyval XNP::roots", "~no Polynomial class (polyfit/polyval/roots ship)"),
    ("convolve / correlate", "NP::convolve NP::correlate", "XNP::convolve XNP::correlate", ""),
    ("interp", "NP::interp", "XNP::interp", ""),
    ("gradient / trapezoid", "NP::gradient NP::trapezoid", "XNP::gradient XNP::trapezoid", ""),
])
area("scipy.linalg", [
    ("lu", "L::lu", "XSL::lu", ""),
    ("solve_triangular", "L::solveTriangular", "XSL::solveTriangular", ""),
    ("solve_banded / solveh_banded / eig_banded", "SL::solveBanded SL::solvehBanded SL::eigBanded", "XSL::solveBanded XSL::solvehBanded XSL::eigBanded", ""),
    ("tril / triu", "L::tril L::triu", "", "~F FFI package only"),
    ("lu_factor/lu_solve, cho_factor/cho_solve, ldl", "SL::luFactor SL::luSolve SL::choFactor SL::choSolve SL::ldl", "XSL::luFactor XSL::luSolve XSL::choFactor XSL::choSolve XSL::ldl", ""),
    ("expm logm sqrtm", "SL::expm SL::logm SL::sqrtm", "XSL::expm XSL::logm XSL::sqrtm", "Schur-Parlett (real spectrum)"),
    ("sinm cosm tanm sinhm coshm tanhm", "SL::sinm SL::cosm SL::tanm SL::sinhm SL::coshm SL::tanhm", "XSL::sinm XSL::cosm XSL::tanm XSL::sinhm XSL::coshm XSL::tanhm", "Schur-Parlett (real spectrum)"),
    ("fractional_matrix_power", "SL::fractionalMatrixPower", "XSL::fractionalMatrixPower", "Schur-Parlett x^t (positive real spectrum)"),
    ("schur qz hessenberg", "SL::schur SL::qz SL::hessenberg", "XSL::schur XSL::qz XSL::hessenberg", ""),
    ("null_space orth block_diag polar", "SL::nullSpace SL::orth SL::blockDiag SL::polar", "XSL::nullSpace XSL::orth XSL::blockDiag XSL::polar", ""),
    ("special matrices: toeplitz circulant companion hadamard hilbert khatri_rao diagsvd",
     "SL::toeplitz SL::circulant SL::companion SL::hadamard SL::hilbert SL::khatriRao SL::diagsvd",
     "XSL::toeplitz XSL::circulant XSL::companion XSL::hadamard XSL::hilbert XSL::khatriRao XSL::diagsvd", ""),
])
area("scipy.optimize", [
    ("minimize: BFGS, L-BFGS-B, Nelder-Mead", "Min::minimize", "", "same iterates as SciPy for BFGS/Nelder-Mead"),
    ("minimize: CG, Newton-CG, Powell, TNC, trust-region", "", "", ""),
    ("minimize with constraints: SLSQP, COBYLA, trust-constr", "", "", ""),
    ("minimize_scalar / fminbound", "Min::minimizeScalar Min::fminbound", "", "brent, bounded (no golden)"),
    ("brentq / newton / secant", "Root::brentq Root::newton", "", "brentq bit-identical to SciPy"),
    ("bisect / brenth / ridder / toms748 / root_scalar", "Root::bisect Root::brenth Root::ridder Root::toms748 Root::rootScalar", "", ""),
    ("fixed_point / bracket", "Root::fixedPoint Min::bracket", "", ""),
    ("root (vector) / fsolve", "Root::root Root::fsolve", "", "damped Newton; matches scipy's root for well-posed systems"),
    ("least_squares / curve_fit", "LS::leastSquares LS::curveFit", "", "~Levenberg-Marquardt (unbounded; no trf/dogbox bounds)"),
    ("linprog", "LP::linprog", "XE::linprog", "~dense simplex; practical to a few thousand rows"),
    ("milp", "LP::milp", "XE::milp", "~dense branch and bound"),
    ("HiGHS for large sparse LP/MILP", "", "", ""),
    ("linear_sum_assignment / nnls / lsq_linear", "ASG::linearSumAssignment LS::nnls LS::lsqLinear", "", "~lsq_linear unbounded only"),
    ("global: differential_evolution, basinhopping, dual_annealing, shgo", "", "", ""),
    ("approx_fprime", "Min::approxGrad", "", ""),
])
area("scipy.sparse", [
    ("csr_matrix (from dense, triplets, arrays)", "S::fromDense S::fromTriplets S::fromArrays", "", "float64"),
    ("sparse @ dense, transpose, diagonal, scale", "S::dot S::transpose S::diagonal S::scale", "", ""),
    ("identity / eye", "S::identity", "", ""),
    ("csc / coo / lil / dok / bsr / dia classes", "", "", ""),
    ("sparse @ sparse, sparse + sparse, kron, hstack/vstack", "", "", ""),
    ("diags / random", "", "", ""),
    ("linalg.cg / bicgstab", "S::cg S::bicgstab", "", "Jacobi-preconditioned"),
    ("linalg.gmres / minres / lsqr / lsmr", "", "", ""),
    ("linalg.spsolve / splu / factorized", "", "", ""),
    ("linalg.eigsh / eigs / svds", "", "", ""),
    ("csgraph (shortest paths, components)", "CSG::connectedComponents", "", "~connected components (no shortest paths yet)"),
])
area("scipy.special", [
    ("erf erfc", "M::erf M::erfc", "XM::erf XM::erfc", ""),
    ("gamma gammaln", "M::gamma M::lgamma", "XM::gamma XM::lgamma", ""),
    ("logsumexp (pairwise: logaddexp)", "SP::logsumexp", "XSP::logsumexp", ""),
    ("expit logit softmax log_softmax xlogy", "SP::expit SP::logit SP::softmax SP::logSoftmax SP::xlogy", "XSP::expit XSP::logit XSP::softmax XSP::logSoftmax XSP::xlogy", ""),
    ("beta betaln digamma polygamma", "SP::beta SP::betaln SP::digamma SP::polygamma", "XSP::beta XSP::betaln XSP::digamma XSP::polygamma", ""),
    ("erfinv ndtr ndtri", "SP::erfinv SP::ndtr SP::ndtri", "XSP::erfinv XSP::ndtr XSP::ndtri", ""),
    ("comb perm factorial", "SP::comb SP::factorial", "XSP::comb XSP::factorial", "~comb and factorial (no perm)"),
    ("bessel functions (jv, iv, kv, yv)", "SP::jv SP::iv SP::kv SP::yv", "XSP::jv XSP::iv XSP::kv XSP::yv", "~real arguments only"),
    ("gammainc / betainc (incomplete)", "SP::gammainc SP::gammaincc SP::betainc", "XSP::gammainc XSP::gammaincc XSP::betainc", ""),
    ("softplus / sinc", "SP::softplus SP::sinc", "XSP::softplus XSP::sinc", ""),
    ("spherical_jn / yn / in / kn (and derivatives)", "SP::sphericalJn SP::sphericalYn SP::sphericalIn SP::sphericalKn",
     "XSP::sphericalJn XSP::sphericalYn XSP::sphericalIn XSP::sphericalKn", ""),
])
area("scipy.stats", [
    ("continuous distributions (norm, t, chi2, f, gamma, beta, lognorm, expon, ...): pdf cdf ppf rvs", "ST::norm ST::t ST::chi2 ST::f ST::gamma ST::beta ST::lognorm ST::expon", "XST::norm XST::t XST::chi2 XST::f XST::gamma XST::beta XST::lognorm XST::expon", "every scipy.stats continuous distribution"),
    ("discrete distributions (poisson, binom, nbinom, geom, poisson_binom, ...)", "ST::poisson ST::binom ST::nbinom ST::geom ST::poissonBinom", "XST::poisson XST::binom XST::nbinom XST::geom XST::poissonBinom", "every scipy.stats discrete distribution"),
    ("describe, zscore, mode, skew, kurtosis", "ST::describe ST::zscore ST::mode ST::skew ST::kurtosis", "XST::describe XST::zscore XST::mode XST::skew XST::kurtosis", ""),
    ("pearsonr, spearmanr, kendalltau, linregress", "ST::pearsonr ST::spearmanr ST::kendalltau ST::linregress", "XST::pearsonr XST::spearmanr XST::kendalltau XST::linregress", ""),
    ("t-tests, chi-square, Mann-Whitney, KS tests", "ST::ttestInd ST::ttestRel ST::ttest1samp ST::chisquare ST::mannwhitneyu ST::kstest", "XST::ttestInd XST::ttestRel XST::ttest1samp XST::chisquare XST::mannwhitneyu XST::kstest", ""),
    ("gaussian_kde", "", "", ""),
])
area("scipy.interpolate", [
    ("interp1d / make_interp_spline", "", "", ""),
    ("CubicSpline / PchipInterpolator / Akima1DInterpolator", "", "", ""),
    ("UnivariateSpline (smoothing)", "", "", ""),
    ("RegularGridInterpolator / griddata / RBFInterpolator", "", "", ""),
])
area("scipy.signal", [
    ("convolve / fftconvolve / oaconvolve", "SIG::convolve", "XSIG::convolve", "~direct convolve (no fftconvolve/oaconvolve)"),
    ("butter / cheby / lfilter / filtfilt / sosfilt", "SIG::lfilter", "XSIG::lfilter", "~lfilter only (no filter design/filtfilt/sosfilt)"),
    ("welch / periodogram / spectrogram / stft", "", "", ""),
    ("find_peaks / savgol_filter / detrend / resample", "", "", ""),
])
area("scipy.signal.windows", [
    ("hann / hamming / blackman / bartlett / boxcar / triang", "SW::hann SW::hamming SW::blackman SW::bartlett SW::boxcar SW::triang",
     "XSW::hann XSW::hamming XSW::blackman XSW::bartlett XSW::boxcar XSW::triang", ""),
    ("blackmanharris / nuttall / flattop / cosine / lanczos / bohman / barthann / parzen",
     "SW::blackmanharris SW::nuttall SW::flattop SW::cosine SW::lanczos SW::bohman SW::barthann SW::parzen",
     "XSW::blackmanharris XSW::nuttall XSW::flattop XSW::cosine XSW::lanczos XSW::bohman XSW::barthann XSW::parzen", ""),
    ("kaiser / gaussian / general_gaussian / exponential / tukey", "SW::kaiser SW::gaussian SW::generalGaussian SW::exponential SW::tukey",
     "XSW::kaiser XSW::gaussian XSW::generalGaussian XSW::exponential XSW::tukey", ""),
    ("general_cosine / general_hamming", "SW::generalCosine SW::generalHamming", "XSW::generalCosine XSW::generalHamming", ""),
    ("chebwin / dpss / taylor / kaiser_bessel_derived / get_window", "", "", ""),
])
area("scipy.integrate", [
    ("quad / dblquad", "Q::quad", "", "~quad only (no dblquad)"),
    ("solve_ivp / odeint", "", "", ""),
    ("trapezoid / simpson / cumulative_trapezoid", "NP::trapezoid NP::simpson NP::cumulativeTrapezoid", "XNP::trapezoid XNP::simpson XNP::cumulativeTrapezoid", ""),
])
area("scipy.spatial, cluster, ndimage", [
    ("distance.cdist / pdist", "DST::cdist DST::pdist", "", ""),
    ("KDTree / cKDTree", "KD::query", "", ""),
    ("ConvexHull / Delaunay / Voronoi", "", "", ""),
    ("cluster.vq.kmeans2 / hierarchy.linkage", "", "", ""),
    ("ndimage filters / labelling / morphology", "NDI::label NDI::maximumFilter NDI::binaryErosion NDI::binaryDilation", "", "~labelling, max filter, binary morphology (no gaussian/median/affine yet)"),
])
area("Beyond NumPy/SciPy (Tessero additions)", [
    ("Markov decision processes (value, policy, modified PI, finite horizon)", "MDP::valueIteration MDP::policyIteration MDP::finiteHorizon",
     "XE::mdpValueIteration XE::mdpPolicyIteration XE::mdpFiniteHorizon", "pymdptoolbox-style, in C, OpenMP"),
    ("native memory budget and accounting", "T::setMemoryBudget T::memoryInUse", "XE::setMemoryBudget XE::memoryInUse", "no NumPy equivalent"),
    ("Laravel integration (casts, rule, facade, Octane-safe settings)", "", "", "tessero/laravel package"),
])


def main(api_path):
    api = {c["class"]: {m["name"] for m in c["methods"]} for c in json.load(open(api_path))}
    ext_loaded = "Tessero\\Ext\\NDArray" in api
    errors = []

    def ok(refs):
        if not refs:
            return False
        for r in refs.split():
            if r.startswith("syntax:"):
                continue
            cls, meth = r.split("::")
            full = ALIAS[cls]
            if full.startswith("Tessero\\Ext\\") and not ext_loaded:
                continue
            if meth not in api.get(full, set()):
                errors.append(f"{r} does not exist")
        return True

    out = []
    total = {"rows": 0, "ffi": 0, "ext": 0, "ffi_full": 0, "partial": 0}
    summary = []
    for name, rows in ROWS.items():
        n = ffi = ext = part = 0
        lines = [f"### {name}\n", "| NumPy / SciPy | FFI package | Extension | Notes |", "|---|:---:|:---:|---|"]
        for fn, fref, xref, note in rows:
            has_f, has_x = ok(fref), ok(xref)
            pf = note.startswith("~") and not note.startswith("~X ")
            px = note.startswith("~") and not note.startswith("~F ")
            partial = pf or px
            note_text = note[3:] if note[:3] in ("~X ", "~F ") else note.lstrip("~")
            mark = lambda has, p: ("◐" if p else "●") if has else "—"
            n += 1
            ffi += has_f
            ext += has_x
            part += (pf and has_f) or (px and has_x)
            refs = " ".join(f"`{r.replace('syntax:', '')}`" for r in (fref + " " + xref).split())
            lines.append(f"| {fn} | {mark(has_f, pf)} | {mark(has_x, px)} | {note_text}{(' — ' if note_text and refs else '') + refs if refs else ''} |")
        summary.append((name, n, ffi, ext))
        total["rows"] += n
        total["ffi"] += ffi
        total["ext"] += ext
        total["partial"] += part
        out.append("\n".join(lines) + "\n")
    if errors:
        sys.exit("gap inventory references missing API:\n" + "\n".join(sorted(set(errors))))

    head = ["| Area | Functions tracked | FFI package | Extension |", "|---|---:|---:|---:|"]
    for name, n, f, x in summary:
        head.append(f"| {name} | {n} | {f} ({round(100 * f / n)} %) | {x} ({round(100 * x / n)} %) |")
    head.append(f"| **All areas** | **{total['rows']}** | **{total['ffi']} ({round(100 * total['ffi'] / total['rows'])} %)** | "
                f"**{total['ext']} ({round(100 * total['ext'] / total['rows'])} %)** |")
    print("<!-- generated by tools/gap_inventory.py from tools/api-dump.php output; do not edit by hand -->\n")
    print("# NumPy / SciPy coverage inventory\n")
    print("Every row is a commonly used NumPy or SciPy function (or group). ● supported, ◐ partly supported "
          "(see notes), — not available. Each Tessero reference in the notes was checked against the public API "
          "of the code when this page was generated, so the page cannot claim a function that does not exist. "
          f"{total['partial']} rows are partial.\n")
    print("## Summary\n")
    print("\n".join(head) + "\n")
    print("Counts are rows, not NumPy symbols: a row can group several related functions. "
          "The [gap analysis](numpy-scipy-gap-analysis.md) interprets these numbers.\n")
    print("## Symbol coverage (M2 to M5)\n")
    print("Every public NumPy/SciPy symbol of the reference versions, from `tools/parity/census.py`, classified by "
          "`tools/parity/report.py`. Where a fixture does not simply hold SciPy's value, the "
          "[reference deviations](reference-deviations.md) page says why.\n")
    print('--8<-- "docs/project/_generated/metrics.md"\n')
    print("## Detail\n")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "build/api.json")
