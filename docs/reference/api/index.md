# API reference

Generated from the code by `tools/gen-api-docs.php` (signatures and docblocks). The C ABI is in [C ABI](c-abi.md).

## Core arrays

| Class | Summary |
|---|---|
| [`Tessero\NDArray`](tessero-ndarray.md) | An n-dimensional, homogeneously typed array in native memory. |
| [`Tessero\Math`](tessero-math.md) | Universal functions (ufuncs): element-wise functions over arrays of any shape, with NumPy's broadcasting and type rules and an optional output array. |
| [`Tessero\DType`](tessero-dtype.md) | Element types. The integer values are the libtessero ABI codes. |
| [`Tessero\Tessero`](tessero-tessero.md) | Process-wide settings and diagnostics. |

## Linear algebra, FFT, random

| Class | Summary |
|---|---|
| [`Tessero\Linalg\Linalg`](tessero-linalg-linalg.md) | numpy.linalg / scipy.linalg on LAPACK (row-major LAPACKE, float64). |
| [`Tessero\Fft\Fft`](tessero-fft-fft.md) | numpy.fft: any length (mixed radix 2/3/4/5 + Bluestein for large prime |
| [`Tessero\Random\Generator`](tessero-random-generator.md) | PCG64 generator seeded through NumPy's SeedSequence. |

## Statistics and special functions

| Class | Summary |
|---|---|
| [`Tessero\Special`](tessero-special.md) | scipy.special: special functions (gamma, Bessel, error functions, orthogonal polynomials, ...). Element-wise: arguments broadcast as in NumPy, all-scalar calls return floats. |
| [`Tessero\Stats`](tessero-stats.md) | scipy.stats: probability distributions and statistical functions. |
| [`Tessero\Stats\Distribution`](tessero-stats-distribution.md) | A frozen scipy.stats distribution: Stats::gamma(2.5, loc: 1.0, scale: 3.0). |
| [`Tessero\Np`](tessero-np.md) | NumPy functions beyond the NDArray methods: statistics, NaN-aware reductions, quantiles, histograms, set operations, searching (numpy.*). |
| [`Tessero\Native\Registry`](tessero-native-registry.md) | Calls into libtessero's function registry (ADR 0011): scipy.special functions, statistics and the other |

## Sparse

| Class | Summary |
|---|---|
| [`Tessero\Sparse\CsrMatrix`](tessero-sparse-csrmatrix.md) | Compressed sparse row matrix of float64 (scipy.sparse.csr_matrix). |

## Optimisation

| Class | Summary |
|---|---|
| [`Tessero\Optimize\Minimize`](tessero-optimize-minimize.md) | scipy.optimize.minimize for smooth problems of modest dimension, with the |
| [`Tessero\Optimize\Root`](tessero-optimize-root.md) | Scalar and vector root finding (scipy.optimize.brentq, newton, root). |
| [`Tessero\Optimize\LeastSquares`](tessero-optimize-leastsquares.md) | Non-linear least squares by Levenberg-Marquardt (scipy.optimize.least_squares |
| [`Tessero\Optimize\LinearProgramming`](tessero-optimize-linearprogramming.md) | scipy.optimize.linprog and scipy.optimize.milp on libtessero's native |
| [`Tessero\Optimize\OptimizeResult`](tessero-optimize-optimizeresult.md) | scipy.optimize.OptimizeResult |
| [`Tessero\Optimize\LinprogResult`](tessero-optimize-linprogresult.md) | Result of LinearProgramming::linprog (field names follow scipy.optimize.OptimizeResult). |
| [`Tessero\Optimize\MilpResult`](tessero-optimize-milpresult.md) | Result of LinearProgramming::milp. |

## Markov decision processes

| Class | Summary |
|---|---|
| [`Tessero\Mdp\MarkovDecisionProcess`](tessero-mdp-markovdecisionprocess.md) | A finite Markov decision process with discounted rewards, solved natively. |
| [`Tessero\Mdp\MdpResult`](tessero-mdp-mdpresult.md) | Solution of a Markov decision process. |

## I/O

| Class | Summary |
|---|---|
| [`Tessero\Io\Npy`](tessero-io-npy.md) | NumPy .npy (format 1.0 / 2.0 / 3.0) and .npz reading and writing, so arrays |

## Native layer (FFI)

| Class | Summary |
|---|---|
| [`Tessero\Native\Library`](tessero-native-library.md) | Loads libtessero once per process. |
| [`Tessero\Native\Buffer`](tessero-native-buffer.md) | A block of native memory owned by exactly one PHP object. |
| [`Tessero\Native\Blas`](tessero-native-blas.md) | Optional BLAS + LAPACKE binding (OpenBLAS, LP64 / 32-bit integers). |
| [`Tessero\Native\Abi`](tessero-native-abi.md) | Operation codes, mirrored from csrc/src/ops.h. AbiTest checks they stay in sync. |

## Exceptions

| Class | Summary |
|---|---|
| [`Tessero\Exceptions\TesseroException`](tessero-exceptions-tesseroexception.md) | Base class for every Tessero error. |
| [`Tessero\Exceptions\ShapeError`](tessero-exceptions-shapeerror.md) | Shapes cannot be broadcast, reshaped or multiplied together. |
| [`Tessero\Exceptions\IndexError`](tessero-exceptions-indexerror.md) | An index or slice lies outside the array. |
| [`Tessero\Exceptions\DTypeError`](tessero-exceptions-dtypeerror.md) | The operation is not defined for this dtype. |
| [`Tessero\Exceptions\MemoryError`](tessero-exceptions-memoryerror.md) | Native allocation failed or the memory budget (Tessero::setMemoryBudget) was exceeded. |
| [`Tessero\Exceptions\LibraryUnavailable`](tessero-exceptions-libraryunavailable.md) | libtessero (or BLAS/LAPACK for the requested routine) could not be loaded in this SAPI. |
| [`Tessero\Exceptions\LinAlgError`](tessero-exceptions-linalgerror.md) | Base class for linear-algebra failures (numpy.linalg.LinAlgError). |
| [`Tessero\Exceptions\SingularMatrix`](tessero-exceptions-singularmatrix.md) | The matrix is exactly singular to working precision. |
| [`Tessero\Exceptions\NotPositiveDefinite`](tessero-exceptions-notpositivedefinite.md) | Cholesky factorisation found a non-positive leading minor. |
| [`Tessero\Exceptions\ConvergenceError`](tessero-exceptions-convergenceerror.md) | An iterative routine (SVD, eigen solver, Krylov, optimiser) did not converge. |

## Laravel bridge

| Class | Summary |
|---|---|
| [`Tessero\Laravel\TesseroManager`](tessero-laravel-tesseromanager.md) | One entry point for applications, whichever backend is installed. |
| [`Tessero\Laravel\TesseroServiceProvider`](tessero-laravel-tesseroserviceprovider.md) | Registers Tessero in a Laravel application: |
| [`Tessero\Laravel\Casts\AsNDArray`](tessero-laravel-casts-asndarray.md) | Eloquent cast between a column and an NDArray. |
| [`Tessero\Laravel\Casts\TensorCast`](tessero-laravel-casts-tensorcast.md) | Alias of AsNDArray under the name used in the original design notes: |
| [`Tessero\Laravel\Rules\NumericArray`](tessero-laravel-rules-numericarray.md) | Validates request input that will become an NDArray: a (nested) list of |

## Native extension (ext-tessero)

| Class | Summary |
|---|---|
| [`Tessero\Ext\NDArray`](tessero-ext-ndarray.md) |  |
| [`Tessero\Ext\Math`](tessero-ext-math.md) |  |
| [`Tessero\Ext\Engine`](tessero-ext-engine.md) |  |
| [`Tessero\Ext\Operand`](tessero-ext-operand.md) |  |
| [`Tessero\Ext\Special`](tessero-ext-special.md) |  |
| [`Tessero\Ext\Stats`](tessero-ext-stats.md) |  |
| [`Tessero\Ext\Distribution`](tessero-ext-distribution.md) |  |
| [`Tessero\Ext\Np`](tessero-ext-np.md) |  |
| [`Tessero\Ext\Random\Generator`](tessero-ext-random-generator.md) |  |
| [`Tessero\Ext\Exception`](tessero-ext-exception.md) |  |
| [`Tessero\Ext\ShapeException`](tessero-ext-shapeexception.md) |  |
| [`Tessero\Ext\IndexException`](tessero-ext-indexexception.md) |  |
| [`Tessero\Ext\DTypeException`](tessero-ext-dtypeexception.md) |  |
| [`Tessero\Ext\MemoryException`](tessero-ext-memoryexception.md) |  |

