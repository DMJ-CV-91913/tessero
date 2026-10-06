# Error codes and exceptions

## Kernel return codes

Every `libtessero` function that can fail returns `int`: 0 on success, a
negative code on error. Solvers also use small positive codes for
non-exceptional outcomes (for example "unbounded"). Those are returned as
result statuses, never thrown.

| Code | Name | Meaning | FFI package throws | Extension throws |
|---:|---|---|---|---|
| 0 | `TSR_OK` | success | | |
| −1 | `TSR_EARG` | invalid argument (null pointer, negative size, NaN in solver data, malformed CSR) | `TesseroException` (LP/MILP: `InvalidArgumentException`) | `Tessero\Ext\Exception` (LP/MILP: `ValueError`) |
| −2 | `TSR_EINDEX` | index out of bounds | `IndexError` | `IndexException` |
| −3 | `TSR_ENOMEM` | out of memory, or over the memory budget | `MemoryError` | `MemoryException` |
| −4 | `TSR_ETYPE` | operation not supported for this dtype | `DTypeError` | `DTypeException` |
| −5 | `TSR_EDIM` | more than 32 dimensions | `ShapeError` | `ShapeException` |
| −6 | `TSR_ESHAPE` | shapes cannot be broadcast or reshaped | `ShapeError` | `ShapeException` |
| −7 | `TSR_ECONTIG` | a view is impossible without a copy | `TesseroException` | (copies instead) |
| −8 | `TSR_ECONVERGE` | an iterative method did not converge, or a singular system | `ConvergenceError` | `Tessero\Ext\Exception` |
| −9 | `TSR_EIO` | an operating-system I/O error while opening, sizing, mapping or flushing a file (`tsr_mmap_error()` has the message) | `TesseroException` ("memmap: …") | `Tessero\Ext\Exception` ("memmap: …") |

## Solver status codes

Returned in results, not thrown.

| Solver | Status | Meaning |
|---|---:|---|
| `linprog`, `milp` | 0 | optimal |
| | 1 | iteration limit (`linprog`) / node limit or unresolved subproblem (`milp`); `x` is the best point found, if any |
| | 2 | infeasible |
| | 3 | unbounded |
| | 4 | numerical difficulties |
| `valueIteration`, `policyIteration` | `converged = true` | stopping rule met |
| | `converged = false` | `maxIter` reached; values and policy are the last iterate |
| `cg`, `bicgstab` | `info = 0` | converged |
| | `info > 0` | iterations run without converging (or `ConvergenceError` with `throw: true`) |

## Exception hierarchy

FFI package (`Tessero\Exceptions`):

```
RuntimeException
└── TesseroException
    ├── ShapeError
    ├── IndexError
    ├── DTypeError
    ├── MemoryError
    ├── LibraryUnavailable
    └── LinAlgError
        ├── SingularMatrix
        ├── NotPositiveDefinite
        └── ConvergenceError
```

Extension (`Tessero\Ext`):

```
RuntimeException
└── Exception
    ├── ShapeException
    ├── IndexException
    ├── DTypeException
    └── MemoryException
```

Argument type errors raise PHP's own `TypeError`/`ValueError` on both backends.

The Laravel manager passes exceptions through unchanged. Catch
`RuntimeException` to handle both backends in one place.
