# Calling Tessero functions from C

Native consumers of `libtessero` (an extension, a Zend module, a test harness) reach its functions in four ways. All
of them compute what the PHP backends compute; the full declarations are in the [C ABI](api/c-abi.md)
(`csrc/include/tessero.h`, included after `<stdint.h>`).

## Typed entry points

Plain C functions with typed arguments, for the operations most often called from C:

| function | what | notes |
|---|---|---|
| `tsr_gemm`, `tsr_axpy` | BLAS-style matrix product and AXPY | canonical round-once FMA, bit-exact on every CPU and thread count |
| `tsr_matmul` | matrix product through the linked BLAS | tolerance-level agreement across BLAS builds |
| `tsr_cholesky`, `tsr_solve`, `tsr_slogdet` | `numpy.linalg.cholesky`, `solve`, `slogdet` (row-major float64) | run the registry routines of the same name, so the results equal the registry's and both PHP backends' byte for byte |
| `tsr_sort`, `tsr_sort_simd`, `tsr_sort_runs`, `tsr_argsort` | sorting | identical results; see [sorting design](../project/sort-design.md) |
| `tsr_unary`, `tsr_binary`, `tsr_reduce_mid` | the core element-wise operations and reductions over strided arrays | the operation is an op code, below |

## The core element-wise operations: op codes by name

`tsr_unary(op, ...)` and `tsr_binary(op, ...)` take an op code. `tsr_unary_find("exp")` / `tsr_binary_find("add")`
return it by NumPy ufunc name (-1 when unknown), `tsr_unary_name(op)` / `tsr_binary_name(op)` go back. The codes are
stable within an ABI generation (`tsr_abi_version()`), and the PHP side mirrors them in `Tessero\Native\Abi`.

- unary: 0 negative, 1 absolute (alias abs), 2 square, 3 sign, 4 sqrt, 5 exp, 6 log, 7 log10, 8 log2, 9 sin, 10 cos,
  11 tan, 12 arcsin, 13 arccos, 14 arctan, 15 sinh, 16 cosh, 17 tanh, 18 floor, 19 ceil, 20 rint, 21 expm1, 22 log1p,
  23 reciprocal, 24 isnan, 25 isfinite, 26 isinf, 27 logical_not, 28 real, 29 imag, 30 cabs, 31 angle,
  32 conjugate (alias conj)
- binary: 0 add, 1 subtract, 2 multiply, 3 divide (alias true_divide), 4 power, 5 maximum, 6 minimum, 7 arctan2,
  8 hypot, 9 mod (alias remainder), 10 equal, 11 not_equal, 12 less, 13 less_equal, 14 greater, 15 greater_equal,
  16 logical_and, 17 logical_or, 18 logical_xor, 19 floor_divide

## Loop ufuncs

A second table of element-wise functions with their own type resolution: `tsr_ufunc_count()`, `tsr_ufunc_name(id)`,
`tsr_ufunc_find(name)`, `tsr_ufunc_resolve()` and `tsr_ufunc()`. In this release: cbrt, exp2, trunc, arcsinh, arccosh,
arctanh, degrees, radians, erf, erfc, gamma, lgamma, signbit, fmax, fmin, copysign, nextafter, heaviside, logaddexp,
logaddexp2, fmod, gcd, lcm, float_power (`floatPower`). Enumerate them at run time rather than relying on this list.

## The function registry

Everything else the PHP backends offer is a registry entry named `module.function`: `special.expit`,
`special.erf`, `linalg.cholesky`, `np.einsum`, `distance.kdtree_query`, ... (1,561 entries in this release; the list is
`tools/parity/registry.json`). `tsr_fn_find(name)` returns an id (-1 when unknown); element-wise entries run through
`tsr_fn_ufunc`, the others through `tsr_fn_routine` with `tsr_arg` arguments and `tsr_result` results the caller
releases with `tsr_fn_results_free`.

**Ids are positions in the registry table.** They are stable for the life of the loaded library, so looking a name
up once per process and keeping the id is safe; they are not stable across releases (new entries shift them), so
never store an id outside the process or compile one in.

| looking for | where |
|---|---|
| exp, log, tanh, sqrt, sin, ... | op codes: `tsr_unary_find` |
| add, power, maximum, comparisons, ... | op codes: `tsr_binary_find` |
| erf, gamma, cbrt, exp2, logaddexp, ... | loop ufuncs: `tsr_ufunc_find` |
| expit, special functions, distributions, linalg, signal, ... | registry: `tsr_fn_find` |
