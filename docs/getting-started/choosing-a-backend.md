# Choosing a backend

Tessero has one kernel (`libtessero`, C) and two ways to reach it from PHP.
They produce identical numbers: `tests/Ext/ExtParityTest.php` replays the FFI
package's NumPy/SciPy fixtures through the extension, and
`tests/Ext/BackendParityTest.php` runs both on the same random inputs
(reductions over every axis combination, ufuncs, file round trips) and
requires identical bytes.

| | FFI package `tessero/tessero` | Extension `tessero/tessero-ext` |
|---|---|---|
| Install | `composer require`; prebuilt binary | `phpize` build (C/C++ compiler, LAPACKE and BLAS headers); `pie install` once registered |
| PHP settings | `ffi.enable=1` (CLI) or `ffi.enable=preload` + preload script (FPM) | `extension=tessero` |
| Small calls (speed sweep, 2026-11-15) | `np.zeros` 1.87 µs, `np.sum` 1.71, `np.dot` 5.43 (NumPy: 0.25, 2.31, 0.86); per-call geomean 0.30 of NumPy/SciPy's time | `np.zeros` 0.17 µs, `np.sum` 0.27, `np.dot` 0.36; per-call geomean 0.074, faster on 966 of 1,022 functions |
| PHP array → native, 1M floats | 26 ms | 9 ms |
| Native → PHP list, 1M floats | 65 ms | 11 ms |
| Operators `$a + $b`, `$a * 2` | with the extension loaded (via `Tessero\Ext\Operand`) | yes |
| `foreach`, `count()`, `json_encode()` | yes | yes (native iterator; `toJson()` byte-identical to `json_encode` with `JSON_PRESERVE_ZERO_FRACTION`) |
| Slicing `$a['1:, ::-1']`, masks, fancy indexing | yes | yes |
| Reductions | any axis or list of axes, `keepdims` | any axis or list of axes, `keepdims` |
| Linear algebra (LAPACK) | yes | yes: `Tessero\Ext\Linalg`, `ScipyLinalg` |
| FFT | full `numpy.fft` API | `Engine::fft` (last axis, complex128 result) |
| Random | full `Generator` API | `Tessero\Ext\Random\Generator`: the same streams; no `choice` yet |
| Sparse CSR, CG, BiCGSTAB | yes | CSR input accepted by the MDP solvers |
| minimize / root / curve_fit | yes | no |
| linprog / milp / MDP | yes | yes |
| complex128 | yes | storage, arithmetic, `real imag conj angle abs`, FFT |
| `.npy` | `NDArray::load/save/openMemmap`, `Npy` class | `NDArray::load/save/openMemmap` |
| `.npz` | yes (`Npy::saveZ/loadZ`) | no |
| Universal functions (`Math::`, `out:` arrays, cbrt/erf/gamma/logaddexp/fmax/fmod …) | `Tessero\Math`: 74 functions | `Tessero\Ext\Math`: the same 74 ([Ufuncs](../guide/ufuncs.md)) |
| Memory-mapped arrays (files larger than RAM) | `NDArray::memmap()` | `NDArray::memmap()` ([Memmap](../guide/memmap.md)) |
| `scipy.special`, `scipy.stats`, NumPy statistics | `Tessero\Special`, `Stats`, `Np` | `Tessero\Ext\Special`, `Stats`, `Np`: the same functions and results ([Statistics](../guide/statistics.md)) |

Timings: median on a 2-core x86-64 VM with PHP 8.4, `bench/run.php`. They vary by
±40 % on shared hardware.

The registry functions (the NumPy and SciPy functions of the
[coverage tables](../project/numpy-scipy-coverage.md)) pass their NumPy/SciPy fixtures on both backends. Rows
above that say "no" are PHP-level APIs of the FFI package. Install it alongside the extension to use them.

## Recommendations

- **Default: the extension**, with the PHP package alongside for its PHP-level APIs. It is faster than FFI on
  small calls and level with it on large ones, and it is the backend that beats NumPy on NumPy's own core functions
  ([speed sweep](../project/speed-sweep.md)).
- **Laravel or any web application under FPM or Octane:** the extension. `tessero/laravel` picks it automatically
  (`TESSERO_BACKEND=auto`).
- **Shared hosting or no compiler:** FFI package only. Under FPM configure the
  preload script ([Deployment](../operations/deployment.md#3-php-fpm-and-apache)).
- **CLI batch jobs on large arrays:** either. Per-call overhead is irrelevant
  when each call touches millions of elements.
- **Many calls on small arrays** (per-request scoring, per-row pricing): the
  extension. At 10 elements an FFI call costs as much as ~30 plain PHP
  additions; the extension costs the same as one.

## Using both at once

Both can be loaded in one process. The extension compiles the kernel with
hidden symbol visibility, so the two copies of `libtessero` do not clash. They
have separate memory accounting: `Tessero\Tessero::memoryInUse()` counts FFI
buffers and `Tessero\Ext\Engine::memoryInUse()` counts extension buffers. The
Laravel manager applies the thread and budget settings to both.

To move data between them without going through PHP arrays:

```php
$ext = Tessero\Ext\NDArray::fromBytes($ffi->toBytes(), $ffi->dtype()->name(), $ffi->shape());
$ffi = Tessero\NDArray::fromBytes($ext->toBytes(), $ext->dtype(), $ext->shape());
```
