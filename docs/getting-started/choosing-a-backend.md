# Choosing a backend

Tessero has one kernel (`libtessero`, C) and two ways to reach it from PHP.
They produce identical numbers: `tests/Ext/ExtParityTest.php` replays the FFI
package's NumPy/SciPy fixtures through the extension, and
`tests/Ext/BackendParityTest.php` runs both on the same random inputs
(reductions over every axis combination, ufuncs, file round trips) and
requires identical bytes.

| | FFI package `tessero/tessero` | Extension `tessero/tessero-ext` |
|---|---|---|
| Install | `composer require`; prebuilt binary | `pie install` or `phpize`; needs a compiler (or a distro package) |
| PHP settings | `ffi.enable=1` (CLI) or `ffi.enable=preload` + preload script (FPM) | `extension=tessero` |
| Call overhead | ~9 µs per call | ~0.3 µs per call |
| PHP array → native, 1M floats | 26 ms | 9 ms |
| Native → PHP list, 1M floats | 65 ms | 11 ms |
| Operators `$a + $b`, `$a * 2` | with the extension loaded (via `Tessero\Ext\Operand`) | yes |
| `foreach`, `count()`, `json_encode()` | yes | yes (native iterator; `toJson()` byte-identical to `json_encode` with `JSON_PRESERVE_ZERO_FRACTION`) |
| Slicing `$a['1:, ::-1']`, masks, fancy indexing | yes | yes |
| Reductions | any axis or list of axes, `keepdims` | any axis or list of axes, `keepdims` |
| Linear algebra (LAPACK) | yes | no. Use the FFI package alongside |
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

## Recommendations

- **Laravel or any web application under FPM or Octane:** install the
  extension, and the FFI package as well if you need linear algebra, FFT or the
  optimisers. `tessero/laravel` picks the extension automatically
  (`TESSERO_BACKEND=auto`).
- **Shared hosting or no compiler:** FFI package only. Under FPM configure the
  preload script ([Deployment](../operations/deployment.md#3-php-fpm-and-apache)).
- **CLI batch jobs on large arrays:** either. Per-call overhead is irrelevant
  when each call touches millions of elements. The FFI package has the wider API.
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
