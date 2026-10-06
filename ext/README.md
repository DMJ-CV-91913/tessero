# ext-tessero — the native backend

A Zend extension that compiles libtessero in and exposes it without FFI:

| Class | What it gives you |
|---|---|
| `Tessero\Ext\NDArray` | typed n-d array in native memory: `+ - * / ** %`, `$a['1:, ::-1']` views, masks, fancy indexing, `foreach`, `count()`, `json_encode()`, `toJson()`, `serialize()`, `clone`, `==`, reductions with axes, `matmul`, sort, `where` |
| `Tessero\Ext\Engine` | `linprog`, `milp`, `mdpValueIteration`, `mdpPolicyIteration`, `mdpFiniteHorizon`, `fft`, NumPy-identical `random`/`normal`/`integers`, threads and memory settings |
| `Tessero\Ext\Math` | 74 universal functions (`sin`, `cbrt`, `erf`, `gamma`, `logaddexp`, `fmax`, `fmod`, …) with broadcasting, NumPy type rules and `out:` arrays |
| `NDArray::memmap()` | arrays backed by a memory-mapped file: larger than RAM, zero-copy views, `flush()` |
| `Tessero\Ext\Operand` | base class that lends `+ - * / ** %` to userland classes (the FFI `Tessero\NDArray` uses it automatically) |
| `Tessero\Ext\*Exception` | `ShapeException`, `IndexException`, `DTypeException`, `MemoryException` (all extend `Tessero\Ext\Exception`, a `RuntimeException`) |

Why use it next to the FFI package:

- **No FFI configuration.** Works on hosts where `ext-ffi` is disabled, and under PHP-FPM without preloading.
- **30x lower call overhead.** An element-wise call on a small array costs ~0.3 µs instead of ~9 µs through FFI, so code that makes many small calls (per-request pricing, per-row scoring) stays fast.
- **Faster PHP boundary.** Converting 1M values to or from PHP arrays is 3–6x faster; `toJson()` is byte-identical to `json_encode(..., JSON_PRESERVE_ZERO_FRACTION)`.

Numerical results are identical to the FFI package: `tests/Ext/ExtParityTest.php` replays the same NumPy/SciPy fixtures through both.

## Build and install

```bash
pie install tessero/tessero-ext                  # PHP Foundation installer
# or from a checkout
cd ext && phpize && ./configure --enable-tessero && make && make test
echo "extension=tessero" > "$(php-config --ini-dir)/30-tessero.ini"
```

`--disable-tessero-openmp` builds without OpenMP (no libgomp dependency).
The kernel sources live in `ext/libtessero/`, a copy of `csrc/` kept in sync by
`tools/sync-ext.sh` (CI fails on drift). Kernel symbols are compiled with
hidden visibility, so the extension and the FFI library can be loaded in the
same process.

## INI settings

| Setting | Default | Meaning |
|---|---|---|
| `tessero.threads` | 1 | OpenMP threads for large element-wise kernels and MDP sweeps (re-applied at each request start) |
| `tessero.memory_budget` | 0 | Native memory cap in bytes (`256M` syntax allowed); allocations beyond it throw `MemoryException` |
| `tessero.epsilon` | 1e-6 | Default tolerance for the MDP solvers |

## Example

```php
use Tessero\Ext\NDArray;
use Tessero\Ext\Engine;

$prices = NDArray::array($hourlyPrices);            // PHP array -> native float64
$z = ($prices - $prices->mean()) / $prices->std();  // operators, broadcasting
$peak = $prices[$z->gt(2.0)];                       // boolean mask
$week = $prices->reshape(-1, 24)->mean(0);          // hour-of-day profile
return response($week->toJson(), 200, ['Content-Type' => 'application/json']);

$plan = Engine::linprog([-40, -30], [[2, 1], [1, 1]], [100, 80], bounds: [[0, 40], [0, null]]);
$plan['ineqlin'];                                   // shadow prices
```

## Tests

`ext/tests/*.phpt` (12 files: lifecycle and view ownership, operators, indexing,
iteration/JSON/serialize, math, solvers, the userland Operand, INI budget, solver
input validation, Math ufuncs, memory-mapped arrays) run
with `make test`. They are also run with the module built with
`-fsanitize=address,undefined` in CI.
