# Tessero

**Numerical computing for PHP.** Tessero gives PHP applications n-dimensional
arrays, linear algebra, FFT, NumPy-identical random numbers, sparse matrices,
optimisation, linear and mixed-integer programming, and Markov decision
processes, all running in one small C library (`libtessero`).

Results are checked against NumPy and SciPy on every commit. Where NumPy
defines the answer bit for bit (random streams, integer arithmetic, slicing,
`json_encode` output) Tessero matches bit for bit. Where only a tolerance makes
sense (floating-point linear algebra, FFT, iterative solvers) the tolerance is
stated and tested.

```php
use Tessero\Ext\NDArray;
use Tessero\Ext\Engine;

$load = NDArray::array($hourlyLoad);               // PHP array -> native float64
$z    = ($load - $load->mean()) / $load->std();    // operators, broadcasting
$peak = $load[$z->gt(2.0)];                        // boolean mask
$day  = $load->reshape(-1, 24)->mean(0);           // hour-of-day profile

$plan = Engine::linprog([-40, -30], [[2, 1], [1, 1]], [100, 80]);
$plan['x'];        // [20, 60]
$plan['ineqlin'];  // shadow prices, same sign convention as SciPy
```

## Three packages, one kernel

| Package | What it is | Install |
|---|---|---|
| `tessero/tessero` | The PHP library. Loads `libtessero` through FFI. Nothing to compile. | `composer require tessero/tessero` |
| `tessero/tessero-ext` | A Zend extension with `libtessero` compiled in: operators (`$a + $b`), native slicing, `foreach`, `json_encode`, 30× lower call overhead, no FFI setup. | `pie install tessero/tessero-ext` |
| `tessero/laravel` | Laravel integration: service provider, facade, Eloquent casts, validation rule, Octane-safe settings, `response()->ndarray()`. | `composer require tessero/laravel` |

Both backends produce identical numbers. The parity suite replays the same
NumPy/SciPy fixtures through both. See [Choosing a backend](getting-started/choosing-a-backend.md).

## Where to go next

<div class="grid cards" markdown>

- **[Getting started](getting-started/installation.md)**: install, check the machine with `tessero doctor`, run a first calculation.
- **[User guide](guide/arrays.md)**: arrays, indexing, maths, linear algebra, FFT, random numbers, sparse, optimisation, LP/MILP, MDPs, I/O, Laravel.
- **[Operations manual](operations/index.md)**: deployment (FPM, Octane, queues, Docker), configuration, memory and threads, monitoring, runbooks, security, upgrades.
- **[Architecture](architecture/overview.md)**: how the kernel, the FFI binding and the extension fit together, numerical methods, testing strategy, and the decision records behind them.
- **[Reference](reference/api/index.md)**: generated API pages for every public class, the C ABI, error codes, configuration keys.
- **[Project](project/roadmap.md)**: roadmap, status against the original gap analysis, governance and how to contribute.

</div>

## Status

Version **0.2.0**, pre-1.0. The API is usable and heavily tested, but it can
still change between minor versions; [Upgrading](operations/upgrading.md) lists
every change. The platform has been built and verified on Linux x86-64 with
PHP 8.4. Other platforms are built by CI and are listed as unverified until CI
has run them ([Roadmap](project/roadmap.md)).

Tessero is released under the BSD-3-Clause licence.
