# Quickstart

Ten minutes with the main features. The examples use the FFI package
(`Tessero\NDArray`); the extension's `Tessero\Ext\NDArray` accepts the same calls
and adds operators, shown at the end.

## Arrays

```php
use Tessero\NDArray;
use function Tessero\{arr, arange, zeros, linspace};

$a = arr([[1, 2, 3], [4, 5, 6]]);        // int64, shape [2, 3]
$b = arr([0.5, 1.5, 2.5]);               // float64, shape [3]

$a->shape();                             // [2, 3]
$a->dtype()->name();                     // 'int64'
$c = $a->mul($b);                        // broadcast [2,3] * [3] -> float64 [2,3]
$c->toArray();                           // [[0.5, 3.0, 7.5], [2.0, 7.5, 15.0]]

$a->sum();                               // 21
$a->sum(axis: 0)->toList();              // [5, 7, 9]
$a->mean(axis: 1, keepdims: true);       // shape [2, 1]
```

## Views and slicing

Slices are views: they share memory with the original array.

```php
$m = arange(12)->reshape(3, 4);
$m['1:, ::2']->toArray();                // [[4, 6], [8, 10]]
$m['..., -1']->toList();                 // last column: [3, 7, 11]
$m['::-1'];                              // rows reversed, no copy
$m[$m->gt(8)] = 0;                       // boolean-mask assignment
$m->t();                                 // transpose view
```

## Linear algebra, FFT, random numbers

```php
use Tessero\Linalg\Linalg;
use Tessero\Fft\Fft;
use Tessero\Random\Generator;

$rng = Generator::defaultRng(42);        // same stream as numpy.random.default_rng(42)
$X = $rng->normal(size: [200, 3]);
$y = $X->matmul(arr([1.5, -2.0, 0.5]))->add($rng->normal(0, 0.1, 200));
[$beta] = Linalg::lstsq($X, $y);         // close to [1.5, -2.0, 0.5]

$spectrum = Fft::rfft(linspace(0, 1, 1024)->mul(2 * M_PI * 50)->sin())->abs();
```

## Optimisation and decisions

```php
use Tessero\Optimize\LinearProgramming as LP;
use Tessero\Mdp\MarkovDecisionProcess;

// maximise 40x + 30y  s.t.  2x + y <= 100,  x + y <= 80
$r = LP::linprog([-40, -30], [[2, 1], [1, 1]], [100, 80]);
$r->x;                   // [20.0, 60.0]
$r->fun;                 // -2600.0
$r->ineqlinMarginals;    // [-10.0, -20.0]: what one more unit of each resource is worth

// two-state machine: action 0 = run, 1 = repair
$P = [
    [[0.9, 0.1], [0.0, 1.0]],            // run
    [[1.0, 0.0], [1.0, 0.0]],            // repair
];
$R = [[10, -5], [0, -5]];                // R[state][action]
$sol = MarkovDecisionProcess::fromDense($P, $R)->policyIteration(gamma: 0.95);
$sol->policy->toList();                  // [0, 1]: run while healthy, repair when broken
```

## With the extension: operators

```php
use Tessero\Ext\NDArray as X;

$p = X::array([42.1, 39.8, 55.0, 61.2]);
$q = X::array([10, 12, 8, 9]);
$revenue = ($p * $q)->sum();             // operators map to the same kernels
$norm    = ($p - $p->mean()) / $p->std();
foreach ($norm as $v) { /* native iterator */ }
echo json_encode($norm);                  // identical to json_encode($norm->toList())
```

With the extension loaded, the FFI `NDArray` gets the same operators through
`Tessero\Ext\Operand`.

## Next

- [Choosing a backend](choosing-a-backend.md)
- [User guide](../guide/arrays.md)
- [Deploying to production](../operations/deployment.md)
