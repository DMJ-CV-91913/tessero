# Random numbers

`Tessero\Random\Generator` reproduces NumPy's default generator **bit for
bit**. PCG64 is seeded through `SeedSequence`, and normals come from the
ziggurat method with NumPy's own tables:

```php
use Tessero\Random\Generator;

$rng = Generator::defaultRng(42);      // numpy.random.default_rng(42)
$rng->random(5);                       // identical to rng.random(5)
$rng->uniform(-1, 1, [2, 3]);
$rng->normal(100, 15, 1000);           // loc, scale, size
$rng->standardNormal([10, 10]);
$rng->integers(0, 10, 20);             // [low, high)
$rng->choice(5, 3, replace: false);
$rng->choice(arr([10, 20, 30]), 1000, p: [0.2, 0.5, 0.3]);
$rng->permutation(10);
$rng->shuffle($array);                 // in place, along axis 0
$rng->exponential(2.0, 100);
$rng->gamma(2.0, size: 1000);          // every numpy.random.Generator distribution method
$rng->binomial(10, 0.3, size: 5);
```

A model fitted in Python with a fixed seed can be rerun in PHP and produce
the same draws. That makes Monte Carlo results reproducible across the two
languages and lets tests compare them exactly.

| Method | Stream-identical to NumPy |
|---|---|
| `random`, `uniform` | yes |
| `normal`, `standardNormal` | yes (ziggurat) |
| `integers` | yes (Lemire's method; 32- and 64-bit paths) |
| `choice` (with/without replacement, with `p`) | yes |
| `permutation`, `shuffle` | yes |
| `exponential`, `gamma`, `beta`, `poisson`, `binomial`, ... (the univariate distribution methods) | yes: NumPy's own distribution code |

The distribution methods run NumPy's `distributions.c`, compiled into the kernel, on the Generator's PCG64
stream. They take NumPy's arguments, with array parameters broadcast, `size:`, and `dtype: 'float32'`
where NumPy has it. They reject invalid parameters as NumPy does. The multivariate methods
(`multivariate_normal`, `dirichlet`, `multinomial`) are not implemented yet.

Seeds: a non-negative int, a list of ints (like NumPy's entropy lists), or
`null` for 128 bits from the operating system.

## Saving and restoring state

```php
$words = $rng->getState();             // 6 decimal strings: safe in JSON, databases, queues
$rng2 = Generator::defaultRng(0);
$rng2->setState($words);               // continues exactly where $rng was
```

Use this to split a long simulation across queued jobs without losing
reproducibility.

## Extension

The extension has the same generator, `Tessero\Ext\Random\Generator`, with the same streams:

```php
use Tessero\Ext\Random\Generator;

$rng = Generator::defaultRng(42);
$rng->random(1000);                    // == default_rng(42).random(1000)
$rng->integers(0, 100, 10);
$rng->gamma(2.0, size: 1000);
$words = $rng->getState();             // the same state format as the FFI Generator
```

It has every method above except `choice`. The stateless `Tessero\Ext\Engine::random()`, `normal()` and
`integers()` remain for existing code.

## Not a CSPRNG

PCG64 is for simulation. Use `random_bytes()` / `random_int()` for tokens,
passwords and anything security-related.
