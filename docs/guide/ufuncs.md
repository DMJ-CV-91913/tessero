# Universal functions (Math)

`Tessero\Math` (FFI package) and `Tessero\Ext\Math` (native extension) are
the universal-function interface, modelled on NumPy's ufuncs. The two classes
have the same 74 functions, signatures and results: both call libtessero's
loops, and a cross-backend test checks that they return identical bytes. Each function works element by element over arrays of any
shape. It broadcasts its operands, follows NumPy's type rules, and can write
into an output array you provide.

```php
use Tessero\Ext\Math;
use Tessero\Ext\NDArray as X;

$x = X::array([[0.25, 1.0, 4.0], [9.0, 16.0, 25.0]]);

Math::sqrt($x);                                // new array
Math::hypot($x, X::array([3.0, 4.0, 0.0]));    // [2,3] with [3]: broadcast
Math::cbrt(X::array([8, -27]));                // ints -> float64: [2.0, -3.0000000000000004], as NumPy
Math::fmax($a, $b);                            // maximum that ignores NaN
Math::logaddexp($logp, $logq);                 // log(e^p + e^q) without overflow
Math::erf($z);  Math::gamma($x);  Math::lgamma($x);

$buf = X::zeros([2, 3]);
Math::exp($x, $buf);                           // written into $buf, returns $buf
Math::sqrt($x, $x);                            // in place
Math::apply('cbrt', $x);                       // by name
Math::ufuncs();                                // ['sin' => ['nin' => 1, 'engine' => 'kernel', 'summary' => ...], ...]
```

The examples use the extension; with the FFI package write
`use Tessero\Math; use Tessero\NDArray as X;` and the same lines run unchanged.

## The catalogue

**Kernel ufuncs** run through the same C kernels as the `NDArray` methods and
operators, so `Math::sin($a)` and `$a->sin()` return identical bytes:

| One input | `negative absolute abs square sign sqrt exp expm1 log log10 log2 log1p sin cos tan arcsin arccos arctan sinh cosh tanh floor ceil rint reciprocal isnan isinf isfinite invert real imag conj angle` |
|---|---|
| Two inputs | `add subtract multiply divide power mod remainder floorDivide maximum minimum arctan2 hypot equal notEqual less lessEqual greater greaterEqual logicalAnd logicalOr logicalXor` |

**Loop ufuncs** are libtessero's typed inner loops (`csrc/src/ufunc.c`), shared by both packages:

| Function | Result | Reference |
|---|---|---|
| `cbrt` | cube root | `numpy.cbrt` |
| `exp2` | 2^x | `numpy.exp2` |
| `trunc` | round towards zero; integers unchanged (same dtype) | `numpy.trunc` |
| `arcsinh`, `arccosh`, `arctanh` | inverse hyperbolic functions | `numpy` |
| `degrees`, `radians` | angle conversion | `numpy` |
| `erf`, `erfc` | error function, complement | `scipy.special` |
| `gamma`, `lgamma` | Γ(x), log\|Γ(x)\| | `scipy.special.gamma`, `gammaln` |
| `signbit` | sign bit set (bool; true for −0.0) | `numpy.signbit` |
| `fmax`, `fmin` | maximum/minimum ignoring NaN; integer inputs stay integer | `numpy.fmax`, `fmin` |
| `copysign`, `nextafter` | magnitude with another sign; next float | `numpy` |
| `logaddexp`, `logaddexp2` | log(e^x + e^y), log2(2^x + 2^y) | `numpy` (same branches) |
| `fmod` | C remainder (sign of x); integers stay integer, x % 0 gives 0 | `numpy.fmod` |

Every loop ufunc is checked against NumPy/SciPy (`tests/fixtures/ufunc.json`,
64 cases: float64, float32, integers, broadcasting, NaN and ±∞) to 1e-13
relative for float64 and 2e-6 for float32. Result dtypes match NumPy's. The one
difference is `uint8` input to a float function, which gives float64 in Tessero
(float16 in NumPy, a dtype Tessero does not have).

## Types

- Integer and bool inputs to floating-point functions produce float64.
- float32 stays float32 and uses float32 functions (`cbrtf`, `erff`, …).
- PHP scalars are weak (NEP 50): `Math::copysign($f32, -1.0)` stays float32.
- Complex input to a loop ufunc throws `DTypeException`. The kernel ufuncs
  `absolute real imag conj angle` and the arithmetic ones accept complex128.

## Output arrays

`out` receives the result:

- When its dtype and shape match the result, the kernel writes straight into
  it: no allocation, no copy. This is the fast path for reusing a buffer in a
  loop, or for writing into a [memory-mapped file](memmap.md).
- When its dtype differs, the result is cast into it if the cast is *same-kind*
  or safer (float64 → float32 is allowed; float → int is not and throws
  `DTypeException`). This is NumPy's `casting='same_kind'`.
- Its shape must equal the broadcast result shape (`ShapeException` otherwise).
- In-place use (`Math::sqrt($x, $x)`) is safe. When `out` overlaps an input in
  any other way (a shifted view of the same array), the engine computes into a
  temporary first, so the result is always what NumPy would produce.
- A read-only array (memmap mode `'r'`) as `out` throws.

## How it runs

```
Math::fmax($a, $b, $out)
  │  resolve: promoted dtype → typed inner loop (float64, float32, int64 …)
  │  cast inputs that don't match the loop's dtype
  │  broadcast: zero strides for stretched axes (no copies)
  │  coalesce dimensions (tsr_iter): a C-contiguous array becomes one run
  └─ for each inner run: loop(args, n, byte_steps)
        contiguous fast path: restrict pointers, direct call, `omp simd`,
        compiled three times (AVX-512 / AVX2 / baseline, chosen at load)
```

- **Inner loops** follow NumPy's design: a table of typed loops per function,
  each handling one run of `n` elements with byte steps. Contiguous runs and
  "second operand is a scalar" runs have dedicated fast paths.
- **SIMD.** Arithmetic-only loops (`degrees`, `radians`, `trunc`, `fmax`,
  `fmin`, `copysign`, `signbit`) vectorise. Transcendental loops call libm per
  element. Vector libm would require `-ffast-math`, which Tessero avoids
  because it breaks NaN handling and summation order.
- **Threads.** Arrays of at least 131 072 elements are split across
  `tessero.threads` OpenMP threads (by rows, or by chunks of a single run).
  Results are byte-identical for any thread count, and a test checks this.
  `lgamma` uses the re-entrant `lgamma_r` so that it is thread-safe.

Measured on a 2-core VM, 2M float64 elements, 1 → 2 threads: `degrees` 2.0 →
0.8 ms (1.2 ms with a preallocated `out`), `erf` 12 → 6 ms, `fmax` 5.5 →
3.5 ms, `cbrt` 48 → 22 ms.

`cbrt` returns the C library's result, as NumPy does on CPUs without AVX-512 and in the pinned reference
([ADR 0012](../architecture/adr/0012-pinned-reference-environment.md)). glibc is one ulp off on some exact
cubes (cbrt(−27) = −3.0000000000000004), and so is NumPy there.
