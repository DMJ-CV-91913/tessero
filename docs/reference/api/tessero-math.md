# Math

`Tessero\Math`

Universal functions (ufuncs): element-wise functions over arrays of any shape, with NumPy's broadcasting and type rules and an optional output array.

    Math::sin($x);                    // new array
    Math::hypot($x, $y, out: $buf);   // written into $buf (no allocation)
    Math::cbrt($a, $a);               // in place

The native extension has the same class as Tessero\Ext\Math, with the same
names, signatures and results: both call libtessero's loops.

Kernel ufuncs (sin, exp, add, maximum, comparisons, ...) run the NDArray
methods, so Math::sin($a) and $a->sin() are identical to the bit. Loop
ufuncs (cbrt, erf, gamma, logaddexp, fmax, copysign, fmod, ...) are
libtessero's typed inner loops (tsr_ufunc), vectorised and threaded like
the other kernels.

With `out`, the result is written into that array: directly when the dtype
and shape match and it does not partially overlap an input, otherwise
through a temporary and a same-kind cast (float64 -> float32 is allowed,
float -> int is not). The result broadcasts into `out`.

## Methods

### abs

```php
static abs(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

alias of absolute.

### absolute

```php
static absolute(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

|x| (modulus for complex).

### add

```php
static add(mixed $x, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray
```

x + y.

### angle

```php
static angle(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

argument of a complex number.

### apply

```php
static apply(string $name, mixed $x, mixed $y = null, ?Tessero\NDArray $out = null): Tessero\NDArray
```

Call a ufunc by name: Math::apply('cbrt', $x), Math::apply('fmax', $x, $y, $out).
For a one-input ufunc the third argument may be the output array.

### arccos

```php
static arccos(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

inverse cosine.

### arccosh

```php
static arccosh(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

inverse hyperbolic cosine.

### arcsin

```php
static arcsin(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

inverse sine.

### arcsinh

```php
static arcsinh(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

inverse hyperbolic sine.

### arctan

```php
static arctan(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

inverse tangent.

### arctan2

```php
static arctan2(mixed $x, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray
```

angle of (x=y2, y=x1): atan2(x1, x2).

### arctanh

```php
static arctanh(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

inverse hyperbolic tangent.

### cbrt

```php
static cbrt(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

cube root.

### ceil

```php
static ceil(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

round up.

### conj

```php
static conj(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

complex conjugate.

### copysign

```php
static copysign(mixed $x, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray
```

magnitude of x with the sign of y.

### cos

```php
static cos(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

cosine (radians).

### cosh

```php
static cosh(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

hyperbolic cosine.

### degrees

```php
static degrees(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

radians to degrees.

### divide

```php
static divide(mixed $x, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray
```

x / y (true division).

### equal

```php
static equal(mixed $x, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray
```

x == y (bool).

### erf

```php
static erf(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

error function (scipy.special.erf).

### erfc

```php
static erfc(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

complementary error function.

### exp

```php
static exp(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

e^x.

### exp2

```php
static exp2(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

2^x.

### expm1

```php
static expm1(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

e^x - 1, accurate near 0.

### floor

```php
static floor(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

round down.

### floorDivide

```php
static floorDivide(mixed $x, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray
```

floor(x / y).

### fmax

```php
static fmax(mixed $x, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray
```

maximum ignoring NaN.

### fmin

```php
static fmin(mixed $x, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray
```

minimum ignoring NaN.

### fmod

```php
static fmod(mixed $x, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray
```

C remainder with the sign of x (numpy.fmod); integer x % 0 gives 0.

### gamma

```php
static gamma(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

gamma function (scipy.special.gamma).

### greater

```php
static greater(mixed $x, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray
```

x > y (bool).

### greaterEqual

```php
static greaterEqual(mixed $x, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray
```

x >= y (bool).

### hypot

```php
static hypot(mixed $x, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray
```

sqrt(x^2 + y^2) without overflow.

### imag

```php
static imag(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

imaginary part.

### invert

```php
static invert(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

bitwise NOT (logical NOT for bool).

### isfinite

```php
static isfinite(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

x is finite (bool).

### isinf

```php
static isinf(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

x is +/-INF (bool).

### isnan

```php
static isnan(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

x is NaN (bool).

### less

```php
static less(mixed $x, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray
```

x < y (bool).

### lessEqual

```php
static lessEqual(mixed $x, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray
```

x <= y (bool).

### lgamma

```php
static lgamma(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

log|gamma(x)| (scipy.special.gammaln).

### log

```php
static log(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

natural logarithm.

### log10

```php
static log10(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

base-10 logarithm.

### log1p

```php
static log1p(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

log(1 + x), accurate near 0.

### log2

```php
static log2(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

base-2 logarithm.

### logaddexp

```php
static logaddexp(mixed $x, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray
```

log(e^x + e^y) without overflow.

### logaddexp2

```php
static logaddexp2(mixed $x, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray
```

log2(2^x + 2^y) without overflow.

### logicalAnd

```php
static logicalAnd(mixed $x, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray
```

x AND y (bool).

### logicalOr

```php
static logicalOr(mixed $x, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray
```

x OR y (bool).

### logicalXor

```php
static logicalXor(mixed $x, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray
```

x XOR y (bool).

### maximum

```php
static maximum(mixed $x, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray
```

element-wise maximum, NaN propagates.

### minimum

```php
static minimum(mixed $x, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray
```

element-wise minimum, NaN propagates.

### mod

```php
static mod(mixed $x, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray
```

remainder with the sign of y (Python %).

### multiply

```php
static multiply(mixed $x, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray
```

x * y.

### negative

```php
static negative(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

-x.

### nextafter

```php
static nextafter(mixed $x, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray
```

next representable value after x towards y.

### notEqual

```php
static notEqual(mixed $x, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray
```

x != y (bool).

### power

```php
static power(mixed $x, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray
```

x ** y.

### radians

```php
static radians(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

degrees to radians.

### real

```php
static real(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

real part.

### reciprocal

```php
static reciprocal(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

1 / x.

### remainder

```php
static remainder(mixed $x, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray
```

alias of mod.

### rint

```php
static rint(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

round to nearest, ties to even.

### sign

```php
static sign(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

-1, 0 or 1.

### signbit

```php
static signbit(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

sign bit is set (bool; true for -0.0).

### sin

```php
static sin(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

sine (radians).

### sinh

```php
static sinh(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

hyperbolic sine.

### sqrt

```php
static sqrt(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

square root.

### square

```php
static square(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

x * x.

### subtract

```php
static subtract(mixed $x, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray
```

x - y.

### tan

```php
static tan(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

tangent (radians).

### tanh

```php
static tanh(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

hyperbolic tangent.

### trunc

```php
static trunc(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray
```

round towards zero.

### ufuncs

```php
static ufuncs(): array
```

Every ufunc: ['sin' => ['nin' => 1, 'engine' => 'kernel'|'loop', 'summary' => ...], ...].
