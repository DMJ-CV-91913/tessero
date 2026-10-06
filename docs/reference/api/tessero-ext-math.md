# Math

`Tessero\Ext\Math` *(native extension)*

## Methods

### abs

```php
static abs($x, ?Tessero\Ext\NDArray $out = …)
```

alias of absolute. Kernel ufunc (same code as the NDArray method).

### absolute

```php
static absolute($x, ?Tessero\Ext\NDArray $out = …)
```

|x| (modulus for complex). Kernel ufunc (same code as the NDArray method).

### add

```php
static add($x, $y, ?Tessero\Ext\NDArray $out = …)
```

x + y. Kernel ufunc (same code as the NDArray method).

### angle

```php
static angle($x, ?Tessero\Ext\NDArray $out = …)
```

argument of a complex number. Kernel ufunc (same code as the NDArray method).

### apply

```php
static apply(string $name, $x, $y = …, ?Tessero\Ext\NDArray $out = …)
```

### arccos

```php
static arccos($x, ?Tessero\Ext\NDArray $out = …)
```

inverse cosine. Kernel ufunc (same code as the NDArray method).

### arccosh

```php
static arccosh($x, ?Tessero\Ext\NDArray $out = …)
```

inverse hyperbolic cosine. Loop ufunc.

### arcsin

```php
static arcsin($x, ?Tessero\Ext\NDArray $out = …)
```

inverse sine. Kernel ufunc (same code as the NDArray method).

### arcsinh

```php
static arcsinh($x, ?Tessero\Ext\NDArray $out = …)
```

inverse hyperbolic sine. Loop ufunc.

### arctan

```php
static arctan($x, ?Tessero\Ext\NDArray $out = …)
```

inverse tangent. Kernel ufunc (same code as the NDArray method).

### arctan2

```php
static arctan2($x, $y, ?Tessero\Ext\NDArray $out = …)
```

angle of (x=y2, y=x1): atan2(x1, x2). Kernel ufunc (same code as the NDArray method).

### arctanh

```php
static arctanh($x, ?Tessero\Ext\NDArray $out = …)
```

inverse hyperbolic tangent. Loop ufunc.

### cbrt

```php
static cbrt($x, ?Tessero\Ext\NDArray $out = …)
```

cube root. Loop ufunc.

### ceil

```php
static ceil($x, ?Tessero\Ext\NDArray $out = …)
```

round up. Kernel ufunc (same code as the NDArray method).

### conj

```php
static conj($x, ?Tessero\Ext\NDArray $out = …)
```

complex conjugate. Kernel ufunc (same code as the NDArray method).

### copysign

```php
static copysign($x, $y, ?Tessero\Ext\NDArray $out = …)
```

magnitude of x with the sign of y. Loop ufunc.

### cos

```php
static cos($x, ?Tessero\Ext\NDArray $out = …)
```

cosine (radians). Kernel ufunc (same code as the NDArray method).

### cosh

```php
static cosh($x, ?Tessero\Ext\NDArray $out = …)
```

hyperbolic cosine. Kernel ufunc (same code as the NDArray method).

### degrees

```php
static degrees($x, ?Tessero\Ext\NDArray $out = …)
```

radians to degrees. Loop ufunc.

### divide

```php
static divide($x, $y, ?Tessero\Ext\NDArray $out = …)
```

x / y (true division). Kernel ufunc (same code as the NDArray method).

### equal

```php
static equal($x, $y, ?Tessero\Ext\NDArray $out = …)
```

x == y (bool). Kernel ufunc (same code as the NDArray method).

### erf

```php
static erf($x, ?Tessero\Ext\NDArray $out = …)
```

error function (scipy.special.erf). Loop ufunc.

### erfc

```php
static erfc($x, ?Tessero\Ext\NDArray $out = …)
```

complementary error function. Loop ufunc.

### exp

```php
static exp($x, ?Tessero\Ext\NDArray $out = …)
```

e^x. Kernel ufunc (same code as the NDArray method).

### exp2

```php
static exp2($x, ?Tessero\Ext\NDArray $out = …)
```

2^x. Loop ufunc.

### expm1

```php
static expm1($x, ?Tessero\Ext\NDArray $out = …)
```

e^x - 1, accurate near 0. Kernel ufunc (same code as the NDArray method).

### floor

```php
static floor($x, ?Tessero\Ext\NDArray $out = …)
```

round down. Kernel ufunc (same code as the NDArray method).

### floorDivide

```php
static floorDivide($x, $y, ?Tessero\Ext\NDArray $out = …)
```

floor(x / y). Kernel ufunc (same code as the NDArray method).

### fmax

```php
static fmax($x, $y, ?Tessero\Ext\NDArray $out = …)
```

maximum ignoring NaN. Loop ufunc.

### fmin

```php
static fmin($x, $y, ?Tessero\Ext\NDArray $out = …)
```

minimum ignoring NaN. Loop ufunc.

### fmod

```php
static fmod($x, $y, ?Tessero\Ext\NDArray $out = …)
```

C remainder with the sign of x (numpy.fmod); integer x % 0 gives 0. Loop ufunc.

### gamma

```php
static gamma($x, ?Tessero\Ext\NDArray $out = …)
```

gamma function (scipy.special.gamma). Loop ufunc.

### greater

```php
static greater($x, $y, ?Tessero\Ext\NDArray $out = …)
```

x > y (bool). Kernel ufunc (same code as the NDArray method).

### greaterEqual

```php
static greaterEqual($x, $y, ?Tessero\Ext\NDArray $out = …)
```

x >= y (bool). Kernel ufunc (same code as the NDArray method).

### hypot

```php
static hypot($x, $y, ?Tessero\Ext\NDArray $out = …)
```

sqrt(x^2 + y^2) without overflow. Kernel ufunc (same code as the NDArray method).

### imag

```php
static imag($x, ?Tessero\Ext\NDArray $out = …)
```

imaginary part. Kernel ufunc (same code as the NDArray method).

### invert

```php
static invert($x, ?Tessero\Ext\NDArray $out = …)
```

bitwise NOT (logical NOT for bool). Kernel ufunc (same code as the NDArray method).

### isfinite

```php
static isfinite($x, ?Tessero\Ext\NDArray $out = …)
```

x is finite (bool). Kernel ufunc (same code as the NDArray method).

### isinf

```php
static isinf($x, ?Tessero\Ext\NDArray $out = …)
```

x is +/-INF (bool). Kernel ufunc (same code as the NDArray method).

### isnan

```php
static isnan($x, ?Tessero\Ext\NDArray $out = …)
```

x is NaN (bool). Kernel ufunc (same code as the NDArray method).

### less

```php
static less($x, $y, ?Tessero\Ext\NDArray $out = …)
```

x < y (bool). Kernel ufunc (same code as the NDArray method).

### lessEqual

```php
static lessEqual($x, $y, ?Tessero\Ext\NDArray $out = …)
```

x <= y (bool). Kernel ufunc (same code as the NDArray method).

### lgamma

```php
static lgamma($x, ?Tessero\Ext\NDArray $out = …)
```

log|gamma(x)| (scipy.special.gammaln). Loop ufunc.

### log

```php
static log($x, ?Tessero\Ext\NDArray $out = …)
```

natural logarithm. Kernel ufunc (same code as the NDArray method).

### log10

```php
static log10($x, ?Tessero\Ext\NDArray $out = …)
```

base-10 logarithm. Kernel ufunc (same code as the NDArray method).

### log1p

```php
static log1p($x, ?Tessero\Ext\NDArray $out = …)
```

log(1 + x), accurate near 0. Kernel ufunc (same code as the NDArray method).

### log2

```php
static log2($x, ?Tessero\Ext\NDArray $out = …)
```

base-2 logarithm. Kernel ufunc (same code as the NDArray method).

### logaddexp

```php
static logaddexp($x, $y, ?Tessero\Ext\NDArray $out = …)
```

log(e^x + e^y) without overflow. Loop ufunc.

### logaddexp2

```php
static logaddexp2($x, $y, ?Tessero\Ext\NDArray $out = …)
```

log2(2^x + 2^y) without overflow. Loop ufunc.

### logicalAnd

```php
static logicalAnd($x, $y, ?Tessero\Ext\NDArray $out = …)
```

x AND y (bool). Kernel ufunc (same code as the NDArray method).

### logicalOr

```php
static logicalOr($x, $y, ?Tessero\Ext\NDArray $out = …)
```

x OR y (bool). Kernel ufunc (same code as the NDArray method).

### logicalXor

```php
static logicalXor($x, $y, ?Tessero\Ext\NDArray $out = …)
```

x XOR y (bool). Kernel ufunc (same code as the NDArray method).

### maximum

```php
static maximum($x, $y, ?Tessero\Ext\NDArray $out = …)
```

element-wise maximum, NaN propagates. Kernel ufunc (same code as the NDArray method).

### minimum

```php
static minimum($x, $y, ?Tessero\Ext\NDArray $out = …)
```

element-wise minimum, NaN propagates. Kernel ufunc (same code as the NDArray method).

### mod

```php
static mod($x, $y, ?Tessero\Ext\NDArray $out = …)
```

remainder with the sign of y (Python %). Kernel ufunc (same code as the NDArray method).

### multiply

```php
static multiply($x, $y, ?Tessero\Ext\NDArray $out = …)
```

x * y. Kernel ufunc (same code as the NDArray method).

### negative

```php
static negative($x, ?Tessero\Ext\NDArray $out = …)
```

-x. Kernel ufunc (same code as the NDArray method).

### nextafter

```php
static nextafter($x, $y, ?Tessero\Ext\NDArray $out = …)
```

next representable value after x towards y. Loop ufunc.

### notEqual

```php
static notEqual($x, $y, ?Tessero\Ext\NDArray $out = …)
```

x != y (bool). Kernel ufunc (same code as the NDArray method).

### power

```php
static power($x, $y, ?Tessero\Ext\NDArray $out = …)
```

x ** y. Kernel ufunc (same code as the NDArray method).

### radians

```php
static radians($x, ?Tessero\Ext\NDArray $out = …)
```

degrees to radians. Loop ufunc.

### real

```php
static real($x, ?Tessero\Ext\NDArray $out = …)
```

real part. Kernel ufunc (same code as the NDArray method).

### reciprocal

```php
static reciprocal($x, ?Tessero\Ext\NDArray $out = …)
```

1 / x. Kernel ufunc (same code as the NDArray method).

### remainder

```php
static remainder($x, $y, ?Tessero\Ext\NDArray $out = …)
```

alias of mod. Kernel ufunc (same code as the NDArray method).

### rint

```php
static rint($x, ?Tessero\Ext\NDArray $out = …)
```

round to nearest, ties to even. Kernel ufunc (same code as the NDArray method).

### sign

```php
static sign($x, ?Tessero\Ext\NDArray $out = …)
```

-1, 0 or 1. Kernel ufunc (same code as the NDArray method).

### signbit

```php
static signbit($x, ?Tessero\Ext\NDArray $out = …)
```

sign bit is set (bool; true for -0.0). Loop ufunc.

### sin

```php
static sin($x, ?Tessero\Ext\NDArray $out = …)
```

sine (radians). Kernel ufunc (same code as the NDArray method).

### sinh

```php
static sinh($x, ?Tessero\Ext\NDArray $out = …)
```

hyperbolic sine. Kernel ufunc (same code as the NDArray method).

### sqrt

```php
static sqrt($x, ?Tessero\Ext\NDArray $out = …)
```

square root. Kernel ufunc (same code as the NDArray method).

### square

```php
static square($x, ?Tessero\Ext\NDArray $out = …)
```

x * x. Kernel ufunc (same code as the NDArray method).

### subtract

```php
static subtract($x, $y, ?Tessero\Ext\NDArray $out = …)
```

x - y. Kernel ufunc (same code as the NDArray method).

### tan

```php
static tan($x, ?Tessero\Ext\NDArray $out = …)
```

tangent (radians). Kernel ufunc (same code as the NDArray method).

### tanh

```php
static tanh($x, ?Tessero\Ext\NDArray $out = …)
```

hyperbolic tangent. Kernel ufunc (same code as the NDArray method).

### trunc

```php
static trunc($x, ?Tessero\Ext\NDArray $out = …)
```

round towards zero. Loop ufunc.

### ufuncs

```php
static ufuncs()
```
