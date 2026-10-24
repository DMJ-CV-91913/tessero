# Fft

`Tessero\Fft\Fft`

numpy.fft: any length (mixed radix 2/3/4/5 + Bluestein for large prime
factors), along any axis, "backward" normalisation (inverse scales by 1/n)
or "ortho"/"forward".

## Methods

### fft

```php
static fft(mixed $a, ?int $n = null, int $axis = -1, string $norm = 'backward'): Tessero\NDArray
```

### fft2

```php
static fft2(mixed $a, string $norm = 'backward'): Tessero\NDArray
```

2-D FFT over the last two axes.

### fftfreq

```php
static fftfreq(int $n, float $d = 1.0): Tessero\NDArray
```

Sample frequencies for fft of length n with sample spacing d.

### fftn

```php
static fftn(mixed $a, ?array $s = null, ?array $axes = null, string $norm = 'backward'): Tessero\NDArray
```

N-D FFT over the given axes (every axis by default).

### fftshift

```php
static fftshift(mixed $a, ?int $axis = null): Tessero\NDArray
```

Move the zero-frequency term to the centre.

### hfft

```php
static hfft(mixed $a, ?int $n = null, int $axis = -1, string $norm = 'backward'): Tessero\NDArray
```

FFT of a Hermitian-symmetric spectrum, giving a real signal (numpy.fft.hfft).

### hfft2

```php
static hfft2(mixed $a, ?array $s = null, ?array $axes = null, string $norm = 'backward'): Tessero\NDArray
```

### hfftn

```php
static hfftn(mixed $a, ?array $s = null, ?array $axes = null, string $norm = 'backward'): Tessero\NDArray
```

N-D FFT of a Hermitian-symmetric spectrum, giving a real signal (scipy.fft.hfftn).

### ifft

```php
static ifft(mixed $a, ?int $n = null, int $axis = -1, string $norm = 'backward'): Tessero\NDArray
```

### ifft2

```php
static ifft2(mixed $a, string $norm = 'backward'): Tessero\NDArray
```

### ifftn

```php
static ifftn(mixed $a, ?array $s = null, ?array $axes = null, string $norm = 'backward'): Tessero\NDArray
```

### ifftshift

```php
static ifftshift(mixed $a, ?int $axis = null): Tessero\NDArray
```

### ihfft

```php
static ihfft(mixed $a, ?int $n = null, int $axis = -1, string $norm = 'backward'): Tessero\NDArray
```

Inverse of hfft: the Hermitian half-spectrum of a real signal (numpy.fft.ihfft).

### ihfft2

```php
static ihfft2(mixed $a, ?array $s = null, ?array $axes = null, string $norm = 'backward'): Tessero\NDArray
```

### ihfftn

```php
static ihfftn(mixed $a, ?array $s = null, ?array $axes = null, string $norm = 'backward'): Tessero\NDArray
```

Inverse of hfftn: the Hermitian half-spectrum of a real n-D signal (scipy.fft.ihfftn).

### irfft

```php
static irfft(mixed $a, ?int $n = null, int $axis = -1, string $norm = 'backward'): Tessero\NDArray
```

Inverse of rfft; n is the length of the real output (default 2 (m - 1)).

### irfft2

```php
static irfft2(mixed $a, ?array $s = null, ?array $axes = null, string $norm = 'backward'): Tessero\NDArray
```

### irfftn

```php
static irfftn(mixed $a, ?array $s = null, ?array $axes = null, string $norm = 'backward'): Tessero\NDArray
```

Inverse of rfftn: complex ifft over all but the last axis, then irfft on the last (numpy.fft.irfftn).

### nextFastLen

```php
static nextFastLen(int $target, bool $real = false): int
```

Smallest efficient FFT length >= target: a product of 2,3,5(,7,11 for complex) (scipy.fft.next_fast_len).

### prevFastLen

```php
static prevFastLen(int $target, bool $real = false): int
```

Largest efficient FFT length <= target (scipy.fft.prev_fast_len).

### rfft

```php
static rfft(mixed $a, ?int $n = null, int $axis = -1, string $norm = 'backward'): Tessero\NDArray
```

FFT of real input: the n/2 + 1 non-negative frequency terms (libtessero's half-length real FFT).

### rfft2

```php
static rfft2(mixed $a, ?array $s = null, ?array $axes = null, string $norm = 'backward'): Tessero\NDArray
```

### rfftfreq

```php
static rfftfreq(int $n, float $d = 1.0): Tessero\NDArray
```

### rfftn

```php
static rfftn(mixed $a, ?array $s = null, ?array $axes = null, string $norm = 'backward'): Tessero\NDArray
```

rfft over the last of the axes, then a complex fft over the rest (numpy.fft.rfftn).
