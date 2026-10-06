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

### fftshift

```php
static fftshift(mixed $a, ?int $axis = null): Tessero\NDArray
```

Move the zero-frequency term to the centre.

### ifft

```php
static ifft(mixed $a, ?int $n = null, int $axis = -1, string $norm = 'backward'): Tessero\NDArray
```

### ifft2

```php
static ifft2(mixed $a, string $norm = 'backward'): Tessero\NDArray
```

### ifftshift

```php
static ifftshift(mixed $a, ?int $axis = null): Tessero\NDArray
```

### irfft

```php
static irfft(mixed $a, ?int $n = null, int $axis = -1, string $norm = 'backward'): Tessero\NDArray
```

Inverse of rfft; n is the length of the real output (default 2 (m - 1)).

### rfft

```php
static rfft(mixed $a, ?int $n = null, int $axis = -1, string $norm = 'backward'): Tessero\NDArray
```

FFT of real input: the n/2 + 1 non-negative frequency terms (libtessero's half-length real FFT).

### rfftfreq

```php
static rfftfreq(int $n, float $d = 1.0): Tessero\NDArray
```
