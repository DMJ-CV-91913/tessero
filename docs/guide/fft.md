# FFT

`Tessero\Fft\Fft` follows `numpy.fft`.

```php
use Tessero\Fft\Fft;

$x = linspace(0, 1, 1000, endpoint: false);
$signal = $x->mul(2 * M_PI * 50)->sin()->add($x->mul(2 * M_PI * 120)->sin()->mul(0.5));

$X = Fft::rfft($signal);              // complex128, length n/2 + 1; half-length transform, ~1.6x faster than fft()
$freq = Fft::rfftfreq(1000, d: 1 / 1000);
$power = $X->abs();                   // peaks at 50 Hz and 120 Hz
$back = Fft::irfft($X, n: 1000);      // equals $signal to ~1e-15

Fft::fft($z); Fft::ifft($Z);          // complex, any axis (axis: -1 default)
Fft::fft2($img); Fft::ifft2($F);
Fft::fftfreq(8); Fft::fftshift($F); Fft::ifftshift($F);
Fft::fft($x, n: 4096);                // zero-pad or truncate
Fft::fft($x, norm: 'ortho');          // 'backward' (default), 'ortho', 'forward'
```

## Any length

Lengths are factored into radix 2, 3, 4 and 5 butterflies plus a generic
odd-radix pass. Lengths with a large prime factor use Bluestein's algorithm, so
cost stays O(n log n) for every n: a transform of 1 000 003 points (prime)
takes about 0.55 s. Twiddle factors are computed directly with `sin`/`cos` for
every index (not by recurrence) and stored per level, so the error stays at
the round-off level.

Checked against `numpy.fft` at 1e-10 relative on 14 lengths, including
primes 17, 97, 127 and 1031. The C tests also compare it with a long-double
direct DFT.

## Performance

About 1.5–2× slower than NumPy's pocketfft (70–98 ms for 2^20 points
against 48 ms). The real-input transform still runs a full complex transform
internally. Both are on the [roadmap](../project/roadmap.md).

## Extension

`Tessero\Ext\Engine::fft($x, inverse: false)` transforms the last axis of a
real or complex array with the same kernel and returns a complex128 array.
`toList()` shows each element as `[re, im]`, and `real()`, `imag()`,
`conj()`, `angle()` and `abs()` work on the result.
