<?php

declare(strict_types=1);

namespace Tessero\Ext\Fft;

use Tessero\Ext\Engine;
use Tessero\Ext\NDArray;
use Tessero\Exceptions\ShapeError;

/**
 * numpy.fft for the extension backend, mirroring Tessero\Fft\Fft but built on the extension's complex FFT
 * (Tessero\Ext\Engine::fft, which transforms the last axis) and Ext\NDArray. Real (rfft/irfft) transforms are
 * derived from the complex FFT; index gathers (roll, Hermitian rebuild) go through flat data since the extension
 * NDArray has no take/concatenate.
 */
final class Fft
{
    private static function toArr(mixed $a): NDArray
    {
        return $a instanceof NDArray ? $a : NDArray::array($a);
    }

    private static function axis(int $axis, int $nd): int
    {
        return $axis < 0 ? $axis + $nd : $axis;
    }

    /** @return list<int> */
    private static function permToLast(int $ax, int $nd): array
    {
        $p = [];
        for ($i = 0; $i < $nd; $i++) {
            if ($i !== $ax) {
                $p[] = $i;
            }
        }
        $p[] = $ax;
        return $p;
    }

    /** @return list<int> */
    private static function permFromLast(int $ax, int $nd): array
    {
        $to = self::permToLast($ax, $nd);
        $inv = array_fill(0, $nd, 0);
        foreach ($to as $i => $v) {
            $inv[$v] = $i;
        }
        return $inv;
    }

    private static function scale(string $norm, int $len, bool $inverse): float
    {
        return match ($norm) {
            'backward' => 1.0,
            'ortho' => $inverse ? sqrt($len) : 1.0 / sqrt($len),
            'forward' => $inverse ? (float) $len : 1.0 / $len,
            default => throw new ShapeError("Unknown norm '{$norm}' (backward, ortho, forward)."),
        };
    }

    public static function fft(mixed $a, ?int $n = null, int $axis = -1, string $norm = 'backward'): NDArray
    {
        return self::transform(self::toArr($a), $n, $axis, false, $norm);
    }

    public static function ifft(mixed $a, ?int $n = null, int $axis = -1, string $norm = 'backward'): NDArray
    {
        return self::transform(self::toArr($a), $n, $axis, true, $norm);
    }

    private static function transform(NDArray $x, ?int $n, int $axis, bool $inverse, string $norm): NDArray
    {
        $nd = count($x->shape());
        if ($nd === 0) {
            throw new ShapeError('fft needs at least one dimension.');
        }
        $ax = self::axis($axis, $nd);
        $x = $x->astype('complex128');
        $len = $n ?? $x->shape()[$ax];
        if ($len < 1) {
            throw new ShapeError("Invalid number of FFT data points ({$len}).");
        }
        if ($len !== $x->shape()[$ax]) {
            $x = self::fitLength($x, $len, $ax);
        }
        $moved = $ax === $nd - 1 ? $x : $x->transpose(self::permToLast($ax, $nd));
        $y = Engine::fft($moved, $inverse);
        $s = self::scale($norm, $len, $inverse);
        if ($s !== 1.0) {
            $y = $y->mul($s);
        }
        return $ax === $nd - 1 ? $y : $y->transpose(self::permFromLast($ax, $nd));
    }

    public static function rfft(mixed $a, ?int $n = null, int $axis = -1, string $norm = 'backward'): NDArray
    {
        $x = self::toArr($a)->astype('float64');
        $nd = count($x->shape());
        if ($nd === 0) {
            throw new ShapeError('fft needs at least one dimension.');
        }
        $ax = self::axis($axis, $nd);
        $len = $n ?? $x->shape()[$ax];
        if ($len < 1) {
            throw new ShapeError("Invalid number of FFT data points ({$len}).");
        }
        $full = self::transform($x, $len, $ax, false, 'backward');
        $h = intdiv($len, 2) + 1;
        $half = self::sliceAxis($full, $ax, $h);
        $s = self::scale($norm, $len, false);
        return $s !== 1.0 ? $half->mul($s) : $half;
    }

    public static function irfft(mixed $a, ?int $n = null, int $axis = -1, string $norm = 'backward'): NDArray
    {
        $x = self::toArr($a)->astype('complex128');
        $nd = count($x->shape());
        if ($nd === 0) {
            throw new ShapeError('fft needs at least one dimension.');
        }
        $ax = self::axis($axis, $nd);
        $shape = $x->shape();
        $m = $shape[$ax];
        $n ??= 2 * ($m - 1);
        if ($n < 1) {
            throw new ShapeError('irfft: invalid output length.');
        }
        $moved = $ax === $nd - 1 ? $x : $x->transpose(self::permToLast($ax, $nd));
        $reIn = $moved->real()->toList();                 // the real/imag parts separately (C-order of moved)
        $imIn = $moved->imag()->toList();
        $rows = $m > 0 ? intdiv((int) ($x->size()), $m) : 0;
        $re = [];
        $im = [];
        for ($r = 0; $r < $rows; $r++) {
            $fr = array_fill(0, $n, 0.0);
            $fi = array_fill(0, $n, 0.0);
            for ($k = 0; $k < $m && $k < $n; $k++) {
                $fr[$k] = $reIn[$r * $m + $k];
                $fi[$k] = $imIn[$r * $m + $k];
            }
            for ($k = 1; $k <= $n - $m; $k++) {           // Hermitian symmetry
                $fr[$n - $k] = $reIn[$r * $m + $k];
                $fi[$n - $k] = -$imIn[$r * $m + $k];
            }
            for ($j = 0; $j < $n; $j++) {
                $re[] = $fr[$j];
                $im[] = $fi[$j];
            }
        }
        $spec = $rows > 0 ? NDArray::complex(NDArray::array($re, 'float64'), NDArray::array($im, 'float64'))->reshape([$rows, $n]) : NDArray::array([], 'float64')->reshape([0, $n]);
        $y = Engine::fft($spec, true)->real();            // ifft (1/n), real part -> (rows, n)
        $movedShape = $moved->shape();
        $movedShape[count($movedShape) - 1] = $n;
        $y = $y->reshape($movedShape);
        if ($ax !== $nd - 1) {
            $y = $y->transpose(self::permFromLast($ax, $nd));
        }
        $s = self::scale($norm, $n, true);
        return $s !== 1.0 ? $y->mul($s) : $y;
    }

    public static function fft2(mixed $a, string $norm = 'backward'): NDArray
    {
        return self::fft(self::fft($a, null, -1, $norm), null, -2, $norm);
    }

    public static function ifft2(mixed $a, string $norm = 'backward'): NDArray
    {
        return self::ifft(self::ifft($a, null, -1, $norm), null, -2, $norm);
    }

    /**
     * N-D FFT over the given axes (every axis by default).
     *
     * @param list<int>|null $s     output length per transformed axis
     * @param list<int>|null $axes  axes to transform
     */
    public static function fftn(mixed $a, ?array $s = null, ?array $axes = null, string $norm = 'backward'): NDArray
    {
        return self::transformN(self::toArr($a), $s, $axes, $norm, false);
    }

    public static function ifftn(mixed $a, ?array $s = null, ?array $axes = null, string $norm = 'backward'): NDArray
    {
        return self::transformN(self::toArr($a), $s, $axes, $norm, true);
    }

    private static function transformN(NDArray $x, ?array $s, ?array $axes, string $norm, bool $inverse): NDArray
    {
        [$axes, $s] = self::axesAndSizes($x, $s, $axes);
        foreach ($axes as $i => $ax) {
            $x = $inverse ? self::ifft($x, $s[$i], $ax, $norm) : self::fft($x, $s[$i], $ax, $norm);
        }

        return $x;
    }

    /** rfft over the last of the axes, then a complex fft over the rest (numpy.fft.rfftn). */
    public static function rfftn(mixed $a, ?array $s = null, ?array $axes = null, string $norm = 'backward'): NDArray
    {
        $x = self::toArr($a);
        [$axes, $s] = self::axesAndSizes($x, $s, $axes);
        $last = count($axes) - 1;
        $x = self::rfft($x, $s[$last], $axes[$last], $norm);
        for ($i = 0; $i < $last; $i++) {
            $x = self::fft($x, $s[$i], $axes[$i], $norm);
        }

        return $x;
    }

    public static function rfft2(mixed $a, ?array $s = null, ?array $axes = null, string $norm = 'backward'): NDArray
    {
        return self::rfftn($a, $s, $axes ?? [-2, -1], $norm);
    }

    /** Inverse of rfftn: complex ifft over all but the last axis, then irfft on the last (numpy.fft.irfftn). */
    public static function irfftn(mixed $a, ?array $s = null, ?array $axes = null, string $norm = 'backward'): NDArray
    {
        $x = self::toArr($a);
        $nd = count($x->shape());
        if ($axes === null) {
            $axes = $s === null ? range(0, $nd - 1) : range($nd - count($s), $nd - 1);
        }
        $axes = array_map(static fn (int $ax): int => self::axis($ax, $nd), $axes);
        $last = count($axes) - 1;
        if ($s === null) {
            $s = array_map(static fn (int $ax): int => $x->shape()[$ax], $axes);
            $s[$last] = 2 * ($x->shape()[$axes[$last]] - 1);
        }
        for ($i = 0; $i < $last; $i++) {
            $x = self::ifft($x, $s[$i], $axes[$i], $norm);
        }

        return self::irfft($x, $s[$last], $axes[$last], $norm);
    }

    public static function irfft2(mixed $a, ?array $s = null, ?array $axes = null, string $norm = 'backward'): NDArray
    {
        return self::irfftn($a, $s, $axes ?? [-2, -1], $norm);
    }

    /** FFT of a Hermitian-symmetric spectrum, giving a real signal (numpy.fft.hfft). */
    public static function hfft(mixed $a, ?int $n = null, int $axis = -1, string $norm = 'backward'): NDArray
    {
        $x = self::toArr($a);
        $ax = self::axis($axis, count($x->shape()));
        $n ??= 2 * ($x->shape()[$ax] - 1);

        return self::irfft($x->conj(), $n, $ax, self::swapNorm($norm));
    }

    /** Inverse of hfft: the Hermitian half-spectrum of a real signal (numpy.fft.ihfft). */
    public static function ihfft(mixed $a, ?int $n = null, int $axis = -1, string $norm = 'backward'): NDArray
    {
        $x = self::toArr($a);
        $ax = self::axis($axis, count($x->shape()));
        $n ??= $x->shape()[$ax];

        return self::rfft($x, $n, $ax, self::swapNorm($norm))->conj();
    }

    /** N-D FFT of a Hermitian-symmetric spectrum, giving a real signal (scipy.fft.hfftn). */
    public static function hfftn(mixed $a, ?array $s = null, ?array $axes = null, string $norm = 'backward'): NDArray
    {
        return self::irfftn(self::toArr($a)->conj(), $s, $axes, self::swapNorm($norm));
    }

    public static function hfft2(mixed $a, ?array $s = null, ?array $axes = null, string $norm = 'backward'): NDArray
    {
        return self::hfftn($a, $s, $axes ?? [-2, -1], $norm);
    }

    /** Inverse of hfftn: the Hermitian half-spectrum of a real n-D signal (scipy.fft.ihfftn). */
    public static function ihfftn(mixed $a, ?array $s = null, ?array $axes = null, string $norm = 'backward'): NDArray
    {
        return self::rfftn(self::toArr($a), $s, $axes, self::swapNorm($norm))->conj();
    }

    public static function ihfft2(mixed $a, ?array $s = null, ?array $axes = null, string $norm = 'backward'): NDArray
    {
        return self::ihfftn($a, $s, $axes ?? [-2, -1], $norm);
    }

    /** Smallest efficient FFT length >= target: a product of 2,3,5(,7,11 for complex) (scipy.fft.next_fast_len). */
    public static function nextFastLen(int $target, bool $real = false): int
    {
        if ($target <= 1) {
            return 1;
        }
        for ($n = $target; ; $n++) {
            if (self::fftGoodSize($n, $real)) {
                return $n;
            }
        }
    }

    /** Largest efficient FFT length <= target (scipy.fft.prev_fast_len). */
    public static function prevFastLen(int $target, bool $real = false): int
    {
        for ($n = max($target, 1); $n > 1; $n--) {
            if (self::fftGoodSize($n, $real)) {
                return $n;
            }
        }

        return 1;
    }

    /** True when n factors entirely into {2,3,5} (real) or {2,3,5,7,11} (complex), pocketfft's "good size". */
    private static function fftGoodSize(int $n, bool $real): bool
    {
        foreach ($real ? [2, 3, 5] : [2, 3, 5, 7, 11] as $p) {
            while ($n % $p === 0) {
                $n = intdiv($n, $p);
            }
        }

        return $n === 1;
    }

    private static function swapNorm(string $norm): string
    {
        return match ($norm) {
            'backward' => 'forward',
            'forward' => 'backward',
            'ortho' => 'ortho',
            default => throw new ShapeError("Unknown norm '{$norm}' (backward, ortho, forward)."),
        };
    }

    /**
     * Resolve the (axes, sizes) pair the way numpy.fft.fftn does: axes default to every axis (or the last
     * count($s) axes when only s is given); each size defaults to the current length of its axis.
     *
     * @param  list<int>|null $s
     * @param  list<int>|null $axes
     * @return array{list<int>, list<int>}
     */
    private static function axesAndSizes(NDArray $x, ?array $s, ?array $axes): array
    {
        $nd = count($x->shape());
        if ($axes === null) {
            $axes = $s === null ? range(0, $nd - 1) : range($nd - count($s), $nd - 1);
        }
        $axes = array_map(static fn (int $ax): int => self::axis($ax, $nd), $axes);
        if ($s === null) {
            $s = array_map(static fn (int $ax): int => $x->shape()[$ax], $axes);
        }

        return [array_values($axes), array_values($s)];
    }

    public static function fftfreq(int $n, float $d = 1.0): NDArray
    {
        $vals = [];
        $pos = intdiv($n - 1, 2) + 1;
        for ($i = 0; $i < $pos; $i++) {
            $vals[] = $i / ($n * $d);
        }
        for ($i = -intdiv($n, 2); $i < 0; $i++) {
            $vals[] = $i / ($n * $d);
        }
        return NDArray::array($vals, 'float64');
    }

    public static function rfftfreq(int $n, float $d = 1.0): NDArray
    {
        $vals = [];
        for ($i = 0, $h = intdiv($n, 2); $i <= $h; $i++) {
            $vals[] = $i / ($n * $d);
        }
        return NDArray::array($vals, 'float64');
    }

    public static function fftshift(mixed $a, ?int $axis = null): NDArray
    {
        return self::roll(self::toArr($a), $axis, static fn (int $len): int => intdiv($len, 2));
    }

    public static function ifftshift(mixed $a, ?int $axis = null): NDArray
    {
        return self::roll(self::toArr($a), $axis, static fn (int $len): int => -intdiv($len, 2));
    }

    /** Slice [0:stop] along one axis via a numpy-style spec string. */
    private static function sliceAxis(NDArray $x, int $ax, int $stop): NDArray
    {
        $nd = count($x->shape());
        $parts = [];
        for ($i = 0; $i < $nd; $i++) {
            $parts[] = $i === $ax ? "0:{$stop}" : ':';
        }
        return $x->slice(implode(',', $parts));
    }

    private static function fitLength(NDArray $x, int $len, int $ax): NDArray
    {
        $cur = $x->shape()[$ax];
        if ($cur === $len) {
            return $x;
        }
        if ($cur > $len) {
            return self::sliceAxis($x, $ax, $len);
        }
        throw new ShapeError('fft: zero-padding to a longer length is not supported on the extension backend.');
    }

    private static function roll(NDArray $x, ?int $axis, callable $shiftFor): NDArray
    {
        $shape = $x->shape();
        $nd = count($shape);
        if ($nd === 0) {
            return $x;
        }
        $axes = $axis === null ? range(0, $nd - 1) : [self::axis($axis, $nd)];
        $flat = $x->toList();
        $total = 1;
        foreach ($shape as $d) {
            $total *= $d;
        }
        $w = $total > 0 ? intdiv(count($flat), $total) : 1;   // 1 = real, 2 = complex (interleaved)
        $stride = array_fill(0, $nd, 1);
        for ($i = $nd - 2; $i >= 0; $i--) {
            $stride[$i] = $stride[$i + 1] * $shape[$i + 1];
        }
        $sh = array_fill(0, $nd, 0);
        foreach ($axes as $ax) {
            $len = $shape[$ax];
            if ($len > 0) {
                $sh[$ax] = ((($shiftFor($len)) % $len) + $len) % $len;
            }
        }
        $re = [];
        $im = [];
        for ($lin = 0; $lin < $total; $lin++) {
            $src = 0;
            for ($d = 0; $d < $nd; $d++) {
                $coord = intdiv($lin, $stride[$d]) % $shape[$d];
                $srcCoord = (($coord - $sh[$d]) % $shape[$d] + $shape[$d]) % $shape[$d];
                $src += $srcCoord * $stride[$d];
            }
            if ($w === 2) {
                $re[] = $flat[2 * $src];
                $im[] = $flat[2 * $src + 1];
            } else {
                $re[] = $flat[$src];
            }
        }
        if ($w === 2) {
            return NDArray::complex(NDArray::array($re, 'float64'), NDArray::array($im, 'float64'))->reshape($shape);
        }
        return NDArray::array($re, 'float64')->reshape($shape);
    }
}
