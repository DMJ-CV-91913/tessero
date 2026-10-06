<?php

declare(strict_types=1);

namespace Tessero\Fft;

use Tessero\DType;
use Tessero\Exceptions\ShapeError;
use Tessero\NDArray;
use Tessero\Native\Library;

/**
 * numpy.fft: any length (mixed radix 2/3/4/5 + Bluestein for large prime
 * factors), along any axis, "backward" normalisation (inverse scales by 1/n)
 * or "ortho"/"forward".
 */
final class Fft
{
    public static function fft(mixed $a, ?int $n = null, int $axis = -1, string $norm = 'backward'): NDArray
    {
        return self::transform(NDArray::asArray($a), $n, $axis, false, $norm);
    }

    public static function ifft(mixed $a, ?int $n = null, int $axis = -1, string $norm = 'backward'): NDArray
    {
        return self::transform(NDArray::asArray($a), $n, $axis, true, $norm);
    }

    /** FFT of real input: the n/2 + 1 non-negative frequency terms (libtessero's half-length real FFT). */
    public static function rfft(mixed $a, ?int $n = null, int $axis = -1, string $norm = 'backward'): NDArray
    {
        $x = NDArray::asArray($a);
        if ($x->dtype() === DType::Complex128) {
            $x = $x->real();
        }
        if ($x->shape() === []) {
            throw new ShapeError('fft needs at least one dimension.');
        }
        $ax = NDArray::normAxis($axis, count($x->shape()));
        $x = $x->astype(DType::Float64);
        $len = $n ?? $x->shape()[$ax];
        if ($len < 1) {
            throw new ShapeError("Invalid number of FFT data points ({$len}).");
        }
        $x = self::fitLength($x, $len, $ax);
        $work = $x->moveAxis($ax, -1)->copy();                  // rows x len, contiguous
        $h = intdiv($len, 2) + 1;
        $rows = intdiv($work->size(), $len);
        $outShape = $work->shape();
        $outShape[count($outShape) - 1] = $h;
        $out = NDArray::empty($outShape, DType::Complex128);
        if ($rows > 0) {
            $ffi = Library::ffi();
            Library::check($ffi->tsr_rfft($len, $rows, $ffi->cast('double*', $work->ptr()), $ffi->cast('double*', $out->ptr())), 'rfft');
        }
        $scale = self::scale($norm, $len, false);
        if ($scale !== 1.0) {
            $out = $out->mul($scale);
        }

        return $out->moveAxis(-1, $ax)->contiguous();
    }

    /** Inverse of rfft; n is the length of the real output (default 2 (m - 1)). */
    public static function irfft(mixed $a, ?int $n = null, int $axis = -1, string $norm = 'backward'): NDArray
    {
        $x = NDArray::asArray($a)->astype(DType::Complex128);
        if ($x->shape() === []) {
            throw new ShapeError('fft needs at least one dimension.');
        }
        $ax = NDArray::normAxis($axis, count($x->shape()));
        $m = $x->shape()[$ax];
        $n ??= 2 * ($m - 1);
        if ($n < 1) {
            throw new ShapeError('irfft: invalid output length.');
        }
        $h = intdiv($n, 2) + 1;
        $work = self::fitLength($x, $h, $ax)->moveAxis($ax, -1)->copy();
        $rows = intdiv($work->size(), $h);
        $outShape = $work->shape();
        $outShape[count($outShape) - 1] = $n;
        $out = NDArray::empty($outShape, DType::Float64);
        if ($rows > 0) {
            $ffi = Library::ffi();
            Library::check($ffi->tsr_irfft($n, $rows, $ffi->cast('double*', $work->ptr()), $ffi->cast('double*', $out->ptr())), 'irfft');
        }
        $scale = self::scale($norm, $n, true);
        if ($scale !== 1.0) {
            $out = $out->mul($scale);
        }

        return $out->moveAxis(-1, $ax)->contiguous();
    }

    /** Extra factor on top of the kernel's convention (forward unscaled, inverse 1/n). */
    private static function scale(string $norm, int $len, bool $inverse): float
    {
        return match ($norm) {
            'backward' => 1.0,
            'ortho' => $inverse ? sqrt($len) : 1.0 / sqrt($len),
            'forward' => $inverse ? (float) $len : 1.0 / $len,
            default => throw new ShapeError("Unknown norm '{$norm}' (backward, ortho, forward)."),
        };
    }

    /** 2-D FFT over the last two axes. */
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
        return self::transformN(NDArray::asArray($a), $s, $axes, $norm, false);
    }

    public static function ifftn(mixed $a, ?array $s = null, ?array $axes = null, string $norm = 'backward'): NDArray
    {
        return self::transformN(NDArray::asArray($a), $s, $axes, $norm, true);
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
        $x = NDArray::asArray($a);
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
        $x = NDArray::asArray($a);
        $nd = count($x->shape());
        if ($axes === null) {
            $axes = $s === null ? range(0, $nd - 1) : range($nd - count($s), $nd - 1);
        }
        $axes = array_map(static fn (int $ax): int => NDArray::normAxis($ax, $nd), $axes);
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
        $x = NDArray::asArray($a);
        $ax = NDArray::normAxis($axis, count($x->shape()));
        $n ??= 2 * ($x->shape()[$ax] - 1);

        return self::irfft($x->conj(), $n, $ax, self::swapNorm($norm));
    }

    /** Inverse of hfft: the Hermitian half-spectrum of a real signal (numpy.fft.ihfft). */
    public static function ihfft(mixed $a, ?int $n = null, int $axis = -1, string $norm = 'backward'): NDArray
    {
        $x = NDArray::asArray($a);
        $ax = NDArray::normAxis($axis, count($x->shape()));
        $n ??= $x->shape()[$ax];

        return self::rfft($x, $n, $ax, self::swapNorm($norm))->conj();
    }

    /** N-D FFT of a Hermitian-symmetric spectrum, giving a real signal (scipy.fft.hfftn). */
    public static function hfftn(mixed $a, ?array $s = null, ?array $axes = null, string $norm = 'backward'): NDArray
    {
        return self::irfftn(NDArray::asArray($a)->conj(), $s, $axes, self::swapNorm($norm));
    }

    public static function hfft2(mixed $a, ?array $s = null, ?array $axes = null, string $norm = 'backward'): NDArray
    {
        return self::hfftn($a, $s, $axes ?? [-2, -1], $norm);
    }

    /** Inverse of hfftn: the Hermitian half-spectrum of a real n-D signal (scipy.fft.ihfftn). */
    public static function ihfftn(mixed $a, ?array $s = null, ?array $axes = null, string $norm = 'backward'): NDArray
    {
        return self::rfftn(NDArray::asArray($a), $s, $axes, self::swapNorm($norm))->conj();
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
        $axes = array_map(static fn (int $ax): int => NDArray::normAxis($ax, $nd), $axes);
        if ($s === null) {
            $s = array_map(static fn (int $ax): int => $x->shape()[$ax], $axes);
        }

        return [array_values($axes), array_values($s)];
    }

    /** Sample frequencies for fft of length n with sample spacing d. */
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

        return NDArray::fromFlat($vals, [$n]);
    }

    public static function rfftfreq(int $n, float $d = 1.0): NDArray
    {
        return NDArray::arange(intdiv($n, 2) + 1, dtype: DType::Float64)->div($n * $d);
    }

    /** Move the zero-frequency term to the centre. */
    public static function fftshift(mixed $a, ?int $axis = null): NDArray
    {
        return self::roll(NDArray::asArray($a), $axis, fn (int $len): int => intdiv($len, 2));
    }

    public static function ifftshift(mixed $a, ?int $axis = null): NDArray
    {
        return self::roll(NDArray::asArray($a), $axis, fn (int $len): int => -intdiv($len, 2));
    }

    private static function roll(NDArray $x, ?int $axis, callable $shiftFor): NDArray
    {
        $axes = $axis === null ? range(0, count($x->shape()) - 1) : [NDArray::normAxis($axis, count($x->shape()))];
        foreach ($axes as $ax) {
            $len = $x->shape()[$ax];
            if ($len === 0) {
                continue;
            }
            $shift = (($shiftFor($len) % $len) + $len) % $len;
            $idx = [];
            for ($i = 0; $i < $len; $i++) {
                $idx[] = ($i - $shift + $len) % $len;
            }
            $x = $x->take($idx, $ax);
        }

        return $x;
    }

    private static function transform(NDArray $x, ?int $n, int $axis, bool $inverse, string $norm): NDArray
    {
        if ($x->shape() === []) {
            throw new ShapeError('fft needs at least one dimension.');
        }
        $ax = NDArray::normAxis($axis, count($x->shape()));
        $fresh = $x->dtype() !== DType::Complex128;
        $x = $x->astype(DType::Complex128);
        $len = $n ?? $x->shape()[$ax];
        if ($len < 1) {
            throw new ShapeError("Invalid number of FFT data points ({$len}).");
        }
        if ($len !== $x->shape()[$ax]) {
            $x = self::fitLength($x, $len, $ax);
            $fresh = false;
        }
        $last = $ax === count($x->shape()) - 1;
        // transform in place when we already own a contiguous complex copy along the last axis
        $work = $fresh && $last && $x->isContiguous() ? $x : $x->moveAxis($ax, -1)->copy();
        $rows = intdiv($work->size(), $len);
        if ($rows > 0) {
            $ffi = Library::ffi();
            $p = $ffi->cast('double*', $work->ptr());
            Library::check($ffi->tsr_fft($len, $rows, $inverse ? 1 : 0, $p, $p), 'fft');
        }
        $scale = match ($norm) {
            'backward' => 1.0,
            'ortho' => $inverse ? sqrt($len) : 1.0 / sqrt($len),
            'forward' => $inverse ? (float) $len : 1.0 / $len,
            default => throw new ShapeError("Unknown norm '{$norm}' (backward, ortho, forward)."),
        };
        if ($scale !== 1.0) {
            $work = $work->mul($scale);
        }

        return $work->moveAxis(-1, $ax)->contiguous();
    }

    private static function fitLength(NDArray $x, int $len, int $ax): NDArray
    {
        $cur = $x->shape()[$ax];
        if ($cur === $len) {
            return $x;
        }
        if ($cur > $len) {
            $spec = array_fill(0, count($x->shape()), ':');
            $spec[$ax] = '0:' . $len;

            return $x->slice(...$spec);
        }
        $shape = $x->shape();
        $shape[$ax] = $len - $cur;

        return NDArray::concatenate([$x, NDArray::zeros($shape, $x->dtype())], $ax);
    }
}
