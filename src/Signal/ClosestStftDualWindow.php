<?php

declare(strict_types=1);

namespace Tessero\Signal;

/**
 * scipy.signal.closest_STFT_dual_window — the STFT dual window of a given window
 * closest to a desired dual window.
 *
 * Standalone port of scipy.signal._short_time_fft.closest_STFT_dual_window (SciPy 1.17.1).
 * Self-contained least-squares computation over a window and its canonical dual window;
 * does NOT require the mutable/stateful ShortTimeFFT machinery.
 *
 * Windows may be real- or complex-valued 1-D arrays. A real entry is a float; a complex
 * entry is a [re, im] pair. Output preserves the input form.
 */
final class ClosestStftDualWindow
{
    public static function compute(array $win, int $hop, ?array $desiredDual = null, bool $scaled = true): array
    {
        $m = count($win);
        $complex = false;
        $wr = [];
        $wi = [];
        for ($i = 0; $i < $m; $i++) {
            [$r, $im] = self::pair($win[$i]);
            if ($im !== 0.0) { $complex = true; }
            $wr[$i] = $r;
            $wi[$i] = $im;
        }

        if ($desiredDual === null) {
            $dr = array_fill(0, $m, 1.0);
            $di = array_fill(0, $m, 0.0);
        } else {
            if (count($desiredDual) !== $m) {
                throw new \InvalidArgumentException('Parameters `win` and `desired_dual` are not 1d arrays of equal length');
            }
            $dr = [];
            $di = [];
            for ($i = 0; $i < $m; $i++) {
                [$r, $im] = self::pair($desiredDual[$i]);
                if ($im !== 0.0) { $complex = true; }
                $dr[$i] = $r;
                $di[$i] = $im;
            }
        }

        foreach ($wr as $i => $r) {
            if (! is_finite($r) || ! is_finite($wi[$i])) {
                throw new \InvalidArgumentException('Parameter win must have finite entries!');
            }
            if (! is_finite($dr[$i]) || ! is_finite($di[$i])) {
                throw new \InvalidArgumentException('Parameter desired_dual must have finite entries!');
            }
        }

        if ($hop < 1 || $hop > $m) {
            throw new \InvalidArgumentException("Parameter hop={$hop} is not an integer between 1 and {$m}!");
        }

        [$wdr, $wdi] = self::canonicalDual($wr, $wi, $hop, $m);

        $qr = [];
        $qi = [];
        for ($i = 0; $i < $m; $i++) {
            $qr[$i] = $wr[$i] * $dr[$i] + $wi[$i] * $di[$i];
            $qi[$i] = $wr[$i] * $di[$i] - $wi[$i] * $dr[$i];
        }

        $accR = $qr;
        $accI = $qi;
        for ($k = $hop; $k < $m; $k += $hop) {
            for ($i = $k; $i < $m; $i++) {
                $accR[$i] += $qr[$i - $k];
                $accI[$i] += $qi[$i - $k];
            }
            for ($i = 0; $i < $m - $k; $i++) {
                $accR[$i] += $qr[$i + $k];
                $accI[$i] += $qi[$i + $k];
            }
        }
        $qdr = [];
        $qdi = [];
        for ($i = 0; $i < $m; $i++) {
            $qdr[$i] = $wdr[$i] * $accR[$i] - $wdi[$i] * $accI[$i];
            $qdi[$i] = $wdr[$i] * $accI[$i] + $wdi[$i] * $accR[$i];
        }

        if (! $scaled) {
            $dual = self::assemble($wdr, $wdi, $dr, $di, $qdr, $qdi, $complex);

            return ['dual_win' => $dual, 'alpha' => $complex ? [1.0, 0.0] : 1.0];
        }

        $numR = 0.0;
        $numI = 0.0;
        $den = 0.0;
        for ($i = 0; $i < $m; $i++) {
            $numR += $qdr[$i] * $wdr[$i] + $qdi[$i] * $wdi[$i];
            $numI += $qdr[$i] * $wdi[$i] - $qdi[$i] * $wdr[$i];
            $den += $qdr[$i] * $qdr[$i] + $qdi[$i] * $qdi[$i];
        }
        $absNum = sqrt($numR * $numR + $numI * $numI);
        if (! ($absNum > 0.0 && $den > 1.0e-15)) {
            throw new \RuntimeException('Unable to calculate scaled closest dual window due to numerically unstable scaling factor! Try setting parameter `scaled` to False.');
        }
        $alphaR = $numR / $den;
        $alphaI = $numI / $den;

        $dual = [];
        for ($i = 0; $i < $m; $i++) {
            $diffR = $dr[$i] - $qdr[$i];
            $diffI = $di[$i] - $qdi[$i];
            $pr = $alphaR * $diffR - $alphaI * $diffI;
            $pi = $alphaR * $diffI + $alphaI * $diffR;
            $vr = $wdr[$i] + $pr;
            $vi = $wdi[$i] + $pi;
            $dual[$i] = $complex ? [$vr, $vi] : $vr;
        }

        return ['dual_win' => $dual, 'alpha' => $complex ? [$alphaR, $alphaI] : $alphaR];
    }

    private static function canonicalDual(array $wr, array $wi, int $hop, int $m): array
    {
        if ($hop > $m) {
            throw new \RuntimeException("hop={$hop} is larger than window length of {$m} => STFT not invertible!");
        }
        $w2 = [];
        for ($i = 0; $i < $m; $i++) {
            $w2[$i] = $wr[$i] * $wr[$i] + $wi[$i] * $wi[$i];
        }
        $DD = $w2;
        for ($k = $hop; $k < $m; $k += $hop) {
            for ($i = $k; $i < $m; $i++) {
                $DD[$i] += $w2[$i - $k];
            }
            for ($i = 0; $i < $m - $k; $i++) {
                $DD[$i] += $w2[$i + $k];
            }
        }
        $maxDD = max($DD);
        $rel = 1.0e-15 * $maxDD;
        foreach ($DD as $d) {
            if ($d < $rel) {
                throw new \RuntimeException('Short-time Fourier Transform not invertible!');
            }
        }
        $wdr = [];
        $wdi = [];
        for ($i = 0; $i < $m; $i++) {
            $wdr[$i] = $wr[$i] / $DD[$i];
            $wdi[$i] = $wi[$i] / $DD[$i];
        }

        return [$wdr, $wdi];
    }

    private static function assemble(array $wdr, array $wdi, array $dr, array $di, array $qdr, array $qdi, bool $complex): array
    {
        $out = [];
        $m = count($wdr);
        for ($i = 0; $i < $m; $i++) {
            $vr = $wdr[$i] + $dr[$i] - $qdr[$i];
            $vi = $wdi[$i] + $di[$i] - $qdi[$i];
            $out[$i] = $complex ? [$vr, $vi] : $vr;
        }

        return $out;
    }

    private static function pair(mixed $v): array
    {
        if (is_array($v)) {
            return [(float) $v[0], (float) ($v[1] ?? 0.0)];
        }

        return [(float) $v, 0.0];
    }
}
