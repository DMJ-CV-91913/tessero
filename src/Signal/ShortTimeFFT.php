<?php

declare(strict_types=1);

namespace Tessero\Signal;

use Tessero\Fft\Fft;

/**
 * scipy.signal.ShortTimeFFT - the modern sliding-window STFT class.
 *
 * Scope (the cleanly-deterministic, fixturable core):
 *   - construction from an explicit window array + hop + fs,
 *   - fft_mode = 'onesided' (the default, real input) and 'twosided',
 *   - scale_to = null (no rescaling); phase_shift = m_num_mid (the default),
 *   - stft(x) / istft(S) / spectrogram(x),
 *   - the canonical dual window (for inversion) and the derived scalar
 *     properties (p_min / p_max / f_pts / delta_t / delta_f / f).
 *
 * Every numeric result follows scipy's _short_time_fft.py line for line, over
 * the landed Tessero\Fft FFT primitives, so both backends agree exactly.
 *
 * Out of scope (mutable / stateful dual-window machinery): scale_to rescaling,
 * closest_STFT_dual_window, from_win_equals_dual, detrending, multi-axis input,
 * 'centered'/'onesided2X' fft modes. These expose scipy's mutable OO surface
 * which does not fit the parity fixture model; the core transform below does.
 */
final class ShortTimeFFT
{
    /** @var float[] analysis window samples */
    public array $win;

    public int $hop;

    /** sampling interval T = 1/fs */
    public float $T;

    public int $mfft;

    public string $fftMode;

    /** @var int|null phase_shift (default m_num_mid) */
    public ?int $phaseShift;

    public int $mNum;

    public int $mNumMid;

    /** @var float[]|null lazily-computed canonical dual window */
    private ?array $dualWin = null;

    /**
     * @param float[]      $win        window samples
     * @param int          $hop        hop size in samples
     * @param float        $fs         sampling frequency (T = 1/fs)
     * @param int|null     $mfft       FFT length (default len(win))
     * @param string       $fftMode    'onesided' (default) or 'twosided'
     * @param int|null     $phaseShift default 0 (scipy default); null disables the roll; int sets explicitly
     */
    public function __construct(
        array $win,
        int $hop,
        float $fs = 1.0,
        ?int $mfft = null,
        string $fftMode = 'onesided',
        int|null $phaseShift = 0
    ) {
        $this->win = array_map('floatval', array_values($win));
        $this->hop = $hop;
        $this->T = 1.0 / $fs;
        $this->mNum = count($this->win);
        $this->mNumMid = intdiv($this->mNum, 2);
        $this->mfft = $mfft ?? $this->mNum;
        if (! in_array($fftMode, ['onesided', 'twosided'], true)) {
            throw new \InvalidArgumentException("ShortTimeFFT: fft_mode '{$fftMode}' not supported (onesided|twosided).");
        }
        $this->fftMode = $fftMode;
        $this->phaseShift = $phaseShift === null ? null : (int) $phaseShift;
    }

    public function onesidedFft(): bool
    {
        return $this->fftMode === 'onesided';
    }

    public function fPts(): int
    {
        return $this->onesidedFft() ? intdiv($this->mfft, 2) + 1 : $this->mfft;
    }

    public function deltaT(): float
    {
        return $this->T * $this->hop;
    }

    public function deltaF(): float
    {
        return 1.0 / ($this->mfft * $this->T);
    }

    /** rfftfreq / fftfreq spacing, as a flat list. */
    public function f(): array
    {
        $df = $this->deltaF();
        $out = [];
        if ($this->onesidedFft()) {
            for ($i = 0; $i < $this->fPts(); $i++) {
                $out[] = $i * $df;
            }

            return $out;
        }
        // twosided fftfreq
        $n = $this->mfft;
        for ($i = 0; $i < $n; $i++) {
            $k = $i < intdiv($n + 1, 2) ? $i : $i - $n;
            $out[] = $k * $df;
        }

        return $out;
    }

    /** _pre_padding -> (k_min, p_min). */
    private function prePadding(): array
    {
        $w2 = $this->winSq();
        $n0 = -$this->mNumMid;
        $p = 0;
        for ($n = $n0; $n >= $n0 - $this->mNum - 1; $n -= $this->hop) {
            $nNext = $n - $this->hop;
            $tail = true;
            for ($i = max(0, $this->mNum + $nNext); $i < $this->mNum; $i++) {
                // w2[nNext:] all zero?  nNext is negative index into a length-mNum array
            }
            // mirror numpy: all(w2[nNext:] == 0) where nNext<0 means last |nNext| entries
            $allZero = true;
            if ($nNext < 0) {
                $start = $this->mNum + $nNext; // numpy negative index
                $start = max(0, $start);
                for ($i = $start; $i < $this->mNum; $i++) {
                    if ($w2[$i] != 0.0) { $allZero = false; break; }
                }
            } else {
                $allZero = false;
            }
            if ($nNext + $this->mNum <= 0 || $allZero) {
                return [$n, -$p];
            }
            $p++;
        }
        throw new \RuntimeException('ShortTimeFFT::prePadding did not converge.');
    }

    public function pMin(): int
    {
        return $this->prePadding()[1];
    }

    public function kMin(): int
    {
        return $this->prePadding()[0];
    }

    /** _post_padding(n) -> (k_max, p_max). */
    private function postPadding(int $n): array
    {
        $m2p = $this->mNum - $this->mNumMid;
        if ($n < $m2p) {
            throw new \InvalidArgumentException("ShortTimeFFT: n must be >= ceil(m_num/2) = {$m2p}.");
        }
        $w2 = $this->winSq();
        $q1 = intdiv($n, $this->hop);
        $k1 = $q1 * $this->hop - $this->mNumMid;
        $q = $q1;
        for ($k = $k1; $k < $n + $this->mNum; $k += $this->hop) {
            $nNext = $k + $this->hop;
            $allZero = true;
            $upper = $n - $nNext; // w2[:n-n_next]
            if ($upper > 0) {
                for ($i = 0; $i < min($upper, $this->mNum); $i++) {
                    if ($w2[$i] != 0.0) { $allZero = false; break; }
                }
            }
            if ($nNext >= $n || $allZero) {
                return [$k + $this->mNum, $q + 1];
            }
            $q++;
        }
        throw new \RuntimeException('ShortTimeFFT::postPadding did not converge.');
    }

    public function pMax(int $n): int
    {
        return $this->postPadding($n)[1];
    }

    public function kMax(int $n): int
    {
        return $this->postPadding($n)[0];
    }

    public function pNum(int $n): int
    {
        return $this->pMax($n) - $this->pMin();
    }

    public function pRange(int $n, ?int $p0 = null, ?int $p1 = null): array
    {
        $pMax = $this->pMax($n);
        $pMin = $this->pMin();
        $p0_ = $p0 ?? $pMin;
        $p1_ = $p1 ?? $pMax;
        if (! ($pMin <= $p0_ && $p0_ < $p1_ && $p1_ <= $pMax)) {
            throw new \InvalidArgumentException("ShortTimeFFT::pRange: invalid p0/p1 for n={$n}.");
        }

        return [$p0_, $p1_];
    }

    /**
     * Short-time Fourier transform of a real 1-D signal.
     *
     * @param float[] $x
     * @return array<int, array<int, array{0: float, 1: float}>> frequency x time grid of [re, im]
     */
    public function stft(array $x): array
    {
        $x = array_map('floatval', array_values($x));
        $n = count($x);
        [$p0, $p1] = $this->pRange($n);
        $fPts = $this->fPts();

        // _x_slices: zero padding (padding='zeros'), k_off=0
        $n1 = ($p1 - $p0) * $this->hop;
        $k0 = $p0 * $this->hop - $this->mNumMid;
        $k1 = $k0 + $n1 + $this->mNum;

        // build padded window of x over [k0, k1)
        $S = []; // [freq][time]
        for ($f = 0; $f < $fPts; $f++) {
            $S[$f] = [];
        }

        $col = 0;
        for ($kStart = $k0; $kStart < $k0 + $n1; $kStart += $this->hop) {
            // slice of length mNum starting at kStart, zero-padded
            $frame = [];
            for ($i = 0; $i < $this->mNum; $i++) {
                $idx = $kStart + $i;
                $xv = ($idx >= 0 && $idx < $n) ? $x[$idx] : 0.0;
                $frame[$i] = $xv * $this->win[$i]; // win is real -> conj == win
            }
            $spec = $this->fftFunc($frame);
            for ($f = 0; $f < $fPts; $f++) {
                $S[$f][$col] = $spec[$f];
            }
            $col++;
        }

        return $S;
    }

    /**
     * _fft_func over a real frame -> list of [re, im] of length f_pts.
     *
     * @param float[] $frame length m_num
     * @return array<int, array{0: float, 1: float}>
     */
    private function fftFunc(array $frame): array
    {
        // zero-pad to mfft if needed, then roll by -(phase_shift + m_num_mid) % m_num
        if ($this->phaseShift !== null) {
            if (count($frame) < $this->mfft) {
                $frame = array_merge($frame, array_fill(0, $this->mfft - count($frame), 0.0));
            }
            $ps = (($this->phaseShift + $this->mNumMid) % $this->mNum + $this->mNum) % $this->mNum;
            // np.roll(frame, -ps): element i moves to (i - ps) mod len
            $len = count($frame);
            $rolled = array_fill(0, $len, 0.0);
            for ($i = 0; $i < $len; $i++) {
                $rolled[(($i - $ps) % $len + $len) % $len] = $frame[$i];
            }
            $frame = $rolled;
        }

        if ($this->fftMode === 'onesided') {
            $r = Fft::rfft($frame, $this->mfft);
        } else { // twosided
            $r = Fft::fft($frame, $this->mfft);
        }

        return self::complexList($r);
    }

    /**
     * Inverse STFT reconstructing the real signal (scipy ShortTimeFFT.istft).
     *
     * @param array<int, array<int, array{0: float, 1: float}>> $S frequency x time grid
     * @return float[] reconstructed signal of length k1 (defaults to full extent)
     */
    public function istft(array $S, int $k0 = 0, ?int $k1 = null): array
    {
        $fPts = $this->fPts();
        if (count($S) !== $fPts) {
            throw new \InvalidArgumentException('ShortTimeFFT::istft: S must have f_pts rows.');
        }
        $nSlices = count($S[0]);
        $nMin = $this->mNum - $this->mNumMid;
        $pMin = $this->pMin();

        $qMax = $nSlices + $pMin;
        $kMax = ($qMax - 1) * $this->hop + $this->mNum - $this->mNumMid;
        $k1 = $k1 ?? $kMax;

        if (! ($this->kMin() <= $k0 && $k0 < $k1 && $k1 <= $kMax)) {
            throw new \InvalidArgumentException('ShortTimeFFT::istft: k0/k1 out of bounds.');
        }

        $q0 = $k0 >= 0 ? intdiv($k0, $this->hop) + $pMin : self::floordiv($k0, $this->hop);
        $q1 = min($this->pMax($k1), $qMax);

        $kq0 = $this->nearestKP($k0, true);
        $kq1 = $this->nearestKP($k1, false);
        $nPts = $kq1 - $kq0 + $this->mNum - $this->mNumMid;

        $dual = $this->dualWin();
        $x = array_fill(0, $nPts, 0.0);

        for ($q = $q0; $q < $q1; $q++) {
            $col = $q - $pMin;
            $spec = [];
            for ($f = 0; $f < $fPts; $f++) {
                $spec[$f] = $S[$f][$col];
            }
            $xs = $this->ifftFunc($spec); // length m_num, real
            for ($i = 0; $i < $this->mNum; $i++) {
                $xs[$i] *= $dual[$i];
            }
            $i0 = $q * $this->hop - $this->mNumMid;
            $i1 = min($i0 + $this->mNum, $nPts + $k0);
            $j0 = 0;
            $j1 = $i1 - $i0;
            if ($i0 < $k0) {
                $j0 += $k0 - $i0;
                $i0 = $k0;
            }
            for ($j = $j0, $ix = $i0 - $k0; $j < $j1; $j++, $ix++) {
                if ($ix >= 0 && $ix < $nPts) {
                    $x[$ix] += $xs[$j];
                }
            }
        }

        return array_slice($x, 0, $k1 - $k0);
    }

    /** _ifft_func: inverse of fftFunc, returns the real m_num-length frame. */
    private function ifftFunc(array $spec): array
    {
        if ($this->fftMode === 'onesided') {
            $r = Fft::irfft(self::toComplexInput($spec), $this->mfft);
            $frame = self::realList($r);
        } else {
            $r = Fft::ifft(self::toComplexInput($spec), $this->mfft);
            $frame = self::realList($r);
        }
        // undo phase roll: np.roll(frame, +ps)
        if ($this->phaseShift !== null) {
            $ps = (($this->phaseShift + $this->mNumMid) % $this->mNum + $this->mNum) % $this->mNum;
            $len = count($frame);
            $rolled = array_fill(0, $len, 0.0);
            for ($i = 0; $i < $len; $i++) {
                $rolled[(($i + $ps) % $len + $len) % $len] = $frame[$i];
            }
            $frame = $rolled;
        }

        return array_slice($frame, 0, $this->mNum);
    }

    /**
     * Spectrogram = |stft|^2 (scipy ShortTimeFFT.spectrogram, default).
     *
     * @param float[] $x
     * @return float[][] frequency x time magnitude-squared
     */
    public function spectrogram(array $x): array
    {
        $S = $this->stft($x);
        $out = [];
        foreach ($S as $f => $row) {
            $out[$f] = [];
            foreach ($row as $t => $c) {
                $out[$f][$t] = $c[0] * $c[0] + $c[1] * $c[1];
            }
        }

        return $out;
    }

    /** Canonical dual window (scipy _calc_dual_canonical_window). */
    public function dualWin(): array
    {
        if ($this->dualWin !== null) {
            return $this->dualWin;
        }
        if ($this->hop > $this->mNum) {
            throw new \RuntimeException('ShortTimeFFT: hop larger than window length => not invertible.');
        }
        $w2 = $this->winSq();
        $DD = $w2;
        for ($k = $this->hop; $k < $this->mNum; $k += $this->hop) {
            // DD[k:] += w2[:-k]
            for ($i = $k; $i < $this->mNum; $i++) {
                $DD[$i] += $w2[$i - $k];
            }
            // DD[:-k] += w2[k:]
            for ($i = 0; $i < $this->mNum - $k; $i++) {
                $DD[$i] += $w2[$i + $k];
            }
        }
        $maxDD = max($DD);
        $rel = 2.220446049250313e-16 * $maxDD; // np.finfo(float64).resolution ~1e-15; use eps*max as DD>0 gate
        $resolution = 1.0e-15 * $maxDD;
        foreach ($DD as $d) {
            if ($d < $resolution) {
                throw new \RuntimeException('ShortTimeFFT: STFT not invertible (dual window singular).');
            }
        }
        $dual = [];
        for ($i = 0; $i < $this->mNum; $i++) {
            $dual[$i] = $this->win[$i] / $DD[$i];
        }
        $this->dualWin = $dual;

        return $dual;
    }

    // ---- helpers ----

    private function winSq(): array
    {
        $w2 = [];
        foreach ($this->win as $w) {
            $w2[] = $w * $w;
        }

        return $w2;
    }

    private function nearestKP(int $k, bool $left = true): int
    {
        // scipy.nearest_k_p: nearest multiple of hop (toward left/right)
        $n = $k - ($k % $this->hop);
        if ($n === $k) {
            return $k;
        }

        return $left ? $n : $n + $this->hop;
    }

    private static function floordiv(int $a, int $b): int
    {
        return (int) floor($a / $b);
    }

    /** NDArray complex -> list of [re,im]. */
    private static function complexList(mixed $r): array
    {
        $arr = is_object($r) && method_exists($r, 'toArray') ? $r->toArray() : $r;
        $out = [];
        foreach ($arr as $v) {
            if (is_array($v)) {
                $out[] = [(float) $v[0], (float) ($v[1] ?? 0.0)];
            } else {
                $out[] = [(float) $v, 0.0];
            }
        }

        return $out;
    }

    /** NDArray real -> flat float list. */
    private static function realList(mixed $r): array
    {
        $arr = is_object($r) && method_exists($r, 'toArray') ? $r->toArray() : $r;
        $out = [];
        foreach ($arr as $v) {
            $out[] = is_array($v) ? (float) $v[0] : (float) $v;
        }

        return $out;
    }

    /** list of [re,im] -> a Complex128 NDArray the FFT accepts. */
    private static function toComplexInput(array $spec): \Tessero\NDArray
    {
        $re = [];
        $im = [];
        foreach ($spec as $c) {
            $re[] = $c[0];
            $im[] = $c[1];
        }

        return \Tessero\NDArray::complex($re, $im);
    }
}
