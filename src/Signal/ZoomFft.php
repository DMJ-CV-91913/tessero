<?php

declare(strict_types=1);

namespace Tessero\Signal;

use Tessero\Signal;

/**
 * scipy.signal.ZoomFFT - callable zoom FFT object.
 *
 * `new ZoomFft($n, $fn, $m, $fs, $endpoint)` builds a transform configured for length-$n inputs that
 * evaluates the DFT only over the frequency band $fn; invoking the object via {@see ZoomFft::transform()}
 * (or as a callable) produces exactly the same result as scipy.signal.zoom_fft(x, fn, m, fs=fs,
 * endpoint=endpoint).
 *
 * This is a thin deterministic container over the already-landed `Tessero\Signal::zoomFft` primitive
 * (ZoomFFT is a special case of the chirp z-transform). Holding the parameters lets the same transform
 * be reused across many inputs; results are bit-identical to zoom_fft.
 *
 * @see \Tessero\Signal::zoomFft
 */
final class ZoomFft
{
    private int $n;

    private mixed $fn;

    private mixed $m;

    private mixed $fs;

    private bool $endpoint;

    /**
     * @param int    $n        The size of the signal the transform will operate on.
     * @param mixed  $fn       A length-2 sequence [f1, f2] giving the frequency range, or a scalar
     *                         for which the range [0, fn] is assumed.
     * @param ?int   $m        The number of points to evaluate. Defaults to $n.
     * @param mixed  $fs       The sampling frequency. If fs=10 represents 10 kHz, for example, then
     *                         the frequency band fn is in kHz. Defaults to 2 (Nyquist = 1).
     * @param bool   $endpoint If true, $fn[1] is the last sample; otherwise it is not included.
     */
    public function __construct(int $n, mixed $fn, mixed $m = null, mixed $fs = 2, bool $endpoint = false)
    {
        $this->n = $n;
        $this->fn = $fn;
        $this->m = $m;
        $this->fs = $fs;
        $this->endpoint = $endpoint;
    }

    /**
     * Apply the zoom FFT to $x (equivalent to scipy.signal.zoom_fft(x, fn, m, fs, endpoint)).
     */
    public function transform(mixed $x): mixed
    {
        return Signal::zoomFft($x, $this->fn, $this->m, $this->fs, $this->endpoint);
    }

    /**
     * Allow the object to be invoked directly as a callable: $zoomFft($x).
     */
    public function __invoke(mixed $x): mixed
    {
        return $this->transform($x);
    }
}
