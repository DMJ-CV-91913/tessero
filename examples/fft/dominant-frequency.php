<?php

/**
 * Find the dominant frequency of a sampled signal with the real FFT. Both backends.
 * NumPy: F = np.fft.rfft(x); freqs = np.fft.rfftfreq(n, d=1/fs); freqs[np.argmax(np.abs(F))]
 *
 * Scenario: a 10 Hz tone sampled at 128 Hz. The real FFT's largest-magnitude bin, mapped
 * through rfftfreq, recovers the frequency - the basis of seasonality/period detection.
 */

declare(strict_types=1);

$fs = 128.0;                                           // sample rate (Hz)
$n = 256;                                              // samples; bin width = fs/n = 0.5 Hz
$f0 = 10.0;                                            // true frequency

$k = ND::arange($n);
$signal = M::sin(M::multiply($k, 2.0 * M_PI * $f0 / $fs));   // sin(2*pi*f0*k/fs)

$spectrum = FFT::rfft($signal);
$magnitude = M::absolute($spectrum);                   // |F| per bin
$peak = $magnitude->argmax();                          // loudest bin index
$freqs = FFT::rfftfreq($n, 1.0 / $fs);                 // Hz for each bin

$peakHz = (float) $freqs->toList()[$peak];
say('dominant frequency = ' . $peakHz . ' Hz  on ' . backend());
check(abs($peakHz - $f0) < 0.6, 'recovered ~10 Hz (within one 0.5 Hz bin)');
