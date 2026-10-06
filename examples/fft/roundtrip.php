<?php

/**
 * The inverse FFT undoes the forward FFT. Both backends.
 * NumPy: np.fft.ifft(np.fft.fft(x)).real  ==  x
 */

declare(strict_types=1);

$x = ND::array([1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0]);

$restored = FFT::ifft(FFT::fft($x));          // complex, with imaginary part ~ 0
$real = $restored->real();                    // take the real part back

check_close($real, $x, 'ifft(fft(x)).real == x', 1e-9, 1e-9);
say('FFT round-trip exact to 1e-9 on ' . backend());
