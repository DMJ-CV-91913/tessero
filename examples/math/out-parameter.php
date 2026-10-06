<?php

/**
 * Write ufunc results into a preallocated array with the out= parameter (no new allocation).
 * NumPy: np.sqrt(x, out=buf); np.multiply(x, 2, out=x)  (in place)
 *
 * Useful in hot loops / queued jobs where you reuse one buffer instead of allocating each call.
 * out is passed positionally here (works identically on both backends).
 */

declare(strict_types=1);

$x = ND::array([1.0, 4.0, 9.0, 16.0, 25.0]);
$buf = ND::zeros([5]);

M::sqrt($x, $buf);                                    // fill buf; returns buf too
check_close($buf->toList(), [1.0, 2.0, 3.0, 4.0, 5.0], 'sqrt into out buffer', 1e-12, 1e-12);

M::multiply($x, 2.0, $x);                             // double x in place
check_close($x->toList(), [2.0, 8.0, 18.0, 32.0, 50.0], 'in-place double', 1e-12, 1e-12);

say('out= (fill + in place) ok on ' . backend());
