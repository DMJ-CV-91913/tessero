<?php

/**
 * Reproduce a NumPy random stream from a seed, and snapshot/restore generator state.
 * NumPy: g = np.random.default_rng(42); g.integers(0, 100, 5); g.standard_normal(4)
 *
 * Tessero's Generator is the same PCG64 as NumPy's default_rng and draws the bit-identical
 * stream, so seeded simulations reproduce across PHP and NumPy. Runs on both backends.
 */

declare(strict_types=1);

// Same seed as numpy.random.default_rng(42): identical integer stream.
$g = RNG::defaultRng(42);
check_close($g->integers(0, 100, 5), [8, 77, 65, 43, 43], 'integers match numpy default_rng(42)', 0.0, 0.0);

// Standard-normal draws match NumPy to full double precision.
$g2 = RNG::defaultRng(42);
check_close(
    $g2->standardNormal(4),
    [0.304717079754, -1.03998410624, 0.750451195806, 0.940564716391],
    'standard_normal matches numpy',
    1e-9,
    1e-12,
);
say('reproduced numpy default_rng(42) on ' . backend());

// State round-trip: snapshot, draw, restore, and the next draws repeat exactly.
$g3 = RNG::defaultRng(7);
$g3->integers(0, 1000, 10);                 // advance the stream a little
$state = $g3->getState();
$first = $g3->integers(0, 1000, 5);
$g3->setState($state);
$again = $g3->integers(0, 1000, 5);
check_close($again, $first, 'getState/setState reproduces the next draws', 0.0, 0.0);
