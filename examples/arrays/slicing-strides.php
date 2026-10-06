<?php

/**
 * Strided and negative slicing (NumPy's start:stop:step on a 1-D array). Runs on both backends.
 * NumPy: v[::2]; v[::-1]; v[-3:]
 */

declare(strict_types=1);

$v = ND::arange(10);                        // [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

$evens = $v->slice('::2');                  // every other element
check_close($evens->toList(), [0, 2, 4, 6, 8], 'v[::2]', 0.0, 0.0);

$reversed = $v->slice('::-1');              // reversed view
check_close($reversed->toList(), [9, 8, 7, 6, 5, 4, 3, 2, 1, 0], 'v[::-1]', 0.0, 0.0);

$lastThree = $v->slice('-3:');             // tail
check_close($lastThree->toList(), [7, 8, 9], 'v[-3:]', 0.0, 0.0);

say('strided slicing ok on ' . backend());
