<?php

/**
 * Create arrays and read elements/rows. Runs on both backends.
 * NumPy: np.arange(12).reshape(3, 4); a[1]; np.zeros((2, 3))
 *
 * ND is Tessero\NDArray on the FFI backend and Tessero\Ext\NDArray on the extension.
 *
 * Note: a *multi-axis* block slice (NumPy a[0:2, 1:3]) is written differently on the two
 * backends today - the FFI package takes one spec per axis, slice('0:2', '1:3'), while the
 * extension takes a single string, slice('0:2, 1:3') - so it is shown in the FFI linear-algebra
 * examples rather than here. Single-axis slicing (below) is identical on both.
 */

declare(strict_types=1);

$a = ND::arange(12)->reshape(3, 4);                 // 0..11 as a 3x4 matrix

say('shape = ' . json_encode($a->shape()) . '  on ' . backend());
check($a->shape() === [3, 4], 'shape is 3x4');
check_close($a->toList(), [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], 'arange + reshape', 0.0, 0.0);

$row = $a->slice('1');                              // NumPy a[1]  ->  [4, 5, 6, 7]
check_close($row->toList(), [4, 5, 6, 7], 'row 1', 0.0, 0.0);

$flat = $a->reshape(12);                            // reshape back to 1-D
check($flat->shape() === [12], 'reshape 3x4 -> 12');

$zeros = ND::zeros([2, 3]);
check_close($zeros->toList(), [[0, 0, 0], [0, 0, 0]], 'zeros (2, 3)', 0.0, 0.0);
