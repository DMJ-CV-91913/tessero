<?php

/**
 * Save and load arrays in NumPy's .npy and .npz formats (interchangeable with NumPy).
 * NumPy: np.save / np.load; np.savez / np.load
 *
 * FFI-only today (the extension has no .npz yet; see TSR-107), so this skips under ext.
 */

declare(strict_types=1);

use Tessero\Io\Npy;

ffi_only('.npy / .npz I/O');

$arr = ND::array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

// .npy single array round-trip.
$npy = tempnam(sys_get_temp_dir(), 'tsr') . '.npy';
Npy::save($npy, $arr);
check_close(Npy::load($npy)->toList(), $arr->toList(), '.npy round-trip', 0.0, 0.0);
@unlink($npy);

// .npz dictionary of named arrays.
$npz = tempnam(sys_get_temp_dir(), 'tsr') . '.npz';
Npy::saveZ($npz, ['weights' => $arr, 'index' => ND::arange(5)]);
$loaded = Npy::loadZ($npz);
check_close($loaded['weights']->toList(), $arr->toList(), '.npz weights', 0.0, 0.0);
check_close($loaded['index']->toList(), [0, 1, 2, 3, 4], '.npz index', 0.0, 0.0);
@unlink($npz);

say('.npy and .npz round-trips ok on ' . backend());
