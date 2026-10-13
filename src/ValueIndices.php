<?php

declare(strict_types=1);

namespace Tessero;

use Tessero\Exceptions\TesseroException;

/**
 * scipy.ndimage.value_indices: the index arrays of every distinct value in an integer array.
 *
 * Hand-written pure-PHP carve-out (like {@see OrthoPoly} / {@see StatsFns}), NOT the generated Ndimage
 * facade. For each distinct integer value v (in sorted-ascending order), it returns the result of
 * Np::nonzero(input == v) -- a tuple (one index array per input dimension, C order) locating every cell
 * holding v. The optional ignore_value is skipped. Integer input only, exactly as SciPy requires.
 *
 * It is expressed as $input->eq($v)->nonzero() -- the NDArray-level form of Np::nonzero(input==v) -- so the
 * mask, the scan and the int64 index arrays all stay on the input's own backend (FFI or ext); one class
 * serves both backends with no kernel-specific code and produces results bit-identical to SciPy (a pure
 * selection: no arithmetic on the data).
 */
final class ValueIndices
{
    /**
     * scipy.ndimage.value_indices(arr, ignore_value=None).
     *
     * @param NDArray|\Tessero\Ext\NDArray $input       an integer array (any rank >= 1); typed `mixed` so the
     *                                                  one carve-out serves both backends' (unrelated) NDArray
     *                                                  classes, exactly as the generated Ndimage facade does
     * @param int|null                     $ignoreValue a value to leave out of the result (SciPy's ignore_value)
     *
     * @return array<int, list<object>> distinct value => tuple of per-dimension int64 index arrays, keys ascending
     */
    public static function valueIndices(mixed $input, ?int $ignoreValue = null): array
    {
        if (! is_object($input) || ! method_exists($input, 'dtype') || ! method_exists($input, 'eq')) {
            throw new TesseroException('value_indices: arr must be a Tessero NDArray.');
        }
        $dtype = $input->dtype();
        $name = strtolower(is_string($dtype) ? $dtype : $dtype->name);
        if (! (str_starts_with($name, 'int') || str_starts_with($name, 'uint'))) {
            // SciPy: "Parameter 'arr' must be an integer array"
            throw new TesseroException("value_indices: arr must be an integer array, got {$name}.");
        }

        // Distinct integer values actually present, ascending. toList() is C-order; recurse to flatten.
        $present = [];
        $collect = static function ($node) use (&$collect, &$present): void {
            if (is_array($node)) {
                foreach ($node as $child) {
                    $collect($child);
                }
                return;
            }
            $present[(int) $node] = true;
        };
        $collect($input->toList());
        $values = array_keys($present);
        sort($values, SORT_NUMERIC);

        $out = [];
        foreach ($values as $v) {
            if ($ignoreValue !== null && $v === $ignoreValue) {
                continue;
            }
            // Np::nonzero(input == v): a tuple of one int64 index array per dimension.
            $idx = $input->eq($v)->nonzero();
            $out[$v] = array_values($idx);
        }

        return $out;
    }
}
