<?php

declare(strict_types=1);

namespace Tessero;

/**
 * scipy.ndimage.labeled_comprehension, pure PHP (one class serves both backends, no kernel call).
 *
 * Roughly equivalent to `[func(input[labels == i]) for i in index]`: the values of `input` sharing
 * each requested label are gathered (in C / row-major order, matching numpy's boolean-mask selection)
 * and handed to the user callable, which returns one scalar per index.
 *
 *  - index is a scalar  -> a single scalar result is returned.
 *  - index is a list    -> a flat list of results, one per index, is returned.
 *  - index is null      -> func is applied once to the set of ALL values whose label is non-zero,
 *                          returning a single scalar (labels must then be non-null).
 *  - labels is null      -> func is applied once to the whole raveled input (index is ignored as a group
 *                          selector; a scalar is returned).
 *  - default             -> substituted for any index not present in labels.
 *  - pass_positions      -> when true, the callable receives a second argument: the flat C-order linear
 *                          positions of the selected elements (same order as the values).
 *
 * Input is an N-D array given as PHP nested numeric lists; labels is a same-shaped nested integer list
 * (or null). Values are compared bit-for-bit to scipy for the arithmetic the callable performs.
 */
final class LabeledComprehension
{
    private function __construct() {}

    /**
     * @param array           $input          N-D nested numeric array
     * @param array|null      $labels         same-shaped nested integer array, or null for whole-input
     * @param int|array|null  $index          scalar / list of label ids / null (all non-zero labels)
     * @param callable        $func           fn(array $values [, array $positions]): float
     * @param float|int|null  $default        value returned for an index absent from labels
     * @param bool            $passPositions  pass flat linear positions as a 2nd argument to $func
     *
     * @return float|array scalar for scalar/null index, else a flat list aligned with $index
     */
    public static function labeledComprehension(
        array $input,
        ?array $labels,
        int|array|null $index,
        callable $func,
        float|int|null $default = null,
        bool $passPositions = false,
    ): float|array {
        $values = self::flatten($input);                        // C-order values
        $n = count($values);

        // labels == null: func over the whole raveled input, single scalar.
        if ($labels === null) {
            if ($passPositions) {
                return (float) $func($values, range(0, $n - 1));
            }
            return (float) $func($values);
        }

        $lab = self::flattenInt($labels);
        if (count($lab) !== $n) {
            throw new \ValueError('labeled_comprehension: labels must have the same shape as input.');
        }

        // index == null: single group of every non-zero-labelled value.
        if ($index === null) {
            $vals = [];
            $pos = [];
            for ($i = 0; $i < $n; $i++) {
                if ($lab[$i] !== 0) {
                    $vals[] = $values[$i];
                    $pos[] = $i;
                }
            }
            return self::callOne($func, $vals, $pos, $passPositions, $default);
        }

        // Group C-order values (and positions) by label id in one pass.
        $byLabel = [];                                          // id => [values...]
        $posByLabel = [];                                       // id => [positions...]
        for ($i = 0; $i < $n; $i++) {
            $id = $lab[$i];
            $byLabel[$id][] = $values[$i];
            if ($passPositions) {
                $posByLabel[$id][] = $i;
            }
        }

        $scalar = ! is_array($index);
        $ids = $scalar ? [(int) $index] : array_map('intval', array_values($index));

        $out = [];
        foreach ($ids as $id) {
            if (! array_key_exists($id, $byLabel)) {
                $out[] = $default === null ? 0.0 : (float) $default;
                continue;
            }
            $out[] = self::callOne($func, $byLabel[$id], $posByLabel[$id] ?? [], $passPositions, $default);
        }

        return $scalar ? $out[0] : $out;
    }

    /** Invoke the callable on one group, optionally passing positions; empty group -> default. */
    private static function callOne(callable $func, array $vals, array $pos, bool $passPositions, float|int|null $default): float
    {
        if (count($vals) === 0) {
            return $default === null ? 0.0 : (float) $default;
        }
        return (float) ($passPositions ? $func($vals, $pos) : $func($vals));
    }

    /** C-order flatten of a nested list to a flat float list. */
    private static function flatten(array $a): array
    {
        $out = [];
        $walk = static function ($v) use (&$walk, &$out): void {
            if (is_array($v)) {
                foreach ($v as $e) {
                    $walk($e);
                }
            } else {
                $out[] = (float) $v;
            }
        };
        $walk($a);
        return $out;
    }

    /** C-order flatten of a nested list to a flat int list. */
    private static function flattenInt(array $a): array
    {
        $out = [];
        $walk = static function ($v) use (&$walk, &$out): void {
            if (is_array($v)) {
                foreach ($v as $e) {
                    $walk($e);
                }
            } else {
                $out[] = (int) $v;
            }
        };
        $walk($a);
        return $out;
    }
}
