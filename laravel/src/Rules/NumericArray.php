<?php

declare(strict_types=1);

namespace Tessero\Laravel\Rules;

use Closure;
use Illuminate\Contracts\Validation\ValidationRule;

/**
 * Validates request input that will become an NDArray: a (nested) list of
 * numbers with a rectangular shape, optional exact dimensions and a size cap.
 *
 *   $request->validate(['prices' => ['required', new NumericArray(ndim: 1, maxElements: 8784)]]);
 *   $request->validate(['matrix' => ['required', new NumericArray(shape: [null, 3])]]);   // n x 3
 */
final class NumericArray implements ValidationRule
{
    /** @param list<int|null>|null $shape null entries match any length */
    public function __construct(
        private readonly ?int $ndim = null,
        private readonly ?array $shape = null,
        private readonly int $maxElements = 1_000_000,
        private readonly bool $allowBool = false,
    ) {
    }

    public function validate(string $attribute, mixed $value, Closure $fail): void
    {
        $error = self::check($value, $this->ndim, $this->shape, $this->maxElements, $this->allowBool);
        if ($error !== null) {
            $fail("The :attribute {$error}");
        }
    }

    /** @return string|null the problem, or null when valid (usable outside Laravel) */
    public static function check(mixed $value, ?int $ndim = null, ?array $shape = null, int $maxElements = 1_000_000, bool $allowBool = false): ?string
    {
        if (! is_array($value)) {
            return 'must be an array of numbers.';
        }
        $dims = [];
        $probe = $value;
        while (is_array($probe)) {
            if (! array_is_list($probe)) {
                return 'must use list arrays (no string keys).';
            }
            $dims[] = count($probe);
            if ($probe === []) {
                break;
            }
            $probe = $probe[0];
        }
        if ($ndim !== null && count($dims) !== $ndim) {
            return "must have {$ndim} dimension(s), got " . count($dims) . '.';
        }
        if ($shape !== null) {
            if (count($shape) !== count($dims)) {
                return 'must have ' . count($shape) . ' dimension(s).';
            }
            foreach ($shape as $i => $len) {
                if ($len !== null && $dims[$i] !== $len) {
                    return "must have length {$len} along axis {$i}, got {$dims[$i]}.";
                }
            }
        }
        $total = (int) array_product($dims);
        if ($total > $maxElements) {
            return "has {$total} elements; the limit is {$maxElements}.";
        }
        $stack = [[$value, 0]];
        while ($stack !== []) {
            [$node, $depth] = array_pop($stack);
            if ($depth === count($dims)) {
                if (! (is_int($node) || is_float($node) || ($allowBool && is_bool($node)))) {
                    return 'must contain only numbers.';
                }
                if (is_float($node) && ! is_finite($node)) {
                    return 'must contain only finite numbers.';
                }
                continue;
            }
            if (! is_array($node) || count($node) !== $dims[$depth]) {
                return 'must be rectangular (every row the same length).';
            }
            foreach ($node as $child) {
                $stack[] = [$child, $depth + 1];
            }
        }

        return null;
    }
}
