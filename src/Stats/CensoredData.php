<?php

declare(strict_types=1);

namespace Tessero\Stats;

/**
 * scipy.stats.CensoredData - deterministic container for censored observations.
 *
 * An observation is *censored* when its precise value is unknown but a bound is known:
 *   - left-censored:     below a known value (unknown by how much)
 *   - right-censored:    above a known value
 *   - interval-censored: somewhere within a known finite interval
 *
 * Construct directly with uncensored / left / right / interval arrays, or via the convenience
 * factories leftCensored(), rightCensored(), intervalCensored().
 *
 * Mirrors scipy.stats.CensoredData, including the canonicalisation performed on `interval` input:
 * a row [a, a] becomes an uncensored value a; a row [-inf, b] becomes a left-censored value b;
 * a row [a, inf] becomes a right-censored value a; only genuinely finite non-degenerate rows remain
 * interval-censored. Interval-derived values are appended *after* the directly supplied ones, matching
 * scipy's concatenation order.
 *
 * Public accessors expose the raw partitioned data:
 *   - uncensored() / left() / right() / interval()
 *   - len()            total number of values
 *   - numCensored()    number of censored values (left + right + interval)
 *
 * @see scipy.stats._censored_data.CensoredData
 */
final class CensoredData
{
    /** @var array<int,float> */
    private array $uncensored;

    /** @var array<int,float> */
    private array $left;

    /** @var array<int,float> */
    private array $right;

    /** @var array<int,array{0:float,1:float}> */
    private array $interval;

    /**
     * @param array<int,float>|mixed|null              $uncensored uncensored observations (1-D)
     * @param array<int,float>|mixed|null              $left       left-censored observations (1-D)
     * @param array<int,float>|mixed|null              $right      right-censored observations (1-D)
     * @param array<int,array{0:float,1:float}>|mixed|null $interval interval-censored observations, shape (m, 2)
     */
    public function __construct(mixed $uncensored = null, mixed $left = null, mixed $right = null, mixed $interval = null)
    {
        $uncensored = self::vec($uncensored);
        $left = self::vec($left);
        $right = self::vec($right);

        self::validate1d($uncensored, 'uncensored', false);
        self::validate1d($left, 'left', false);
        self::validate1d($right, 'right', false);

        [$unc2, $left2, $right2, $iv2] = self::validateInterval($interval);

        // scipy: concatenate((direct, interval-derived))
        $this->uncensored = array_merge($uncensored, $unc2);
        $this->left = array_merge($left, $left2);
        $this->right = array_merge($right, $right2);
        $this->interval = $iv2;
    }

    /**
     * Create a CensoredData instance of right-censored data from measurements and a boolean mask.
     *
     * @param array<int,float>|mixed $x        observed measurements (finite)
     * @param array<int,bool>|mixed  $censored true where the corresponding x is right-censored
     */
    public static function rightCensored(mixed $x, mixed $censored): self
    {
        [$x, $c] = self::validateXCensored($x, $censored);
        $unc = [];
        $rc = [];
        foreach ($x as $i => $v) {
            if ($c[$i]) {
                $rc[] = $v;
            } else {
                $unc[] = $v;
            }
        }
        return new self($unc, null, $rc, null);
    }

    /**
     * Create a CensoredData instance of left-censored data from measurements and a boolean mask.
     *
     * @param array<int,float>|mixed $x        observed measurements (finite)
     * @param array<int,bool>|mixed  $censored true where the corresponding x is left-censored
     */
    public static function leftCensored(mixed $x, mixed $censored): self
    {
        [$x, $c] = self::validateXCensored($x, $censored);
        $unc = [];
        $lc = [];
        foreach ($x as $i => $v) {
            if ($c[$i]) {
                $lc[] = $v;
            } else {
                $unc[] = $v;
            }
        }
        return new self($unc, $lc, null, null);
    }

    /**
     * Create a CensoredData instance of interval-censored data from low/high bound arrays.
     *
     * @param array<int,float>|mixed $low  low ends of the intervals (may be -inf)
     * @param array<int,float>|mixed $high high ends of the intervals (may be +inf)
     */
    public static function intervalCensored(mixed $low, mixed $high): self
    {
        $low = self::vec($low);
        $high = self::vec($high);
        self::validate1d($low, 'low', true);
        self::validate1d($high, 'high', true);
        if (count($low) !== count($high)) {
            throw new \InvalidArgumentException('`low` and `high` must have the same length.');
        }
        $iv = [];
        foreach ($low as $i => $lo) {
            $iv[] = [(float) $lo, (float) $high[$i]];
        }
        return new self(null, null, null, $iv);
    }

    /** Uncensored observations. @return array<int,float> */
    public function uncensored(): array
    {
        return $this->uncensored;
    }

    /** Left-censored observations. @return array<int,float> */
    public function left(): array
    {
        return $this->left;
    }

    /** Right-censored observations. @return array<int,float> */
    public function right(): array
    {
        return $this->right;
    }

    /** Interval-censored observations (finite, non-degenerate rows only). @return array<int,array{0:float,1:float}> */
    public function interval(): array
    {
        return $this->interval;
    }

    /** Total number of values (censored and not censored). */
    public function len(): int
    {
        return count($this->uncensored) + count($this->left) + count($this->right) + count($this->interval);
    }

    /** Number of censored values. */
    public function numCensored(): int
    {
        return count($this->left) + count($this->right) + count($this->interval);
    }

    /** Human-readable summary, matching scipy's str(CensoredData). */
    public function __toString(): string
    {
        $numNc = count($this->uncensored);
        $numLc = count($this->left);
        $numRc = count($this->right);
        $numIc = count($this->interval);
        $n = $numNc + $numLc + $numRc + $numIc;
        $parts = ["{$numNc} not censored"];
        if ($numLc > 0) {
            $parts[] = "{$numLc} left-censored";
        }
        if ($numRc > 0) {
            $parts[] = "{$numRc} right-censored";
        }
        if ($numIc > 0) {
            $parts[] = "{$numIc} interval-censored";
        }
        return "CensoredData({$n} values: " . implode(', ', $parts) . ')';
    }

    // ---- validation / canonicalisation --------------------------------------------------------

    /**
     * Partition an interval array into (uncensored, left, right, interval) following scipy.
     *
     * @return array{0:array<int,float>,1:array<int,float>,2:array<int,float>,3:array<int,array{0:float,1:float}>}
     */
    private static function validateInterval(mixed $interval): array
    {
        if ($interval === null) {
            return [[], [], [], []];
        }
        if (is_object($interval) && method_exists($interval, 'toArray')) {
            $interval = $interval->toArray();
        }
        $interval = array_values((array) $interval);
        if ($interval === []) {
            return [[], [], [], []];
        }
        foreach ($interval as $row) {
            if (! is_array($row) || count($row) !== 2) {
                throw new \InvalidArgumentException('`interval` must be a two-dimensional array with shape (m, 2)');
            }
        }

        $unc = [];
        $left = [];
        $right = [];
        $iv = [];
        foreach ($interval as $row) {
            $a = (float) $row[0];
            $b = (float) $row[1];
            if (is_nan($a) || is_nan($b)) {
                throw new \InvalidArgumentException('`interval` must not contain nan.');
            }
            $aInf = is_infinite($a);
            $bInf = is_infinite($b);
            if ($aInf && $bInf) {
                throw new \InvalidArgumentException('In each row in `interval`, both values must not be infinite.');
            }
            if ($a > $b) {
                throw new \InvalidArgumentException('In each row of `interval`, the left value must not exceed the right value.');
            }
            if ($a === $b) {
                $unc[] = $a;                 // degenerate interval -> uncensored
            } elseif ($aInf) {
                $left[] = $b;                // [-inf, b] -> left-censored at b
            } elseif ($bInf) {
                $right[] = $a;               // [a, inf] -> right-censored at a
            } else {
                $iv[] = [$a, $b];            // genuine finite interval
            }
        }
        return [$unc, $left, $right, $iv];
    }

    /**
     * @param array<int,float> $a
     */
    private static function validate1d(array $a, string $name, bool $allowInf): void
    {
        foreach ($a as $v) {
            if (is_nan($v)) {
                throw new \InvalidArgumentException("`{$name}` must not contain nan.");
            }
            if (! $allowInf && is_infinite($v)) {
                throw new \InvalidArgumentException("`{$name}` must contain only finite values.");
            }
        }
    }

    /**
     * @return array{0:array<int,float>,1:array<int,bool>}
     */
    private static function validateXCensored(mixed $x, mixed $censored): array
    {
        $x = self::vec($x);
        $c = self::vec($censored, false);
        foreach ($x as $v) {
            if (! is_finite($v)) {
                throw new \InvalidArgumentException('`x` must not contain nan or inf.');
            }
        }
        if (count($c) !== count($x)) {
            throw new \InvalidArgumentException('`x` and `censored` must have the same length.');
        }
        $cb = [];
        foreach ($c as $v) {
            $cb[] = (bool) $v;
        }
        return [$x, $cb];
    }

    /**
     * Coerce an NDArray / array / null into a flat PHP list.
     *
     * @return array<int,mixed>
     */
    private static function vec(mixed $a, bool $asFloat = true): array
    {
        if ($a === null) {
            return [];
        }
        if (is_object($a) && method_exists($a, 'toArray')) {
            $a = $a->toArray();
        }
        $a = array_values((array) $a);
        return $asFloat ? array_map('floatval', $a) : $a;
    }
}
