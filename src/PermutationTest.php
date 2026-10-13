<?php

declare(strict_types=1);

namespace Tessero;

/**
 * scipy.stats.permutation_test exact (full-enumeration) path, pure PHP.
 * Reproduces scipy.stats._resampling for exact_test=True (adjustment=0):
 *   independent -> all partitions concatenated (scipy _all_partitions_concatenated)
 *   pairings    -> Cartesian product of per-sample index permutations
 *   samples     -> all 2^n sign flips of one paired sample
 * p-value (gamma = |eps * observed|, eps = numpy float64 eps):
 *   less=mean(null<=obs+gamma) greater=mean(null>=obs-gamma) two-sided=min(less,greater)*2 clipped.
 */
final class PermutationTest
{
    private const EPS = 2.220446049250313e-16;

    public static function permutationTest(
        array $data,
        callable $statistic,
        string $permutationType = 'independent',
        string $alternative = 'two-sided'
    ): array {
        $data = array_map(static fn ($x) => array_map('floatval', $x), $data);
        $observed = (float) $statistic(...$data);
        if ($permutationType === 'independent') {
            $null = self::nullIndependent($data, $statistic);
        } elseif ($permutationType === 'pairings') {
            $null = self::nullPairings($data, $statistic);
        } elseif ($permutationType === 'samples') {
            $null = self::nullSamples($data, $statistic);
        } else {
            throw new \InvalidArgumentException('unknown permutation_type');
        }
        return ['statistic' => $observed, 'pvalue' => self::pvalue($null, $observed, $alternative)];
    }

    private static function pvalue(array $null, float $observed, string $alternative): float
    {
        $n = count($null);
        $gamma = abs(self::EPS * $observed);
        $cl = 0;
        $cg = 0;
        foreach ($null as $v) {
            if ($v <= $observed + $gamma) {
                $cl++;
            }
            if ($v >= $observed - $gamma) {
                $cg++;
            }
        }
        $less = $cl / $n;
        $greater = $cg / $n;
        if ($alternative === 'less') {
            $p = $less;
        } elseif ($alternative === 'greater') {
            $p = $greater;
        } elseif ($alternative === 'two-sided') {
            $p = min($less, $greater) * 2.0;
        } else {
            throw new \InvalidArgumentException('unknown alternative');
        }
        return max(0.0, min(1.0, $p));
    }

    private static function nullIndependent(array $data, callable $statistic): array
    {
        $sizes = array_map('count', $data);
        $pool = [];
        foreach ($data as $sample) {
            foreach ($sample as $v) {
                $pool[] = (float) $v;
            }
        }
        $null = [];
        $indices = range(0, count($pool) - 1);
        self::partition($pool, $indices, $sizes, [], $statistic, $null);
        return $null;
    }

    private static function partition(array $pool, array $remaining, array $sizes, array $groups, callable $statistic, array &$null): void
    {
        if (count($sizes) === 0) {
            $null[] = (float) $statistic(...$groups);
            return;
        }
        $k = $sizes[0];
        $restSizes = array_slice($sizes, 1);
        foreach (self::combinations($remaining, $k) as [$chosen, $rest]) {
            $group = [];
            foreach ($chosen as $idx) {
                $group[] = $pool[$idx];
            }
            self::partition($pool, $rest, $restSizes, [...$groups, $group], $statistic, $null);
        }
    }

    private static function combinations(array $items, int $k): array
    {
        $n = count($items);
        $out = [];
        if ($k > $n) {
            return $out;
        }
        if ($k === 0) {
            return [[[], array_values($items)]];
        }
        $idx = range(0, $k - 1);
        while (true) {
            $chosen = [];
            $chosenPos = [];
            foreach ($idx as $i) {
                $chosen[] = $items[$i];
                $chosenPos[$i] = true;
            }
            $rest = [];
            for ($i = 0; $i < $n; $i++) {
                if (! isset($chosenPos[$i])) {
                    $rest[] = $items[$i];
                }
            }
            $out[] = [$chosen, $rest];
            $i = $k - 1;
            while ($i >= 0 && $idx[$i] === $i + $n - $k) {
                $i--;
            }
            if ($i < 0) {
                break;
            }
            $idx[$i]++;
            for ($j = $i + 1; $j < $k; $j++) {
                $idx[$j] = $idx[$j - 1] + 1;
            }
        }
        return $out;
    }

    private static function nullPairings(array $data, callable $statistic): array
    {
        $perSample = [];
        foreach ($data as $sample) {
            $perSample[] = self::permutations(range(0, count($sample) - 1));
        }
        $null = [];
        self::cartesian($data, $perSample, 0, [], $statistic, $null);
        return $null;
    }

    private static function cartesian(array $data, array $perSample, int $depth, array $acc, callable $statistic, array &$null): void
    {
        if ($depth === count($data)) {
            $null[] = (float) $statistic(...$acc);
            return;
        }
        foreach ($perSample[$depth] as $perm) {
            $permuted = [];
            foreach ($perm as $idx) {
                $permuted[] = $data[$depth][$idx];
            }
            self::cartesian($data, $perSample, $depth + 1, [...$acc, $permuted], $statistic, $null);
        }
    }

    private static function nullSamples(array $data, callable $statistic): array
    {
        $sample = $data[0];
        $n = count($sample);
        $total = 1 << $n;
        $null = [];
        for ($mask = 0; $mask < $total; $mask++) {
            $flipped = [];
            for ($i = 0; $i < $n; $i++) {
                $flipped[] = (($mask >> $i) & 1) ? -$sample[$i] : $sample[$i];
            }
            $null[] = (float) $statistic($flipped);
        }
        return $null;
    }

    private static function permutations(array $items): array
    {
        if (count($items) <= 1) {
            return [$items];
        }
        $out = [];
        foreach ($items as $pos => $first) {
            $rest = $items;
            array_splice($rest, $pos, 1);
            foreach (self::permutations(array_values($rest)) as $sub) {
                $out[] = [$first, ...$sub];
            }
        }
        return $out;
    }
}

