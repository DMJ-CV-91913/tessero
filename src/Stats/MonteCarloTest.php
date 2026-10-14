<?php

declare(strict_types=1);

namespace Tessero\Stats;

use Tessero\Random\Generator;

/**
 * scipy.stats.monte_carlo_test, pure PHP over a PHP callable statistic and
 * Tessero's PCG64 Generator. Bit-exact against scipy when the same seeded
 * default_rng and the same null-distribution sampler are used:
 *
 *   - each null sample set is drawn by one rvs_i(size=[n_resamples, n]) call per
 *     sample, in sequence (single batch: batch defaults to n_resamples), using a
 *     Tessero Generator distribution whose draws reproduce numpy's exactly;
 *   - the p-value is Phipson & Smyth's (sum(cmp) + 1) / (n_resamples + 1), with the
 *     same gamma = |100*eps*observed| tolerance as scipy.
 *
 * $rvs is a list (one per sample in $data) of callables rvs(int $rows, int $n): list<list<float>>
 * returning an $rows-by-$n null sample. $statistic accepts len(data) sample arrays
 * (each list<float>) and returns a float. Returns ['statistic' => float,
 * 'pvalue' => float, 'null_distribution' => list<float>].
 */
final class MonteCarloTest
{
    private const EPS100 = 2.220446049250313e-16 * 100.0;

    public static function monteCarloTest(
        array $data,
        array $rvs,
        callable $statistic,
        int $nResamples = 9999,
        string $alternative = 'two-sided'
    ): array {
        if (! in_array($alternative, ['two-sided', 'less', 'greater'], true)) {
            throw new \InvalidArgumentException("monte_carlo_test: unknown alternative '{$alternative}'.");
        }
        $data = array_map(static fn ($x) => array_map('floatval', array_values($x)), array_values($data));
        if (count($rvs) !== count($data)) {
            throw new \InvalidArgumentException('monte_carlo_test: len(rvs) must equal len(data).');
        }

        $observed = (float) $statistic(...$data);

        // Null distribution: one rvs_i(size=[nResamples, n]) draw per sample, in order.
        $nObs = array_map('count', $data);
        $samples = [];
        foreach ($rvs as $i => $rvsI) {
            $samples[] = $rvsI($nResamples, $nObs[$i]);   // list<list<float>> of shape (nResamples, n)
        }
        $null = [];
        for ($r = 0; $r < $nResamples; $r++) {
            $row = [];
            foreach ($samples as $s) {
                $row[] = $s[$r];
            }
            $null[] = (float) $statistic(...$row);
        }

        $gamma = abs(self::EPS100 * $observed);
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
        $less = ($cl + 1.0) / ($nResamples + 1.0);
        $greater = ($cg + 1.0) / ($nResamples + 1.0);
        if ($alternative === 'less') {
            $p = $less;
        } elseif ($alternative === 'greater') {
            $p = $greater;
        } else {
            $p = min($less, $greater) * 2.0;
        }
        $p = max(0.0, min(1.0, $p));

        return ['statistic' => $observed, 'pvalue' => $p, 'null_distribution' => $null];
    }

    /**
     * Build a named null-distribution sampler over a Generator, returning a
     * callable(rows, n): list<list<float>> that draws an (rows, n) block in
     * numpy's row-major order (one standardNormal/normal/random/uniform call).
     *
     * @param array<string,float> $params
     */
    public static function makeRvs(Generator $gen, string $kind, array $params = []): callable
    {
        return static function (int $rows, int $n) use ($gen, $kind, $params): array {
            // toArray() on the (rows, n) draw is already a list of rows in numpy row-major order.
            return match ($kind) {
                'standard_normal' => $gen->standardNormal([$rows, $n])->toArray(),
                'normal' => $gen->normal($params['loc'] ?? 0.0, $params['scale'] ?? 1.0, [$rows, $n])->toArray(),
                'random' => $gen->random([$rows, $n])->toArray(),
                'uniform' => $gen->uniform($params['low'] ?? 0.0, $params['high'] ?? 1.0, [$rows, $n])->toArray(),
                default => throw new \InvalidArgumentException("monte_carlo_test: unknown rvs kind '{$kind}'."),
            };
        };
    }
}
