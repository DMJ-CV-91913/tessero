<?php

declare(strict_types=1);

namespace Tessero\Stats;

/**
 * The callable family class returned by {@see MakeDistribution::of()} — SciPy's
 * make_distribution(stats.<name>). Invoke it with the base distribution's shape parameters to freeze a
 * new-API {@see MakeDistribution} object, mirroring make_distribution(stats.X)(**shapes).
 */
final class MakeDistributionFactory
{
    public function __construct(private readonly string $name)
    {
    }

    /**
     * Freeze the family with shape parameters.
     *
     * @param array<string, mixed> $shape
     */
    public function __invoke(array $shape = []): MakeDistribution
    {
        return new MakeDistribution($this->name, $shape);
    }

    /** The scipy base-distribution name this family wraps. */
    public function name(): string
    {
        return $this->name;
    }
}
