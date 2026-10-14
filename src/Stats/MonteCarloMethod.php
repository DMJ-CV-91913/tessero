<?php

declare(strict_types=1);

namespace Tessero\Stats;

/**
 * scipy.stats.MonteCarloMethod: configuration dataclass for the Monte-Carlo
 * hypothesis-test machinery. Pure data, no math; field defaults mirror scipy
 * exactly (n_resamples=9999, batch=None, rvs=None, rng=None). One class, both
 * backends.
 */
final class MonteCarloMethod
{
    public function __construct(
        public readonly int $n_resamples = 9999,
        public readonly ?int $batch = null,
        public readonly mixed $rvs = null,
        public readonly mixed $rng = null,
    ) {
    }
}
