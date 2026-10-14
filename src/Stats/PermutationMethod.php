<?php

declare(strict_types=1);

namespace Tessero\Stats;

/**
 * scipy.stats.PermutationMethod: configuration dataclass for the permutation-test
 * machinery. Pure data, no math; field defaults mirror scipy exactly
 * (n_resamples=9999, batch=None, rng=None). One class, both backends.
 */
final class PermutationMethod
{
    public function __construct(
        public readonly int $n_resamples = 9999,
        public readonly ?int $batch = null,
        public readonly mixed $rng = null,
    ) {
    }
}
