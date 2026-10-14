<?php

declare(strict_types=1);

namespace Tessero\Stats;

/**
 * scipy.stats.BootstrapMethod: a configuration dataclass carrying the resampling
 * options for the bootstrap-based confidence-interval machinery (e.g. the `method`
 * argument of scipy.stats distribution CI helpers). Pure data, no math; field
 * defaults mirror scipy exactly (n_resamples=9999, batch=None, rng=None,
 * method='BCa'). One class, both backends.
 */
final class BootstrapMethod
{
    public function __construct(
        public readonly int $n_resamples = 9999,
        public readonly ?int $batch = null,
        public readonly mixed $rng = null,
        public readonly string $method = 'BCa',
    ) {
    }
}
