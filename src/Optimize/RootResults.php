<?php

declare(strict_types=1);

namespace Tessero\Optimize;

/**
 * scipy.optimize.RootResults: the result object returned by root_scalar, with the same
 * attribute surface (root, iterations, function_calls, converged, flag, method).
 */
final class RootResults implements \JsonSerializable
{
    public function __construct(
        public readonly float $root,
        public readonly int $iterations,
        public readonly int $function_calls,
        public readonly bool $converged,
        public readonly string $flag,
        public readonly string $method = '',
    ) {
    }

    public function jsonSerialize(): array
    {
        return get_object_vars($this);
    }
}
