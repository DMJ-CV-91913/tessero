<?php

declare(strict_types=1);

namespace Tessero\Optimize;

/** Result of LinearProgramming::milp. */
final class MilpResult implements \JsonSerializable
{
    /** @param list<float>|null $x */
    public function __construct(
        public readonly ?array $x,
        public readonly ?float $fun,
        public readonly bool $success,
        public readonly int $status,
        public readonly string $message,
        public readonly int $nodes,
        public readonly float $bestBound,
    ) {
    }

    /** Relative gap between the incumbent and the best bound (0 when proven optimal). */
    public function gap(): float
    {
        if ($this->fun === null || ! is_finite($this->bestBound)) {
            return INF;
        }

        return abs($this->fun - $this->bestBound) / max(1.0, abs($this->fun));
    }

    public function jsonSerialize(): array
    {
        return get_object_vars($this) + ['gap' => $this->gap()];
    }
}
