<?php

declare(strict_types=1);

namespace Tessero\Mdp;

use Tessero\NDArray;

/** Solution of a Markov decision process. */
final class MdpResult implements \JsonSerializable
{
    public function __construct(
        public readonly NDArray $values,      // V (S,)
        public readonly NDArray $policy,      // action index per state (S,), int64
        public readonly int $iterations,
        public readonly bool $converged,
        public readonly string $method,
        public readonly ?float $delta = null, // last sup-norm change (value iteration)
    ) {
    }

    public function jsonSerialize(): array
    {
        return [
            'values' => $this->values->toList(),
            'policy' => $this->policy->toList(),
            'iterations' => $this->iterations,
            'converged' => $this->converged,
            'method' => $this->method,
            'delta' => $this->delta,
        ];
    }
}
