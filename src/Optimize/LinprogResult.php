<?php

declare(strict_types=1);

namespace Tessero\Optimize;

/** Result of LinearProgramming::linprog (field names follow scipy.optimize.OptimizeResult). */
final class LinprogResult implements \JsonSerializable
{
    /**
     * @param list<float>|null $x
     * @param list<float> $ineqlinMarginals d fun / d b_ub (<= 0 for a minimisation)
     * @param list<float> $eqlinMarginals d fun / d b_eq
     * @param list<float> $reducedCosts per variable, in the variable's own direction
     */
    public function __construct(
        public readonly ?array $x,
        public readonly ?float $fun,
        public readonly bool $success,
        public readonly int $status,
        public readonly string $message,
        public readonly int $nit,
        public readonly array $ineqlinMarginals = [],
        public readonly array $eqlinMarginals = [],
        public readonly array $reducedCosts = [],
    ) {
    }

    /** Slack of each A_ub row at the solution: b_ub - A_ub x. */
    public function slack(array $A_ub, array $b_ub): array
    {
        $out = [];
        foreach ($A_ub as $i => $row) {
            $s = (float) $b_ub[$i];
            foreach ($row as $j => $a) {
                $s -= $a * ($this->x[$j] ?? 0.0);
            }
            $out[] = $s;
        }

        return $out;
    }

    public function jsonSerialize(): array
    {
        return get_object_vars($this);
    }
}
