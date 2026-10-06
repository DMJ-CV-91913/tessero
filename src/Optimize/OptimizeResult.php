<?php

declare(strict_types=1);

namespace Tessero\Optimize;

/** scipy.optimize.OptimizeResult */
final class OptimizeResult implements \JsonSerializable
{
    /**
     * @param list<float> $x
     * @param list<float>|null $jac
     * @param list<list<float>>|null $hessInv
     */
    public function __construct(
        public readonly array $x,
        public readonly float $fun,
        public readonly bool $success,
        public readonly int $status,
        public readonly string $message,
        public readonly int $nit,
        public readonly int $nfev,
        public readonly int $njev = 0,
        public readonly ?array $jac = null,
        public readonly ?array $hessInv = null,
        public readonly string $method = '',
    ) {
    }

    public function jsonSerialize(): array
    {
        return array_filter(get_object_vars($this), static fn ($v): bool => $v !== null);
    }
}
