<?php

declare(strict_types=1);

namespace Tessero\Signal;

use Tessero\Signal;

/**
 * scipy.signal.ZerosPolesGain - LTI system given by zeros, poles and gain.
 *
 * Zeros / poles are accepted as real lists or [re, im] complex pairs (the same
 * shape the landed routines emit from their complex NDArrays).
 */
final class ZerosPolesGain extends Lti
{
    /** @var array zeros as real values or [re, im] pairs */
    public array $z;

    /** @var array poles as real values or [re, im] pairs */
    public array $p;

    public float $k;

    public function __construct(mixed $z, mixed $p, mixed $k, ?float $dt = null)
    {
        $this->z = self::coerce($z);
        $this->p = self::coerce($p);
        $this->k = (float) $k;
        $this->dt = $dt;
    }

    public function poles(): array
    {
        return $this->p;
    }

    public function zeros(): array
    {
        return $this->z;
    }

    public function toTf(): TransferFunction
    {
        $r = Signal::zpk2tf($this->z, $this->p, $this->k);

        return new TransferFunction(self::toList($r['b']), self::toList($r['a']), $this->dt);
    }

    public function toZpk(): ZerosPolesGain
    {
        return $this;
    }

    public function toSs(): StateSpace
    {
        $r = Signal::zpk2ss($this->z, $this->p, $this->k);

        return new StateSpace(
            self::toList($r['A']),
            self::toList($r['B']),
            self::toList($r['C']),
            self::toList($r['D']),
            $this->dt
        );
    }

    private static function coerce(mixed $v): array
    {
        if (is_object($v) && method_exists($v, 'toArray')) {
            return $v->toArray();
        }
        if (is_array($v)) {
            return $v;
        }

        return [$v];
    }
}
