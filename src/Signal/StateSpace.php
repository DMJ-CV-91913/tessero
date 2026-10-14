<?php

declare(strict_types=1);

namespace Tessero\Signal;

use Tessero\Signal;

/**
 * scipy.signal.StateSpace - LTI system in state-space form (A, B, C, D).
 *
 * Because a state-space realization is NOT unique (any similarity transform
 * yields the same transfer function), the matrices themselves are only
 * well-defined up to a basis change. The transfer-function / zpk conversions
 * (ss2tf / ss2zpk) are invariants and match scipy exactly; equality of two
 * realizations is tested through the impulse response, not the raw matrices.
 */
final class StateSpace extends Lti
{
    public array $A;
    public array $B;
    public array $C;
    public array $D;

    public function __construct(mixed $A, mixed $B, mixed $C, mixed $D, ?float $dt = null)
    {
        $this->A = self::coerce($A);
        $this->B = self::coerce($B);
        $this->C = self::coerce($C);
        $this->D = self::coerce($D);
        $this->dt = $dt;
    }

    public function poles(): array
    {
        // Poles are the poles of the equivalent transfer function (invariant).
        return $this->toZpk()->p;
    }

    public function zeros(): array
    {
        return $this->toZpk()->z;
    }

    public function toTf(): TransferFunction
    {
        $r = Signal::ss2tf($this->A, $this->B, $this->C, $this->D);
        $num = self::toList($r['num']);
        // ss2tf returns num as (n_out, n_coef); a single-output system is squeezed
        // to a flat 1-D numerator, matching scipy StateSpace.to_tf().
        if (count($num) === 1 && is_array($num[0])) {
            $num = $num[0];
        }

        return new TransferFunction($num, self::toList($r['den']), $this->dt);
    }

    public function toZpk(): ZerosPolesGain
    {
        $r = Signal::ss2zpk($this->A, $this->B, $this->C, $this->D);

        return new ZerosPolesGain(
            self::toList($r['z']),
            self::toList($r['p']),
            (float) $r['k'],
            $this->dt
        );
    }

    public function toSs(): StateSpace
    {
        return $this;
    }

    private static function coerce(mixed $v): array
    {
        if (is_object($v) && method_exists($v, 'toArray')) {
            return $v->toArray();
        }
        if (is_array($v)) {
            return $v;
        }

        return [[(float) $v]];
    }
}
