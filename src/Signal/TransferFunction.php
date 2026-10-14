<?php

declare(strict_types=1);

namespace Tessero\Signal;

use Tessero\Signal;

/**
 * scipy.signal.TransferFunction - LTI system given by numerator/denominator polynomials.
 *
 * Continuous-time when dt is null (default), discrete-time when dt is a float
 * (matching scipy.signal.TransferFunction(..., dt=...), i.e. the Discrete variant).
 *
 * All conversions / responses delegate to the landed Tessero\Signal routines.
 */
final class TransferFunction extends Lti
{
    /** @var float[]|float[][] numerator coefficients */
    public array $num;

    /** @var float[]|float[][] denominator coefficients */
    public array $den;

    public function __construct(mixed $num, mixed $den, ?float $dt = null)
    {
        $num = self::coerce($num);
        $den = self::coerce($den);
        // scipy.signal.TransferFunction.__init__ always runs the coefficients
        // through normalize(): den is scaled so den[0] == 1 and leading
        // near-zero numerator coefficients are trimmed. Mirror that here so
        // the stored num/den match scipy exactly (SISO: both are 1-D).
        if (! self::is2d($num) && ! self::is2d($den)) {
            $r = Signal::normalize($num, $den);
            $num = self::toList($r['b']);
            $den = self::toList($r['a']);
            // normalize may return num as (1, k); flatten a single row.
            if (count($num) === 1 && is_array($num[0])) {
                $num = $num[0];
            }
        }
        $this->num = $num;
        $this->den = $den;
        $this->dt = $dt;
    }

    private static function is2d(array $a): bool
    {
        return isset($a[0]) && is_array($a[0]);
    }

    public function poles(): array
    {
        return self::toList(Signal::tf2zpk($this->num, $this->den)['p']);
    }

    public function zeros(): array
    {
        return self::toList(Signal::tf2zpk($this->num, $this->den)['z']);
    }

    public function toTf(): TransferFunction
    {
        return $this;
    }

    public function toZpk(): ZerosPolesGain
    {
        $r = Signal::tf2zpk($this->num, $this->den);

        return new ZerosPolesGain(
            self::toList($r['z']),
            self::toList($r['p']),
            (float) $r['k'],
            $this->dt
        );
    }

    public function toSs(): StateSpace
    {
        $r = Signal::tf2ss($this->num, $this->den);

        return new StateSpace(
            self::toList($r['A']),
            self::toList($r['B']),
            self::toList($r['C']),
            self::toList($r['D']),
            $this->dt
        );
    }

    /** @return float[]|float[][] */
    private static function coerce(mixed $v): array
    {
        if (is_object($v) && method_exists($v, 'toArray')) {
            return $v->toArray();
        }
        if (is_array($v)) {
            return $v;
        }

        return [(float) $v];
    }
}
