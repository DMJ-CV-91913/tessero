<?php

declare(strict_types=1);

namespace Tessero\Signal;

use Tessero\Signal;

/**
 * scipy.signal.lti - continuous-time linear time-invariant system factory base.
 *
 * Mirrors scipy: `lti(*system)` dispatches on the number of positional arguments:
 *   - 2 args (num, den)   -> TransferFunction
 *   - 3 args (z, p, k)    -> ZerosPolesGain
 *   - 4 args (A, B, C, D) -> StateSpace
 *
 * The returned object exposes the three cross-conversions (toTf / toZpk / toSs)
 * and the continuous-time response methods (impulse, step, bode, freqresp),
 * all delegating to the already-landed Tessero\Signal routines.
 *
 * This is a thin, deterministic OO layer: every numeric result is produced by
 * the landed primitives, so both backends agree exactly.
 */
abstract class Lti
{
    /** Discrete sampling interval; null for continuous-time systems. */
    public ?float $dt = null;

    /**
     * Factory: construct the appropriate representation from a variable argument list.
     *
     * @return TransferFunction|ZerosPolesGain|StateSpace
     */
    public static function create(mixed ...$system): Lti
    {
        switch (count($system)) {
            case 1:
                $s = $system[0];
                if ($s instanceof Lti) {
                    return $s;
                }
                throw new \InvalidArgumentException('lti: single-argument form must be an existing Lti instance.');
            case 2:
                return new TransferFunction($system[0], $system[1]);
            case 3:
                return new ZerosPolesGain($system[0], $system[1], $system[2]);
            case 4:
                return new StateSpace($system[0], $system[1], $system[2], $system[3]);
            default:
                throw new \InvalidArgumentException(
                    'lti: needs 2 (num,den), 3 (z,p,k) or 4 (A,B,C,D) arguments, got ' . count($system) . '.'
                );
        }
    }

    /** Poles of the system (roots of the denominator / eigenvalues of A), as [re, im] pairs. */
    abstract public function poles(): array;

    /** Zeros of the system, as [re, im] pairs. */
    abstract public function zeros(): array;

    /** Convert to TransferFunction representation. */
    abstract public function toTf(): TransferFunction;

    /** Convert to ZerosPolesGain representation. */
    abstract public function toZpk(): ZerosPolesGain;

    /** Convert to StateSpace representation. */
    abstract public function toSs(): StateSpace;

    /**
     * Continuous-time impulse response (scipy LTI.impulse).
     *
     * @return array{tout: float[], yout: float[]}
     */
    public function impulse(mixed $T): array
    {
        $ss = $this->toSs();
        $r = Signal::impulse($ss->A, $ss->B, $ss->C, $ss->D, $T);

        return ['tout' => self::toList($r['tout']), 'yout' => self::toList($r['yout'])];
    }

    /**
     * Continuous-time step response (scipy LTI.step).
     *
     * @return array{tout: float[], yout: float[]}
     */
    public function step(mixed $T): array
    {
        $ss = $this->toSs();
        $r = Signal::step($ss->A, $ss->B, $ss->C, $ss->D, $T);

        return ['tout' => self::toList($r['tout']), 'yout' => self::toList($r['yout'])];
    }

    /**
     * Bode magnitude (dB) and phase (deg) (scipy LTI.bode).
     *
     * @return array{w: float[], mag: float[], phase: float[]}
     */
    public function bode(mixed $w): array
    {
        $tf = $this->toTf();
        $r = Signal::bode($tf->num, $tf->den, $w);

        return [
            'w' => self::toList($r['w']),
            'mag' => self::toList($r['mag']),
            'phase' => self::toList($r['phase']),
        ];
    }

    /**
     * Complex frequency response H(jw) (scipy LTI.freqresp), returned as [re, im] pairs.
     *
     * @return array{w: float[], H: array<int, array{0: float, 1: float}>}
     */
    public function freqresp(mixed $w): array
    {
        $tf = $this->toTf();
        $r = Signal::freqs($tf->num, $tf->den, $w);

        // freqs returns {w, h}; normalise key casing.
        $wv = $r['w'] ?? ($r['W'] ?? null);
        $hv = $r['h'] ?? ($r['H'] ?? null);

        return ['w' => self::toList($wv), 'H' => self::toList($hv)];
    }

    /**
     * Discrete-time impulse response (scipy dlti.impulse / dimpulse).
     *
     * @return array{tout: float[], yout: float[]}
     */
    public function dimpulse(int $n = 100): array
    {
        $ss = $this->toSs();
        $dt = $this->dt ?? 1.0;
        $r = Signal::dimpulse($ss->A, $ss->B, $ss->C, $ss->D, $dt, $n);

        return ['tout' => self::toList($r['tout']), 'yout' => self::flatten(self::toList($r['yout']))];
    }

    /**
     * Discrete-time step response (scipy dlti.step / dstep).
     *
     * @return array{tout: float[], yout: float[]}
     */
    public function dstep(int $n = 100): array
    {
        $ss = $this->toSs();
        $dt = $this->dt ?? 1.0;
        $r = Signal::dstep($ss->A, $ss->B, $ss->C, $ss->D, $dt, $n);

        return ['tout' => self::toList($r['tout']), 'yout' => self::flatten(self::toList($r['yout']))];
    }

    /**
     * Discrete-time Bode magnitude (dB) and phase (deg) (scipy dlti.bode).
     *
     * @return array{w: float[], mag: float[], phase: float[]}
     */
    public function dbode(mixed $w): array
    {
        $tf = $this->toTf();
        $r = Signal::dbode($tf->num, $tf->den, $w);

        return [
            'w' => self::toList($r['w']),
            'mag' => self::toList($r['mag']),
            'phase' => self::toList($r['phase']),
        ];
    }

    /** Flatten an (n,1) response to a flat list of floats. */
    protected static function flatten(array $v): array
    {
        $out = [];
        foreach ($v as $row) {
            $out[] = is_array($row) ? (float) $row[0] : (float) $row;
        }

        return $out;
    }

    /** Coerce an NDArray / nested PHP array to a flat/2d PHP list. */
    protected static function toList(mixed $v): array
    {
        if (is_object($v) && method_exists($v, 'toArray')) {
            return $v->toArray();
        }

        return (array) $v;
    }
}
