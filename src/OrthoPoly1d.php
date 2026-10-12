<?php

declare(strict_types=1);

namespace Tessero;

/**
 * An orthogonal polynomial returned by the {@see OrthoPoly} factories, mirroring
 * scipy.special's `orthopoly1d` (a `numpy.poly1d` subclass). It carries the power-basis
 * coefficients (highest degree first, like `numpy.poly1d.c`) and is callable: invoking it
 * evaluates the polynomial at a point or array of points via Horner's method.
 */
final class OrthoPoly1d
{
    /** @var list<float> Coefficients, highest degree first (so c[0] is the leading coefficient). */
    public array $c;

    /** @param list<float> $c */
    public function __construct(array $c)
    {
        $this->c = array_values(array_map('floatval', $c));
    }

    /** Degree of the polynomial. */
    public function order(): int
    {
        return count($this->c) - 1;
    }

    /**
     * Evaluate the polynomial. A float returns a float; a list of floats returns a list.
     *
     * @param float|list<float> $x
     * @return float|list<float>
     */
    public function __invoke(float|array $x): float|array
    {
        if (is_array($x)) {
            return array_map(fn ($xi): float => $this->at((float) $xi), array_values($x));
        }

        return $this->at($x);
    }

    private function at(float $x): float
    {
        $v = 0.0;
        foreach ($this->c as $cc) {
            $v = $v * $x + $cc;
        }

        return $v;
    }
}
