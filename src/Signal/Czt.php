<?php

declare(strict_types=1);

namespace Tessero\Signal;

use Tessero\Signal;

/**
 * scipy.signal.CZT - callable chirp z-transform object.
 *
 * `new Czt($n, $m, $w, $a)` builds a transform configured for length-$n inputs; invoking the object
 * via {@see Czt::transform()} (or treating it as callable) applies the chirp z-transform, producing
 * exactly the same result as the free function scipy.signal.czt(x, m, w, a).
 *
 * This is a thin deterministic container over the already-landed `Tessero\Signal::czt` primitive:
 * CZT holds the parameters so the same transform can be reused across many inputs. Because
 * CZT(...)($x) is defined by scipy as czt(x, m, w, a), results are bit-identical to czt.
 *
 * @see \Tessero\Signal::czt
 */
final class Czt
{
    private int $n;

    private mixed $m;

    private mixed $w;

    private mixed $a;

    /**
     * @param int   $n The size of the signal the transform will operate on.
     * @param ?int  $m The number of output points desired. Defaults to $n.
     * @param mixed $w The ratio between points in each step (complex). Defaults to equally spaced
     *                 points around the entire unit circle.
     * @param mixed $a The starting point in the complex plane. Defaults to 1+0j.
     */
    public function __construct(int $n, mixed $m = null, mixed $w = null, mixed $a = 1)
    {
        $this->n = $n;
        $this->m = $m;
        $this->w = $w;
        $this->a = $a;
    }

    /**
     * Apply the chirp z-transform to $x (equivalent to scipy.signal.czt(x, m, w, a)).
     */
    public function transform(mixed $x): mixed
    {
        return Signal::czt($x, $this->m, $this->w, $this->a);
    }

    /**
     * Allow the object to be invoked directly as a callable: $czt($x).
     */
    public function __invoke(mixed $x): mixed
    {
        return $this->transform($x);
    }

    /**
     * Points in the complex plane at which the chirp z-transform is sampled
     * (scipy.signal.CZT.points), i.e. czt_points(m, w, a).
     */
    public function points(): mixed
    {
        return Signal::cztPoints($this->m ?? $this->n, $this->w, $this->a);
    }
}
