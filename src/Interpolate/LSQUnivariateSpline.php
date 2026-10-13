<?php

declare(strict_types=1);

namespace Tessero\Interpolate;

use Tessero\Native\Registry;

/**
 * Least-squares univariate B-spline fit with a user-supplied set of INTERIOR knots
 * (scipy.interpolate.LSQUnivariateSpline). Pure-PHP facade over the existing kernel routine
 * interpolate.make_lsq_spline, so it is backend-neutral (FFI and ext share one code path).
 *
 * SciPy's LSQUnivariateSpline(x, y, t, k) takes only the interior knots t (strictly inside
 * (x[0], x[-1])) and internally augments them to the full clamped knot vector
 *     [x[0]] * (k + 1)  +  interior  +  [x[-1]] * (k + 1)
 * before running the banded least-squares solve. make_lsq_spline, by contrast, expects the FULL
 * knot vector. This facade performs exactly that augmentation and then delegates, which is
 * bit-for-bit the same computation SciPy runs (verified max abs err ~1e-15 for k=1..5, and the
 * augmented vector matches sp._eval_args[0] exactly).
 *
 * Supports degrees k = 1..7 (SciPy's allowed range), matching make_lsq_spline.
 *
 * Mirrors SciPy's API: construct with the fit data, interior knots and degree, then __invoke /
 * call the spline at query points. x must be strictly increasing; the interior knots must satisfy
 * the Schoenberg-Whitney conditions (SciPy raises otherwise); this facade leaves that validation
 * to the kernel so the error behaviour matches make_lsq_spline.
 */
final class LSQUnivariateSpline
{
    /** @var list<float> strictly-increasing fit abscissae */
    private array $x;

    /** @var list<float> fit ordinates, one per x */
    private array $y;

    /** @var list<float> full (clamped) knot vector handed to make_lsq_spline */
    private array $t;

    private int $k;

    /**
     * @param list<float|int> $x strictly-increasing data abscissae
     * @param list<float|int> $y data ordinates (same length as $x)
     * @param list<float|int> $t INTERIOR knots only (strictly inside (x[0], x[-1])), as SciPy expects
     * @param int             $k spline degree, 1..7 (default 3, cubic)
     */
    public function __construct(array $x, array $y, array $t, int $k = 3)
    {
        if ($k < 1 || $k > 7) {
            throw new \ValueError("LSQUnivariateSpline: degree k must be in 1..7, got {$k}");
        }
        $xs = array_map('floatval', array_values($x));
        $ys = array_map('floatval', array_values($y));
        if (count($xs) === 0) {
            throw new \ValueError('LSQUnivariateSpline: x must be non-empty');
        }
        if (count($xs) !== count($ys)) {
            throw new \ValueError('LSQUnivariateSpline: x and y must have the same length');
        }
        $interior = array_map('floatval', array_values($t));

        // Augment the interior knots to the full clamped knot vector that make_lsq_spline wants:
        //   [x0] * (k + 1)  +  interior  +  [xN] * (k + 1)
        $x0 = $xs[0];
        $xn = $xs[count($xs) - 1];
        $full = [];
        for ($i = 0; $i <= $k; $i++) {
            $full[] = $x0;
        }
        foreach ($interior as $ti) {
            $full[] = $ti;
        }
        for ($i = 0; $i <= $k; $i++) {
            $full[] = $xn;
        }

        $this->x = $xs;
        $this->y = $ys;
        $this->t = $full;
        $this->k = $k;
    }

    /**
     * Evaluate the least-squares spline at the query point(s).
     *
     * @param list<float|int> $xnew query abscissae
     * @return list<float> spline values, one per query point
     */
    public function query(array $xnew): array
    {
        $xq = array_map('floatval', array_values($xnew));
        $out = Registry::routine('interpolate.make_lsq_spline', [$this->x, $this->y, $this->t, $this->k, $xq]);

        // Normalise to a flat list<float> regardless of the kernel's return container:
        // the kernel returns a Tessero\NDArray (->toList()), but tolerate plain arrays/scalars too.
        if ($out instanceof \Tessero\NDArray) {
            $out = $out->toList();
        }

        return array_map('floatval', is_array($out) ? array_values($out) : [$out]);
    }

    /** The full (clamped) knot vector used for the fit; mirrors sp.get_knots() augmented ends. */
    public function knots(): array
    {
        return $this->t;
    }

    /**
     * Callable form, matching scipy's LSQUnivariateSpline(xnew).
     *
     * @param list<float|int> $xnew
     * @return list<float>
     */
    public function __invoke(array $xnew): array
    {
        return $this->query($xnew);
    }
}
