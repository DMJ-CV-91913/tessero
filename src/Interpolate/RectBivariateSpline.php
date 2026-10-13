<?php

declare(strict_types=1);

namespace Tessero\Interpolate;

use Tessero\Interpolate as Kernel;

/**
 * Tensor-product interpolating spline on a regular grid (scipy.interpolate.RectBivariateSpline, s = 0).
 *
 * Pure PHP. The s = 0 (exactly interpolating) rectangular bivariate spline is the tensor product of two
 * 1-D interpolating B-splines: fit a spline along the x axis for every column of the grid, then fit a
 * spline along the y axis through those x-coefficients. The two 1-D solves and the FITPACK IUS / not-a-knot
 * knot pattern are built here; the final tensor de Boor evaluation delegates to the existing kernel routine
 * Tessero\Interpolate::NdBSpline (identical on both the FFI and extension backends).
 *
 * Scope: s = 0 only, and degrees kx, ky in {1, 3} (linear or not-a-knot cubic), matching the 1-D
 * interpolating spline that Tessero supports. The coefficient block passed to NdBSpline is the x-then-y
 * solve transposed back to shape (ncx, ncy) (see the two-solve construction in __construct).
 */
final class RectBivariateSpline
{
    /** @var list<float> strictly increasing x grid coordinates */
    private array $x;

    /** @var list<float> strictly increasing y grid coordinates */
    private array $y;

    private int $kx;

    private int $ky;

    /** @var list<float> x knot vector (length ncx + kx + 1) */
    private array $tx;

    /** @var list<float> y knot vector (length ncy + ky + 1) */
    private array $ty;

    /** @var list<list<float>> coefficient block, shape (ncx, ncy) */
    private array $c;

    private int $ncx;

    private int $ncy;

    /**
     * @param list<float|int>       $x  grid x coordinates (strictly increasing), length nx
     * @param list<float|int>       $y  grid y coordinates (strictly increasing), length ny
     * @param list<list<float|int>> $z  grid values, shape (nx, ny): z[i][j] = f(x[i], y[j])
     * @param int                   $kx x degree, 1 or 3 (default 3)
     * @param int                   $ky y degree, 1 or 3 (default 3)
     * @param float                 $s  smoothing factor; only s = 0 (exact interpolation) is supported
     */
    public function __construct(array $x, array $y, array $z, int $kx = 3, int $ky = 3, float $s = 0.0)
    {
        if ($s !== 0.0) {
            throw new \ValueError('RectBivariateSpline: only s=0 (interpolating) is supported');
        }
        if ($kx !== 1 && $kx !== 3) {
            throw new \ValueError('RectBivariateSpline: kx must be 1 or 3');
        }
        if ($ky !== 1 && $ky !== 3) {
            throw new \ValueError('RectBivariateSpline: ky must be 1 or 3');
        }
        $this->x = array_map('floatval', array_values($x));
        $this->y = array_map('floatval', array_values($y));
        $this->kx = $kx;
        $this->ky = $ky;
        $nx = count($this->x);
        $ny = count($this->y);
        if ($nx < $kx + 1) {
            throw new \ValueError('RectBivariateSpline: need at least kx+1 points along x');
        }
        if ($ny < $ky + 1) {
            throw new \ValueError('RectBivariateSpline: need at least ky+1 points along y');
        }
        $zz = array_map(static fn (array $row): array => array_map('floatval', array_values($row)), array_values($z));
        if (count($zz) !== $nx) {
            throw new \ValueError('RectBivariateSpline: z must have x length rows');
        }
        foreach ($zz as $row) {
            if (count($row) !== $ny) {
                throw new \ValueError('RectBivariateSpline: z must have y length columns');
            }
        }

        // --- solve 1: interpolate along x for every column j of z -> cx, shape (ncx, ny) ---
        $this->tx = self::interpKnots($this->x, $kx);
        $this->ncx = $nx;                                  // interpolating spline: ncx == nx
        // column-wise system A_x is shared across all columns; factor once, solve for each column rhs.
        $ax = self::collocation($this->tx, $kx, $this->x);
        $cx = [];                                          // (ncx, ny)
        for ($j = 0; $j < $ny; $j++) {
            $rhs = [];
            for ($i = 0; $i < $nx; $i++) {
                $rhs[$i] = $zz[$i][$j];
            }
            $sol = self::solve($ax, $rhs);
            for ($i = 0; $i < $this->ncx; $i++) {
                $cx[$i][$j] = $sol[$i];
            }
        }

        // --- solve 2: interpolate along y through the x-coefficient rows ---
        // feed the transpose (ny, ncx) so each x-coefficient index becomes a column, solve over y,
        // then transpose back to (ncx, ncy). (This is the 2nd-axis coeff-block transpose in the proto.)
        $this->ty = self::interpKnots($this->y, $ky);
        $this->ncy = $ny;
        $ay = self::collocation($this->ty, $ky, $this->y);
        $this->c = [];                                     // (ncx, ncy)
        for ($i = 0; $i < $this->ncx; $i++) {
            $rhs = [];
            for ($j = 0; $j < $ny; $j++) {
                $rhs[$j] = $cx[$i][$j];
            }
            $sol = self::solve($ay, $rhs);
            for ($j = 0; $j < $this->ncy; $j++) {
                $this->c[$i][$j] = $sol[$j];
            }
        }
    }

    /**
     * Evaluate on the tensor grid xq x yq, returning a (len(xq), len(yq)) array
     * (matching scipy.interpolate.RectBivariateSpline.__call__(xq, yq)).
     *
     * @param list<float|int> $xq
     * @param list<float|int> $yq
     * @return list<list<float>>
     */
    public function __invoke(array $xq, array $yq): array
    {
        $xqa = array_map('floatval', array_values($xq));
        $yqa = array_map('floatval', array_values($yq));
        $pts = [];
        foreach ($xqa as $px) {
            foreach ($yqa as $py) {
                $pts[] = [$px, $py];
            }
        }
        // tensor de Boor eval via the existing kernel routine (both backends identical)
        $res = Kernel::NdBSpline($this->tx, $this->ty, $this->c, $this->kx, $this->ky, $pts);
        if (is_object($res) && method_exists($res, 'toList')) {
            $flat = array_values($res->toList());          // NDArray 1-D result -> flat PHP list
        } elseif (is_array($res)) {
            $flat = array_values($res);
        } else {
            $flat = [$res];
        }
        $out = [];
        $nxq = count($xqa);
        $nyq = count($yqa);
        $p = 0;
        for ($i = 0; $i < $nxq; $i++) {
            $row = [];
            for ($j = 0; $j < $nyq; $j++) {
                $row[$j] = (float) $flat[$p++];
            }
            $out[$i] = $row;
        }
        return $out;
    }

    /** @return list<list<float>> coefficient block (ncx, ncy) */
    public function getCoeffs(): array
    {
        return $this->c;
    }

    /** @return array{0: list<float>, 1: list<float>} the (tx, ty) knot vectors */
    public function getKnots(): array
    {
        return [$this->tx, $this->ty];
    }

    /**
     * FITPACK IUS / not-a-knot knot vector for an interpolating spline of degree k through points x.
     * k = 1: [x0, x0, x[1..n-2], x_{n-1}, x_{n-1}].
     * k = 3: [x0 x4, x[2..n-3], x_{n-1} x4]  (not-a-knot), yielding exactly n coefficients.
     *
     * @param list<float> $x
     * @return list<float>
     */
    private static function interpKnots(array $x, int $k): array
    {
        $n = count($x);
        $t = [];
        if ($k === 1) {
            $t[] = $x[0];
            for ($i = 0; $i < $n; $i++) {
                $t[] = $x[$i];
            }
            $t[] = $x[$n - 1];
            return $t;
        }
        // k === 3
        for ($r = 0; $r < 4; $r++) {
            $t[] = $x[0];
        }
        for ($i = 2; $i <= $n - 3; $i++) {
            $t[] = $x[$i];
        }
        for ($r = 0; $r < 4; $r++) {
            $t[] = $x[$n - 1];
        }
        return $t;
    }

    /**
     * B-spline basis functions N_{0..k} at xv for the knot span containing xv (de Boor, Cox recurrence).
     *
     * @param list<float> $t knot vector
     * @param int         $mu knot span index: t[mu] <= xv < t[mu+1] (clamped to the base interval)
     * @return list<float> k+1 basis values, aligned so N[j] multiplies coefficient (mu-k+j)
     */
    private static function basis(array $t, int $k, int $mu, float $xv): array
    {
        $N = array_fill(0, $k + 1, 0.0);
        $N[0] = 1.0;
        $left = array_fill(0, $k + 1, 0.0);
        $right = array_fill(0, $k + 1, 0.0);
        for ($r = 1; $r <= $k; $r++) {
            $left[$r] = $xv - $t[$mu + 1 - $r];
            $right[$r] = $t[$mu + $r] - $xv;
            $saved = 0.0;
            for ($i = 0; $i < $r; $i++) {
                $denom = $right[$i + 1] + $left[$r - $i];
                $temp = $denom != 0.0 ? $N[$i] / $denom : 0.0;
                $N[$i] = $saved + $right[$i + 1] * $temp;
                $saved = $left[$r - $i] * $temp;
            }
            $N[$r] = $saved;
        }
        return $N;
    }

    /**
     * Collocation matrix A (n x n) of B-spline basis values at the data points x, for the interpolating
     * knot vector t (n == number of coefficients == number of data points).
     *
     * @param list<float> $t
     * @param list<float> $x
     * @return list<list<float>>
     */
    private static function collocation(array $t, int $k, array $x): array
    {
        $n = count($t) - $k - 1;                           // number of coefficients
        $m = count($x);
        $A = [];
        for ($r = 0; $r < $m; $r++) {
            $xv = $x[$r];
            // find span mu: t[mu] <= xv < t[mu+1], clamped so (mu-k .. mu) are valid coefficient indices
            if ($xv >= $t[$n]) {
                $mu = $n - 1;
            } else {
                $mu = $k;
                while ($mu < $n - 1 && $t[$mu + 1] <= $xv) {
                    $mu++;
                }
            }
            $N = self::basis($t, $k, $mu, $xv);
            $row = array_fill(0, $n, 0.0);
            for ($j = 0; $j <= $k; $j++) {
                $row[$mu - $k + $j] = $N[$j];
            }
            $A[$r] = $row;
        }
        return $A;
    }

    /**
     * Solve A x = b for a square matrix via Gaussian elimination with partial pivoting.
     * A is small and banded (bandwidth k+1); a dense solve is more than fast enough here and keeps the
     * code dependency-free and backend-neutral.
     *
     * @param list<list<float>> $A
     * @param list<float>       $b
     * @return list<float>
     */
    private static function solve(array $A, array $b): array
    {
        $n = count($A);
        // working copies
        $M = [];
        for ($i = 0; $i < $n; $i++) {
            $M[$i] = $A[$i];
            $M[$i][$n] = $b[$i];                            // augmented column
        }
        for ($col = 0; $col < $n; $col++) {
            // partial pivot
            $piv = $col;
            $best = abs($M[$col][$col]);
            for ($r = $col + 1; $r < $n; $r++) {
                $v = abs($M[$r][$col]);
                if ($v > $best) {
                    $best = $v;
                    $piv = $r;
                }
            }
            if ($piv !== $col) {
                $tmp = $M[$piv];
                $M[$piv] = $M[$col];
                $M[$col] = $tmp;
            }
            $pivot = $M[$col][$col];
            if ($pivot == 0.0) {
                throw new \RuntimeException('RectBivariateSpline: singular collocation matrix');
            }
            for ($r = $col + 1; $r < $n; $r++) {
                $f = $M[$r][$col] / $pivot;
                if ($f == 0.0) {
                    continue;
                }
                for ($cc = $col; $cc <= $n; $cc++) {
                    $M[$r][$cc] -= $f * $M[$col][$cc];
                }
            }
        }
        // back substitution
        $x = array_fill(0, $n, 0.0);
        for ($i = $n - 1; $i >= 0; $i--) {
            $sum = $M[$i][$n];
            for ($j = $i + 1; $j < $n; $j++) {
                $sum -= $M[$i][$j] * $x[$j];
            }
            $x[$i] = $sum / $M[$i][$i];
        }
        return $x;
    }
}
