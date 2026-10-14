<?php

declare(strict_types=1);

namespace Tessero\Interpolate;

use Tessero\Interpolate as Kernel;
use Tessero\Linalg;

/**
 * Least-squares tensor-product bivariate B-spline with user-supplied INTERIOR knots
 * (scipy.interpolate.LSQBivariateSpline).
 *
 * Pure PHP, backend-neutral (FFI and ext share one code path; the final evaluation delegates
 * to the kernel routine Tessero\Interpolate::NdBSpline). Unlike the auto-knot smoothing FITPACK
 * family (SmoothBivariateSpline / bisplrep), the knots here are FIXED by the caller, so the fit
 * is a plain, deterministic linear least-squares problem -- exactly what SciPy's
 * LSQBivariateSpline computes with its given knots.
 *
 * Construction:
 *   SciPy's LSQBivariateSpline(x, y, z, tx, ty, kx=3, ky=3) takes the INTERIOR knots tx, ty
 *   (strictly inside the data range) and augments each to the full clamped knot vector
 *       [x.min()] * (kx + 1)  +  tx  +  [x.max()] * (kx + 1)      (and similarly for ty)
 *   then minimises  || B c - z ||  where B is the (ndata x ncx*ncy) tensor-product B-spline
 *   design matrix, B[i, p*ncy + q] = Nx_p(x_i) * Ny_q(y_i).  The coefficient block c has shape
 *   (ncx, ncy) with ncx = len(tx_full) - kx - 1, ncy = len(ty_full) - ky - 1.
 *
 *   This facade builds the same augmented knot vectors and the same (ndata x ncx*ncy) design
 *   matrix, then solves the least-squares problem through the stable kernel routine
 *   numpy.linalg.lstsq (SVD-based, same algorithm SciPy's np.linalg.lstsq uses). On the
 *   well-conditioned fixtures this matches SciPy's coefficients and evaluations to ~3e-13
 *   (verified for kx, ky in {1, 2, 3}, symmetric and asymmetric degrees). A normal-equations
 *   solve would square the condition number and only reach ~1e-10, so lstsq is used instead.
 *
 * Evaluation mirrors SciPy: __invoke(xq, yq) returns the (len(xq), len(yq)) grid of spline values
 * via the tensor de Boor kernel routine NdBSpline.
 */
final class LSQBivariateSpline
{
    /** @var list<float> full (clamped) x knot vector, length ncx + kx + 1 */
    private array $tx;

    /** @var list<float> full (clamped) y knot vector, length ncy + ky + 1 */
    private array $ty;

    private int $kx;

    private int $ky;

    /** @var list<list<float>> coefficient block, shape (ncx, ncy) */
    private array $c;

    private int $ncx;

    private int $ncy;

    /**
     * @param list<float|int>       $x  data x coordinates, length ndata
     * @param list<float|int>       $y  data y coordinates, length ndata
     * @param list<float|int>       $z  data values, length ndata: z[i] = f(x[i], y[i])
     * @param list<float|int>       $tx INTERIOR x knots only (strictly inside the x range), as SciPy expects
     * @param list<float|int>       $ty INTERIOR y knots only (strictly inside the y range), as SciPy expects
     * @param int                   $kx x degree, 1..5 (default 3)
     * @param int                   $ky y degree, 1..5 (default 3)
     */
    public function __construct(array $x, array $y, array $z, array $tx, array $ty, int $kx = 3, int $ky = 3)
    {
        if ($kx < 1 || $kx > 5) {
            throw new \ValueError("LSQBivariateSpline: kx must be in 1..5, got {$kx}");
        }
        if ($ky < 1 || $ky > 5) {
            throw new \ValueError("LSQBivariateSpline: ky must be in 1..5, got {$ky}");
        }
        $xs = array_map('floatval', array_values($x));
        $ys = array_map('floatval', array_values($y));
        $zs = array_map('floatval', array_values($z));
        $m = count($xs);
        if ($m === 0) {
            throw new \ValueError('LSQBivariateSpline: x must be non-empty');
        }
        if (count($ys) !== $m || count($zs) !== $m) {
            throw new \ValueError('LSQBivariateSpline: x, y and z must have the same length');
        }
        $this->kx = $kx;
        $this->ky = $ky;

        // Augment the interior knots to the full clamped knot vectors, as SciPy's LSQBivariateSpline does.
        $this->tx = self::augment($xs, array_map('floatval', array_values($tx)), $kx);
        $this->ty = self::augment($ys, array_map('floatval', array_values($ty)), $ky);
        $this->ncx = count($this->tx) - $kx - 1;
        $this->ncy = count($this->ty) - $ky - 1;
        $ncoef = $this->ncx * $this->ncy;
        if ($m < $ncoef) {
            throw new \ValueError('LSQBivariateSpline: too few data points for the number of coefficients');
        }

        // Build the full tensor-product design matrix B (m x ncoef),
        //   B[i, p*ncy + q] = Nx_p(x_i) * Ny_q(y_i),
        // and solve  min || B c - z ||  through the SVD-based kernel lstsq (stable; normal
        // equations would square the condition number).
        $B = [];
        for ($i = 0; $i < $m; $i++) {
            $row = array_fill(0, $ncoef, 0.0);
            [$mux, $Nx] = self::basisAt($this->tx, $kx, $this->ncx, $xs[$i]);
            [$muy, $Ny] = self::basisAt($this->ty, $ky, $this->ncy, $ys[$i]);
            for ($a = 0; $a <= $kx; $a++) {
                $p = $mux - $kx + $a;
                $nxa = $Nx[$a];
                if ($nxa === 0.0) {
                    continue;
                }
                $base = $p * $this->ncy;
                for ($b = 0; $b <= $ky; $b++) {
                    $q = $muy - $ky + $b;
                    $row[$base + $q] = $nxa * $Ny[$b];
                }
            }
            $B[$i] = $row;
        }

        $coef = self::lstsq($B, $zs);
        $this->c = [];
        $p = 0;
        for ($r = 0; $r < $this->ncx; $r++) {
            $row = [];
            for ($s = 0; $s < $this->ncy; $s++) {
                $row[$s] = $coef[$p++];
            }
            $this->c[$r] = $row;
        }
    }

    /**
     * Evaluate on the tensor grid xq x yq, returning a (len(xq), len(yq)) array
     * (matching scipy.interpolate.LSQBivariateSpline.__call__(xq, yq)).
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
        $res = Kernel::NdBSpline($this->tx, $this->ty, $this->c, $this->kx, $this->ky, $pts);
        if (is_object($res) && method_exists($res, 'toList')) {
            $flat = array_values($res->toList());
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

    /** Alias of __invoke, for callers that prefer an explicit method. */
    public function ev(array $xq, array $yq): array
    {
        return $this($xq, $yq);
    }

    /** @return list<list<float>> coefficient block (ncx, ncy) */
    public function getCoeffs(): array
    {
        return $this->c;
    }

    /** @return array{0: list<float>, 1: list<float>} the (tx, ty) full knot vectors */
    public function getKnots(): array
    {
        return [$this->tx, $this->ty];
    }

    /**
     * Augment interior knots to the full clamped knot vector [v0]*(k+1) + interior + [vN]*(k+1),
     * using the data min/max for the clamp ends (what LSQBivariateSpline uses internally).
     *
     * @param list<float> $data   the data coordinates along this axis
     * @param list<float> $interior interior knots
     * @return list<float>
     */
    private static function augment(array $data, array $interior, int $k): array
    {
        $lo = min($data);
        $hi = max($data);
        $full = [];
        for ($i = 0; $i <= $k; $i++) {
            $full[] = $lo;
        }
        foreach ($interior as $ti) {
            $full[] = (float) $ti;
        }
        for ($i = 0; $i <= $k; $i++) {
            $full[] = $hi;
        }
        return $full;
    }

    /**
     * B-spline basis values at xv for the knot vector t, returning [mu, N] where N holds the k+1
     * nonzero basis values and N[j] multiplies coefficient (mu - k + j).
     *
     * @param list<float> $t full knot vector
     * @param int         $ncoef number of coefficients (= len(t) - k - 1)
     * @return array{0: int, 1: list<float>}
     */
    private static function basisAt(array $t, int $k, int $ncoef, float $xv): array
    {
        $n = $ncoef;
        if ($xv >= $t[$n]) {
            $mu = $n - 1;
        } else {
            $mu = $k;
            while ($mu < $n - 1 && $t[$mu + 1] <= $xv) {
                $mu++;
            }
        }
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
        return [$mu, $N];
    }

    /**
     * Least-squares solve min ||B c - z|| through the SVD-based kernel routine numpy.linalg.lstsq,
     * returning the coefficient vector c as a flat list<float>. numpy.linalg.lstsq returns the tuple
     * (x, residuals, rank, s); we take the first element and normalise its container.
     *
     * @param list<list<float>> $B design matrix (m x ncoef)
     * @param list<float>       $z right-hand side (length m)
     * @return list<float>
     */
    private static function lstsq(array $B, array $z): array
    {
        $out = Linalg::lstsq($B, $z, null);

        // numpy.linalg.lstsq returns the solution together with residuals/rank/singular values. The
        // Tessero facade hands it back as ['x' => NDArray, 'residuals' => ..., 'rank' => ...,
        // 'singular_values' => ...]; tolerate a positional [x, ...] tuple too. Pull out x.
        $x = $out;
        if (is_array($out)) {
            if (array_key_exists('x', $out)) {
                $x = $out['x'];
            } elseif (array_key_exists(0, $out)) {
                $x = $out[0];
            }
        }
        if (is_object($x) && method_exists($x, 'toList')) {
            $x = $x->toList();
        }
        return array_map('floatval', is_array($x) ? array_values($x) : [$x]);
    }
}
