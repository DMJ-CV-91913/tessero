<?php

declare(strict_types=1);

namespace Tessero\Interpolate;

use Tessero\Interpolate as InterpolateFacade;
use Tessero\Optimize\Minimize;
use Tessero\ScipyLinalg;

/**
 * Penalised (smoothing) cubic B-spline (scipy.interpolate.make_smoothing_spline). Fits the natural cubic
 * smoothing spline of Pollock / de Hoog & Hutchinson: it minimises
 *
 *     sum_i w_i (y_i - f(x_i))^2  +  lam * integral f''(u)^2 du
 *
 * over natural cubic splines whose knots are exactly the (triple-clamped) data abscissae. This is a pure-PHP
 * facade: it builds the banded (2,2) collocation+penalty system natively (de Boor-Cox basis + the divided-
 * difference penalty operator E) and reuses the EXISTING Tessero kernel banded LAPACK primitives
 * (slinalg.solve_banded for the final solve, slinalg.cholesky_banded for the GCV trace), plus
 * Tessero\Optimize\Minimize::fminbound for the lam=None generalised-cross-validation search and
 * Tessero\Interpolate::BSpline (de Boor) for evaluation. No new kernel code; backend-neutral (FFI and ext
 * share one path, since all three prims are kernel-shared).
 *
 * This mirrors SciPy 1.17's implementation routine-for-routine (banded design matrix X, weighted penalty wE,
 * normal-equation bands X^T W X and X^T W E for GCV, de Hoog banded-inverse trace, fminbound at xtol=1e-5
 * on [0, n]); verified max abs err ~1e-9 on query-point evaluation against scipy.interpolate.make_smoothing_spline.
 *
 * API mirrors SciPy: construct with (x, y, w=null, lam=null); lam=null triggers the GCV search, otherwise the
 * supplied lam is used directly. x must be strictly increasing with at least 5 points (SciPy's minimum for the
 * natural-spline penalty). Evaluate with query()/__invoke(); tck() returns the clamped (t, c, k=3) triple.
 */
final class MakeSmoothingSpline
{
    /** @var list<float> clamped knot vector (triple-repeated end knots + interior = x) */
    private array $t;

    /** @var list<float> B-spline coefficients */
    private array $c;

    private int $k = 3;

    /**
     * @param list<float|int>      $x strictly-increasing data abscissae (n >= 5)
     * @param list<float|int>      $y data ordinates, one per x
     * @param list<float|int>|null $w per-point weights (default: all ones)
     * @param float|null           $lam smoothing parameter; null => choose by GCV (fminbound on [0, n])
     */
    public function __construct(array $x, array $y, ?array $w = null, ?float $lam = null)
    {
        $xv = array_map('floatval', array_values($x));
        $yv = array_map('floatval', array_values($y));
        $n = count($xv);
        if ($n !== count($yv)) {
            throw new \ValueError('make_smoothing_spline: x and y must have the same length');
        }
        if ($n < 5) {
            throw new \ValueError('make_smoothing_spline: need at least 5 data points for the natural cubic penalty');
        }
        for ($i = 1; $i < $n; $i++) {
            if ($xv[$i] <= $xv[$i - 1]) {
                throw new \ValueError('make_smoothing_spline: x must be strictly increasing');
            }
        }
        if ($w === null) {
            $wv = array_fill(0, $n, 1.0);
        } else {
            $wv = array_map('floatval', array_values($w));
            if (count($wv) !== $n) {
                throw new \ValueError('make_smoothing_spline: w must have the same length as x');
            }
        }
        if ($lam !== null && $lam < 0.0) {
            throw new \ValueError('make_smoothing_spline: lam must be non-negative');
        }

        // Clamped knot vector: t = [x0]*3 + x + [xN]*3  (triple-clamped ends, interior = data abscissae).
        $t = [];
        for ($i = 0; $i < 3; $i++) { $t[] = $xv[0]; }
        foreach ($xv as $xi) { $t[] = $xi; }
        for ($i = 0; $i < 3; $i++) { $t[] = $xv[$n - 1]; }

        [$X, $wE] = $this->buildXwE($xv, $wv, $t, $n);   // both (5 x n) banded (2,2) operators

        if ($lam === null) {
            $XtWX = $this->bdedXtWY($X, $wv, $X);        // (4 x n) upper symmetric band, kd=2
            $XtE  = $this->bdedXtWY($X, $wv, $wE);
            $self = $this;
            $func = static function (float $l) use ($self, $X, $XtWX, $wE, $XtE, $yv): float {
                return $self->gcv($l, $X, $XtWX, $wE, $XtE, $yv);
            };
            $lam = Minimize::fminbound($func, 0.0, (float) $n, 1e-5);
        }

        // Final solve of the banded (2,2) system (X + lam*wE) c = y.
        $AB = $this->addScaled($X, $wE, $lam);           // (5 x n)
        $cSol = $this->toFlat(ScipyLinalg::solveBanded(2, 2, $AB, $yv));

        // Reconstruct the full (n+2) coefficient vector for the clamped BSpline, exactly as SciPy does.
        $c0 = $cSol[0]; $c1 = $cSol[1];
        $cm0 = $cSol[$n - 1]; $cm1 = $cSol[$n - 2];
        $nt = count($t);
        $cFull = [];
        $cFull[] = $c0 * ($t[5] + $t[4] - 2.0 * $t[3]) + $c1;
        $cFull[] = $c0 * ($t[5] - $t[3]) + $c1;
        for ($i = 1; $i < $n - 1; $i++) { $cFull[] = $cSol[$i]; }
        $cFull[] = $cm0 * ($t[$nt - 4] - $t[$nt - 6]) + $cm1;
        $cFull[] = $cm0 * (2.0 * $t[$nt - 4] - $t[$nt - 5] - $t[$nt - 6]) + $cm1;

        $this->t = $t;
        $this->c = $cFull;
    }

    /**
     * Reciprocal of the product of pairwise differences (the divided-difference weights 1/prod_{k!=i}(x_i-x_k)).
     *
     * @param list<float> $x
     * @return list<float>
     */
    private function coeffDd(array $x): array
    {
        $m = count($x);
        $res = array_fill(0, $m, 0.0);
        for ($i = 0; $i < $m; $i++) {
            $pp = 1.0;
            for ($kk = 0; $kk < $m; $kk++) {
                if ($kk !== $i) { $pp *= ($x[$i] - $x[$kk]); }
            }
            $res[$i] = 1.0 / $pp;
        }
        return $res;
    }

    /**
     * Cubic B-spline design matrix (n x (n+2)) via de Boor-Cox (P&T A2.2), dense.
     *
     * @param list<float> $x
     * @param list<float> $t
     * @return list<list<float>>
     */
    private function designMatrix(array $x, array $t, int $n): array
    {
        $k = 3;
        $nc = $n + 2;
        $Xb = [];
        for ($i = 0; $i < $n; $i++) { $Xb[$i] = array_fill(0, $nc, 0.0); }
        for ($i = 0; $i < $n; $i++) {
            $xi = $x[$i];
            $mu = $k;
            while ($mu < $nc - 1 && $t[$mu + 1] <= $xi) { $mu++; }
            $N = array_fill(0, $k + 1, 0.0);
            $left = array_fill(0, $k + 1, 0.0);
            $right = array_fill(0, $k + 1, 0.0);
            $N[0] = 1.0;
            for ($rr = 1; $rr <= $k; $rr++) {
                $left[$rr] = $xi - $t[$mu + 1 - $rr];
                $right[$rr] = $t[$mu + $rr] - $xi;
                $saved = 0.0;
                for ($ii = 0; $ii < $rr; $ii++) {
                    $denom = $right[$ii + 1] + $left[$rr - $ii];
                    $temp = $denom !== 0.0 ? $N[$ii] / $denom : 0.0;
                    $N[$ii] = $saved + $right[$ii + 1] * $temp;
                    $saved = $left[$rr - $ii] * $temp;
                }
                $N[$rr] = $saved;
            }
            for ($rr = 0; $rr <= $k; $rr++) {
                $Xb[$i][$mu - $k + $rr] = $N[$rr];
            }
        }
        return $Xb;
    }

    /**
     * Build the banded (5 x n) design operator X and weighted penalty operator wE, exactly as SciPy's
     * _compute_optimal_gcv_parameter/make_smoothing_spline do. Both are (l=2, u=2) banded, stored with the
     * diagonal in row 2, super-diagonals in rows 1,0 and sub-diagonals in rows 3,4 (scipy solve_banded form).
     *
     * @param list<float> $x
     * @param list<float> $w
     * @param list<float> $t
     * @return array{0: list<list<float>>, 1: list<list<float>>}
     */
    private function buildXwE(array $x, array $w, array $t, int $n): array
    {
        $Xb = $this->designMatrix($x, $t, $n);

        $X = [];
        for ($r = 0; $r < 5; $r++) { $X[$r] = array_fill(0, $n, 0.0); }

        // X[1, 2:-2], X[2, 2:-2], X[3, 2:-2] = interior diagonal bands of the dense design matrix.
        //   Python: X[i, 2:-2] = Xb[i:i-4, 3:-3] diag,  i in 1..3
        // For band row i and column index j in [2, n-3], the contribution is Xb[row, col] with
        //   row = (i) + (j-2),  col = 3 + (j-2),  reproduced directly from the numpy slicing below.
        for ($i = 1; $i <= 3; $i++) {
            // Xb[i : i-4] has (n-4) rows (since i-4 is negative, Python slice i..n-4+i). Diagonal of the
            // (n-4)x(n-4) block Xb[i:i-4, 3:-3] -> entries Xb[i+d, 3+d], d=0..n-5, placed at X[i, 2+d].
            for ($d = 0; $d < $n - 4; $d++) {
                $X[$i][2 + $d] = $Xb[$i + $d][3 + $d];
            }
        }
        $X[1][1] = $Xb[0][0];
        $X[2][0] = ($x[2] + $x[1] - 2.0 * $x[0]) * $Xb[0][0];
        $X[2][1] = $Xb[1][1] + $Xb[1][2];
        $X[3][0] = ($x[2] - $x[0]) * $Xb[1][1];
        $X[3][1] = $Xb[2][2];
        $X[1][$n - 2] = $Xb[$n - 3][$n - 1];                               // Xb[-3,-3]
        $X[1][$n - 1] = ($x[$n - 1] - $x[$n - 3]) * $Xb[$n - 2][$n];       // Xb[-2,-2]
        $X[2][$n - 2] = $Xb[$n - 2][$n - 1] + $Xb[$n - 2][$n];             // Xb[-2,-3]+Xb[-2,-2]
        $X[2][$n - 1] = (2.0 * $x[$n - 1] - $x[$n - 2] - $x[$n - 3]) * $Xb[$n - 1][$n + 1]; // *Xb[-1,-1]
        $X[3][$n - 2] = $Xb[$n - 1][$n + 1];                               // Xb[-1,-1]

        // Penalty operator wE (6 * divided-difference second-derivative jumps, weighted by 1/w).
        $wE = [];
        for ($r = 0; $r < 5; $r++) { $wE[$r] = array_fill(0, $n, 0.0); }

        // wE[2:, 0] = coeff_dd(x[:3]) / w[:3]
        $dd = $this->coeffDd(array_slice($x, 0, 3));
        for ($r = 0; $r < 3; $r++) { $wE[2 + $r][0] = $dd[$r] / $w[$r]; }
        // wE[1:, 1] = coeff_dd(x[:4]) / w[:4]
        $dd = $this->coeffDd(array_slice($x, 0, 4));
        for ($r = 0; $r < 4; $r++) { $wE[1 + $r][1] = $dd[$r] / $w[$r]; }
        // wE[:, j] = (x[j+2]-x[j-2]) * coeff_dd(x[j-2:j+3]) / w[j-2:j+3],  j in 2..n-3
        for ($j = 2; $j < $n - 2; $j++) {
            $dd = $this->coeffDd(array_slice($x, $j - 2, 5));
            $scale = $x[$j + 2] - $x[$j - 2];
            for ($r = 0; $r < 5; $r++) {
                $wE[$r][$j] = $scale * $dd[$r] / $w[$j - 2 + $r];
            }
        }
        // wE[:-1, -2] = -coeff_dd(x[-4:]) / w[-4:]   (rows 0..3, column n-2)
        $dd = $this->coeffDd(array_slice($x, $n - 4, 4));
        for ($r = 0; $r < 4; $r++) { $wE[$r][$n - 2] = -$dd[$r] / $w[$n - 4 + $r]; }
        // wE[:-2, -1] = coeff_dd(x[-3:]) / w[-3:]    (rows 0..2, column n-1)
        $dd = $this->coeffDd(array_slice($x, $n - 3, 3));
        for ($r = 0; $r < 3; $r++) { $wE[$r][$n - 1] = $dd[$r] / $w[$n - 3 + $r]; }

        for ($r = 0; $r < 5; $r++) {
            for ($j = 0; $j < $n; $j++) { $wE[$r][$j] *= 6.0; }
        }

        return [$X, $wE];
    }

    /**
     * Banded X^T W Y collapse to a (4 x n) upper symmetric band (kd=2), as SciPy's _banded_matmul does.
     * X and Y are (5 x n) solve_banded-form bands; W is a per-column diagonal weight vector.
     *
     * @param list<list<float>> $X  (5 x n)
     * @param list<float>       $w  (n)
     * @param list<list<float>> $Y  (5 x n)
     * @return list<list<float>> (4 x n) upper band for cholesky_banded (row 3 = diagonal)
     */
    private function bdedXtWY(array $X, array $w, array $Y): array
    {
        $n = count($w);
        // W_Y = copy(Y); W_Y[2] *= w; then the off-diagonal weighting below.
        $WY = [];
        for ($r = 0; $r < 5; $r++) { $WY[$r] = $Y[$r]; }   // deep-copy per row (value arrays)
        for ($j = 0; $j < $n; $j++) { $WY[2][$j] *= $w[$j]; }
        // for i in range(2): W_Y[i, 2-i:] *= w[:-2+i];  W_Y[3+i, :-1-i] *= w[1+i:]
        for ($i = 0; $i < 2; $i++) {
            // W_Y[i, 2-i:] *= w[0 : n-2+i]
            for ($p = 0; $p < $n - 2 + $i; $p++) {
                $WY[$i][2 - $i + $p] *= $w[$p];
            }
            // W_Y[3+i, :-1-i] *= w[1+i:]   -> columns 0 .. n-2-i, weight w[1+i + col]
            for ($p = 0; $p < $n - 1 - $i; $p++) {
                $WY[3 + $i][$p] *= $w[1 + $i + $p];
            }
        }
        $res = [];
        for ($r = 0; $r < 4; $r++) { $res[$r] = array_fill(0, $n, 0.0); }
        // for i in range(n): for j in range(min(n-i,4)): res[-j-1, i+j] = sum(X[j:, i] * W_Y[:5-j, i+j])
        for ($i = 0; $i < $n; $i++) {
            $jmax = min($n - $i, 4);
            for ($j = 0; $j < $jmax; $j++) {
                $s = 0.0;
                for ($p = 0; $p < 5 - $j; $p++) {
                    $s += $X[$j + $p][$i] * $WY[$p][$i + $j];
                }
                $res[3 - $j][$i + $j] = $s;
            }
        }
        return $res;
    }

    /**
     * de Hoog banded inverse: given a (4 x n) SPD upper band A (kd=2), return the (4 x n) band of A^{-1}
     * restricted to the bandwidth, via the Cholesky factor (cholesky_banded) and the Hutchinson recursion.
     * Pure arithmetic on the kernel-produced Cholesky factor.
     *
     * @param list<list<float>> $A (4 x n) upper band, row 3 = diagonal
     * @return list<list<float>> (4 x n) band of the inverse
     */
    private function computeBInv(array $A): array
    {
        $U = $this->toBand(ScipyLinalg::choleskyBanded($A, false));   // (4 x n) upper Cholesky factor
        $n = count($U[0]);
        // for i in 2..4: U[-i, i-1:] /= U[-1, :n-i+1]    (U[-1] is row 3)
        for ($i = 2; $i <= 4; $i++) {
            $row = 4 - $i;                 // U[-i]
            for ($c = $i - 1; $c < $n; $c++) {
                $U[$row][$c] /= $U[3][$c - ($i - 1)];
            }
        }
        $D = [];
        for ($c = 0; $c < $n; $c++) { $D[$c] = 1.0 / ($U[3][$c] * $U[3][$c]); }
        for ($c = 0; $c < $n; $c++) { $U[3][$c] = 1.0; }

        $B = [];
        for ($r = 0; $r < 4; $r++) { $B[$r] = array_fill(0, $n, 0.0); }

        // elem(i,j): fills B[-j-1, i+j] from already-computed entries (back-substitution recursion).
        $elem = function (int $i, int $j) use (&$B, $U, $D, $n): void {
            $rng = min(3, $n - $i - 1);
            $sum = 0.0;
            if ($j === 0) {
                for ($kk = 1; $kk <= $rng; $kk++) {
                    // U[-k-1, i+k] * B[-k-1, i+k]
                    $sum -= $U[3 - $kk][$i + $kk] * $B[3 - $kk][$i + $kk];
                }
                $sum += $D[$i];
                $B[3][$i] = $sum;           // B[-1, i]
            } else {
                for ($kk = 1; $kk <= $rng; $kk++) {
                    $diag = abs($kk - $j);
                    $ind = $i + min($kk, $j);
                    // U[-k-1, i+k] * B[-diag-1, ind+diag]
                    $sum -= $U[3 - $kk][$i + $kk] * $B[3 - $diag][$ind + $diag];
                }
                $B[3 - $j][$i + $j] = $sum;  // B[-j-1, i+j]
            }
        };
        for ($i = $n - 1; $i >= 0; $i--) {
            $jhi = min(3, $n - $i - 1);
            for ($j = $jhi; $j >= 0; $j--) { $elem($i, $j); }
        }
        for ($c = 0; $c < $n; $c++) { $B[0][$c] = 0.0; }   // B[0] = 0.0
        return $B;
    }

    /**
     * Generalised cross-validation objective at smoothing parameter lam (SciPy's _gcv).
     *
     * @param list<list<float>> $X    (5 x n)
     * @param list<list<float>> $XtWX (4 x n)
     * @param list<list<float>> $wE   (5 x n)
     * @param list<list<float>> $XtE  (4 x n)
     * @param list<float>       $y    (n)
     */
    public function gcv(float $lam, array $X, array $XtWX, array $wE, array $XtE, array $y): float
    {
        $n = count($y);
        $AB = $this->addScaled($X, $wE, $lam);                 // (5 x n)
        $c = $this->toFlat(ScipyLinalg::solveBanded(2, 2, $AB, $y));

        // res[i] = sum over band rows of (wE * c) aligned as in SciPy's loop.
        $res = array_fill(0, $n, 0.0);
        for ($i = 0; $i < $n; $i++) {
            $jlo = max(0, $i - $n + 3);
            $jhi = min(5, $i + 3);
            for ($j = $jlo; $j < $jhi; $j++) {
                // tmp = wE * c ; res[i] += tmp[j, i+2-j]
                $col = $i + 2 - $j;
                $res[$i] += $wE[$j][$col] * $c[$col];
            }
        }
        $sq = 0.0;
        for ($i = 0; $i < $n; $i++) {
            $v = $lam * $res[$i];
            $sq += $v * $v;
        }
        $numer = $sq / $n;

        $lhs = $this->addScaled4($XtWX, $XtE, $lam);           // (4 x n)
        $b = $this->computeBInv($lhs);
        // tr = b * XtWX; tr[:-1] *= 2; trace = sum(tr)
        $trace = 0.0;
        for ($r = 0; $r < 4; $r++) {
            $mult = ($r < 3) ? 2.0 : 1.0;                      // tr[:-1] (rows 0..2) doubled
            for ($cc = 0; $cc < $n; $cc++) {
                $trace += $mult * $b[$r][$cc] * $XtWX[$r][$cc];
            }
        }
        $d = 1.0 - $trace / $n;
        $denom = $d * $d;
        return $numer / $denom;
    }

    /**
     * X + lam*Y for two (5 x n) bands.
     *
     * @param list<list<float>> $X
     * @param list<list<float>> $Y
     * @return list<list<float>>
     */
    private function addScaled(array $X, array $Y, float $lam): array
    {
        $out = [];
        $n = count($X[0]);
        for ($r = 0; $r < 5; $r++) {
            $out[$r] = array_fill(0, $n, 0.0);
            for ($c = 0; $c < $n; $c++) { $out[$r][$c] = $X[$r][$c] + $lam * $Y[$r][$c]; }
        }
        return $out;
    }

    /**
     * A + lam*B for two (4 x n) bands.
     *
     * @param list<list<float>> $A
     * @param list<list<float>> $B
     * @return list<list<float>>
     */
    private function addScaled4(array $A, array $B, float $lam): array
    {
        $out = [];
        $n = count($A[0]);
        for ($r = 0; $r < 4; $r++) {
            $out[$r] = array_fill(0, $n, 0.0);
            for ($c = 0; $c < $n; $c++) { $out[$r][$c] = $A[$r][$c] + $lam * $B[$r][$c]; }
        }
        return $out;
    }

    /**
     * Normalise a kernel banded-factor return (NDArray or nested array) to a list<list<float>> of rows.
     *
     * @return list<list<float>>
     */
    private function toBand(mixed $v): array
    {
        if (is_object($v) && method_exists($v, 'toArray')) {
            $v = $v->toArray();
        }
        $out = [];
        foreach ($v as $row) {
            $out[] = array_map('floatval', array_values(is_array($row) ? $row : [$row]));
        }
        return $out;
    }

    /**
     * Normalise a kernel 1-D return (NDArray or array) to a flat list<float>.
     *
     * @return list<float>
     */
    private function toFlat(mixed $v): array
    {
        if (is_object($v) && method_exists($v, 'toArray')) {
            $v = $v->toArray();
        }
        $flat = [];
        $arr = is_array($v) ? $v : [$v];
        array_walk_recursive($arr, static function ($x) use (&$flat): void { $flat[] = (float) $x; });
        return $flat;
    }

    /**
     * Evaluate the smoothing spline at the query point(s) (de Boor, via Tessero\Interpolate::BSpline).
     *
     * @param list<float|int> $xq query abscissae
     * @return list<float> spline value at each query point
     */
    public function query(array $xq): array
    {
        $q = array_map('floatval', array_values($xq));
        if (count($q) === 0) { return []; }
        return $this->toFlat(InterpolateFacade::BSpline($this->t, $this->c, $this->k, $q));
    }

    /**
     * The clamped (t, c, k) triple, mirroring the BSpline returned by scipy.interpolate.make_smoothing_spline.
     *
     * @return array{0: list<float>, 1: list<float>, 2: int}
     */
    public function tck(): array
    {
        return [$this->t, $this->c, $this->k];
    }

    /**
     * Callable form, matching scipy: make_smoothing_spline(x, y, w, lam)(xq).
     *
     * @param list<float|int> $xq
     * @return list<float>
     */
    public function __invoke(array $xq): array
    {
        return $this->query($xq);
    }
}
