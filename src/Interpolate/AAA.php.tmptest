<?php

declare(strict_types=1);

namespace Tessero\Interpolate;

use Tessero\Linalg\Linalg;

/**
 * AAA real rational approximation (scipy.interpolate.AAA): the adaptive Antoulas-Anderson algorithm
 * builds a barycentric rational interpolant r(z) = N(z)/D(z) of sample data (x_i, y_i) by a greedy
 * iteration. Each step picks the next support point as the sample where the current approximant has
 * the largest residual, appends a column to the Loewner matrix, and solves for the barycentric weights
 * as the trailing right-singular vector of the (masked) Loewner matrix via SVD; it stops when the
 * max residual falls below rtol * max|y| (default rtol = eps**0.75) or max_terms support points are used.
 *
 * Faithful, backend-neutral pure-PHP port of SciPy's _bary_rational.py AAA._compute_weights plus the
 * barycentric evaluation __call__. The weight vector is the right-singular vector of the smallest
 * singular value; it is unique only up to sign, but the evaluated value N/D is sign-invariant, so the
 * approximant reproduces SciPy to machine precision regardless of the LAPACK SVD sign convention. SVD
 * is taken through the shared Tessero\Linalg\Linalg::svd kernel (LAPACKE dgesdd, identical on both
 * backends and to scipy.linalg.svd). Verified machine-precision vs SciPy 1.17.1 (VERIFY.py).
 *
 * Scope: real-valued data in the well-conditioned regime SciPy's own examples use (smooth functions,
 * default rtol), where the smallest singular value is simple and no Froissart-doublet clean-up is
 * triggered -- matching the campaign's documented general-position scoping. The clean_up (Froissart
 * doublet removal) and complex-data paths are out of scope here; the default clean_up is a no-op on
 * such data, so AAA(x, y) still reproduces SciPy exactly.
 */
final class AAA
{
    /** @var list<float> selected support points z_j */
    public array $supportPoints = [];

    /** @var list<float> data values f_j at the support points */
    public array $supportValues = [];

    /** @var list<float> barycentric weights w_j */
    public array $weights = [];

    /** @var list<float> per-iteration max-residual history */
    public array $errors = [];

    /**
     * @param list<float|int> $x sample abscissae (distinct real points)
     * @param list<float|int> $y sample values y_i = f(x_i)
     * @param float|null      $rtol relative tolerance; default null -> eps**0.75
     * @param int             $maxTerms maximum number of support points (SciPy default 100)
     */
    public function __construct(array $x, array $y, ?float $rtol = null, int $maxTerms = 100)
    {
        $z = array_map('floatval', array_values($x));
        $f = array_map('floatval', array_values($y));
        $M = count($z);
        if ($M === 0 || count($f) !== $M) {
            throw new \ValueError('AAA: x and y must be non-empty and of equal length');
        }

        $eps = 2.220446049250313e-16;
        $rtol ??= $eps ** 0.75;
        $atol = $rtol * self::normInf($f);

        $mask = array_fill(0, $M, true);           // samples not yet chosen as support points
        $mean = array_sum($f) / $M;
        $R = array_fill(0, $M, $mean);             // current approximant values at all samples

        /** @var list<list<float>> Cauchy matrix columns: C[col] is a length-M column vector */
        $C = [];
        /** @var list<list<float>> Loewner matrix columns: A[col] is a length-M column vector */
        $A = [];
        $zj = [];
        $fj = [];
        $wj = [];

        $m = 0;
        for ($m = 0; $m < $maxTerms; $m++) {
            // 1) greedy: pick the masked sample with the largest |f - R| (first max, as np.argmax)
            $best = -1.0;
            $jjGlobal = -1;
            for ($i = 0; $i < $M; $i++) {
                if (! $mask[$i]) {
                    continue;
                }
                $r = abs($f[$i] - $R[$i]);
                if ($r > $best) {
                    $best = $r;
                    $jjGlobal = $i;
                }
            }
            $zj[$m] = $z[$jjGlobal];
            $fj[$m] = $f[$jjGlobal];

            // 2) next Cauchy column C[:, m] = 1 / (z - zj[m]); inf at the support row itself
            $col = [];
            for ($i = 0; $i < $M; $i++) {
                $den = $z[$i] - $zj[$m];
                $col[$i] = $den === 0.0 ? INF : 1.0 / $den;
            }
            $C[$m] = $col;

            // remove the chosen sample from the mask
            $mask[$jjGlobal] = false;

            // 3) Loewner column A[:, m] = (f - fj[m]) * C[:, m]
            $acol = [];
            for ($i = 0; $i < $M; $i++) {
                $acol[$i] = ($f[$i] - $fj[$m]) * $col[$i];
            }
            $A[$m] = $acol;

            // 4) solve for weights as the trailing right-singular vector of the masked Loewner matrix
            $rows = 0;
            for ($i = 0; $i < $M; $i++) {
                if ($mask[$i]) {
                    $rows++;
                }
            }
            // assemble the masked submatrix A[mask, 0..m] (rows x (m+1))
            $sub = [];
            for ($i = 0; $i < $M; $i++) {
                if (! $mask[$i]) {
                    continue;
                }
                $r = [];
                for ($cc = 0; $cc <= $m; $cc++) {
                    $r[] = $A[$cc][$i];
                }
                $sub[] = $r;
            }

            if ($rows >= $m + 1) {
                // tall-skinny: full_matrices=False SVD; weight = conj(V) row(s) of the min singular value
                [$u, $s, $vt] = Linalg::svd($sub, false);
                $sv = self::toList($s);
                $vtArr = self::toMatrix($vt);              // shape (m+1, m+1), rows are right-singular vectors
                $smin = min($sv);
                // rows with s == min(s); for well-conditioned data this is a single simple row
                $mmRows = [];
                foreach ($sv as $idx => $val) {
                    if ($val === $smin) {
                        $mmRows[] = $idx;
                    }
                }
                $mm = count($mmRows);
                $wj = array_fill(0, $m + 1, 0.0);
                foreach ($mmRows as $idx) {
                    for ($cc = 0; $cc <= $m; $cc++) {
                        $wj[$cc] += $vtArr[$idx][$cc];
                    }
                }
                $inv = 1.0 / sqrt((float) $mm);
                for ($cc = 0; $cc <= $m; $cc++) {
                    $wj[$cc] *= $inv;
                }
            } else {
                // fewer rows than columns: null space basis summed (SciPy's rare branch)
                $wj = self::nullSpaceSum($sub, $m + 1);
            }

            // 5) rational approximant at all samples: R = N / D, interpolating at support rows
            $i0 = [];
            for ($cc = 0; $cc <= $m; $cc++) {
                if ($wj[$cc] !== 0.0) {
                    $i0[] = $cc;
                }
            }
            $N = array_fill(0, $M, 0.0);
            $D = array_fill(0, $M, 0.0);
            foreach ($i0 as $cc) {
                $wf = $wj[$cc] * $fj[$cc];
                $w = $wj[$cc];
                $Ccc = $C[$cc];
                for ($i = 0; $i < $M; $i++) {
                    $N[$i] += $Ccc[$i] * $wf;
                    $D[$i] += $Ccc[$i] * $w;
                }
            }
            for ($i = 0; $i < $M; $i++) {
                if (is_infinite($D[$i]) || is_nan($D[$i])) {
                    $D[$i] = 1.0;
                    $N[$i] = $f[$i];
                }
                $R[$i] = $N[$i] / $D[$i];
            }

            // 6) convergence
            $maxErr = 0.0;
            for ($i = 0; $i < $M; $i++) {
                $e = abs($f[$i] - $R[$i]);
                if ($e > $maxErr) {
                    $maxErr = $e;
                }
            }
            $this->errors[$m] = $maxErr;
            if ($maxErr <= $atol) {
                break;
            }
        }
        if ($m >= $maxTerms) {
            $m = $maxTerms - 1;
        }

        // drop support points with zero weight
        $zjOut = [];
        $fjOut = [];
        $wjOut = [];
        for ($cc = 0; $cc <= $m; $cc++) {
            if ($wj[$cc] !== 0.0) {
                $zjOut[] = $zj[$cc];
                $fjOut[] = $fj[$cc];
                $wjOut[] = $wj[$cc];
            }
        }
        $this->supportPoints = $zjOut;
        $this->supportValues = $fjOut;
        $this->weights = $wjOut;
    }

    /** max |v| over the list */
    private static function normInf(array $v): float
    {
        $m = 0.0;
        foreach ($v as $x) {
            $a = abs((float) $x);
            if ($a > $m) {
                $m = $a;
            }
        }
        return $m;
    }

    /** normalise an NDArray|array result to a flat list<float> */
    private static function toList(mixed $x): array
    {
        $out = is_object($x) && method_exists($x, 'toArray') ? $x->toArray() : $x;
        $flat = [];
        $arr = is_array($out) ? $out : [$out];
        array_walk_recursive($arr, static function ($v) use (&$flat): void {
            $flat[] = (float) $v;
        });
        return $flat;
    }

    /** normalise an NDArray|array 2-D result to list<list<float>> */
    private static function toMatrix(mixed $x): array
    {
        $out = is_object($x) && method_exists($x, 'toArray') ? $x->toArray() : $x;
        $rows = [];
        foreach ($out as $r) {
            $row = [];
            $rv = is_array($r) ? $r : [$r];
            array_walk_recursive($rv, static function ($v) use (&$row): void {
                $row[] = (float) $v;
            });
            $rows[] = $row;
        }
        return $rows;
    }

    /**
     * Sum of a null-space basis (right-singular vectors with zero singular value) of the (rows < cols)
     * Loewner submatrix, divided by sqrt of the basis dimension -- SciPy's scipy.linalg.null_space branch.
     *
     * @param list<list<float>> $sub masked Loewner submatrix (rows x ncols), rows < ncols
     */
    private static function nullSpaceSum(array $sub, int $ncols): array
    {
        [$u, $s, $vt] = Linalg::svd($sub, true); // full_matrices=True so Vt is ncols x ncols
        $sv = self::toList($s);
        $vtArr = self::toMatrix($vt);
        $rows = count($sub);
        $eps = 2.220446049250313e-16;
        $smax = $sv === [] ? 0.0 : max($sv);
        $tol = $smax * max($rows, $ncols) * $eps;
        // null space = rows of Vt whose singular value is below tol (indices >= rank)
        $basis = [];
        for ($idx = 0; $idx < $ncols; $idx++) {
            $sval = $idx < count($sv) ? $sv[$idx] : 0.0;
            if ($sval <= $tol) {
                $basis[] = $vtArr[$idx];
            }
        }
        $nm = count($basis);
        $wj = array_fill(0, $ncols, 0.0);
        foreach ($basis as $b) {
            for ($cc = 0; $cc < $ncols; $cc++) {
                $wj[$cc] += $b[$cc];
            }
        }
        $inv = $nm > 0 ? 1.0 / sqrt((float) $nm) : 1.0;
        for ($cc = 0; $cc < $ncols; $cc++) {
            $wj[$cc] *= $inv;
        }
        return $wj;
    }

    /**
     * Evaluate the barycentric rational approximant at the query point(s).
     *
     * @param list<float|int> $z query points
     * @return list<float> approximant value at each query point
     */
    public function query(array $z): array
    {
        $Z = array_map('floatval', array_values($z));
        $zj = $this->supportPoints;
        $fj = $this->supportValues;
        $wj = $this->weights;
        $m = count($zj);
        $out = [];
        foreach ($Z as $zi) {
            // exact hit at a support point -> return its value
            $hit = null;
            for ($j = 0; $j < $m; $j++) {
                if ($zi === $zj[$j]) {
                    $hit = $fj[$j];
                    break;
                }
            }
            if ($hit !== null) {
                $out[] = $hit;
                continue;
            }
            $num = 0.0;
            $den = 0.0;
            for ($j = 0; $j < $m; $j++) {
                $cc = 1.0 / ($zi - $zj[$j]);
                $num += $cc * $wj[$j] * $fj[$j];
                $den += $cc * $wj[$j];
            }
            $out[] = $num / $den;
        }
        return $out;
    }

    /**
     * Callable form, matching scipy's AAA(z).
     *
     * @param list<float|int> $z
     * @return list<float>
     */
    public function __invoke(array $z): array
    {
        return $this->query($z);
    }
}
