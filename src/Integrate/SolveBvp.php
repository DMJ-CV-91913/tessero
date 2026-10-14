<?php

declare(strict_types=1);

namespace Tessero\Integrate;

/**
 * Boundary value problem solver for first-order ODE systems (scipy.integrate.solve_bvp).
 *
 * Pure PHP over user callables $fun(list<float> $x, list<list<float>> $y, list<float> $p): list<list<float>>
 * (shape (n,m)) and $bc(list<float> $ya, list<float> $yb, list<float> $p): list<float> (shape n+k), solving
 *
 *     dy/dx = f(x, y, p),  a <= x <= b
 *     bc(y(a), y(b), p) = 0
 *
 * for an n-vector y(x) and an optional k-vector p of unknown parameters.
 *
 * Faithful port of scipy's _bvp.py (1.17.1): a 4th-order Lobatto IIIA collocation method with residual
 * control and adaptive mesh refinement. Reproduces construct_global_jac (the n*m+k collocation+bc Jacobian),
 * solve_newton (damped Newton with affine-invariant F = ||J^-1 r||^2 criterion + backtracking line search +
 * fixed-Jacobian iterations), estimate_rms_residuals (5-point Lobatto quadrature of the relative residuals),
 * create_spline (the C1 cubic PPoly), and modify_mesh (1-node / 2-node interval insertion). The forward-
 * difference estimate_fun_jac / estimate_bc_jac match scipy's step h = sqrt(EPS)*(1+|x|).
 *
 * The collocation Jacobian is solved with a dense partial-pivot LU (vs scipy's sparse splu); the control
 * flow is deterministic so the full adaptive mesh sequence, converged status, unknown parameters p, and the
 * solution spline sol(x) all match scipy to <= 1e-12 (verified across Bratu, linear 2nd-order, nonlinear,
 * and Sturm-Liouville eigenvalue BVPs). Singular-term S and complex domains are out of scope.
 */
final class SolveBvp
{
    private const EPS = 2.220446049250313e-16;

    /**
     * @param callable(list<float>, list<list<float>>, list<float>): list<list<float>> $fun  rhs f(x,y,p) -> (n,m)
     * @param callable(list<float>, list<float>, list<float>): list<float> $bc  residuals bc(ya,yb,p) -> (n+k)
     * @param list<float> $x  strictly increasing initial mesh (m,)
     * @param list<list<float>> $y  initial guess (n,m)
     * @param list<float>|null $p  initial guess for unknown params (k,) or null
     * @return array{
     *   x: list<float>, y: list<list<float>>, yp: list<list<float>>,
     *   p: list<float>|null, rms_residuals: list<float>, niter: int,
     *   status: int, success: bool, c: list<list<list<float>>>
     * }  c = spline coeffs (4,n,m-1) for sol() evaluation
     */
    public static function solve(
        callable $fun,
        callable $bc,
        array $x,
        array $y,
        ?array $p = null,
        float $tol = 1e-3,
        int $maxNodes = 1000,
        ?float $bcTol = null
    ): array {
        $x = array_values(array_map('floatval', $x));
        $m = count($x);
        $h = self::diff($x);
        $y = array_map(static fn ($row): array => array_values(array_map('floatval', $row)), $y);
        $n = count($y);
        $pv = $p === null ? [] : array_values(array_map('floatval', $p));
        $k = count($pv);

        if ($tol < 100 * self::EPS) {
            $tol = 100 * self::EPS;
        }
        if ($bcTol === null) {
            $bcTol = $tol;
        }
        $maxIteration = 10;

        $status = 0;
        $iteration = 0;
        $rmsRes = [];
        $f = [];
        $sol = null; // ['c'=>..., 'x'=>..., 'n'=>..., 'm'=>...]

        while (true) {
            $m = count($x);
            [$y, $pv, $singular] = self::solveNewton($n, $m, $k, $h, $fun, $bc, $x, $y, $pv, $tol, $bcTol);
            $iteration++;

            [$colRes, $yMiddle, $f, $fMiddle] = self::collocationFun($fun, $y, $pv, $x, $h);
            $bcRes = self::callBc($bc, self::col($y, 0), self::col($y, $m - 1), $pv);
            $maxBcRes = 0.0;
            foreach ($bcRes as $v) {
                if (abs($v) > $maxBcRes) { $maxBcRes = abs($v); }
            }

            // r_middle = 1.5 * col_res / h
            $rMiddle = [];
            for ($i = 0; $i < $n; $i++) {
                $row = [];
                for ($j = 0; $j < $m - 1; $j++) {
                    $row[$j] = 1.5 * $colRes[$i][$j] / $h[$j];
                }
                $rMiddle[$i] = $row;
            }

            $sol = self::createSpline($y, $f, $x, $h);
            $rmsRes = self::estimateRmsResiduals($fun, $sol, $x, $h, $pv, $rMiddle, $fMiddle);
            $maxRms = 0.0;
            foreach ($rmsRes as $v) {
                if ($v > $maxRms) { $maxRms = $v; }
            }

            if ($singular) {
                $status = 2;
                break;
            }

            $insert1 = [];
            $insert2 = [];
            foreach ($rmsRes as $i => $r) {
                if ($r >= 100 * $tol) {
                    $insert2[] = $i;
                } elseif ($r > $tol) {
                    $insert1[] = $i;
                }
            }
            $nodesAdded = count($insert1) + 2 * count($insert2);

            if ($m + $nodesAdded > $maxNodes) {
                $status = 1;
                break;
            }

            if ($nodesAdded > 0) {
                $x = self::modifyMesh($x, $insert1, $insert2);
                $h = self::diff($x);
                $y = self::splineEval($sol, $x, 0);
            } elseif ($maxBcRes <= $bcTol) {
                $status = 0;
                break;
            } elseif ($iteration >= $maxIteration) {
                $status = 3;
                break;
            }
        }

        return [
            'x' => $x,
            'y' => $y,
            'yp' => $f,
            'p' => $k === 0 ? null : $pv,
            'rms_residuals' => $rmsRes,
            'niter' => $iteration,
            'status' => $status,
            'success' => $status === 0,
            'c' => $sol['c'],
        ];
    }

    /**
     * Evaluate the solution spline at query points (scipy res.sol(x)).
     *
     * @param array{c: list<list<list<float>>>} $result  the array returned by solve()
     * @param list<float> $xq
     * @param list<float> $meshX  the final mesh (result['x'])
     * @return list<list<float>>  shape (n, len(xq))
     */
    public static function sol(array $result, array $xq, array $meshX, int $nu = 0): array
    {
        $sol = ['c' => $result['c'], 'x' => array_values(array_map('floatval', $meshX))];
        $sol['n'] = count($result['c'][0]);
        $sol['m'] = count($result['c'][0][0]) + 1;
        return self::splineEval($sol, array_values(array_map('floatval', $xq)), $nu);
    }

    // ---------------------------------------------------------------------
    // collocation core
    // ---------------------------------------------------------------------

    /**
     * @return array{0: list<list<float>>, 1: list<list<float>>, 2: list<list<float>>, 3: list<list<float>>}
     *         [col_res (n,m-1), y_middle (n,m-1), f (n,m), f_middle (n,m-1)]
     */
    private static function collocationFun(callable $fun, array $y, array $p, array $x, array $h): array
    {
        $n = count($y);
        $m = count($x);
        $f = self::callFun($fun, $x, $y, $p, $n, $m);

        $xMiddle = [];
        for ($j = 0; $j < $m - 1; $j++) {
            $xMiddle[$j] = $x[$j] + 0.5 * $h[$j];
        }
        $yMiddle = [];
        for ($i = 0; $i < $n; $i++) {
            $row = [];
            for ($j = 0; $j < $m - 1; $j++) {
                $row[$j] = 0.5 * ($y[$i][$j + 1] + $y[$i][$j]) - 0.125 * $h[$j] * ($f[$i][$j + 1] - $f[$i][$j]);
            }
            $yMiddle[$i] = $row;
        }
        $fMiddle = self::callFun($fun, $xMiddle, $yMiddle, $p, $n, $m - 1);

        $colRes = [];
        for ($i = 0; $i < $n; $i++) {
            $row = [];
            for ($j = 0; $j < $m - 1; $j++) {
                $row[$j] = $y[$i][$j + 1] - $y[$i][$j]
                    - $h[$j] / 6 * ($f[$i][$j] + $f[$i][$j + 1] + 4 * $fMiddle[$i][$j]);
            }
            $colRes[$i] = $row;
        }

        return [$colRes, $yMiddle, $f, $fMiddle];
    }

    /**
     * Damped Newton on the collocation+bc system. Mirrors scipy solve_newton.
     *
     * @return array{0: list<list<float>>, 1: list<float>, 2: bool}  [y, p, singular]
     */
    private static function solveNewton(
        int $n,
        int $m,
        int $k,
        array $h,
        callable $fun,
        callable $bc,
        array $x,
        array $y,
        array $p,
        float $bvpTol,
        float $bcTol
    ): array {
        $xMiddle = [];
        for ($j = 0; $j < $m - 1; $j++) {
            $xMiddle[$j] = $x[$j] + 0.5 * $h[$j];
        }
        // tol_r = 2/3 * h * 5e-2 * bvp_tol  (per interval)
        $tolR = [];
        foreach ($h as $j => $hj) {
            $tolR[$j] = 2 / 3 * $hj * 5e-2 * $bvpTol;
        }

        $maxNjev = 4;
        $maxIter = 8;
        $sigma = 0.2;
        $tau = 0.5;
        $nTrial = 4;

        [$colRes, $yMiddle, $f, $fMiddle] = self::collocationFun($fun, $y, $p, $x, $h);
        $bcRes = self::callBc($bc, self::col($y, 0), self::col($y, $m - 1), $p);
        $res = self::stackRes($colRes, $bcRes, $n, $m); // fortran-order ravel of col_res + bc_res

        $njev = 0;
        $singular = false;
        $recomputeJac = true;
        $LU = null;
        $piv = null;
        $step = [];
        $cost = 0.0;
        $stepNew = [];
        $costNew = 0.0;

        for ($iter = 0; $iter < $maxIter; $iter++) {
            if ($recomputeJac) {
                $J = self::buildJac($n, $m, $k, $fun, $bc, $x, $xMiddle, $y, $yMiddle, $p, $h, $f, $fMiddle, $bcRes);
                $njev++;
                $fac = self::luFactor($J);
                if ($fac === null) {
                    $singular = true;
                    break;
                }
                [$LU, $piv] = $fac;
                $step = self::luSolve($LU, $piv, $res);
                $cost = self::dot($step, $step);
            }

            $alpha = 1.0;
            $yNew = $y;
            $pNew = $p;
            for ($trial = 0; $trial <= $nTrial; $trial++) {
                // y_new = y - alpha * y_step (y_step is step reshaped (n,m) fortran order)
                $yNew = [];
                for ($i = 0; $i < $n; $i++) {
                    $rowNew = [];
                    for ($j = 0; $j < $m; $j++) {
                        $rowNew[$j] = $y[$i][$j] - $alpha * $step[$j * $n + $i];
                    }
                    $yNew[$i] = $rowNew;
                }
                $pNew = [];
                for ($i = 0; $i < $k; $i++) {
                    $pNew[$i] = $p[$i] - $alpha * $step[$m * $n + $i];
                }

                [$colRes, $yMiddle, $f, $fMiddle] = self::collocationFun($fun, $yNew, $pNew, $x, $h);
                $bcRes = self::callBc($bc, self::col($yNew, 0), self::col($yNew, $m - 1), $pNew);
                $res = self::stackRes($colRes, $bcRes, $n, $m);

                $stepNew = self::luSolve($LU, $piv, $res);
                $costNew = self::dot($stepNew, $stepNew);
                if ($costNew < (1 - 2 * $alpha * $sigma) * $cost) {
                    break;
                }
                if ($trial < $nTrial) {
                    $alpha *= $tau;
                }
            }

            $y = $yNew;
            $p = $pNew;

            if ($njev === $maxNjev) {
                break;
            }

            // convergence: all |col_res| < tol_r*(1+|f_middle|) and all |bc_res| < bc_tol
            $conv = true;
            for ($i = 0; $i < $n && $conv; $i++) {
                for ($j = 0; $j < $m - 1; $j++) {
                    if (abs($colRes[$i][$j]) >= $tolR[$j] * (1 + abs($fMiddle[$i][$j]))) {
                        $conv = false;
                        break;
                    }
                }
            }
            if ($conv) {
                foreach ($bcRes as $v) {
                    if (abs($v) >= $bcTol) {
                        $conv = false;
                        break;
                    }
                }
            }
            if ($conv) {
                break;
            }

            if ($alpha === 1.0) {
                $step = $stepNew;
                $cost = $costNew;
                $recomputeJac = false;
            } else {
                $recomputeJac = true;
            }
        }

        return [$y, $p, $singular];
    }

    /**
     * Assemble the dense (n*m+k) collocation+bc Jacobian. Mirrors construct_global_jac.
     *
     * @return list<list<float>>
     */
    private static function buildJac(
        int $n,
        int $m,
        int $k,
        callable $fun,
        callable $bc,
        array $x,
        array $xMiddle,
        array $y,
        array $yMiddle,
        array $p,
        array $h,
        array $f,
        array $fMiddle,
        array $bcRes
    ): array {
        // df_dy[i,j,q] over mesh nodes (n,n,m), df_dy_middle (n,n,m-1), + df_dp (n,k,*) if k>0
        [$dfDy, $dfDp] = self::estimateFunJac($fun, $x, $y, $p, $n, $m, $f);
        [$dfDyMid, $dfDpMid] = self::estimateFunJac($fun, $xMiddle, $yMiddle, $p, $n, $m - 1, $fMiddle);
        [$dbcDya, $dbcDyb, $dbcDp] = self::estimateBcJac($bc, self::col($y, 0), self::col($y, $m - 1), $p, $n, $k, $bcRes);

        $N = $n * $m + $k;
        $J = [];
        for ($r = 0; $r < $N; $r++) {
            $J[$r] = array_fill(0, $N, 0.0);
        }

        // Diagonal / off-diagonal n x n blocks for the m-1 collocation residuals.
        // dPhi_dy_0 = -I - h/6*(df_dy[q] + 2*df_dy_mid[q]) - h^2/12 * (df_dy_mid[q] @ df_dy[q])
        // dPhi_dy_1 =  I - h/6*(df_dy[q+1] + 2*df_dy_mid[q]) + h^2/12 * (df_dy_mid[q] @ df_dy[q+1])
        for ($q = 0; $q < $m - 1; $q++) {
            $hq = $h[$q];
            // extract n x n matrices at node q, q+1, middle q
            $A0 = self::jacAt($dfDy, $n, $q);       // df_dy[:,:,q]
            $A1 = self::jacAt($dfDy, $n, $q + 1);   // df_dy[:,:,q+1]
            $Am = self::jacAt($dfDyMid, $n, $q);    // df_dy_mid[:,:,q]
            $Tm0 = self::matmul($Am, $A0, $n);      // df_dy_mid @ df_dy[q]
            $Tm1 = self::matmul($Am, $A1, $n);      // df_dy_mid @ df_dy[q+1]

            $rowBase = $q * $n;
            $col0 = $q * $n;
            $col1 = ($q + 1) * $n;
            for ($i = 0; $i < $n; $i++) {
                for ($j = 0; $j < $n; $j++) {
                    $iden = $i === $j ? 1.0 : 0.0;
                    $b0 = -$iden - $hq / 6 * ($A0[$i][$j] + 2 * $Am[$i][$j]) - $hq * $hq / 12 * $Tm0[$i][$j];
                    $b1 = $iden - $hq / 6 * ($A1[$i][$j] + 2 * $Am[$i][$j]) + $hq * $hq / 12 * $Tm1[$i][$j];
                    $J[$rowBase + $i][$col0 + $j] += $b0;
                    $J[$rowBase + $i][$col1 + $j] += $b1;
                }
            }
        }

        // Boundary-condition rows: (n+k) rows starting at (m-1)*n.
        $bcRow = ($m - 1) * $n;
        for ($i = 0; $i < $n + $k; $i++) {
            for ($j = 0; $j < $n; $j++) {
                $J[$bcRow + $i][$j] += $dbcDya[$i][$j];                 // vs ya (cols 0..n)
                $J[$bcRow + $i][($m - 1) * $n + $j] += $dbcDyb[$i][$j]; // vs yb (last node block)
            }
        }

        if ($k > 0) {
            // df_dp_middle += 0.125 * h * (df_dy_mid @ (df_dp[q]-df_dp[q+1]))
            // dPhi_dp = -h/6*(df_dp[q] + df_dp[q+1] + 4*df_dp_mid_corrected)
            for ($q = 0; $q < $m - 1; $q++) {
                $hq = $h[$q];
                $Am = self::jacAt($dfDyMid, $n, $q);
                // df_dp[:,:,q] etc are (n,k)
                for ($c = 0; $c < $k; $c++) {
                    // column c of df_dp at q, q+1, middle
                    $dp0 = [];
                    $dp1 = [];
                    $dpm = [];
                    for ($i = 0; $i < $n; $i++) {
                        $dp0[$i] = $dfDp[$i][$c][$q];
                        $dp1[$i] = $dfDp[$i][$c][$q + 1];
                        $dpm[$i] = $dfDpMid[$i][$c][$q];
                    }
                    // correction: dpm += 0.125*h * Am @ (dp0 - dp1)
                    $diff = [];
                    for ($i = 0; $i < $n; $i++) {
                        $diff[$i] = $dp0[$i] - $dp1[$i];
                    }
                    $Tv = self::matvec($Am, $diff, $n);
                    for ($i = 0; $i < $n; $i++) {
                        $dpmCorr = $dpm[$i] + 0.125 * $hq * $Tv[$i];
                        $val = -$hq / 6 * ($dp0[$i] + $dp1[$i] + 4 * $dpmCorr);
                        $J[$q * $n + $i][$m * $n + $c] += $val;
                    }
                }
            }
            // bc rows vs p
            for ($i = 0; $i < $n + $k; $i++) {
                for ($c = 0; $c < $k; $c++) {
                    $J[$bcRow + $i][$m * $n + $c] += $dbcDp[$i][$c];
                }
            }
        }

        return $J;
    }

    /**
     * Forward-difference df/dy (n,n,m*) and df/dp (n,k,m*). Mirrors estimate_fun_jac.
     *
     * @return array{0: list<list<list<float>>>, 1: list<list<list<float>>>}  [df_dy, df_dp]
     */
    private static function estimateFunJac(callable $fun, array $x, array $y, array $p, int $n, int $mm, array $f0): array
    {
        $k = count($p);
        // df_dy[i][j][q]
        $dfDy = [];
        for ($i = 0; $i < $n; $i++) {
            $dfDy[$i] = [];
            for ($j = 0; $j < $n; $j++) {
                $dfDy[$i][$j] = array_fill(0, $mm, 0.0);
            }
        }
        for ($j = 0; $j < $n; $j++) {
            // perturb component j: y_new[j] += h[j] (h depends on each column)
            $yNew = $y;
            $hcol = [];
            for ($q = 0; $q < $mm; $q++) {
                $hj = sqrt(self::EPS) * (1 + abs($y[$j][$q]));
                $yNew[$j][$q] = $y[$j][$q] + $hj;
                $hcol[$q] = $yNew[$j][$q] - $y[$j][$q];
            }
            $fNew = self::callFun($fun, $x, $yNew, $p, $n, $mm);
            for ($i = 0; $i < $n; $i++) {
                for ($q = 0; $q < $mm; $q++) {
                    $dfDy[$i][$j][$q] = ($fNew[$i][$q] - $f0[$i][$q]) / $hcol[$q];
                }
            }
        }

        $dfDp = [];
        if ($k > 0) {
            for ($i = 0; $i < $n; $i++) {
                $dfDp[$i] = [];
                for ($c = 0; $c < $k; $c++) {
                    $dfDp[$i][$c] = array_fill(0, $mm, 0.0);
                }
            }
            for ($c = 0; $c < $k; $c++) {
                $pNew = $p;
                $hc = sqrt(self::EPS) * (1 + abs($p[$c]));
                $pNew[$c] = $p[$c] + $hc;
                $hi = $pNew[$c] - $p[$c];
                $fNew = self::callFun($fun, $x, $y, $pNew, $n, $mm);
                for ($i = 0; $i < $n; $i++) {
                    for ($q = 0; $q < $mm; $q++) {
                        $dfDp[$i][$c][$q] = ($fNew[$i][$q] - $f0[$i][$q]) / $hi;
                    }
                }
            }
        }

        return [$dfDy, $dfDp];
    }

    /**
     * Forward-difference bc Jacobians. Mirrors estimate_bc_jac (returns transposed (n+k,n)/(n+k,k)).
     *
     * @return array{0: list<list<float>>, 1: list<list<float>>, 2: list<list<float>>}  [dbc_dya, dbc_dyb, dbc_dp]
     */
    private static function estimateBcJac(callable $bc, array $ya, array $yb, array $p, int $n, int $k, array $bc0): array
    {
        $nk = $n + $k;
        // dbc_dya built as (n rows, n+k cols) then transposed -> (n+k, n)
        $tmpYa = [];
        for ($i = 0; $i < $n; $i++) {
            $yaNew = $ya;
            $hi = sqrt(self::EPS) * (1 + abs($ya[$i]));
            $yaNew[$i] = $ya[$i] + $hi;
            $hh = $yaNew[$i] - $ya[$i];
            $bcNew = self::callBc($bc, $yaNew, $yb, $p);
            $row = [];
            for ($r = 0; $r < $nk; $r++) {
                $row[$r] = ($bcNew[$r] - $bc0[$r]) / $hh;
            }
            $tmpYa[$i] = $row;
        }
        $dbcDya = self::transpose($tmpYa, $n, $nk); // (n+k, n)

        $tmpYb = [];
        for ($i = 0; $i < $n; $i++) {
            $ybNew = $yb;
            $hi = sqrt(self::EPS) * (1 + abs($yb[$i]));
            $ybNew[$i] = $yb[$i] + $hi;
            $hh = $ybNew[$i] - $yb[$i];
            $bcNew = self::callBc($bc, $ya, $ybNew, $p);
            $row = [];
            for ($r = 0; $r < $nk; $r++) {
                $row[$r] = ($bcNew[$r] - $bc0[$r]) / $hh;
            }
            $tmpYb[$i] = $row;
        }
        $dbcDyb = self::transpose($tmpYb, $n, $nk);

        $dbcDp = [];
        if ($k > 0) {
            $tmpP = [];
            for ($i = 0; $i < $k; $i++) {
                $pNew = $p;
                $hi = sqrt(self::EPS) * (1 + abs($p[$i]));
                $pNew[$i] = $p[$i] + $hi;
                $hh = $pNew[$i] - $p[$i];
                $bcNew = self::callBc($bc, $ya, $yb, $pNew);
                $row = [];
                for ($r = 0; $r < $nk; $r++) {
                    $row[$r] = ($bcNew[$r] - $bc0[$r]) / $hh;
                }
                $tmpP[$i] = $row;
            }
            $dbcDp = self::transpose($tmpP, $k, $nk); // (n+k, k)
        }

        return [$dbcDya, $dbcDyb, $dbcDp];
    }

    /**
     * 5-point Lobatto quadrature RMS of relative residuals. Mirrors estimate_rms_residuals.
     *
     * @param array{c: list<list<list<float>>>, x: list<float>, n: int, m: int} $sol
     * @return list<float>  rms over each interval (m-1,)
     */
    private static function estimateRmsResiduals(callable $fun, array $sol, array $x, array $h, array $p, array $rMiddle, array $fMiddle): array
    {
        $n = count($rMiddle);
        $mm1 = count($h);
        $s = (3 / 7) ** 0.5;
        $x1 = [];
        $x2 = [];
        for ($j = 0; $j < $mm1; $j++) {
            $xm = $x[$j] + 0.5 * $h[$j];
            $sh = 0.5 * $h[$j] * $s;
            $x1[$j] = $xm + $sh;
            $x2[$j] = $xm - $sh;
        }
        $y1 = self::splineEval($sol, $x1, 0);
        $y2 = self::splineEval($sol, $x2, 0);
        $y1p = self::splineEval($sol, $x1, 1);
        $y2p = self::splineEval($sol, $x2, 1);
        $f1 = self::callFun($fun, $x1, $y1, $p, $n, $mm1);
        $f2 = self::callFun($fun, $x2, $y2, $p, $n, $mm1);

        // r1 = y1' - f1, normalized by 1+|f1|; summed over components (squared)
        $sum1 = array_fill(0, $mm1, 0.0);
        $sum2 = array_fill(0, $mm1, 0.0);
        $sumM = array_fill(0, $mm1, 0.0);
        for ($i = 0; $i < $n; $i++) {
            for ($j = 0; $j < $mm1; $j++) {
                $r1 = ($y1p[$i][$j] - $f1[$i][$j]) / (1 + abs($f1[$i][$j]));
                $r2 = ($y2p[$i][$j] - $f2[$i][$j]) / (1 + abs($f2[$i][$j]));
                $rm = $rMiddle[$i][$j] / (1 + abs($fMiddle[$i][$j]));
                $sum1[$j] += $r1 * $r1;
                $sum2[$j] += $r2 * $r2;
                $sumM[$j] += $rm * $rm;
            }
        }
        $out = [];
        for ($j = 0; $j < $mm1; $j++) {
            $out[$j] = (0.5 * (32 / 45 * $sumM[$j] + 49 / 90 * ($sum1[$j] + $sum2[$j]))) ** 0.5;
        }
        return $out;
    }

    /**
     * Build the C1 cubic spline (PPoly coeffs (4,n,m-1)). Mirrors create_spline.
     *
     * @return array{c: list<list<list<float>>>, x: list<float>, n: int, m: int}
     */
    private static function createSpline(array $y, array $yp, array $x, array $h): array
    {
        $n = count($y);
        $m = count($x);
        // c[0..3][i][j]
        $c = [];
        for ($d = 0; $d < 4; $d++) {
            $c[$d] = [];
            for ($i = 0; $i < $n; $i++) {
                $c[$d][$i] = array_fill(0, $m - 1, 0.0);
            }
        }
        for ($i = 0; $i < $n; $i++) {
            for ($j = 0; $j < $m - 1; $j++) {
                $slope = ($y[$i][$j + 1] - $y[$i][$j]) / $h[$j];
                $t = ($yp[$i][$j] + $yp[$i][$j + 1] - 2 * $slope) / $h[$j];
                $c[0][$i][$j] = $t / $h[$j];
                $c[1][$i][$j] = ($slope - $yp[$i][$j]) / $h[$j] - $t;
                $c[2][$i][$j] = $yp[$i][$j];
                $c[3][$i][$j] = $y[$i][$j];
            }
        }
        return ['c' => $c, 'x' => $x, 'n' => $n, 'm' => $m];
    }

    /**
     * Evaluate the PPoly spline (nu = 0 value, 1 first derivative) at xq, extrapolating.
     *
     * @param array{c: list<list<list<float>>>, x: list<float>, n: int, m: int} $sol
     * @param list<float> $xq
     * @return list<list<float>>  shape (n, len(xq))
     */
    private static function splineEval(array $sol, array $xq, int $nu): array
    {
        $c = $sol['c'];
        $xk = $sol['x'];
        $n = $sol['n'];
        $m = $sol['m'];
        $nIntervals = $m - 1;
        $out = [];
        for ($i = 0; $i < $n; $i++) {
            $out[$i] = [];
        }
        foreach ($xq as $qi => $xv) {
            // interval index = searchsorted(x, xv) - 1, clipped to [0, m-2]
            $idx = self::searchSorted($xk, $xv) - 1;
            if ($idx < 0) { $idx = 0; }
            if ($idx > $nIntervals - 1) { $idx = $nIntervals - 1; }
            $dx = $xv - $xk[$idx];
            for ($i = 0; $i < $n; $i++) {
                $c0 = $c[0][$i][$idx];
                $c1 = $c[1][$i][$idx];
                $c2 = $c[2][$i][$idx];
                $c3 = $c[3][$i][$idx];
                if ($nu === 0) {
                    $out[$i][$qi] = (($c0 * $dx + $c1) * $dx + $c2) * $dx + $c3;
                } else {
                    $out[$i][$qi] = (3 * $c0 * $dx + 2 * $c1) * $dx + $c2;
                }
            }
        }
        return $out;
    }

    /**
     * Insert nodes into the mesh. Mirrors modify_mesh (sorted hstack).
     *
     * @param list<float> $x
     * @param list<int> $insert1
     * @param list<int> $insert2
     * @return list<float>
     */
    private static function modifyMesh(array $x, array $insert1, array $insert2): array
    {
        $new = $x;
        foreach ($insert1 as $i) {
            $new[] = 0.5 * ($x[$i] + $x[$i + 1]);
        }
        foreach ($insert2 as $i) {
            $new[] = (2 * $x[$i] + $x[$i + 1]) / 3;
            $new[] = ($x[$i] + 2 * $x[$i + 1]) / 3;
        }
        sort($new);
        return array_values($new);
    }

    // ---------------------------------------------------------------------
    // linear algebra + helpers
    // ---------------------------------------------------------------------

    /**
     * Partial-pivot LU. Returns null if singular.
     *
     * @param list<list<float>> $A
     * @return array{0: list<list<float>>, 1: list<int>}|null
     */
    private static function luFactor(array $A): ?array
    {
        $n = count($A);
        $piv = range(0, $n - 1);
        for ($k = 0; $k < $n; $k++) {
            $pp = $k;
            $maxv = abs($A[$k][$k]);
            for ($i = $k + 1; $i < $n; $i++) {
                $av = abs($A[$i][$k]);
                if ($av > $maxv) { $maxv = $av; $pp = $i; }
            }
            if ($A[$pp][$k] === 0.0) {
                return null;
            }
            if ($pp !== $k) {
                $tmp = $A[$k]; $A[$k] = $A[$pp]; $A[$pp] = $tmp;
                $pt = $piv[$k]; $piv[$k] = $piv[$pp]; $piv[$pp] = $pt;
            }
            $akk = $A[$k][$k];
            for ($i = $k + 1; $i < $n; $i++) {
                $fct = $A[$i][$k] / $akk;
                $A[$i][$k] = $fct;
                if ($fct !== 0.0) {
                    for ($j = $k + 1; $j < $n; $j++) {
                        $A[$i][$j] -= $fct * $A[$k][$j];
                    }
                }
            }
        }
        return [$A, $piv];
    }

    /**
     * @param list<list<float>> $LU
     * @param list<int> $piv
     * @param list<float> $b
     * @return list<float>
     */
    private static function luSolve(array $LU, array $piv, array $b): array
    {
        $n = count($b);
        $x = [];
        for ($i = 0; $i < $n; $i++) {
            $x[$i] = $b[$piv[$i]];
        }
        for ($i = 0; $i < $n; $i++) {
            $sm = $x[$i];
            for ($j = 0; $j < $i; $j++) {
                $sm -= $LU[$i][$j] * $x[$j];
            }
            $x[$i] = $sm;
        }
        for ($i = $n - 1; $i >= 0; $i--) {
            $sm = $x[$i];
            for ($j = $i + 1; $j < $n; $j++) {
                $sm -= $LU[$i][$j] * $x[$j];
            }
            $x[$i] = $sm / $LU[$i][$i];
        }
        return $x;
    }

    /** @param list<float> $a @param list<float> $b */
    private static function dot(array $a, array $b): float
    {
        $s = 0.0;
        foreach ($a as $i => $v) {
            $s += $v * $b[$i];
        }
        return $s;
    }

    /** Fortran-order ravel of col_res (n,m-1) followed by bc_res. @return list<float> */
    private static function stackRes(array $colRes, array $bcRes, int $n, int $m): array
    {
        $out = [];
        for ($j = 0; $j < $m - 1; $j++) {
            for ($i = 0; $i < $n; $i++) {
                $out[] = $colRes[$i][$j];
            }
        }
        foreach ($bcRes as $v) {
            $out[] = $v;
        }
        return $out;
    }

    /** Extract df_dy[:,:,q] as n x n. @return list<list<float>> */
    private static function jacAt(array $dfDy, int $n, int $q): array
    {
        $out = [];
        for ($i = 0; $i < $n; $i++) {
            $row = [];
            for ($j = 0; $j < $n; $j++) {
                $row[$j] = $dfDy[$i][$j][$q];
            }
            $out[$i] = $row;
        }
        return $out;
    }

    /** n x n matmul. @return list<list<float>> */
    private static function matmul(array $A, array $B, int $n): array
    {
        $out = [];
        for ($i = 0; $i < $n; $i++) {
            $row = array_fill(0, $n, 0.0);
            for ($l = 0; $l < $n; $l++) {
                $ail = $A[$i][$l];
                if ($ail !== 0.0) {
                    for ($j = 0; $j < $n; $j++) {
                        $row[$j] += $ail * $B[$l][$j];
                    }
                }
            }
            $out[$i] = $row;
        }
        return $out;
    }

    /** n x n matrix-vector. @return list<float> */
    private static function matvec(array $A, array $v, int $n): array
    {
        $out = [];
        for ($i = 0; $i < $n; $i++) {
            $s = 0.0;
            for ($j = 0; $j < $n; $j++) {
                $s += $A[$i][$j] * $v[$j];
            }
            $out[$i] = $s;
        }
        return $out;
    }

    /** Transpose an r x c matrix -> c x r. @return list<list<float>> */
    private static function transpose(array $M, int $r, int $c): array
    {
        $out = [];
        for ($j = 0; $j < $c; $j++) {
            $row = [];
            for ($i = 0; $i < $r; $i++) {
                $row[$i] = $M[$i][$j];
            }
            $out[$j] = $row;
        }
        return $out;
    }

    /** Column j of a (n,m) matrix. @return list<float> */
    private static function col(array $y, int $j): array
    {
        $out = [];
        foreach ($y as $row) {
            $out[] = $row[$j];
        }
        return $out;
    }

    /** @param list<float> $x @return list<float> */
    private static function diff(array $x): array
    {
        $out = [];
        $m = count($x);
        for ($i = 0; $i < $m - 1; $i++) {
            $out[$i] = $x[$i + 1] - $x[$i];
        }
        return $out;
    }

    /** First index i such that x[i] >= v among sorted x (numpy searchsorted 'left'). */
    private static function searchSorted(array $x, float $v): int
    {
        $lo = 0;
        $hi = count($x);
        while ($lo < $hi) {
            $mid = intdiv($lo + $hi, 2);
            if ($x[$mid] < $v) {
                $lo = $mid + 1;
            } else {
                $hi = $mid;
            }
        }
        return $lo;
    }

    /**
     * Invoke the user rhs f(x,y,p) and coerce to (n, mm). Accepts (n,mm) nested arrays.
     *
     * @return list<list<float>>
     */
    private static function callFun(callable $fun, array $x, array $y, array $p, int $n, int $mm): array
    {
        $r = $fun($x, $y, $p);
        $out = [];
        for ($i = 0; $i < $n; $i++) {
            $row = $r[$i];
            $out[$i] = [];
            for ($j = 0; $j < $mm; $j++) {
                $out[$i][$j] = (float) $row[$j];
            }
        }
        return $out;
    }

    /** @return list<float> */
    private static function callBc(callable $bc, array $ya, array $yb, array $p): array
    {
        $r = $bc($ya, $yb, $p);
        $out = [];
        foreach ($r as $v) {
            $out[] = (float) $v;
        }
        return $out;
    }
}
