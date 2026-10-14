<?php

declare(strict_types=1);

namespace Tessero\Spatial\Transform;

/**
 * scipy.spatial.transform.RotationSpline: C2 interpolation of rotations with continuous angular
 * rate and acceleration (the rotation-vector between consecutive keys is a cubic in time).
 * Pure PHP (no kernel), identical on both backends; built on Tessero\Spatial\Transform\Rotation.
 *
 * A faithful port of scipy's _rotation_spline: it forms the inter-key rotation vectors, solves the
 * block-tridiagonal system for the interior angular rates by Newton iteration (MAX_ITER=10,
 * TOL=1e-9, dense Gaussian elimination on the banded matrix), then builds per-segment cubic
 * coefficients. Calling the instance with query time(s) and order 0/1/2 returns the interpolated
 * Rotation, angular rate (rad/s) or angular acceleration (rad/s^2). Query values match scipy to
 * ~1e-14.
 */
final class RotationSpline
{
    private const MAX_ITER = 10;
    private const TOL = 1e-9;

    /** @var list<float> key times */
    private array $times;

    /** @var Rotation key rotations (stack) */
    private Rotation $rotations;

    /** @var list<Rotation> per-key single Rotation (for order-0 base) */
    private array $keyRots;

    /** @var array<int,array<int,array<int,float>>> PPoly coefficients [power][segment][dim] (power 0..3, highest first) */
    private array $coeff;

    public function __construct(array $times, Rotation $rotations)
    {
        $times = array_values(array_map('floatval', $times));
        $n = count($times);
        if ($n < 2) {
            throw new \InvalidArgumentException('`rotations` must contain at least 2 rotations.');
        }
        for ($i = 1; $i < $n; $i++) {
            if ($times[$i] <= $times[$i - 1]) {
                throw new \InvalidArgumentException('Values in `times` must be in a strictly increasing order.');
            }
        }
        $quats = $rotations->asQuat();
        if ($quats !== [] && !is_array($quats[0])) {
            $quats = [$quats];
        }
        if (count($quats) !== $n) {
            throw new \InvalidArgumentException('Expected number of rotations to equal number of timestamps.');
        }
        $this->times = $times;
        $this->rotations = $rotations;
        $this->keyRots = [];
        foreach ($quats as $q) {
            $this->keyRots[] = Rotation::fromQuat($q);
        }

        $dt = [];
        for ($i = 0; $i < $n - 1; $i++) {
            $dt[$i] = $times[$i + 1] - $times[$i];
        }

        // relative rotation vectors: (R_i^{-1} * R_{i+1}).as_rotvec()
        $rel = [];
        for ($i = 0; $i < $n - 1; $i++) {
            $rr = $this->keyRots[$i]->inv()->compose($this->keyRots[$i + 1]);
            $rel[$i] = $rr->asRotvec();
        }

        // angular_rates[i] = rel[i] / dt[i]
        $ar = [];
        for ($i = 0; $i < $n - 1; $i++) {
            $ar[$i] = [$rel[$i][0] / $dt[$i], $rel[$i][1] / $dt[$i], $rel[$i][2] / $dt[$i]];
        }

        if ($n === 2) {
            $rvd = [$ar[0]];
            $arFull = $ar;
        } else {
            [$arFull, $rvd] = $this->solveForAngularRates($dt, $ar, $rel);
        }

        // per-segment cubic coefficients
        $coeff = [[], [], [], []];
        for ($i = 0; $i < $n - 1; $i++) {
            $D = $dt[$i];
            for ($k = 0; $k < 3; $k++) {
                $coeff[0][$i][$k] = (-2 * $rel[$i][$k] + $D * $arFull[$i][$k] + $D * $rvd[$i][$k]) / ($D * $D * $D);
                $coeff[1][$i][$k] = (3 * $rel[$i][$k] - 2 * $D * $arFull[$i][$k] - $D * $rvd[$i][$k]) / ($D * $D);
                $coeff[2][$i][$k] = $arFull[$i][$k];
                $coeff[3][$i][$k] = 0.0;
            }
        }
        $this->coeff = $coeff;
    }

    /**
     * Compute interpolated values.
     *
     * @param float|array<int,float> $times
     * @return Rotation|array<int,float>|array<int,array<int,float>>  order 0 -> Rotation; 1/2 -> rate/accel
     */
    public function __invoke(float|array $times, int $order = 0): Rotation|array
    {
        if (!in_array($order, [0, 1, 2], true)) {
            throw new \InvalidArgumentException('`order` must be 0, 1 or 2.');
        }
        $single = !is_array($times);
        $qs = is_array($times) ? array_values(array_map('floatval', $times)) : [(float) $times];
        $n = count($this->times);

        if ($order === 0) {
            $outQ = [];
            foreach ($qs as $t) {
                $i = $this->segment($t);
                $rv = $this->evalPoly($i, $t, 0);
                $base = $this->keyRots[$i]->compose(Rotation::fromRotvec($rv));
                $outQ[] = $base->asQuat();
            }
            return $single ? Rotation::fromQuat($outQ[0]) : Rotation::fromQuat($outQ);
        }

        $out = [];
        foreach ($qs as $t) {
            $i = $this->segment($t);
            $rv = $this->evalPoly($i, $t, 0);
            $rvd = $this->evalPoly($i, $t, 1);
            if ($order === 1) {
                $out[] = self::matVec(self::rvdotToAngularRate($rv), $rvd);
            } else {
                $rvdd = $this->evalPoly($i, $t, 2);
                $rate = self::matVec(self::rvdotToAngularRate($rv), $rvdd);
                $nl = self::nonlinearTerm($rv, $rvd);
                $out[] = [$rate[0] + $nl[0], $rate[1] + $nl[1], $rate[2] + $nl[2]];
            }
        }
        return $single ? $out[0] : $out;
    }

    private function segment(float $t): int
    {
        $n = count($this->times);
        $i = 0;
        for ($k = 0; $k < $n; $k++) {
            if ($this->times[$k] <= $t) {
                $i = $k;
            } else {
                break;
            }
        }
        if ($i > $n - 2) {
            $i = $n - 2;
        }
        if ($i < 0) {
            $i = 0;
        }
        return $i;
    }

    /** Evaluate the segment cubic (rotvec, order-th derivative) at absolute time $t. */
    private function evalPoly(int $i, float $t, int $deriv): array
    {
        $x = $t - $this->times[$i];
        $c = $this->coeff;
        $out = [0.0, 0.0, 0.0];
        for ($k = 0; $k < 3; $k++) {
            $a3 = $c[0][$i][$k]; $a2 = $c[1][$i][$k]; $a1 = $c[2][$i][$k]; $a0 = $c[3][$i][$k];
            if ($deriv === 0) {
                $out[$k] = (($a3 * $x + $a2) * $x + $a1) * $x + $a0;
            } elseif ($deriv === 1) {
                $out[$k] = (3 * $a3 * $x + 2 * $a2) * $x + $a1;
            } else {
                $out[$k] = 6 * $a3 * $x + 2 * $a2;
            }
        }
        return $out;
    }

    /**
     * Solve the block-tridiagonal system for interior angular rates by Newton iteration.
     *
     * @param list<float> $dt
     * @param list<array<int,float>> $ar  initial angular rates (length n-1)
     * @param list<array<int,float>> $rel rotation vectors (length n-1)
     * @return array{0:list<array<int,float>>,1:list<array<int,float>>}  [angular_rates(n-1), rotvecs_dot(n-1)]
     */
    private function solveForAngularRates(array $dt, array $ar, array $rel): array
    {
        $m = count($ar); // n-1
        $A = [];
        $Ainv = [];
        for ($i = 0; $i < $m; $i++) {
            $A[$i] = self::angularRateToRvdot($rel[$i]);
            $Ainv[$i] = self::rvdotToAngularRate($rel[$i]);
        }

        // diagonal scalars d (length m-1): 4*(1/dt[i] + 1/dt[i+1])
        $d = [];
        for ($i = 0; $i < $m - 1; $i++) {
            $d[$i] = 4 * (1 / $dt[$i] + 1 / $dt[$i + 1]);
        }
        // sub-diagonal A-blocks and super-diagonal B-blocks (length m-2)
        $Ablk = [];
        $Bblk = [];
        for ($i = 0; $i < $m - 2; $i++) {
            $s = 2 / $dt[1 + $i];
            $Ablk[$i] = self::scaleMat($Ainv[1 + $i], $s);
            $Bblk[$i] = self::scaleMat($A[1 + $i], $s);
        }

        // b0 (length m-1)
        $b0 = [];
        for ($i = 0; $i < $m - 1; $i++) {
            for ($k = 0; $k < 3; $k++) {
                $b0[$i][$k] = 6 * ($rel[$i][$k] / ($dt[$i] * $dt[$i]) + $rel[$i + 1][$k] / ($dt[$i + 1] * $dt[$i + 1]));
            }
        }
        $corr0 = self::matVec($Ainv[0], $ar[0]);
        for ($k = 0; $k < 3; $k++) {
            $b0[0][$k] -= 2 / $dt[0] * $corr0[$k];
        }
        $corrL = self::matVec($A[$m - 1], $ar[$m - 1]);
        for ($k = 0; $k < 3; $k++) {
            $b0[$m - 2][$k] -= 2 / $dt[$m - 1] * $corrL[$k];
        }

        $Mb = self::buildBanded($Ablk, $Bblk, $d);

        // working angular rates (copy)
        $arw = [];
        for ($i = 0; $i < $m; $i++) {
            $arw[$i] = $ar[$i];
        }

        for ($it = 0; $it < self::MAX_ITER; $it++) {
            $rvdAll = [];
            for ($i = 0; $i < $m; $i++) {
                $rvdAll[$i] = self::matVec($A[$i], $arw[$i]);
            }
            $b = [];
            for ($i = 0; $i < $m - 1; $i++) {
                $nl = self::nonlinearTerm($rel[$i], $rvdAll[$i]);
                for ($k = 0; $k < 3; $k++) {
                    $b[$i][$k] = $b0[$i][$k] - $nl[$k];
                }
            }
            $bflat = [];
            for ($i = 0; $i < $m - 1; $i++) {
                for ($k = 0; $k < 3; $k++) {
                    $bflat[] = $b[$i][$k];
                }
            }
            $sol = self::solveBanded($Mb, $bflat);
            $newar = [];
            for ($i = 0; $i < $m - 1; $i++) {
                $newar[$i] = [$sol[3 * $i], $sol[3 * $i + 1], $sol[3 * $i + 2]];
            }
            $tolMet = true;
            for ($i = 0; $i < $m - 1 && $tolMet; $i++) {
                for ($k = 0; $k < 3; $k++) {
                    if (abs($newar[$i][$k] - $arw[$i][$k]) >= self::TOL * (1 + abs($newar[$i][$k]))) {
                        $tolMet = false;
                        break;
                    }
                }
            }
            for ($i = 0; $i < $m - 1; $i++) {
                $arw[$i] = $newar[$i];
            }
            if ($tolMet) {
                break;
            }
        }

        $rvd = [];
        for ($i = 0; $i < $m; $i++) {
            $rvd[$i] = self::matVec($A[$i], $arw[$i]);
        }
        // angular_rates = [ar_first, arw[0..m-2]]  (the last working entry is never updated)
        $arFull = [];
        $arFull[0] = $ar[0];
        for ($i = 0; $i < $m - 1; $i++) {
            $arFull[$i + 1] = $arw[$i];
        }
        return [$arFull, $rvd];
    }

    // ---- math kernels (ported from scipy _rotation_spline) -------------

    private static function angularRateToRvdot(array $rv): array
    {
        $nm = sqrt($rv[0] * $rv[0] + $rv[1] * $rv[1] + $rv[2] * $rv[2]);
        if ($nm > 1e-4) {
            $k = (1 - 0.5 * $nm / tan(0.5 * $nm)) / ($nm * $nm);
        } else {
            $k = 1 / 12 + 1 / 720 * $nm * $nm;
        }
        $S = self::skew($rv);
        $SS = self::mm3($S, $S);
        return self::madd(self::madd(self::eye3(), $S, 0.5), $SS, $k);
    }

    private static function rvdotToAngularRate(array $rv): array
    {
        $nm = sqrt($rv[0] * $rv[0] + $rv[1] * $rv[1] + $rv[2] * $rv[2]);
        if ($nm > 1e-4) {
            $k1 = (1 - cos($nm)) / ($nm * $nm);
            $k2 = ($nm - sin($nm)) / ($nm * $nm * $nm);
        } else {
            $k1 = 0.5 - $nm * $nm / 24;
            $k2 = 1 / 6 - $nm * $nm / 120;
        }
        $S = self::skew($rv);
        $SS = self::mm3($S, $S);
        return self::madd(self::madd(self::eye3(), $S, -$k1), $SS, $k2);
    }

    private static function nonlinearTerm(array $rv, array $rvd): array
    {
        $nm = sqrt($rv[0] * $rv[0] + $rv[1] * $rv[1] + $rv[2] * $rv[2]);
        $dp = $rv[0] * $rvd[0] + $rv[1] * $rvd[1] + $rv[2] * $rvd[2];
        $cp = self::cross($rv, $rvd);
        $ccp = self::cross($rv, $cp);
        $dccp = self::cross($rvd, $cp);
        if ($nm > 1e-4) {
            $k1 = (-$nm * sin($nm) - 2 * (cos($nm) - 1)) / ($nm ** 4);
            $k2 = (-2 * $nm + 3 * sin($nm) - $nm * cos($nm)) / ($nm ** 5);
            $k3 = ($nm - sin($nm)) / ($nm ** 3);
        } else {
            $k1 = 1 / 12 - $nm * $nm / 180;
            $k2 = -1 / 60 + $nm * $nm / 12604;
            $k3 = 1 / 6 - $nm * $nm / 120;
        }
        $out = [0.0, 0.0, 0.0];
        for ($i = 0; $i < 3; $i++) {
            $out[$i] = $dp * ($k1 * $cp[$i] + $k2 * $ccp[$i]) + $k3 * $dccp[$i];
        }
        return $out;
    }

    private static function skew(array $x): array
    {
        return [
            [0.0, -$x[2], $x[1]],
            [$x[2], 0.0, -$x[0]],
            [-$x[1], $x[0], 0.0],
        ];
    }

    private static function cross(array $a, array $b): array
    {
        return [
            $a[1] * $b[2] - $a[2] * $b[1],
            $a[2] * $b[0] - $a[0] * $b[2],
            $a[0] * $b[1] - $a[1] * $b[0],
        ];
    }

    private static function mm3(array $A, array $B): array
    {
        $out = [];
        for ($i = 0; $i < 3; $i++) {
            for ($j = 0; $j < 3; $j++) {
                $out[$i][$j] = $A[$i][0] * $B[0][$j] + $A[$i][1] * $B[1][$j] + $A[$i][2] * $B[2][$j];
            }
        }
        return $out;
    }

    private static function madd(array $A, array $B, float $s): array
    {
        $out = [];
        for ($i = 0; $i < 3; $i++) {
            for ($j = 0; $j < 3; $j++) {
                $out[$i][$j] = $A[$i][$j] + $s * $B[$i][$j];
            }
        }
        return $out;
    }

    private static function scaleMat(array $A, float $s): array
    {
        $out = [];
        for ($i = 0; $i < 3; $i++) {
            for ($j = 0; $j < 3; $j++) {
                $out[$i][$j] = $A[$i][$j] * $s;
            }
        }
        return $out;
    }

    private static function eye3(): array
    {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }

    private static function matVec(array $A, array $v): array
    {
        return [
            $A[0][0] * $v[0] + $A[0][1] * $v[1] + $A[0][2] * $v[2],
            $A[1][0] * $v[0] + $A[1][1] * $v[1] + $A[1][2] * $v[2],
            $A[2][0] * $v[0] + $A[2][1] * $v[1] + $A[2][2] * $v[2],
        ];
    }

    /**
     * Build the banded matrix (upper=lower=5) for the block-tridiagonal system.
     *
     * @param list<array<int,array<int,float>>> $Ablk sub-diagonal blocks (length m-2)
     * @param list<array<int,array<int,float>>> $Bblk super-diagonal blocks (length m-2)
     * @param list<float> $d diagonal scalars (length m-1)
     * @return array{banded:array<int,array<int,float>>,n:int,u:int,l:int}
     */
    private static function buildBanded(array $Ablk, array $Bblk, array $d): array
    {
        $u = 5; $l = 5;
        $N = 3 * count($d);
        $res = [];
        for ($r = 0; $r < $u + $l + 1; $r++) {
            $res[$r] = array_fill(0, $N, 0.0);
        }
        $put = static function (int $i, int $j, float $val) use (&$res, $u): void {
            $r = $u + $i - $j;
            $res[$r][$j] = $val;
        };
        $nblk = count($Ablk);
        for ($bi = 0; $bi < $nblk; $bi++) {
            for ($ii = 0; $ii < 3; $ii++) {
                for ($jj = 0; $jj < 3; $jj++) {
                    $put(3 * (1 + $bi) + $ii, 3 * $bi + $jj, $Ablk[$bi][$ii][$jj]);
                    $put(3 * $bi + $ii, 3 * (1 + $bi) + $jj, $Bblk[$bi][$ii][$jj]);
                }
            }
        }
        for ($di = 0; $di < count($d); $di++) {
            for ($ii = 0; $ii < 3; $ii++) {
                $put(3 * $di + $ii, 3 * $di + $ii, $d[$di]);
            }
        }
        return ['banded' => $res, 'n' => $N, 'u' => $u, 'l' => $l];
    }

    /**
     * Solve the banded system (reconstruct dense, Gaussian elimination with partial pivoting).
     * N is small (3*(n-2)); this matches scipy's solve_banded result to machine precision.
     *
     * @param array{banded:array<int,array<int,float>>,n:int,u:int,l:int} $M
     * @param list<float> $b
     * @return list<float>
     */
    private static function solveBanded(array $M, array $b): array
    {
        $N = $M['n']; $u = $M['u']; $l = $M['l']; $banded = $M['banded'];
        $A = [];
        for ($i = 0; $i < $N; $i++) {
            $A[$i] = array_fill(0, $N + 1, 0.0);
        }
        for ($j = 0; $j < $N; $j++) {
            for ($i = 0; $i < $N; $i++) {
                $r = $u + $i - $j;
                if ($r >= 0 && $r < $u + $l + 1) {
                    $A[$i][$j] = $banded[$r][$j];
                }
            }
        }
        for ($i = 0; $i < $N; $i++) {
            $A[$i][$N] = $b[$i];
        }
        for ($c = 0; $c < $N; $c++) {
            $p = $c;
            $best = abs($A[$c][$c]);
            for ($r = $c + 1; $r < $N; $r++) {
                if (abs($A[$r][$c]) > $best) {
                    $best = abs($A[$r][$c]);
                    $p = $r;
                }
            }
            if ($p !== $c) {
                $tmp = $A[$c]; $A[$c] = $A[$p]; $A[$p] = $tmp;
            }
            $pv = $A[$c][$c];
            for ($r = 0; $r < $N; $r++) {
                if ($r !== $c && $A[$r][$c] != 0.0) {
                    $f = $A[$r][$c] / $pv;
                    for ($k = $c; $k <= $N; $k++) {
                        $A[$r][$k] -= $f * $A[$c][$k];
                    }
                }
            }
        }
        $x = [];
        for ($i = 0; $i < $N; $i++) {
            $x[$i] = $A[$i][$N] / $A[$i][$i];
        }
        return $x;
    }
}
