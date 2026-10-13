<?php

declare(strict_types=1);

namespace Tessero\Integrate;

/**
 * Adaptive integration of a vector-valued function (scipy.integrate.quad_vec).
 *
 * A global adaptive scheme that, following QUADPACK's DQAG*, subdivides the N sub-intervals of
 * largest error at once (batch size 128) over a Gauss-Kronrod rule, until the global error estimate
 * drops below tol/8. The returned integral matches scipy.integrate.quad_vec's `res` bit-for-bit for
 * finite bounds; the error estimate is computed by the identical formula.
 *
 * The integrand $f(x) may return a float (scalar integrand) or a list<float> (vector integrand); the
 * result mirrors that shape. Only finite [a, b] bounds are supported (scipy's (semi-)infinite argument
 * transforms are out of scope here).
 *
 * Default quadrature is 'gk15' (scipy's default for the finite-interval path), NOT 'gk21'.
 */
final class QuadVec
{
    // 21-point Gauss-Kronrod abscissae
    private const GK21_X = [
        0.995657163025808080735527280689003, 0.973906528517171720077964012084452,
        0.930157491355708226001207180059508, 0.865063366688984510732096688423493,
        0.780817726586416897063717578345042, 0.679409568299024406234327365114874,
        0.562757134668604683339000099272694, 0.433395394129247190799265943165784,
        0.294392862701460198131126603103866, 0.148874338981631210884826001129720,
        0.0,
        -0.148874338981631210884826001129720, -0.294392862701460198131126603103866,
        -0.433395394129247190799265943165784, -0.562757134668604683339000099272694,
        -0.679409568299024406234327365114874, -0.780817726586416897063717578345042,
        -0.865063366688984510732096688423493, -0.930157491355708226001207180059508,
        -0.973906528517171720077964012084452, -0.995657163025808080735527280689003,
    ];

    // 10-point Gauss weights (21-point rule)
    private const GK21_W = [
        0.066671344308688137593568809893332, 0.149451349150580593145776339657697,
        0.219086362515982043995534934228163, 0.269266719309996355091226921569469,
        0.295524224714752870173892994651338, 0.295524224714752870173892994651338,
        0.269266719309996355091226921569469, 0.219086362515982043995534934228163,
        0.149451349150580593145776339657697, 0.066671344308688137593568809893332,
    ];

    // 21-point Kronrod weights
    private const GK21_V = [
        0.011694638867371874278064396062192, 0.032558162307964727478818972459390,
        0.054755896574351996031381300244580, 0.075039674810919952767043140916190,
        0.093125454583697605535065465083366, 0.109387158802297641899210590325805,
        0.123491976262065851077958109831074, 0.134709217311473325928054001771707,
        0.142775938577060080797094273138717, 0.147739104901338491374841515972068,
        0.149445554002916905664936468389821, 0.147739104901338491374841515972068,
        0.142775938577060080797094273138717, 0.134709217311473325928054001771707,
        0.123491976262065851077958109831074, 0.109387158802297641899210590325805,
        0.093125454583697605535065465083366, 0.075039674810919952767043140916190,
        0.054755896574351996031381300244580, 0.032558162307964727478818972459390,
        0.011694638867371874278064396062192,
    ];

    // 15-point Gauss-Kronrod abscissae
    private const GK15_X = [
        0.991455371120812639206854697526329, 0.949107912342758524526189684047851,
        0.864864423359769072789712788640926, 0.741531185599394439863864773280788,
        0.586087235467691130294144838258730, 0.405845151377397166906606412076961,
        0.207784955007898467600689403773245,
        0.000000000000000000000000000000000,
        -0.207784955007898467600689403773245, -0.405845151377397166906606412076961,
        -0.586087235467691130294144838258730, -0.741531185599394439863864773280788,
        -0.864864423359769072789712788640926, -0.949107912342758524526189684047851,
        -0.991455371120812639206854697526329,
    ];

    // 7-point Gauss weights (15-point rule)
    private const GK15_W = [
        0.129484966168869693270611432679082, 0.279705391489276667901467771423780,
        0.381830050505118944950369775488975, 0.417959183673469387755102040816327,
        0.381830050505118944950369775488975, 0.279705391489276667901467771423780,
        0.129484966168869693270611432679082,
    ];

    // 15-point Kronrod weights
    private const GK15_V = [
        0.022935322010529224963732008058970, 0.063092092629978553290700663189204,
        0.104790010322250183839876322541518, 0.140653259715525918745189590510238,
        0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
        0.204432940075298892414161999234649, 0.209482141084727828012999174891714,
        0.204432940075298892414161999234649, 0.190350578064785409913256402421014,
        0.169004726639267902826583426598550, 0.140653259715525918745189590510238,
        0.104790010322250183839876322541518, 0.063092092629978553290700663189204,
        0.022935322010529224963732008058970,
    ];

    /**
     * Integrate $f over [$a, $b].
     *
     * @param callable(float): (float|list<float>) $f
     * @param string $norm        '2' (Euclidean) or 'max' (infinity norm)
     * @param string|null $quadrature  'gk21' | 'gk15' (default). null => 'gk15'.
     * @return array{0: float|list<float>, 1: float}  [res, err], mirroring scipy.integrate.quad_vec.
     */
    public static function quadVec(
        callable $f,
        float $a,
        float $b,
        float $epsabs = 1e-200,
        float $epsrel = 1e-8,
        string $norm = '2',
        ?string $quadrature = null
    ): array {
        $a = (float) $a;
        $b = (float) $b;

        if (! (is_finite($a) && is_finite($b))) {
            throw new \InvalidArgumentException("invalid integration bounds a={$a}, b={$b}");
        }

        $rule = $quadrature ?? 'gk15';          // scipy default for finite bounds is gk15, NOT gk21
        if ($rule === 'gk21') {
            $x = self::GK21_X;
            $w = self::GK21_W;
            $v = self::GK21_V;
        } elseif ($rule === 'gk15') {
            $x = self::GK15_X;
            $w = self::GK15_W;
            $v = self::GK15_V;
        } else {
            throw new \InvalidArgumentException("unknown quadrature '{$rule}'");
        }

        // Normalizer selection; scipy maps null/'max' -> max norm, '2' -> np.linalg.norm.
        $normKind = ($norm === '2') ? '2' : 'max';

        $parallelCount = 128;
        $minIntervals = 2;

        // Initial panel.
        [$ig, $err, $rnd, $scalar] = self::gk($a, $b, $f, $x, $w, $v, $normKind, false);

        // Scalar specialization: scipy replaces the max/2-norm with plain abs() once it sees a float
        // result. For our internal representation a scalar is a 1-element vector whose 2-norm and
        // max-norm both already equal abs(value), so no behavioural switch is needed -- but we keep
        // the flag for clarity and to mirror scipy's control flow.
        $globalIntegral = $ig;                  // list<float> internally
        $globalError = (float) $err;
        $roundingError = (float) $rnd;

        // interval_cache: key "a|b" -> integral (list<float>)
        $cache = [];
        $cache[self::key($a, $b)] = $ig;

        // Min-heap of (-err, a, b) replicating Python heapq tuple ordering.
        $heap = new _QvHeap();
        $heap->push(-$err, $a, $b);

        while ($heap->count() > 0 && $heap->count() < 10000) {
            $tol = max($epsabs, $epsrel * self::norm($globalIntegral, $normKind));

            $toProcess = [];
            $errSum = 0.0;
            for ($j = 0; $j < $parallelCount; $j++) {
                if ($heap->count() === 0) {
                    break;
                }
                if ($j > 0 && $errSum > $globalError - $tol / 8) {
                    break;
                }
                [$negOldErr, $x1, $x2] = $heap->pop();
                $ck = self::key($x1, $x2);
                $oldInt = $cache[$ck] ?? null;
                unset($cache[$ck]);
                $toProcess[] = [-$negOldErr, $x1, $x2, $oldInt];
                $errSum += -$negOldErr;
            }

            foreach ($toProcess as [$oldErr, $x1, $x2, $oldInt]) {
                $c = 0.5 * ($x1 + $x2);
                [$s1, $e1, $r1] = self::gk($x1, $c, $f, $x, $w, $v, $normKind, false);
                [$s2, $e2, $r2] = self::gk($c, $x2, $f, $x, $w, $v, $normKind, false);
                if ($oldInt === null) {
                    [$oldInt] = self::gk($x1, $x2, $f, $x, $w, $v, $normKind, false);
                }

                // global_integral += s1 + s2 - old_int  (elementwise)
                $globalIntegral = self::vadd($globalIntegral, self::vsub(self::vadd($s1, $s2), $oldInt));
                $globalError += ($e1 + $e2) - $oldErr;
                $roundingError += $r1 + $r2;

                $cache[self::key($x1, $c)] = $s1;
                $cache[self::key($c, $x2)] = $s2;
                $heap->push(-$e1, $x1, $c);
                $heap->push(-$e2, $c, $x2);
            }

            if ($heap->count() >= $minIntervals) {
                $tol = max($epsabs, $epsrel * self::norm($globalIntegral, $normKind));
                if ($globalError < $tol / 8) {
                    break;
                }
                if ($globalError < $roundingError) {
                    break;
                }
            }

            if (! (is_finite($globalError) && is_finite($roundingError))) {
                break;
            }
        }

        $res = $scalar ? $globalIntegral[0] : array_values($globalIntegral);
        $errOut = $globalError + $roundingError;

        return [$res, $errOut];
    }

    /**
     * One Gauss-Kronrod panel on [$a, $b]. Returns [integral(list<float>), err(float), round_err(float), scalar(bool)].
     *
     * @param callable(float): (float|list<float>) $f
     * @param list<float> $x  abscissae
     * @param list<float> $w  Gauss weights
     * @param list<float> $v  Kronrod weights
     * @return array{0: list<float>, 1: float, 2: float, 3: bool}
     */
    private static function gk(float $a, float $b, callable $f, array $x, array $w, array $v, string $normKind, bool $ignored): array
    {
        $c = 0.5 * ($a + $b);
        $h = 0.5 * ($b - $a);
        $n = count($x);

        $fv = [];                 // list of list<float>
        $scalar = false;
        $dim = 0;
        $sK = null;               // list<float>
        $sKabs = null;

        for ($i = 0; $i < $n; $i++) {
            $ff = $f($c + $h * $x[$i]);
            if (! is_array($ff)) {
                $ff = [(float) $ff];
                $scalar = true;
            } else {
                $ff = array_values(array_map('floatval', $ff));
            }
            if ($sK === null) {
                $dim = count($ff);
                $sK = array_fill(0, $dim, 0.0);
                $sKabs = array_fill(0, $dim, 0.0);
            }
            $fv[$i] = $ff;
            $vv = $v[$i];
            for ($d = 0; $d < $dim; $d++) {
                $sK[$d] += $vv * $ff[$d];
                $sKabs[$d] += $vv * abs($ff[$d]);
            }
        }

        // Gauss sum over fv[2*i+1].
        $sG = array_fill(0, $dim, 0.0);
        $m = count($w);
        for ($i = 0; $i < $m; $i++) {
            $fvi = $fv[2 * $i + 1];
            for ($d = 0; $d < $dim; $d++) {
                $sG[$d] += $w[$i] * $fvi[$d];
            }
        }

        // abs-deviation from average y0 = s_k / 2
        $y0 = [];
        for ($d = 0; $d < $dim; $d++) {
            $y0[$d] = $sK[$d] / 2.0;
        }
        $sKdabs = array_fill(0, $dim, 0.0);
        for ($i = 0; $i < $n; $i++) {
            $fvi = $fv[$i];
            $vv = $v[$i];
            for ($d = 0; $d < $dim; $d++) {
                $sKdabs[$d] += $vv * abs($fvi[$d] - $y0[$d]);
            }
        }

        // err = norm((s_k - s_g) * h)
        $diffH = [];
        for ($d = 0; $d < $dim; $d++) {
            $diffH[$d] = ($sK[$d] - $sG[$d]) * $h;
        }
        $err = self::norm($diffH, $normKind);

        $dabsVec = [];
        for ($d = 0; $d < $dim; $d++) {
            $dabsVec[$d] = $sKdabs[$d] * $h;
        }
        $dabs = self::norm($dabsVec, $normKind);

        if ($dabs != 0.0 && $err != 0.0) {
            $err = $dabs * min(1.0, (200.0 * $err / $dabs) ** 1.5);
        }

        $eps = 2.220446049250313e-16;         // sys.float_info.epsilon (IEEE double)
        $tiny = 2.2250738585072014e-308;      // sys.float_info.min
        $roundVec = [];
        for ($d = 0; $d < $dim; $d++) {
            $roundVec[$d] = 50 * $eps * $h * $sKabs[$d];
        }
        $roundErr = self::norm($roundVec, $normKind);

        if ($roundErr > $tiny) {
            $err = max($err, $roundErr);
        }

        // integral = h * s_k
        $integral = [];
        for ($d = 0; $d < $dim; $d++) {
            $integral[$d] = $h * $sK[$d];
        }

        return [$integral, $err, $roundErr, $scalar];
    }

    /**
     * Vector norm matching numpy. '2' => sqrt(sum sq) (np.linalg.norm of a 1-D array), 'max' => max(|.|).
     * For a 1-element vector both equal abs(value) -- the scalar specialization.
     *
     * @param list<float> $vec
     */
    private static function norm(array $vec, string $normKind): float
    {
        if ($normKind === 'max') {
            $m = 0.0;
            foreach ($vec as $vv) {
                $av = abs($vv);
                if ($av > $m) {
                    $m = $av;
                }
            }
            return $m;
        }
        // 2-norm: sqrt(sum of squares). np.linalg.norm does not use a special hypot path for the
        // general case; it computes sqrt(dot(x, x)) which is sum of v*v then sqrt.
        $s = 0.0;
        foreach ($vec as $vv) {
            $s += $vv * $vv;
        }
        return sqrt($s);
    }

    /** @param list<float> $x @param list<float> $y @return list<float> */
    private static function vadd(array $x, array $y): array
    {
        $o = [];
        foreach ($x as $i => $xi) {
            $o[$i] = $xi + $y[$i];
        }
        return $o;
    }

    /** @param list<float> $x @param list<float> $y @return list<float> */
    private static function vsub(array $x, array $y): array
    {
        $o = [];
        foreach ($x as $i => $xi) {
            $o[$i] = $xi - $y[$i];
        }
        return $o;
    }

    private static function key(float $a, float $b): string
    {
        // Pack the exact IEEE bits so distinct floats never collide (c = 0.5*(a+b) can repeat values).
        return pack('d', $a) . '|' . pack('d', $b);
    }
}

/**
 * Minimal binary min-heap over (negErr, a, b) replicating Python heapq's tuple ordering:
 * compare negErr, then a, then b (all floats). heapq is not stable, but with the full 3-tuple key the
 * ordering is total for the values quad_vec produces, so the pop sequence matches scipy's.
 *
 * @internal
 */
final class _QvHeap
{
    /** @var list<array{0: float, 1: float, 2: float}> */
    private array $h = [];

    public function count(): int
    {
        return count($this->h);
    }

    public function push(float $negErr, float $a, float $b): void
    {
        $this->h[] = [$negErr, $a, $b];
        $i = count($this->h) - 1;
        while ($i > 0) {
            $p = ($i - 1) >> 1;
            if ($this->less($this->h[$i], $this->h[$p])) {
                [$this->h[$i], $this->h[$p]] = [$this->h[$p], $this->h[$i]];
                $i = $p;
            } else {
                break;
            }
        }
    }

    /** @return array{0: float, 1: float, 2: float} */
    public function pop(): array
    {
        $n = count($this->h);
        $top = $this->h[0];
        $last = array_pop($this->h);
        $n--;
        if ($n > 0) {
            $this->h[0] = $last;
            $i = 0;
            while (true) {
                $l = 2 * $i + 1;
                $r = 2 * $i + 2;
                $sm = $i;
                if ($l < $n && $this->less($this->h[$l], $this->h[$sm])) {
                    $sm = $l;
                }
                if ($r < $n && $this->less($this->h[$r], $this->h[$sm])) {
                    $sm = $r;
                }
                if ($sm === $i) {
                    break;
                }
                [$this->h[$i], $this->h[$sm]] = [$this->h[$sm], $this->h[$i]];
                $i = $sm;
            }
        }
        return $top;
    }

    /**
     * @param array{0: float, 1: float, 2: float} $x
     * @param array{0: float, 1: float, 2: float} $y
     */
    private function less(array $x, array $y): bool
    {
        if ($x[0] !== $y[0]) {
            return $x[0] < $y[0];
        }
        if ($x[1] !== $y[1]) {
            return $x[1] < $y[1];
        }
        return $x[2] < $y[2];
    }
}
