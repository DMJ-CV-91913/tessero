<?php

declare(strict_types=1);

namespace Tessero\Integrate;

/**
 * Adaptive multidimensional cubature (scipy.integrate.cubature).
 *
 * Globally adaptive hypercube integration of a scalar-valued function f: R^ndim -> R over the box
 * [a_i, b_i]. A base cubature rule provides an estimate and a local error estimate for each region; the
 * region of greatest error is popped from a max-heap, split into 2^ndim equal subregions, and each
 * subregion is re-estimated and pushed back, until the global error falls below atol + rtol*|est|.
 *
 * Two rule families are supported, matching scipy:
 *   - "gk21" (default), "gk15", "gauss-kronrod": the ndim-fold Cartesian product of a 1-D Gauss-Kronrod
 *     rule. The higher rule (21- or 15-point Gauss-Kronrod) gives the estimate; the error is formed from
 *     the embedded lower Gauss-Legendre rule (npoints//2 nodes).
 *   - "genz-malik": the degree-7 Genz-Malik cubature with its embedded degree-5 rule (ndim >= 2 only).
 *
 * The 1-D Gauss-Kronrod nodes/weights are the QUADPACK dqk21/dqk15 constants; the embedded Gauss-Legendre
 * nodes are scipy roots_legendre(n//2). The Genz-Malik nodes/weights follow Genz & Malik (1980). The
 * returned estimate matches scipy.integrate.cubature(...).estimate to full working precision. Pure PHP,
 * host-independent (no kernel, no LAPACK).
 *
 * @see A.C. Genz, A.A. Malik, "Remarks on algorithm 006...", J. Comput. Appl. Math. 6 (1980) 295-302.
 */
final class Cubature
{
    private const EPS = 2.220446049250313e-16;

    // QUADPACK dqk21 higher-rule nodes (21-point Gauss-Kronrod on [-1, 1]).
    private const GK21_NODES = [
        0.995657163025808080735527280689003, 0.973906528517171720077964012084452,
        0.930157491355708226001207180059508, 0.865063366688984510732096688423493,
        0.780817726586416897063717578345042, 0.679409568299024406234327365114874,
        0.562757134668604683339000099272694, 0.433395394129247190799265943165784,
        0.294392862701460198131126603103866, 0.148874338981631210884826001129720,
        0,
        -0.148874338981631210884826001129720, -0.294392862701460198131126603103866,
        -0.433395394129247190799265943165784, -0.562757134668604683339000099272694,
        -0.679409568299024406234327365114874, -0.780817726586416897063717578345042,
        -0.865063366688984510732096688423493, -0.930157491355708226001207180059508,
        -0.973906528517171720077964012084452, -0.995657163025808080735527280689003,
    ];
    private const GK21_WEIGHTS = [
        0.011694638867371874278064396062192, 0.032558162307964727478818972459390,
        0.054755896574351996031381300244580, 0.075039674810919952767043140916190,
        0.093125454583697605535065465083366, 0.109387158802297641899210590325805,
        0.123491976262065851077958109831074, 0.134709217311473325928054001771707,
        0.142775938577060080797094273138717, 0.147739104901338491374841515972068,
        0.149445554002916905664936468389821,
        0.147739104901338491374841515972068, 0.142775938577060080797094273138717,
        0.134709217311473325928054001771707, 0.123491976262065851077958109831074,
        0.109387158802297641899210590325805, 0.093125454583697605535065465083366,
        0.075039674810919952767043140916190, 0.054755896574351996031381300244580,
        0.032558162307964727478818972459390, 0.011694638867371874278064396062192,
    ];
    // QUADPACK dqk15 higher-rule nodes (15-point Gauss-Kronrod on [-1, 1]).
    private const GK15_NODES = [
        0.991455371120812639206854697526329, 0.949107912342758524526189684047851,
        0.864864423359769072789712788640926, 0.741531185599394439863864773280788,
        0.586087235467691130294144838258730, 0.405845151377397166906606412076961,
        0.207784955007898467600689403773245, 0.000000000000000000000000000000000,
        -0.207784955007898467600689403773245, -0.405845151377397166906606412076961,
        -0.586087235467691130294144838258730, -0.741531185599394439863864773280788,
        -0.864864423359769072789712788640926, -0.949107912342758524526189684047851,
        -0.991455371120812639206854697526329,
    ];
    private const GK15_WEIGHTS = [
        0.022935322010529224963732008058970, 0.063092092629978553290700663189204,
        0.104790010322250183839876322541518, 0.140653259715525918745189590510238,
        0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
        0.204432940075298892414161999234649, 0.209482141084727828012999174891714,
        0.204432940075298892414161999234649, 0.190350578064785409913256402421014,
        0.169004726639267902826583426598550, 0.140653259715525918745189590510238,
        0.104790010322250183839876322541518, 0.063092092629978553290700663189204,
        0.022935322010529224963732008058970,
    ];
    // Embedded Gauss-Legendre lower rule: roots_legendre(10) for GK21, roots_legendre(7) for GK15.
    private const GL10_NODES = [
        -0.9739065285171717, -0.8650633666889844, -0.6794095682990244, -0.4333953941292472,
        -0.14887433898163116, 0.14887433898163116, 0.4333953941292472, 0.6794095682990244,
        0.8650633666889844, 0.9739065285171717,
    ];
    private const GL10_WEIGHTS = [
        0.06667134430868714, 0.14945134915058053, 0.21908636251598224, 0.26926671930999674,
        0.2955242247147533, 0.2955242247147533, 0.26926671930999674, 0.21908636251598224,
        0.14945134915058053, 0.06667134430868714,
    ];
    private const GL7_NODES = [
        -0.9491079123427584, -0.7415311855993945, -0.4058451513773972, 0.0,
        0.4058451513773972, 0.7415311855993945, 0.9491079123427584,
    ];
    private const GL7_WEIGHTS = [
        0.12948496616886992, 0.2797053914892766, 0.38183005050511876, 0.4179591836734691,
        0.38183005050511876, 0.2797053914892766, 0.12948496616886992,
    ];

    /**
     * Integrate $f over the box [$a, $b] (elementwise lower/upper limits).
     *
     * @param callable(list<float>): float $f    scalar integrand of an ndim-vector
     * @param list<float>                   $a    lower limits
     * @param list<float>                   $b    upper limits
     * @param string                        $rule "gk21" (default), "gk15", "gauss-kronrod", "genz-malik"
     * @return array{estimate: float, error: float, status: string, subdivisions: int}
     */
    public static function cubature(
        callable $f,
        array $a,
        array $b,
        string $rule = 'gk21',
        float $rtol = 1e-8,
        float $atol = 0.0,
        int $maxSubdivisions = 10000
    ): array {
        $ndim = count($a);

        // Flip any reversed limits, tracking the overall sign.
        $sign = 1.0;
        $al = [];
        $bl = [];
        for ($i = 0; $i < $ndim; $i++) {
            if ($a[$i] > $b[$i]) { $sign = -$sign; $al[$i] = $b[$i]; $bl[$i] = $a[$i]; }
            else { $al[$i] = $a[$i]; $bl[$i] = $b[$i]; }
        }

        // Build the rule node/weight sets (nodes on [-1, 1]^ndim).
        [$nodes, $weights, $errNodes, $errWeights] = self::buildRule($rule, $ndim);

        // Initial region.
        $estK = self::applyRule($f, $al, $bl, $nodes, $weights, $ndim);
        $errK = self::applyErr($f, $al, $bl, $errNodes, $errWeights, $ndim);
        $regions = [[$errK, $estK, $al, $bl]];
        $est = $estK;
        $err = $errK;

        $subdiv = 0;
        $success = true;
        while ($err > $atol + $rtol * abs($est)) {
            // Pop the region of greatest error (linear scan; region counts here are modest).
            $maxI = 0;
            for ($i = 1, $nr = count($regions); $i < $nr; $i++) {
                if ($regions[$i][0] > $regions[$maxI][0]) { $maxI = $i; }
            }
            [$rErr, $rEst, $ra, $rb] = $regions[$maxI];
            array_splice($regions, $maxI, 1);
            $est -= $rEst;
            $err -= $rErr;

            foreach (self::splitSubregions($ra, $rb, $ndim) as [$sa, $sb]) {
                $se = self::applyRule($f, $sa, $sb, $nodes, $weights, $ndim);
                $serr = self::applyErr($f, $sa, $sb, $errNodes, $errWeights, $ndim);
                $est += $se;
                $err += $serr;
                $regions[] = [$serr, $se, $sa, $sb];
            }
            $subdiv++;
            if ($subdiv >= $maxSubdivisions) { $success = false; break; }
        }

        return [
            'estimate' => $sign * $est,
            'error' => $err,
            'status' => $success ? 'converged' : 'not_converged',
            'subdivisions' => $subdiv,
        ];
    }

    /**
     * Assemble (higher nodes, higher weights, error nodes, error weights) for the chosen rule, all on the
     * reference box [-1, 1]^ndim. Nodes are lists of ndim-vectors; weights are parallel lists of floats.
     *
     * @return array{0: list<list<float>>, 1: list<float>, 2: list<list<float>>, 3: list<float>}
     */
    private static function buildRule(string $rule, int $ndim): array
    {
        if ($rule === 'genz-malik') {
            if ($ndim < 2) {
                throw new \RuntimeException('Tessero\Integrate\Cubature: genz-malik requires ndim >= 2');
            }
            [$hn, $hw] = self::genzMalikHigher($ndim);
            [$ln, $lw] = self::genzMalikLower($ndim);
            // Error = |higher - lower|: concat with negated lower weights.
            $en = array_merge($hn, $ln);
            $ew = array_merge($hw, array_map(static fn ($v) => -$v, $lw));
            return [$hn, $hw, $en, $ew];
        }

        // Product Gauss-Kronrod.
        if ($rule === 'gk15') {
            $n1 = self::GK15_NODES; $w1 = self::GK15_WEIGHTS; $ln1 = self::GL7_NODES; $lw1 = self::GL7_WEIGHTS;
        } elseif ($rule === 'gk21' || $rule === 'gauss-kronrod') {
            $n1 = self::GK21_NODES; $w1 = self::GK21_WEIGHTS; $ln1 = self::GL10_NODES; $lw1 = self::GL10_WEIGHTS;
        } else {
            throw new \RuntimeException("Tessero\Integrate\Cubature: unknown rule '{$rule}'");
        }

        [$hn, $hw] = self::productRule($n1, $w1, $ndim);
        [$lnN, $lnW] = self::productRule($ln1, $lw1, $ndim);
        $en = array_merge($hn, $lnN);
        $ew = array_merge($hw, array_map(static fn ($v) => -$v, $lnW));
        return [$hn, $hw, $en, $ew];
    }

    /**
     * ndim-fold Cartesian product of a 1-D node/weight set (ij ordering: last axis varies fastest).
     *
     * @param list<float> $nodes1
     * @param list<float> $weights1
     * @return array{0: list<list<float>>, 1: list<float>}
     */
    private static function productRule(array $nodes1, array $weights1, int $ndim): array
    {
        $accNodes = [[]];
        $accW = [1.0];
        for ($d = 0; $d < $ndim; $d++) {
            $newNodes = [];
            $newW = [];
            foreach ($accNodes as $ci => $combo) {
                foreach ($nodes1 as $ni => $nv) {
                    $newNodes[] = array_merge($combo, [$nv]);
                    $newW[] = $accW[$ci] * $weights1[$ni];
                }
            }
            $accNodes = $newNodes;
            $accW = $newW;
        }
        return [$accNodes, $accW];
    }

    /**
     * Degree-7 Genz-Malik higher rule nodes/weights on [-1, 1]^ndim.
     *
     * @return array{0: list<list<float>>, 1: list<float>}
     */
    private static function genzMalikHigher(int $ndim): array
    {
        $l2 = sqrt(9.0 / 70.0);
        $l3 = sqrt(9.0 / 10.0);
        $l4 = sqrt(9.0 / 10.0);
        $l5 = sqrt(9.0 / 19.0);

        $nodes = [];
        $nodes[] = array_fill(0, $ndim, 0.0);                 // centre
        self::appendOnePerm($nodes, $l2, $ndim);              // +-l2 e_i
        self::appendOnePerm($nodes, -$l2, $ndim);
        self::appendOnePerm($nodes, $l3, $ndim);              // +-l3 e_i
        self::appendOnePerm($nodes, -$l3, $ndim);
        self::appendTwoPerm($nodes, $l4, $l4, $ndim);         // (l4,l4),(l4,-l4),(-l4,-l4) over pairs
        self::appendTwoPerm($nodes, $l4, -$l4, $ndim);
        self::appendTwoPerm($nodes, -$l4, -$l4, $ndim);
        self::appendFullProduct($nodes, $l5, $ndim);          // (+-l5)^ndim

        $pw = 2 ** $ndim;
        $w1 = $pw * (12824 - 9120 * $ndim + 400 * $ndim * $ndim) / 19683.0;
        $w2 = $pw * 980.0 / 6561.0;
        $w3 = $pw * (1820 - 400 * $ndim) / 19683.0;
        $w4 = $pw * (200.0 / 19683.0);
        $w5 = 6859.0 / 19683.0;

        $weights = [];
        $weights[] = $w1;
        for ($i = 0; $i < 2 * $ndim; $i++) { $weights[] = $w2; }
        for ($i = 0; $i < 2 * $ndim; $i++) { $weights[] = $w3; }
        for ($i = 0; $i < 2 * ($ndim - 1) * $ndim; $i++) { $weights[] = $w4; }
        for ($i = 0; $i < $pw; $i++) { $weights[] = $w5; }

        return [$nodes, $weights];
    }

    /**
     * Degree-5 Genz-Malik lower rule (same nodes as higher minus the l5 full-product block).
     *
     * @return array{0: list<list<float>>, 1: list<float>}
     */
    private static function genzMalikLower(int $ndim): array
    {
        $l2 = sqrt(9.0 / 70.0);
        $l3 = sqrt(9.0 / 10.0);
        $l4 = sqrt(9.0 / 10.0);

        $nodes = [];
        $nodes[] = array_fill(0, $ndim, 0.0);
        self::appendOnePerm($nodes, $l2, $ndim);
        self::appendOnePerm($nodes, -$l2, $ndim);
        self::appendOnePerm($nodes, $l3, $ndim);
        self::appendOnePerm($nodes, -$l3, $ndim);
        self::appendTwoPerm($nodes, $l4, $l4, $ndim);
        self::appendTwoPerm($nodes, $l4, -$l4, $ndim);
        self::appendTwoPerm($nodes, -$l4, -$l4, $ndim);

        $pw = 2 ** $ndim;
        $w1 = $pw * (729 - 950 * $ndim + 50 * $ndim * $ndim) / 729.0;
        $w2 = $pw * (245.0 / 486.0);
        $w3 = $pw * (265 - 100 * $ndim) / 1458.0;
        $w4 = $pw * (25.0 / 729.0);

        $weights = [];
        $weights[] = $w1;
        for ($i = 0; $i < 2 * $ndim; $i++) { $weights[] = $w2; }
        for ($i = 0; $i < 2 * $ndim; $i++) { $weights[] = $w3; }
        for ($i = 0; $i < 2 * ($ndim - 1) * $ndim; $i++) { $weights[] = $w4; }

        return [$nodes, $weights];
    }

    /**
     * Append distinct permutations of (v, 0, ..., 0) (one non-zero coordinate), in the ascending-sort
     * permutation order scipy _distinct_permutations uses.
     *
     * @param list<list<float>> $nodes
     */
    private static function appendOnePerm(array &$nodes, float $v, int $ndim): void
    {
        $tuple = array_merge([$v], array_fill(0, $ndim - 1, 0.0));
        foreach (self::distinctPermutations($tuple) as $p) { $nodes[] = $p; }
    }

    /**
     * Append distinct permutations of (v1, v2, 0, ..., 0) (two non-zero coordinates).
     *
     * @param list<list<float>> $nodes
     */
    private static function appendTwoPerm(array &$nodes, float $v1, float $v2, int $ndim): void
    {
        $tuple = array_merge([$v1, $v2], array_fill(0, $ndim - 2, 0.0));
        foreach (self::distinctPermutations($tuple) as $p) { $nodes[] = $p; }
    }

    /**
     * Append the full Cartesian product (+-v)^ndim, last axis varying fastest (itertools.product order).
     *
     * @param list<list<float>> $nodes
     */
    private static function appendFullProduct(array &$nodes, float $v, int $ndim): void
    {
        $combos = [[]];
        for ($d = 0; $d < $ndim; $d++) {
            $next = [];
            foreach ($combos as $c) {
                $next[] = array_merge($c, [$v]);
                $next[] = array_merge($c, [-$v]);
            }
            $combos = $next;
        }
        foreach ($combos as $c) { $nodes[] = $c; }
    }

    /**
     * Distinct permutations of a tuple in the lexicographic (ascending-sort) order scipy uses
     * (Wikipedia next-permutation on the sorted sequence).
     *
     * @param list<float> $iterable
     * @return list<list<float>>
     */
    private static function distinctPermutations(array $iterable): array
    {
        $items = $iterable;
        sort($items);                       // ascending
        $size = count($items);
        $out = [];
        while (true) {
            $out[] = $items;
            // largest i with items[i] < items[i+1]
            $i = $size - 2;
            while ($i >= 0 && ! ($items[$i] < $items[$i + 1])) { $i--; }
            if ($i < 0) { break; }
            // largest j > i with items[i] < items[j]
            $j = $size - 1;
            while (! ($items[$i] < $items[$j])) { $j--; }
            $t = $items[$i]; $items[$i] = $items[$j]; $items[$j] = $t;
            // reverse items[i+1:]
            $tail = array_reverse(array_slice($items, $i + 1));
            array_splice($items, $i + 1, count($tail), $tail);
        }
        return $out;
    }

    /**
     * Weighted-sum estimate of f over [a, b] for nodes given on [-1, 1]^ndim (scipy _apply_fixed_rule:
     * affine map node -> (node+1)*(len/2)+a, weight scale prod(len)/2^ndim).
     *
     * @param callable(list<float>): float $f
     * @param list<float>                  $a
     * @param list<float>                  $b
     * @param list<list<float>>            $nodes
     * @param list<float>                  $weights
     */
    private static function applyRule(callable $f, array $a, array $b, array $nodes, array $weights, int $ndim): float
    {
        $len = [];
        $prodLen = 1.0;
        for ($i = 0; $i < $ndim; $i++) { $len[$i] = $b[$i] - $a[$i]; $prodLen *= $len[$i]; }
        $scale = $prodLen / (2 ** $ndim);

        $est = 0.0;
        foreach ($nodes as $ni => $node) {
            $x = [];
            for ($i = 0; $i < $ndim; $i++) { $x[$i] = ($node[$i] + 1.0) * ($len[$i] * 0.5) + $a[$i]; }
            $est += ($weights[$ni] * $scale) * (float) $f($x);
        }
        return $est;
    }

    /**
     * Error estimate: |weighted sum over the concatenated (higher, -lower) node set|.
     *
     * @param callable(list<float>): float $f
     * @param list<float>                  $a
     * @param list<float>                  $b
     * @param list<list<float>>            $errNodes
     * @param list<float>                  $errWeights
     */
    private static function applyErr(callable $f, array $a, array $b, array $errNodes, array $errWeights, int $ndim): float
    {
        return abs(self::applyRule($f, $a, $b, $errNodes, $errWeights, $ndim));
    }

    /**
     * Yield the 2^ndim subregions formed by bisecting [a, b] along every axis (ij cartesian order).
     *
     * @param list<float> $a
     * @param list<float> $b
     * @return list<array{0: list<float>, 1: list<float>}>
     */
    private static function splitSubregions(array $a, array $b, int $ndim): array
    {
        $mid = [];
        for ($i = 0; $i < $ndim; $i++) { $mid[$i] = ($a[$i] + $b[$i]) / 2.0; }
        $regions = [[[], []]];
        for ($i = 0; $i < $ndim; $i++) {
            $next = [];
            foreach ($regions as [$ra, $rb]) {
                $next[] = [array_merge($ra, [$a[$i]]), array_merge($rb, [$mid[$i]])];
                $next[] = [array_merge($ra, [$mid[$i]]), array_merge($rb, [$b[$i]])];
            }
            $regions = $next;
        }
        return $regions;
    }
}
