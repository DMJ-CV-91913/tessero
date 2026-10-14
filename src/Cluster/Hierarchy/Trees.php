<?php

declare(strict_types=1);

namespace Tessero\Cluster\Hierarchy;

/**
 * scipy.cluster.hierarchy data-structure helpers over a linkage matrix Z: to_tree (build a ClusterNode tree),
 * dendrogram (the deterministic DATA payload of scipy.cluster.hierarchy.dendrogram with no_plot=True) and
 * set_link_color_palette (the module-level link colour palette). Pure PHP over the already-landed linkage;
 * identical on both backends. The plotting itself (matplotlib) is out of scope; only the returned dict is.
 */
final class Trees
{
    /** scipy's default link-line colour cycle (_link_line_colors). C0 is the above-threshold colour. */
    private const DEFAULT_PALETTE = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'];

    /** @var list<string>|null the active palette, or null for the default */
    private static ?array $palette = null;

    /**
     * Normalize an NDArray / nested array linkage matrix into a (m x 4) nested PHP float array.
     *
     * @return list<array{0: float, 1: float, 2: float, 3: float}>
     */
    private static function asMatrix(mixed $Z): array
    {
        if (is_object($Z) && method_exists($Z, 'toArray')) {
            $Z = $Z->toArray();
        }
        $out = [];
        foreach ($Z as $row) {
            if (is_object($row) && method_exists($row, 'toArray')) {
                $row = $row->toArray();
            }
            $r = array_values($row);
            $out[] = [(float) $r[0], (float) $r[1], (float) $r[2], (float) $r[3]];
        }

        return $out;
    }

    /**
     * Build the binary ClusterNode tree from a linkage matrix Z (scipy.cluster.hierarchy.to_tree).
     * With $rd = false returns the root node; with $rd = true returns [root, nodes] where nodes[i] is the
     * ClusterNode with id i (0..2n-2).
     *
     * @return ClusterNode|array{0: ClusterNode, 1: list<ClusterNode>}
     */
    public static function toTree(mixed $Z, bool $rd = false): ClusterNode|array
    {
        $Zm = self::asMatrix($Z);
        $m = count($Zm);
        $n = $m + 1;
        if ($n < 1) {
            throw new \ValueError('The number of observations cannot be determined on an empty linkage matrix.');
        }
        /** @var list<ClusterNode|null> $d */
        $d = array_fill(0, $n * 2 - 1, null);
        for ($i = 0; $i < $n; $i++) {
            $d[$i] = new ClusterNode($i);
        }
        $nd = null;
        for ($i = 0; $i < $m; $i++) {
            $row = $Zm[$i];
            $fi = (int) $row[0];
            $fj = (int) $row[1];
            if ($fi > $i + $n || $fj > $i + $n) {
                throw new \ValueError('Corrupt matrix Z. Index to derivative cluster is used before it is formed.');
            }
            $nd = new ClusterNode($i + $n, $d[$fi], $d[$fj], (float) $row[2]);
            if ((int) $row[3] !== $nd->count) {
                throw new \ValueError("Corrupt matrix Z. The count Z[{$i},3] is incorrect.");
            }
            $d[$i + $n] = $nd;
        }

        if ($rd) {
            return [$nd, $d];
        }

        return $nd;
    }

    /** Set (or reset, with null) the module link-colour palette. scipy set_link_color_palette. */
    public static function setLinkColorPalette(?array $palette): void
    {
        if ($palette === null) {
            self::$palette = null;

            return;
        }
        $p = array_values($palette);
        foreach ($p as $c) {
            if (! is_string($c)) {
                throw new \TypeError('all elements of listOfColors must be strings');
            }
        }
        self::$palette = $p;
    }

    private static function palette(): array
    {
        return self::$palette ?? self::DEFAULT_PALETTE;
    }

    /**
     * The deterministic data payload of scipy.cluster.hierarchy.dendrogram(Z, no_plot=True) with default
     * ordering (no count_sort / distance_sort), default colour_threshold = 0.7 * max(Z[:,2]) and
     * above_threshold_color = 'C0'. Returns the dict keys icoord, dcoord, ivl, leaves, color_list,
     * leaves_color_list.
     *
     * @return array{icoord: list<list<float>>, dcoord: list<list<float>>, ivl: list<string>,
     *     leaves: list<int>, color_list: list<string>, leaves_color_list: list<string|null>}
     */
    public static function dendrogram(mixed $Z): array
    {
        $Zm = self::asMatrix($Z);
        $n = count($Zm) + 1;
        $palette = self::palette();
        $np = count($palette);

        $maxd = 0.0;
        foreach ($Zm as $row) {
            if ($row[2] > $maxd) {
                $maxd = $row[2];
            }
        }
        $colorThreshold = $maxd * 0.7;

        $icoord = [];
        $dcoord = [];
        $colorList = [];
        $ivl = [];
        $lvs = [];
        $currentColor = 0;
        $below = false;

        // iterative post-order over the implicit tree rooted at 2n-2, replicating scipy's recursion.
        // frame: [i, iv, stage, uiva, uwa, uah, uamd]
        $root = 2 * $n - 2;
        $stack = [[$root, 0.0, 0, 0.0, 0.0, 0.0, 0.0]];
        $ret = [0.0, 0.0, 0.0, 0.0];   // (left, w, h, md) returned by the last completed frame

        while ($stack) {
            $top = &$stack[count($stack) - 1];
            $i = $top[0];
            $iv = $top[1];
            $stage = $top[2];

            if ($i < $n) {                       // leaf
                $ivl[] = (string) $i;
                $lvs[] = $i;
                $ret = [$iv + 5.0, 10.0, 0.0, 0.0];
                array_pop($stack);
                unset($top);

                continue;
            }

            $aa = (int) $Zm[$i - $n][0];
            $ab = (int) $Zm[$i - $n][1];

            if ($stage === 0) {                  // descend into left child (ua = aa)
                $top[2] = 1;
                unset($top);
                $stack[] = [$aa, $iv, 0, 0.0, 0.0, 0.0, 0.0];

                continue;
            }

            if ($stage === 1) {                  // left child returned; pick colour, descend right
                [$uiva, $uwa, $uah, $uamd] = $ret;
                $top[3] = $uiva;
                $top[4] = $uwa;
                $top[5] = $uah;
                $top[6] = $uamd;
                $h = $Zm[$i - $n][2];
                if ($h >= $colorThreshold || $colorThreshold <= 0) {
                    $c = 'C0';
                    if ($below) {
                        $currentColor = ($currentColor + 1) % $np;
                    }
                    $below = false;
                } else {
                    $below = true;
                    $c = $palette[$currentColor % $np];
                }
                $top[2] = 2;
                $top[7] = $c;                     // colour is fixed BEFORE the right subtree recurses
                $uwaLocal = $uwa;
                unset($top);
                $stack[] = [$ab, $iv + $uwaLocal, 0, 0.0, 0.0, 0.0, 0.0];

                continue;
            }

            // stage === 2: right child returned; emit this link
            $uiva = $top[3];
            $uwa = $top[4];
            $uah = $top[5];
            $uamd = $top[6];
            $c = $top[7];
            [$uivb, $uwb, $ubh, $ubmd] = $ret;
            $h = $Zm[$i - $n][2];
            $maxDist = max($uamd, $ubmd, $h);
            $icoord[] = [$uiva, $uiva, $uivb, $uivb];
            $dcoord[] = [$uah, $h, $h, $ubh];
            $colorList[] = $c;
            $ret = [($uiva + $uivb) / 2, $uwa + $uwb, $h, $maxDist];
            array_pop($stack);
            unset($top);
        }

        // leaves_color_list: each leaf takes the colour of the link touching it at height 0.
        $leavesColor = array_fill(0, count($lvs), null);
        foreach ($icoord as $k => $lx) {
            $ly = $dcoord[$k];
            $lc = $colorList[$k];
            foreach ($lx as $pos => $xi) {
                $yi = $ly[$pos];
                if ($yi === 0.0 && fmod($xi, 5.0) === 0.0 && fmod($xi, 2.0) === 1.0) {
                    $leavesColor[(int) (((int) $xi - 5) / 10)] = $lc;
                }
            }
        }

        return [
            'icoord' => $icoord,
            'dcoord' => $dcoord,
            'ivl' => $ivl,
            'leaves' => $lvs,
            'color_list' => $colorList,
            'leaves_color_list' => $leavesColor,
        ];
    }
}
