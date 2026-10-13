<?php

declare(strict_types=1);

namespace Tessero\Spatial\Transform;

/**
 * scipy.spatial.transform.Rotation: 3-D rotations with interchangeable representations.
 * Pure PHP (no kernel), identical on both backends. Quaternions use scipy's SCALAR-LAST
 * convention [x, y, z, w]. A Rotation holds a stack of quaternions (shape (N,4)); a single
 * rotation is stored as one row and its as_* accessors return the un-stacked shape.
 *
 * REPRESENTATIONS (all round-trip exactly through the rotation matrix, the sign-unambiguous
 * invariant): from_quat/as_quat, from_matrix/as_matrix, from_rotvec/as_rotvec,
 * from_euler/as_euler, from_mrp/as_mrp. OPERATIONS: apply, inv, magnitude, compose, identity.
 *
 * as_quat returns the STORED quaternion sign (matching scipy); fixtures compare quaternions
 * up to the global q/-q sign and compare rotation matrices exactly.
 */
final class Rotation
{
    /** @var list<array{0:float,1:float,2:float,3:float}> stacked unit quaternions [x,y,z,w] */
    private array $quat;

    /** @var bool whether this represents a single (un-stacked) rotation */
    private bool $single;

    private function __construct(array $quat, bool $single)
    {
        $this->quat = $quat;
        $this->single = $single;
    }

    public function single(): bool
    {
        return $this->single;
    }

    public function count(): int
    {
        return count($this->quat);
    }

    // ---- constructors --------------------------------------------------

    public static function fromQuat(array $quat): self
    {
        [$rows, $single] = self::as2d($quat, 4, 'quat');
        $out = [];
        foreach ($rows as $q) {
            $n = sqrt($q[0] * $q[0] + $q[1] * $q[1] + $q[2] * $q[2] + $q[3] * $q[3]);
            if ($n == 0.0) {
                throw new \InvalidArgumentException('zero-norm quaternion');
            }
            $out[] = [$q[0] / $n, $q[1] / $n, $q[2] / $n, $q[3] / $n];
        }
        return new self($out, $single);
    }

    public static function fromMatrix(array $matrix): self
    {
        $single = false;
        $mats = self::as3x3($matrix, $single);
        $out = [];
        foreach ($mats as $m) {
            $out[] = self::matToQuat($m);
        }
        return new self($out, $single);
    }

    public static function fromRotvec(array $rotvec, bool $degrees = false): self
    {
        [$rows, $single] = self::as2d($rotvec, 3, 'rotvec');
        $scale = $degrees ? M_PI / 180.0 : 1.0;
        $out = [];
        foreach ($rows as $v) {
            $x = $v[0] * $scale; $y = $v[1] * $scale; $z = $v[2] * $scale;
            $angle = sqrt($x * $x + $y * $y + $z * $z);
            if ($angle < 1e-6) {
                $s = 0.5 - $angle * $angle / 48.0;
            } else {
                $s = sin($angle / 2.0) / $angle;
            }
            $out[] = [$x * $s, $y * $s, $z * $s, cos($angle / 2.0)];
        }
        return new self($out, $single);
    }

    public static function fromMrp(array $mrp): self
    {
        [$rows, $single] = self::as2d($mrp, 3, 'mrp');
        $out = [];
        foreach ($rows as $p) {
            $n2 = $p[0] * $p[0] + $p[1] * $p[1] + $p[2] * $p[2];
            $d = 1.0 + $n2;
            $out[] = [2.0 * $p[0] / $d, 2.0 * $p[1] / $d, 2.0 * $p[2] / $d, (1.0 - $n2) / $d];
        }
        return new self($out, $single);
    }

    public static function fromEuler(string $seq, array|float $angles, bool $degrees = false): self
    {
        $axes = str_split($seq);
        $n = count($axes);
        if ($n < 1 || $n > 3) {
            throw new \InvalidArgumentException('Euler sequence must have 1 to 3 axes');
        }
        if (strtolower($seq) !== $seq && strtoupper($seq) !== $seq) {
            throw new \InvalidArgumentException('Euler axes must be all upper- or all lower-case');
        }
        $intrinsic = ctype_upper($seq[0]);
        $flat = !is_array($angles) || !is_array(array_values($angles)[0] ?? 0.0);
        $rows = self::anglesAs2d($angles, $n);
        $scale = $degrees ? M_PI / 180.0 : 1.0;
        $out = [];
        foreach ($rows as $ang) {
            $q = [0.0, 0.0, 0.0, 1.0];
            for ($i = 0; $i < $n; $i++) {
                $e = self::elementaryQuat(strtolower($axes[$i]), $ang[$i] * $scale);
                // extrinsic: pre-multiply (left); intrinsic: post-multiply (right)
                $q = $intrinsic ? self::quatMul($q, $e) : self::quatMul($e, $q);
            }
            $out[] = $q;
        }
        return new self($out, $flat);
    }

    public static function identity(?int $num = null): self
    {
        if ($num === null) {
            return new self([[0.0, 0.0, 0.0, 1.0]], true);
        }
        return new self(array_fill(0, $num, [0.0, 0.0, 0.0, 1.0]), false);
    }

    // ---- accessors -----------------------------------------------------

    public function asQuat(): array
    {
        return $this->unstack($this->quat);
    }

    public function asMatrix(): array
    {
        $out = [];
        foreach ($this->quat as $q) {
            $out[] = self::quatToMat($q);
        }
        return $this->unstack($out);
    }

    public function asRotvec(bool $degrees = false): array
    {
        $scale = $degrees ? 180.0 / M_PI : 1.0;
        $out = [];
        foreach ($this->quat as $q) {
            [$x, $y, $z, $w] = $q;
            if ($w < 0) { $x = -$x; $y = -$y; $z = -$z; $w = -$w; }
            $sinHalf = sqrt($x * $x + $y * $y + $z * $z);
            $angle = 2.0 * atan2($sinHalf, $w);
            if ($sinHalf < 1e-6) {
                $k = 2.0 + $angle * $angle / 12.0;
            } else {
                $k = $angle / $sinHalf;
            }
            $out[] = [$x * $k * $scale, $y * $k * $scale, $z * $k * $scale];
        }
        return $this->unstack($out);
    }

    public function asMrp(): array
    {
        $out = [];
        foreach ($this->quat as $q) {
            [$x, $y, $z, $w] = $q;
            if ($w < 0) { $x = -$x; $y = -$y; $z = -$z; $w = -$w; }
            $d = 1.0 + $w;
            $out[] = [$x / $d, $y / $d, $z / $d];
        }
        return $this->unstack($out);
    }

    public function asEuler(string $seq, bool $degrees = false): array
    {
        if (strlen($seq) !== 3) {
            throw new \InvalidArgumentException('as_euler requires a 3-axis sequence');
        }
        $intrinsic = ctype_upper($seq[0]);
        $axes = array_map('strtolower', str_split($seq));
        $scale = $degrees ? 180.0 / M_PI : 1.0;
        $out = [];
        foreach ($this->quat as $q) {
            $e = self::quatToEuler($q, $axes, $intrinsic);
            $out[] = [$e[0] * $scale, $e[1] * $scale, $e[2] * $scale];
        }
        return $this->unstack($out);
    }

    // ---- operations ----------------------------------------------------

    public function inv(): self
    {
        $out = [];
        foreach ($this->quat as $q) {
            $out[] = [-$q[0], -$q[1], -$q[2], $q[3]];
        }
        return new self($out, $this->single);
    }

    public function magnitude(): array|float
    {
        $out = [];
        foreach ($this->quat as $q) {
            $w = $q[3] < 0 ? -$q[3] : $q[3];
            $s = sqrt($q[0] * $q[0] + $q[1] * $q[1] + $q[2] * $q[2]);
            $out[] = 2.0 * atan2($s, $w);
        }
        return $this->single ? $out[0] : $out;
    }

    public function apply(array $vectors, bool $inverse = false): array
    {
        [$vrows, $vsingle] = self::as2d($vectors, 3, 'vectors');
        $mats = [];
        foreach ($this->quat as $q) {
            $mats[] = self::quatToMat($q);
        }
        $nr = count($mats);
        $nv = count($vrows);
        $out = [];
        $count = max($nr, $nv);
        for ($i = 0; $i < $count; $i++) {
            $m = $mats[$nr === 1 ? 0 : $i];
            $v = $vrows[$nv === 1 ? 0 : $i];
            $r = [0.0, 0.0, 0.0];
            for ($a = 0; $a < 3; $a++) {
                for ($b = 0; $b < 3; $b++) {
                    $r[$a] += ($inverse ? $m[$b][$a] : $m[$a][$b]) * $v[$b];
                }
            }
            $out[] = $r;
        }
        if ($nr === 1 && $vsingle) {
            return $out[0];
        }
        return $out;
    }

    public function compose(self $other): self
    {
        $a = $this->quat; $b = $other->quat;
        $na = count($a); $nb = count($b);
        $n = max($na, $nb);
        $out = [];
        for ($i = 0; $i < $n; $i++) {
            $out[] = self::quatMul($a[$na === 1 ? 0 : $i], $b[$nb === 1 ? 0 : $i]);
        }
        return new self($out, $this->single && $other->single);
    }

    // ---- quaternion / matrix / euler kernels ---------------------------

    private static function quatToMat(array $q): array
    {
        [$x, $y, $z, $w] = $q;
        $xx = $x * $x; $yy = $y * $y; $zz = $z * $z;
        $xy = $x * $y; $xz = $x * $z; $yz = $y * $z;
        $wx = $w * $x; $wy = $w * $y; $wz = $w * $z;
        return [
            [1 - 2 * ($yy + $zz), 2 * ($xy - $wz), 2 * ($xz + $wy)],
            [2 * ($xy + $wz), 1 - 2 * ($xx + $zz), 2 * ($yz - $wx)],
            [2 * ($xz - $wy), 2 * ($yz + $wx), 1 - 2 * ($xx + $yy)],
        ];
    }

    /** Shepperd's method (largest-component branch), matching scipy's matrix-to-quat. */
    private static function matToQuat(array $m): array
    {
        $t = $m[0][0] + $m[1][1] + $m[2][2];
        $d = [$m[0][0], $m[1][1], $m[2][2], $t];
        $i = 0;
        for ($k = 1; $k < 4; $k++) {
            if ($d[$k] > $d[$i]) {
                $i = $k;
            }
        }
        $q = [0.0, 0.0, 0.0, 0.0];
        if ($i === 3) {
            $q[3] = 1 + $t;
            $q[0] = $m[2][1] - $m[1][2];
            $q[1] = $m[0][2] - $m[2][0];
            $q[2] = $m[1][0] - $m[0][1];
        } else {
            $j = ($i + 1) % 3; $k = ($j + 1) % 3;
            $q[$i] = 1 - $t + 2 * $m[$i][$i];
            $q[$j] = $m[$j][$i] + $m[$i][$j];
            $q[$k] = $m[$k][$i] + $m[$i][$k];
            $q[3] = $m[$k][$j] - $m[$j][$k];
        }
        $n = sqrt($q[0] * $q[0] + $q[1] * $q[1] + $q[2] * $q[2] + $q[3] * $q[3]);
        return [$q[0] / $n, $q[1] / $n, $q[2] / $n, $q[3] / $n];
    }

    private static function elementaryQuat(string $axis, float $angle): array
    {
        $s = sin($angle / 2.0); $c = cos($angle / 2.0);
        return match ($axis) {
            'x' => [$s, 0.0, 0.0, $c],
            'y' => [0.0, $s, 0.0, $c],
            'z' => [0.0, 0.0, $s, $c],
            default => throw new \InvalidArgumentException("bad Euler axis '$axis'"),
        };
    }

    /** Hamilton product, scalar-last. */
    private static function quatMul(array $a, array $b): array
    {
        [$ax, $ay, $az, $aw] = $a;
        [$bx, $by, $bz, $bw] = $b;
        return [
            $aw * $bx + $ax * $bw + $ay * $bz - $az * $by,
            $aw * $by - $ax * $bz + $ay * $bw + $az * $bx,
            $aw * $bz + $ax * $by - $ay * $bx + $az * $bw,
            $aw * $bw - $ax * $bx - $ay * $by - $az * $bz,
        ];
    }

    /**
     * Quaternion -> Euler angles, a direct port of scipy's array-API `as_euler`/`_get_angles`
     * (Bernardes-Viollet 2022). $axes are lowercase x/y/z; $intrinsic selects intrinsic vs
     * extrinsic. Returns [first, second, third] in the order scipy reports.
     */
    private static function quatToEuler(array $quat, array $axes, bool $intrinsic): array
    {
        $extrinsic = !$intrinsic;
        $idx = ['x' => 0, 'y' => 1, 'z' => 2];
        $seq = [$idx[$axes[0]], $idx[$axes[1]], $idx[$axes[2]]];
        // scipy: axes = axes if extrinsic else axes[::-1]
        if (!$extrinsic) {
            $seq = array_reverse($seq);
        }
        [$i, $j, $k] = $seq;
        $symmetric = ($i === $k);
        if ($symmetric) {
            $k = 3 - $i - $j;
        }
        $sign = intdiv(($i - $j) * ($j - $k) * ($k - $i), 2);

        $q = $quat; // [x,y,z,w]
        if ($symmetric) {
            $a = $q[3];
            $b = $q[$i];
            $c = $q[$j];
            $d = $q[$k] * $sign;
        } else {
            $a = $q[3] - $q[$j];
            $b = $q[$i] + $q[$k] * $sign;
            $c = $q[$j] + $q[3];
            $d = $q[$k] * $sign - $q[$i];
        }

        $eps = 1e-7;
        $halfSum = atan2($b, $a);
        $halfDiff = atan2($d, $c);

        $angles = [0.0, 0.0, 0.0];
        $angles[1] = 2.0 * atan2(hypot($c, $d), hypot($a, $b));

        $angleFirst = $extrinsic ? 0 : 2;
        $angleThird = $extrinsic ? 2 : 0;

        $case1 = abs($angles[1]) <= $eps;
        $case2 = abs($angles[1] - M_PI) <= $eps;
        $case0 = !($case1 || $case2);

        // a0 written to slot 0 (covers the singular cases)
        $angles[0] = $case1 ? 2.0 * $halfSum : 2.0 * $halfDiff * ($extrinsic ? -1.0 : 1.0);

        // overwrite the non-singular case into angle_first / angle_third
        if ($case0) {
            $angles[$angleFirst] = $halfSum - $halfDiff;
        }
        $a3 = $case0 ? ($halfSum + $halfDiff) : $angles[$angleThird];
        if (!$symmetric) {
            $a3 *= $sign;
            $angles[1] -= M_PI / 2.0;
        }
        $angles[$angleThird] = $a3;

        foreach ($angles as &$r) {
            $r = fmod($r + M_PI, 2 * M_PI);
            if ($r < 0) { $r += 2 * M_PI; }
            $r -= M_PI;
        }
        unset($r);
        return $angles;
    }

    // ---- shape helpers -------------------------------------------------

    private static function as2d(array $a, int $width, string $what): array
    {
        $a = array_values($a);
        if ($a === []) {
            throw new \InvalidArgumentException("empty $what");
        }
        if (!is_array($a[0])) {
            if (count($a) !== $width) {
                throw new \InvalidArgumentException("$what must have length $width");
            }
            return [[array_map('floatval', $a)], true];
        }
        $rows = [];
        foreach ($a as $row) {
            $row = array_values($row);
            if (count($row) !== $width) {
                throw new \InvalidArgumentException("$what rows must have length $width");
            }
            $rows[] = array_map('floatval', $row);
        }
        return [$rows, false];
    }

    private static function as3x3(array $a, ?bool &$single): array
    {
        $a = array_values($a);
        if (count($a) === 3 && is_array($a[0]) && !is_array($a[0][0] ?? 0.0)) {
            $single = true;
            return [self::floatMat($a)];
        }
        $single = false;
        $out = [];
        foreach ($a as $m) {
            $out[] = self::floatMat($m);
        }
        return $out;
    }

    private static function floatMat(array $m): array
    {
        $out = [];
        foreach (array_values($m) as $row) {
            $out[] = array_map('floatval', array_values($row));
        }
        return $out;
    }

    private static function anglesAs2d(array|float $angles, int $n): array
    {
        if (!is_array($angles)) {
            if ($n !== 1) {
                throw new \InvalidArgumentException('scalar angle only valid for single-axis sequence');
            }
            return [[(float) $angles]];
        }
        $angles = array_values($angles);
        if ($angles !== [] && is_array($angles[0])) {
            $rows = [];
            foreach ($angles as $row) {
                $rows[] = array_map('floatval', array_values($row));
            }
            return $rows;
        }
        return [array_map('floatval', $angles)];
    }

    private function unstack(array $stacked): array
    {
        return $this->single ? $stacked[0] : $stacked;
    }
}
