<?php

declare(strict_types=1);

namespace Tessero\Spatial\Transform;

/**
 * scipy.spatial.transform.RigidTransform: proper rigid transformations in 3-D (the SE(3) group:
 * a rotation followed by a translation). Pure PHP (no kernel), identical on both backends; built
 * on Tessero\Spatial\Transform\Rotation.
 *
 * Internally a RigidTransform is stored as a stack of 4x4 homogeneous matrices
 *     [ R  t ]
 *     [ 0  1 ]
 * where applying the transform to a point p gives R @ p + t. Composition A * B is the matrix
 * product A @ B (apply B first, then A), matching scipy. Factories: from_matrix, from_rotation,
 * from_translation, from_components(translation, rotation), identity. Accessors: as_matrix,
 * rotation, translation. Operations: apply, inv, compose (*).
 */
final class RigidTransform
{
    /** @var list<array<int,array<int,float>>> stack of 4x4 homogeneous matrices */
    private array $mats;

    /** @var bool whether this represents a single (un-stacked) transform */
    private bool $single;

    /**
     * @param list<array<int,array<int,float>>> $mats
     */
    private function __construct(array $mats, bool $single)
    {
        $this->mats = $mats;
        $this->single = $single;
    }

    public function single(): bool
    {
        return $this->single;
    }

    public function count(): int
    {
        return count($this->mats);
    }

    // ---- constructors --------------------------------------------------

    public static function fromMatrix(array $matrix): self
    {
        [$mats, $single] = self::as4x4($matrix);
        return new self($mats, $single);
    }

    public static function fromRotation(Rotation $rotation): self
    {
        $rm = $rotation->asMatrix();
        if ($rm !== [] && is_array($rm[0]) && !is_array($rm[0][0] ?? 0.0)) {
            // single rotation: asMatrix returned a 3x3
            return new self([self::homog($rm, [0.0, 0.0, 0.0])], true);
        }
        $out = [];
        foreach ($rm as $m) {
            $out[] = self::homog($m, [0.0, 0.0, 0.0]);
        }
        return new self($out, false);
    }

    public static function fromTranslation(array $translation): self
    {
        [$rows, $single] = self::as2d($translation, 3);
        $out = [];
        foreach ($rows as $t) {
            $out[] = self::homog(self::eye3(), $t);
        }
        return new self($out, $single);
    }

    public static function fromComponents(array $translation, Rotation $rotation): self
    {
        [$trows, $tsingle] = self::as2d($translation, 3);
        $rm = $rotation->asMatrix();
        $isSingleRot = ($rm !== [] && is_array($rm[0]) && !is_array($rm[0][0] ?? 0.0));
        $rmats = $isSingleRot ? [$rm] : $rm;
        $nr = count($rmats);
        $nt = count($trows);
        $n = max($nr, $nt);
        $out = [];
        for ($i = 0; $i < $n; $i++) {
            $out[] = self::homog($rmats[$nr === 1 ? 0 : $i], $trows[$nt === 1 ? 0 : $i]);
        }
        return new self($out, $tsingle && $isSingleRot);
    }

    public static function identity(?int $num = null): self
    {
        if ($num === null) {
            return new self([self::homog(self::eye3(), [0.0, 0.0, 0.0])], true);
        }
        $out = [];
        for ($i = 0; $i < $num; $i++) {
            $out[] = self::homog(self::eye3(), [0.0, 0.0, 0.0]);
        }
        return new self($out, false);
    }

    // ---- accessors -----------------------------------------------------

    public function asMatrix(): array
    {
        return $this->single ? $this->mats[0] : $this->mats;
    }

    public function rotation(): Rotation
    {
        $out = [];
        foreach ($this->mats as $m) {
            $out[] = [
                [$m[0][0], $m[0][1], $m[0][2]],
                [$m[1][0], $m[1][1], $m[1][2]],
                [$m[2][0], $m[2][1], $m[2][2]],
            ];
        }
        return Rotation::fromMatrix($this->single ? $out[0] : $out);
    }

    public function translation(): array
    {
        $out = [];
        foreach ($this->mats as $m) {
            $out[] = [$m[0][3], $m[1][3], $m[2][3]];
        }
        return $this->single ? $out[0] : $out;
    }

    // ---- operations ----------------------------------------------------

    public function apply(array $vectors, bool $inverse = false): array
    {
        [$vrows, $vsingle] = self::as2d($vectors, 3);
        $src = $inverse ? self::invMats($this->mats) : $this->mats;
        $nr = count($src);
        $nv = count($vrows);
        $n = max($nr, $nv);
        $out = [];
        for ($i = 0; $i < $n; $i++) {
            $m = $src[$nr === 1 ? 0 : $i];
            $v = $vrows[$nv === 1 ? 0 : $i];
            $out[] = [
                $m[0][0] * $v[0] + $m[0][1] * $v[1] + $m[0][2] * $v[2] + $m[0][3],
                $m[1][0] * $v[0] + $m[1][1] * $v[1] + $m[1][2] * $v[2] + $m[1][3],
                $m[2][0] * $v[0] + $m[2][1] * $v[1] + $m[2][2] * $v[2] + $m[2][3],
            ];
        }
        if ($nr === 1 && $vsingle) {
            return $out[0];
        }
        return $out;
    }

    public function inv(): self
    {
        return new self(self::invMats($this->mats), $this->single);
    }

    public function compose(self $other): self
    {
        $a = $this->mats;
        $b = $other->mats;
        $na = count($a);
        $nb = count($b);
        $n = max($na, $nb);
        $out = [];
        for ($i = 0; $i < $n; $i++) {
            $out[] = self::matMul4($a[$na === 1 ? 0 : $i], $b[$nb === 1 ? 0 : $i]);
        }
        return new self($out, $this->single && $other->single);
    }

    // ---- kernels / helpers --------------------------------------------

    /** @return list<array<int,array<int,float>>> */
    private static function invMats(array $mats): array
    {
        $out = [];
        foreach ($mats as $m) {
            // R^-1 = R^T ; t' = -R^T t
            $rt = [
                [$m[0][0], $m[1][0], $m[2][0]],
                [$m[0][1], $m[1][1], $m[2][1]],
                [$m[0][2], $m[1][2], $m[2][2]],
            ];
            $t = [$m[0][3], $m[1][3], $m[2][3]];
            $ti = [
                -($rt[0][0] * $t[0] + $rt[0][1] * $t[1] + $rt[0][2] * $t[2]),
                -($rt[1][0] * $t[0] + $rt[1][1] * $t[1] + $rt[1][2] * $t[2]),
                -($rt[2][0] * $t[0] + $rt[2][1] * $t[1] + $rt[2][2] * $t[2]),
            ];
            $out[] = self::homog($rt, $ti);
        }
        return $out;
    }

    private static function matMul4(array $a, array $b): array
    {
        $out = [];
        for ($i = 0; $i < 4; $i++) {
            $row = [];
            for ($j = 0; $j < 4; $j++) {
                $s = 0.0;
                for ($k = 0; $k < 4; $k++) {
                    $s += $a[$i][$k] * $b[$k][$j];
                }
                $row[] = $s;
            }
            $out[] = $row;
        }
        return $out;
    }

    private static function homog(array $r, array $t): array
    {
        return [
            [$r[0][0], $r[0][1], $r[0][2], $t[0]],
            [$r[1][0], $r[1][1], $r[1][2], $t[1]],
            [$r[2][0], $r[2][1], $r[2][2], $t[2]],
            [0.0, 0.0, 0.0, 1.0],
        ];
    }

    private static function eye3(): array
    {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }

    /** @return array{0:list<array<int,array<int,float>>>,1:bool} */
    private static function as4x4(array $a): array
    {
        $a = array_values($a);
        if (count($a) === 4 && is_array($a[0]) && !is_array($a[0][0] ?? 0.0)) {
            return [[self::floatMat($a)], true];
        }
        $out = [];
        foreach ($a as $m) {
            $out[] = self::floatMat($m);
        }
        return [$out, false];
    }

    private static function floatMat(array $m): array
    {
        $out = [];
        foreach (array_values($m) as $row) {
            $out[] = array_map('floatval', array_values($row));
        }
        return $out;
    }

    /** @return array{0:list<array<int,float>>,1:bool} */
    private static function as2d(array $a, int $width): array
    {
        $a = array_values($a);
        if ($a === []) {
            throw new \InvalidArgumentException('empty input');
        }
        if (!is_array($a[0])) {
            if (count($a) !== $width) {
                throw new \InvalidArgumentException("vector must have length $width");
            }
            return [[array_map('floatval', $a)], true];
        }
        $rows = [];
        foreach ($a as $row) {
            $row = array_values($row);
            if (count($row) !== $width) {
                throw new \InvalidArgumentException("rows must have length $width");
            }
            $rows[] = array_map('floatval', $row);
        }
        return [$rows, false];
    }
}
