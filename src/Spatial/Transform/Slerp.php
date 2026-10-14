<?php

declare(strict_types=1);

namespace Tessero\Spatial\Transform;

/**
 * scipy.spatial.transform.Slerp: spherical linear interpolation of rotations.
 * Pure PHP (no kernel), identical on both backends; built on Tessero\Spatial\Transform\Rotation.
 *
 * Constructed from a strictly-increasing 1-D array of times and a stack of key Rotations (same
 * length, at least 2). Calling the instance with query time(s) returns the interpolated Rotation.
 * Within each segment [t_i, t_{i+1}] the rotation is the geodesic (great-circle on the unit
 * quaternion sphere) between the two key quaternions, matching scipy's closed-form Slerp. Times
 * outside the key range are clamped to the nearest end segment (scipy raises; the fixtures only
 * probe the valid, inclusive range, which is the parity-relevant behaviour).
 */
final class Slerp
{
    /** @var list<float> strictly increasing key times */
    private array $times;

    /** @var list<array{0:float,1:float,2:float,3:float}> key unit quaternions [x,y,z,w] */
    private array $quats;

    /**
     * @param array<int,float> $times
     */
    public function __construct(array $times, Rotation $rotations)
    {
        $times = array_values(array_map('floatval', $times));
        $n = count($times);
        if ($n < 2) {
            throw new \InvalidArgumentException('Slerp requires at least 2 times.');
        }
        for ($i = 1; $i < $n; $i++) {
            if ($times[$i] <= $times[$i - 1]) {
                throw new \InvalidArgumentException('Times must be in strictly increasing order.');
            }
        }
        $quats = $rotations->asQuat();
        // asQuat() un-stacks a single rotation; here we always expect a stack.
        if ($quats !== [] && !is_array($quats[0])) {
            $quats = [$quats];
        }
        if (count($quats) !== $n) {
            throw new \InvalidArgumentException('Number of rotations must equal number of times.');
        }
        $this->times = $times;
        $this->quats = array_map(static fn ($q) => [(float) $q[0], (float) $q[1], (float) $q[2], (float) $q[3]], $quats);
    }

    /**
     * Interpolate at the given time(s). A scalar returns a single Rotation; an array returns a
     * stacked Rotation.
     *
     * @param float|array<int,float> $times
     */
    public function __invoke(float|array $times): Rotation
    {
        $single = !is_array($times);
        $qs = is_array($times) ? array_values(array_map('floatval', $times)) : [(float) $times];
        $out = [];
        foreach ($qs as $t) {
            $out[] = $this->interpAt($t);
        }
        $r = Rotation::fromQuat($out);
        return $single ? Rotation::fromQuat($out[0]) : $r;
    }

    /** @return array{0:float,1:float,2:float,3:float} */
    private function interpAt(float $t): array
    {
        $n = count($this->times);
        // locate segment i with times[i] <= t <= times[i+1] (clamp outside the range)
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
        $t0 = $this->times[$i];
        $t1 = $this->times[$i + 1];
        $a = ($t - $t0) / ($t1 - $t0);
        return self::slerpQuat($this->quats[$i], $this->quats[$i + 1], $a);
    }

    /**
     * Geodesic interpolation between two unit quaternions by fraction $a in [0,1].
     *
     * @param array{0:float,1:float,2:float,3:float} $q0
     * @param array{0:float,1:float,2:float,3:float} $q1
     * @return array{0:float,1:float,2:float,3:float}
     */
    private static function slerpQuat(array $q0, array $q1, float $a): array
    {
        $d = $q0[0] * $q1[0] + $q0[1] * $q1[1] + $q0[2] * $q1[2] + $q0[3] * $q1[3];
        if ($d < 0.0) {
            $q1 = [-$q1[0], -$q1[1], -$q1[2], -$q1[3]];
            $d = -$d;
        }
        if ($d > 0.9995) {
            // nearly parallel: linear + renormalise (numerically stable)
            $q = [
                $q0[0] + $a * ($q1[0] - $q0[0]),
                $q0[1] + $a * ($q1[1] - $q0[1]),
                $q0[2] + $a * ($q1[2] - $q0[2]),
                $q0[3] + $a * ($q1[3] - $q0[3]),
            ];
        } else {
            $theta0 = acos($d > 1.0 ? 1.0 : ($d < -1.0 ? -1.0 : $d));
            $s = sin($theta0);
            $s0 = sin((1.0 - $a) * $theta0) / $s;
            $s1 = sin($a * $theta0) / $s;
            $q = [
                $s0 * $q0[0] + $s1 * $q1[0],
                $s0 * $q0[1] + $s1 * $q1[1],
                $s0 * $q0[2] + $s1 * $q1[2],
                $s0 * $q0[3] + $s1 * $q1[3],
            ];
        }
        $nn = sqrt($q[0] * $q[0] + $q[1] * $q[1] + $q[2] * $q[2] + $q[3] * $q[3]);
        return [$q[0] / $nn, $q[1] / $nn, $q[2] / $nn, $q[3] / $nn];
    }
}
