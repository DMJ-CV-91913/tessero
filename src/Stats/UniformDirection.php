<?php

declare(strict_types=1);

namespace Tessero\Stats;

/**
 * scipy.stats.uniform_direction — uniform random directions on the surface of the unit
 * (dim-1)-sphere S(dim-1), as a pure-PHP facade over Tessero's bit-exact PCG64
 * {@see Generator} (numpy.random.default_rng). No kernel of its own, so it is identical on
 * both backends.
 *
 * Algorithm (Marsaglia 1972, exactly as SciPy's _sample_uniform_direction): draw an
 * array of standard normals of shape (*size, dim) from the Generator in C-order, then
 * normalise each length-`dim` vector (the last axis) by its Euclidean norm. Because the
 * draw order and the PCG64 stream match numpy exactly, the result is bit-identical to
 *   scipy.stats.uniform_direction.rvs(dim, size, random_state=np.random.default_rng(seed)).
 *
 * `size` mirrors SciPy: null → a single length-`dim` vector; an int m → shape (m, dim);
 * a list [m, n, ...] → shape (m, n, ..., dim).
 */
final class UniformDirection
{
    /**
     * Draw random direction vectors on S(dim-1).
     *
     * @param int             $dim  dimension of the ambient space (N >= 1)
     * @param int|array|null  $size output sample shape (excluding the trailing dim axis)
     * @param object          $rng  a seeded Tessero Generator (numpy default_rng equivalent);
     *                              either backend's Generator, used by duck-typed standardNormal()
     *
     * @return array nested PHP array of shape (*size, dim)
     */
    public static function rvs(int $dim, int|array|null $size, object $rng): array
    {
        if ($dim < 1) {
            throw new \InvalidArgumentException('Dimension of vector must be an integer greater than 0.');
        }

        // SciPy: size None -> np.array([]) -> trailing dim only; atleast_1d otherwise.
        if ($size === null) {
            $lead = [];
        } elseif (is_int($size)) {
            $lead = [$size];
        } else {
            $lead = array_values(array_map('intval', $size));
        }
        $shape = array_merge($lead, [$dim]);

        // Draw (*size, dim) standard normals in one call: C-order fill matches numpy's draw order.
        $drawShape = count($shape) === 1 ? $shape[0] : $shape;
        $samples = $rng->standardNormal($drawShape);

        // Flatten to a nested PHP array and normalise each trailing length-`dim` vector.
        $arr = is_object($samples) && method_exists($samples, 'toArray') ? $samples->toArray() : $samples;

        return self::normalizeLastAxis($arr, $dim);
    }

    /**
     * Recursively descend until the innermost length-`dim` vectors, dividing each by its L2 norm.
     *
     * @param array $a   nested array whose innermost dimension has length `dim`
     * @param int   $dim length of the innermost (vector) axis
     */
    private static function normalizeLastAxis(array $a, int $dim): array
    {
        // Innermost vector: a flat list of `dim` floats.
        if (count($a) === $dim && (! isset($a[0]) || ! is_array($a[0]))) {
            $ss = 0.0;
            foreach ($a as $v) {
                $f = (float) $v;
                $ss += $f * $f;
            }
            $n = sqrt($ss);
            $out = [];
            foreach ($a as $v) {
                $out[] = ((float) $v) / $n;
            }

            return $out;
        }

        $out = [];
        foreach ($a as $sub) {
            $out[] = self::normalizeLastAxis($sub, $dim);
        }

        return $out;
    }
}
