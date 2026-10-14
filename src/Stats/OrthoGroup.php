<?php

declare(strict_types=1);

namespace Tessero\Stats;

use Tessero\Linalg\Linalg;
use Tessero\NDArray;
use Tessero\Random\Generator;

/**
 * scipy.stats.ortho_group and scipy.stats.special_ortho_group — random matrices drawn
 * (Haar-)uniformly from the orthogonal group O(N) resp. the special orthogonal group SO(N).
 *
 * Pure-PHP facade over Tessero's bit-exact PCG64 {@see Generator} and {@see Linalg::qr}. It is
 * identical on both backends (FFI and the native extension): the only RNG draw is a single
 * dim×dim standard-normal matrix, drawn in NumPy's exact C-order via Generator::normal(0,1,[dim,dim]),
 * and QR is backend-neutral (LAPACK dgeqrf/dorgqr through either backend).
 *
 * Algorithm (identical to scipy.stats._multivariate.ortho_group_gen.rvs):
 *   z = rng.normal(size=(dim, dim))
 *   q, r = np.linalg.qr(z)
 *   d = diag(r);  q *= d / |d|            # fix the sign of each column
 * This "QR with positive-diagonal" normalisation makes q independent of the QR implementation's
 * internal sign choice, so any correct QR reproduces scipy's matrix bit-for-bit. For SO(N),
 * special_ortho_group additionally scales row 0 by 1/det(q) so det becomes +1.
 *
 * Bit-exact against scipy when the SAME seed is used with rng=np.random.default_rng(seed),
 * because Generator reproduces NumPy's PCG64 standard-normal stream exactly. (An int seed passed
 * to scipy builds a legacy MT19937 RandomState and would NOT match — fixtures therefore use
 * default_rng(seed).)
 */
final class OrthoGroup
{
    /**
     * Draw a random orthogonal matrix from O(dim) (scipy.stats.ortho_group.rvs, size=1).
     *
     * @param int                   $dim dimension of the square matrix (N >= 1)
     * @param int|array<int>|null   $rng seed for a PCG64 Generator, or a Generator instance
     *
     * @return NDArray the dim×dim orthogonal matrix (float64)
     */
    public static function rvs(int $dim, int|array|Generator|null $rng = null): NDArray
    {
        return self::draw($dim, $rng, false);
    }

    /**
     * Draw a random special-orthogonal matrix from SO(dim), det = +1
     * (scipy.stats.special_ortho_group.rvs, size=1).
     *
     * @param int                   $dim dimension of the square matrix (N >= 1)
     * @param int|array<int>|null   $rng seed for a PCG64 Generator, or a Generator instance
     *
     * @return NDArray the dim×dim special-orthogonal matrix (float64)
     */
    public static function rvsSpecial(int $dim, int|array|Generator|null $rng = null): NDArray
    {
        return self::draw($dim, $rng, true);
    }

    private static function draw(int $dim, int|array|Generator|null $rng, bool $special): NDArray
    {
        if ($dim < 1) {
            throw new \InvalidArgumentException('Dimension of rotation must be specified, and must be a scalar greater than 1.');
        }

        $gen = ($rng instanceof Generator) ? $rng : Generator::defaultRng($rng);

        // z = rng.normal(size=(dim, dim)); NumPy fills C-order, which Generator::normal matches.
        $z = $gen->normal(0.0, 1.0, [$dim, $dim]);

        // q, r = qr(z)  (reduced == complete here since z is square)
        [$q, $r] = Linalg::qr($z, 'reduced');

        // d = diag(r); column j of q is scaled by sign(d[j]) = d[j] / |d[j]|.
        $rArr = $r->toArray();
        $signs = [];
        for ($j = 0; $j < $dim; $j++) {
            $d = (float) $rArr[$j][$j];
            $signs[$j] = $d / abs($d);
        }

        $qArr = $q->toArray();
        for ($i = 0; $i < $dim; $i++) {
            for ($j = 0; $j < $dim; $j++) {
                $qArr[$i][$j] = (float) $qArr[$i][$j] * $signs[$j];
            }
        }

        if ($special) {
            // det(q) is ±1 for an orthogonal matrix; scipy scales row 0 by 1/det so det = +1.
            $det = Linalg::det(NDArray::array($qArr, 'float64'));
            $d = (float) $det;
            for ($j = 0; $j < $dim; $j++) {
                $qArr[0][$j] = $qArr[0][$j] / $d;
            }
        }

        return NDArray::array($qArr, 'float64');
    }
}
