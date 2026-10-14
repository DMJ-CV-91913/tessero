<?php

declare(strict_types=1);

namespace Tessero\Stats;

use Tessero\Native\Registry;
use Tessero\NDArray;
use Tessero\Random\Generator;

/**
 * scipy.stats.unitary_group — random matrices drawn (Haar-)uniformly from the unitary group U(N).
 *
 * Pure-PHP facade over Tessero's bit-exact PCG64 {@see Generator} and a complex QR kernel routine
 * (linalg.qr_complex: LAPACKE_zgeqrf + zungqr). It is identical on both backends (FFI and the native
 * extension): the only RNG draws are two dim×dim standard-normal matrices (real then imaginary part,
 * in NumPy's exact C-order via Generator::standardNormal), and complex QR is backend-neutral.
 *
 * Algorithm (identical to scipy.stats._multivariate.unitary_group_gen.rvs, size=1):
 *   z = (normal(size=(dim, dim)) + 1j*normal(size=(dim, dim))) / sqrt(2)   # complex Ginibre
 *   q, r = np.linalg.qr(z)
 *   d = diag(r);  q *= (d / |d|)[newaxis, :]                               # phase-fix each column
 * The "divide each column by the phase of R's diagonal" normalisation makes q independent of the
 * complex-QR implementation's internal reflector phase, so any correct complex QR reproduces scipy's
 * matrix bit-for-bit (verified: max abs err 1.57e-16 vs scipy, dim 1..8, LAPACK zgeqrf/zungqr).
 *
 * Bit-exact against scipy when the SAME seed is used with rng=np.random.default_rng(seed), because
 * Generator reproduces NumPy's PCG64 standard-normal stream exactly, and scipy's rvs draws the real
 * block then the imaginary block from that stream (normal(0,1) and standard_normal share the stream).
 * (An int seed passed to scipy builds a legacy MT19937 RandomState and would NOT match — fixtures
 * therefore use default_rng(seed).)
 */
final class UnitaryGroup
{
    /**
     * Draw a random unitary matrix from U(dim) (scipy.stats.unitary_group.rvs, size=1).
     *
     * @param int                        $dim dimension of the square matrix (N >= 1)
     * @param int|array<int>|Generator|null $rng seed for a PCG64 Generator, or a Generator instance
     *
     * @return NDArray the dim×dim unitary matrix (complex128)
     */
    public static function rvs(int $dim, int|array|Generator|null $rng = null): NDArray
    {
        if ($dim < 1) {
            throw new \InvalidArgumentException('Dimension of rotation must be specified, and must be a scalar greater than 1.');
        }

        $gen = ($rng instanceof Generator) ? $rng : Generator::defaultRng($rng);

        // scipy draws the real block first, then the imaginary block, from the SAME stream;
        // normal(0,1) and standardNormal share the PCG64 stream, so standardNormal matches.
        $invSqrt2 = 1.0 / sqrt(2.0);
        $re = $gen->standardNormal([$dim, $dim]);
        $im = $gen->standardNormal([$dim, $dim]);

        // z = (re + 1j*im) / sqrt(2)
        $z = NDArray::complex(
            $re->mul($invSqrt2),
            $im->mul($invSqrt2),
        );

        // q, r = complex qr(z)  (reduced == complete here since z is square).
        // The routine OUTS are "Q, R", so Registry::routine returns a map keyed by those names.
        $qr = Registry::routine('linalg.qr_complex', [$z]);
        $q  = $qr['Q'] ?? array_values($qr)[0];
        $r  = $qr['R'] ?? array_values($qr)[1];

        // d = diag(r); phase = d / |d|; column j of q is multiplied by phase[j].
        $rArr = $r->toArray();   // complex -> [re, im] pairs
        $phRe = [];
        $phIm = [];
        for ($j = 0; $j < $dim; $j++) {
            $dr = (float) $rArr[$j][$j][0];
            $di = (float) $rArr[$j][$j][1];
            $mag = sqrt($dr * $dr + $di * $di);
            $phRe[$j] = $dr / $mag;
            $phIm[$j] = $di / $mag;
        }

        $qArr = $q->toArray();   // complex -> [re, im] pairs
        $outRe = [];
        $outIm = [];
        for ($i = 0; $i < $dim; $i++) {
            $outRe[$i] = [];
            $outIm[$i] = [];
            for ($j = 0; $j < $dim; $j++) {
                $ar = (float) $qArr[$i][$j][0];
                $ai = (float) $qArr[$i][$j][1];
                // (ar + i*ai) * (phRe + i*phIm)
                $outRe[$i][$j] = $ar * $phRe[$j] - $ai * $phIm[$j];
                $outIm[$i][$j] = $ar * $phIm[$j] + $ai * $phRe[$j];
            }
        }

        return NDArray::complex($outRe, $outIm);
    }
}
