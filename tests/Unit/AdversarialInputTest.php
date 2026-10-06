<?php

declare(strict_types=1);

namespace Tessero\Tests\Unit;

use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;

/**
 * Adversarial arguments found by the sanitizer fuzz pass and the independent verification (2026-09-28). Each call
 * once crashed the PHP process, read out of bounds, never returned, or returned uninitialised memory. Now it
 * must return what NumPy/SciPy return, raise where they raise, or return NaN where SciPy itself crashes
 * (docs/project/reference-deviations.md, "Guards"). A regression here usually kills the test process, which
 * fails the suite loudly.
 *
 * Every case runs on the FFI classes and, when ext-tessero is loaded, on the extension's.
 */
final class AdversarialInputTest extends TestCase
{
    /** @return array<string, array{string, string, string, string}> [module, expression, expect, backend] */
    public static function cases(): array
    {
        $raise = 'raise';
        $nan = 'nan';
        $calls = [
            // out-of-bounds reads (np.partition's kth bounds, SciPy raises)
            ['stats', 'trimMean(range(1.0, 10.0), -0.5)', $raise],
            ['stats', 'trimMean(array_fill(0, 200000, 1.0), -40.0)', $raise],
            ['stats', 'trimboth(range(1.0, 10.0), -0.1)', $raise],
            ['stats', 'trim1(range(0.0, 9.0), -1.5, tail: "left")', $raise],
            ['stats', 'levene(range(0.0, 9.0), range(0.0, 22.0, 2), center: "trimmed", proportiontocut: -0.2)', $raise],
            ['stats', 'lmoment([1.5, 2.0, 3.7, 0.2], order: 3e9)', $raise],
            ['stats', 'lmoment([1.5, 2.0, 3.7, 0.2], order: [2, 2147483648])', $raise],
            // NaN or infinite integer parameters (int(nan) in SciPy; an infinite moment order never returns there)
            ['stats', 'kstat([1.0, 2.0, 4.0], n: NAN)', $raise],
            ['stats', 'trimMean([1.0, 2.0, 3.0], NAN)', $raise],
            ['stats', 'moment([1.0, 2.0, 4.0], order: INF)', $raise],
            ['stats', 'fisherExact([[1e300, 1], [1, 1e300]])', $raise],
            // allocation sizes from parameters: MemoryError / "Maximum allowed size exceeded", never an abort
            ['stats', 'cumfreq([1.5, 2.0, 3.7], numbins: 1e300)', $raise],
            ['stats', 'binnedStatistic([1.0, 2.0], [1.0, 2.0], bins: 2147483648)', $raise],
            ['stats', 'binnedStatistic2d([1.0, 2.0], [1.0, 2.0], [1.0, 2.0], bins: 1000000)', $raise],
            ['stats', 'quantile([1.0, 2.0, 3.0], [])', '[]'],
            ['stats', 'zipf(1.5)->cdf(1e300)', $raise],
            ['stats', 'logser(0.5)->cdf(1e300)', $raise],
            ['stats', 'binom(1e10, 0.5)->entropy()', $raise],
            ['stats', 'irwinhall(1e10)->pdf(0.5)', $raise],
            ['stats', 'genextreme(0.5)->moment(PHP_INT_MAX)', $raise],
            // NumPy's sampler parameter checks behind scipy.stats rvs (SciPy: ValueError / OverflowError)
            ['stats', 'poisson(INF)->rvs(size: 3, randomState: \\Tessero\\Random\\Generator::defaultRng(1))', $raise],
            ['stats', 'poisson(1e20)->rvs(size: 2, randomState: \\Tessero\\Random\\Generator::defaultRng(1))', $raise],
            ['stats', 'skellam(INF, 1.0)->rvs(size: 3, randomState: \\Tessero\\Random\\Generator::defaultRng(1))', $raise],
            ['stats', 'nchypergeomFisher(3e9, 5, 3, 2.0)->stats()', $raise],
            ['stats', 'hypergeom(1e10, 5, 3)->entropy()', $raise],
            // unbounded recursion (SciPy: RecursionError)
            ['stats', 'levyStable(1.0, 0.5)->cdf(-1e300)', $raise],
            // Boost overflow (SciPy: OverflowError)
            ['stats', 'ncf(1.0, 1.0, 0.0)->pdf(0.0)', $raise],
            ['stats', 'ncf(3.0, 11.0, 2.5)->isf(1e-300)', $raise],
            // where SciPy itself crashes: NaN
            ['special', 'mathieuCem(0.0, 1e300, 0.5)["y"]', $nan],
            ['special', 'mathieuCem(3e9, 1.0, 0.5)["y"]', $nan],
            ['special', 'mathieuModsem2(3.0, 1e8, 0.5)["y"]', $nan],
            ['special', 'hyp2f1(1e300, 0.5, 1e300, NAN)', $nan],
            ['special', 'hyp1f1(-1e300, -INF, -INF)', $nan],
            ['special', 'evalJacobi(2147483648.0, 0.5, 1e300, 0.3)', $nan],
            // NumPy input validation
            ['np', 'bincount([INF, 1.0])', $raise],
            ['np', 'bincount({ND}::array([1.0, 2.0, 2.0]))', $raise],   // a float ndarray fails the 'safe' cast
            ['np', 'bincount([1.0, 2.0, 2.0])', '[0,1,2]'],             // a Python list is converted by int()
            ['np', 'bincount([0, 1, 2, 1], minlength: 2.5)', $raise],
            ['np', 'ptp([true, false])', $raise],
            ['np', 'average([[1.0, 2.0], [3.0, 4.0]], weights: [[1.0, 2.0]], axis: 0)', $raise],
            // uninitialised output (include_initial is a boolean)
            ['np', 'cumulativeSum([1.0, 2.0, 3.0], includeInitial: 2)', '[0,1,3,6]'],
            // exact int64 arithmetic with NumPy's wrap-around
            ['np', 'ptp([PHP_INT_MAX, 0])', '9223372036854775807'],
            ['np', 'ptp([PHP_INT_MAX, -PHP_INT_MAX])', '-2'],
            ['np', 'correlate([PHP_INT_MAX, 2], [2, 1], mode: "full")', '[9223372036854775807,0,4]'],
            // array construction and manipulation (PAR-2.3): sizes from parameters raise before anything is
            // allocated; indices out of range raise; in-place writers take only writable NDArrays
            ['np', 'tile([1.0], [4294967296, 4294967296])', $raise],
            ['np', 'repeat([1.0, 2.0], PHP_INT_MAX)', $raise],
            ['np', 'repeat([1.0, 2.0], [PHP_INT_MAX, PHP_INT_MAX])', $raise],
            ['np', 'pad([1.0], 1000000000000000)', $raise],
            ['np', 'pad([1.0], PHP_INT_MAX)', $raise],
            ['np', 'identity(3037000500)', $raise],
            ['np', 'tri(4294967296)', $raise],
            ['np', 'broadcastTo([1.0], [4294967296, 4294967296])', $raise],
            ['np', 'resize([1.0], [4611686018427387904])', $raise],
            ['np', 'arraySplit([1.0, 2.0, 3.0], 1073741824)', $raise],
            ['np', 'diagIndices(1000000000, 1000000)', $raise],
            ['np', 'logspace(0.0, 1.0, num: 1000000000000000)', $raise],
            ['np', 'hanning(100000000000000.0)', $raise],
            ['np', 'meshgrid(range(1, 1000000), range(1, 1000000))', $raise],
            ['np', 'indices([4294967296, 4294967296])', $raise],
            ['np', 'diag([1.0], k: PHP_INT_MAX)', $raise],
            ['np', 'take([1.0, 2.0], [PHP_INT_MAX])', $raise],
            ['np', 'unravelIndex(PHP_INT_MAX, [2, 3])', $raise],
            ['np', 'ravelMultiIndex([[1], [1]], [4294967296, 4294967296])', $raise],
            ['np', 'concatenate([[1.0], [2.0]], axis: 5)', $raise],
            ['np', 'polyint([1.0], m: 1e18)', $raise],
            ['np', 'gradient([1.0, 2.0], edgeOrder: 2)', $raise],
            ['np', 'interp([1.0], [], [])', $raise],
            ['np', 'put({ND}::array([1.0, 2.0]), [PHP_INT_MAX], [1.0])', $raise],
            ['np', 'put([1.0, 2.0], [0], [1.0])', $raise],
            ['np', 'putAlongAxis({ND}::array([[1.0, 2.0]]), {ND}::array([[5]]), 1.0, 1)', $raise],
            ['np', 'roll([1.0, 2.0, 3.0], PHP_INT_MAX)', '[3,1,2]'],
            ['np', 'insert([1, 2, 3], 1, [9, 8])', '[1,9,8,2,3]'],
            // the independent verification of PAR-2.3: exact int64, Python-value conversions, NumPy's k arithmetic
            ['np', 'trace([[4611686018427387905, 1], [2, 1152921504606846979]])', '5764607523034234884'],
            ['np', 'unique([4611686018427387905, 4611686018427387904, 4611686018427387905])', '[4611686018427387904,4611686018427387905]'],
            ['np', 'unique(5)', '[5]'],
            ['np', 'quantile([3, 1, 4], 0.5, method: \'lower\')', '3'],
            ['np', 'quantile([true, false], 0.5)', $raise],
            ['np', 'put({ND}::array([1, 2]), [0], NAN)', $raise],
            ['np', 'put({ND}::array([1, 2]), [0], INF)', $raise],
            ['np', 'put({ND}::array([1, 2]), [0], 1e19)', $raise],
            ['np', 'putmask({ND}::array([1, 2]), [true, false], [NAN])', $raise],
            ['np', 'searchsorted([[1.0, 2.0], [3.0, 4.0]], 3.0)', $raise],
            ['np', 'tensordot([[1.0]], [1.0], axes: [[], []])', '[1]'],            // shape (1, 1, 1); toList() is flat
            ['np', 'rot90([[0, 1], [2, 3]], 1.5)', '[2,0,3,1]'],
            ['np', 'tri(2, 2, -0.5, \'int64\')', '[1,0,1,1]'],
            ['np', 'tri(2, 2, NAN)', $raise],
            ['np', 'tri(2, 2, PHP_INT_MIN)', $raise],
            ['np', 'trilIndices(3, PHP_INT_MIN + 1, 4)', $raise],
            ['np', 'trace([[1.0]], 2147483648)', $raise],
            ['np', 'bincount([1.5, 2.2])', '[0,1,1]'],
            ['np', 'bincount([NAN])', $raise],
            ['np', 'quantile({ND}::array([5, 1, 3], \'int32\'), 0.5, method: \'lower\')', '3'],
            ['np', 'nanquantile({ND}::array([], \'int64\'), 0.5, method: \'lower\')', $nan],
            ['np', 'ravelMultiIndex([1], [4611686018427387904])', '1'],
            ['np', 'trilIndices(0, PHP_INT_MIN)', '[[],[]]'],
            ['np', 'setdiff1d({ND}::array([PHP_INT_MAX, 9007199254740993], \'int64\'), [1.5])', '[9007199254740993,9223372036854775807]'],
            ['np', 'put({ND}::array([1, 2]), [0], -9.223372036854776e18)', 'null'],
        ];
        $out = [];
        $backends = ['ffi'];
        if (extension_loaded('tessero')) {
            $backends[] = 'ext';
        }
        foreach ($backends as $b) {
            foreach ($calls as [$m, $expr, $expect]) {
                $out["{$b}: {$m}::{$expr}"] = [$m, $expr, $expect, $b];
            }
        }

        return $out;
    }

    #[DataProvider('cases')]
    public function testAdversarialInput(string $module, string $expr, string $expect, string $backend): void
    {
        $class = [
            'ffi' => ['stats' => '\\Tessero\\Stats', 'special' => '\\Tessero\\Special', 'np' => '\\Tessero\\Np'],
            'ext' => ['stats' => '\\Tessero\\Ext\\Stats', 'special' => '\\Tessero\\Ext\\Special', 'np' => '\\Tessero\\Ext\\Np'],
        ][$backend][$module];
        $expr = str_replace('{ND}', $backend === 'ext' ? '\\Tessero\\Ext\\NDArray' : '\\Tessero\\NDArray', $expr);
        $raised = null;
        $value = null;
        try {
            $value = eval("return {$class}::{$expr};");
        } catch (\Throwable $e) {
            $raised = $e;
        }
        if ($expect === 'raise') {
            $this->assertNotNull($raised, "{$expr} must raise");

            return;
        }
        $this->assertNull($raised, $raised ? get_class($raised) . ': ' . $raised->getMessage() : '');
        if (is_object($value)) {
            $value = $value->toList();
        } elseif (is_array($value)) {                       // a tuple of arrays (tril_indices, nonzero)
            $value = array_map(static fn (mixed $v): mixed => is_object($v) ? $v->toList() : $v, $value);
        }
        if ($expect === 'nan') {
            $this->assertTrue(is_float($value) && is_nan($value), "{$expr} must be NaN");

            return;
        }
        $this->assertSame($expect, json_encode($value));
    }
}
