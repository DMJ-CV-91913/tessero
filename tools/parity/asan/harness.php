<?php
// Adversarial-argument harness for libtessero under ASan/UBSan.
// usage: php harness.php <kind> <method> [start] [--list]
//   kind: stats | np | special | dist | gen
// Prints "@@CASE <i> <desc>" to stderr before each call so sanitizer reports can be attributed.
require dirname(__DIR__, 3) . '/tests/autoload.php';

use Tessero\NDArray;
use Tessero\DType;
use Tessero\Stats;
use Tessero\Np;
use Tessero\Special;
use Tessero\Random\Generator;

ini_set('memory_limit', '4G');
error_reporting(E_ALL & ~E_DEPRECATED & ~E_WARNING & ~E_NOTICE);

$kind = $argv[1];
$method = $argv[2];
$start = (int) ($argv[3] ?? 0);
$listOnly = in_array('--list', $argv, true);

function E(string $s): void { fwrite(STDERR, $s . "\n"); }

const POISON = "\xbe\xbe\xbe\xbe\xbe\xbe\xbe\xbe";

function flat(mixed $r, array &$acc, int $depth = 0): void
{
    if (count($acc) > 300000) return;
    if ($r instanceof NDArray) {
        try { $r = $r->toArray(); } catch (\Throwable $e) { $acc[] = 'toArray:' . get_class($e); return; }
    }
    if (is_array($r)) { foreach ($r as $k => $v) { $acc[] = "k:$k"; flat($v, $acc, $depth + 1); } return; }
    if (is_float($r)) { $acc[] = is_nan($r) ? 'nan' : pack('d', $r); return; }
    if (is_int($r)) { $acc[] = 'i' . $r; return; }
    if (is_bool($r)) { $acc[] = $r ? 'T' : 'F'; return; }
    if ($r === null) { $acc[] = 'N'; return; }
    $acc[] = is_object($r) ? get_class($r) : (string) $r;
}

function poisoned(array $acc): bool
{
    $pi = unpack('q', POISON)[1];
    foreach ($acc as $v) {
        if ($v === POISON) return true;
        if ($v === 'i' . $pi) return true;
    }
    return false;
}

function desc(mixed $v): string
{
    if ($v instanceof NDArray) return 'NDArray(' . implode('x', $v->shape()) . ',' . $v->dtype()->value . ')';
    if (is_array($v)) {
        $s = json_encode($v, JSON_PARTIAL_OUTPUT_ON_ERROR | JSON_PRESERVE_ZERO_FRACTION);
        if ($s === false || strlen($s) > 80) return 'array[' . count($v) . ']' . substr((string) $s, 0, 60);
        return $s;
    }
    if (is_float($v)) return is_nan($v) ? 'NAN' : (is_infinite($v) ? ($v > 0 ? 'INF' : '-INF') : var_export($v, true));
    return var_export($v, true);
}

// ------------------------------------------------------------------ value pools
mt_srand(12345);
function rnd(int $n, float $lo = -3, float $hi = 3): array { $a = []; for ($i = 0; $i < $n; $i++) $a[] = $lo + ($hi - $lo) * mt_rand() / mt_getrandmax(); return $a; }
$BIG = rnd(100000);
$MID = rnd(3000);
$SAMPLE = [1.5, 2.0, 3.7, 0.2, 5.0, 2.2, 4.1, 0.9];
$ARRAYS = [
    'empty' => [], 'one' => [1.5], 'two' => [1.5, 2.5], 'three' => [0.3, 0.1, 0.2],
    'ties' => [1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0], 'alleq' => [2.0, 2.0, 2.0, 2.0, 2.0],
    'nan' => [NAN, 1.0, 2.0, 3.0, NAN], 'allnan' => [NAN, NAN, NAN], 'inf' => [INF, -INF, 1.0, 2.0],
    'neg' => [-3.0, -1.0, 0.0, 1.0, 2.5], 'ints' => [0, 1, 2, 3, 5, 3], 'negints' => [-5, -1, 0, 2],
    'bools' => [true, false, true], 'huge' => [1e300, -1e300, 1.7e308, 5.0], 'tiny' => [5e-324, 0.0, -0.0, 1e-310],
    'probs' => [0.01, 0.2, 0.5, 0.99, 1.0, 0.0], 'zeros' => [0.0, 0.0, 0.0, 0.0],
    '2d' => [[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 5.0, 3.0], [0.5, 7.0, 2.0, 2.0]],
    '2d_int' => [[3, 1, 2], [1, 4, 0]], '2d_row' => [[1.0, 2.0, 3.0]], '2d_col' => [[1.0], [2.0], [3.0]],
    '2d_0x3' => fn() => NDArray::zeros([0, 3]), '2d_3x0' => fn() => NDArray::zeros([3, 0]), '2d_1x1' => [[4.0]],
    '3d' => fn() => NDArray::array(array_map(fn($i) => array_map(fn($j) => array_map(fn($k) => (float) (($i * 7 + $j * 3 + $k * 5) % 11), range(0, 3)), range(0, 2)), range(0, 1))),
    '3d_0' => fn() => NDArray::zeros([2, 0, 3]),
    'mid' => $MID, 'big' => $BIG, 'bigint' => fn() => NDArray::array(array_map(fn($v) => (int) floor($v * 1000), $GLOBALS['BIG'])),
    'ragged' => [[1.0, 2.0], [3.0]], 'nested_list' => [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0, 8.0]],
    // NDArray objects (fresh per call): the in-place writers take only these
    'nd_1d' => fn() => NDArray::array([1.5, 2.0, 3.7, 0.2, 5.0]), 'nd_int' => fn() => NDArray::array([3, 1, 2, 7]),
    'nd_bool' => fn() => NDArray::array([true, false, true]), 'nd_2d' => fn() => NDArray::array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    'nd_empty' => fn() => NDArray::zeros([0]), 'nd_3d' => fn() => NDArray::zeros([2, 2, 2]),
    'nd_view' => fn() => NDArray::array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])->transpose(),
    'nd_bigidx' => fn() => NDArray::array([PHP_INT_MAX, PHP_INT_MIN, -1, 0]),
];
$SCALARS = [-1, 0, 0.5, 1, 2, 3, 7, -0.5, 1.5, 100, 1e300, -1e300, PHP_INT_MAX, PHP_INT_MIN, NAN, INF, -INF, true, false, null, 1e18, 2147483648, -2147483649, 1e-300, 1000000];
$STRINGS = ['bogus', '', 'two-sided', 'less', 'greater', 'omit', 'raise', 'propagate', 'x', str_repeat('y', 500)];
$AXES = [-2, 5, -5, [0, 1], [0, 0], [], [0, 1, 2], [-1, 5]];

if (getenv('EXTRA')) {
    $ARRAYS = [
        'l8_bigidx' => [0, 1, 2, 3, 4, 5, 6, 100], 'l8_negidx' => [-1, 0, 1, 2, 3, 4, 5, -100], 'l8_perm' => [7, 6, 5, 4, 3, 2, 1, 0],
        'l8_perm1' => [8, 7, 6, 5, 4, 3, 2, 1], 'l8_dup' => [0, 0, 0, 0, 1, 1, 1, 1], 'l8_nan' => [NAN, 1.0, 2.0, NAN, 3.0, 4.0, 5.0, 6.0],
        'l8_huge' => [1e300, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1e300], 'l8_inf' => [INF, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -INF],
        'l8_zero' => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'l8_neg' => [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0],
        'l8_frac' => [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], 'l8_bigint' => [PHP_INT_MAX, 0, 1, 2, 3, 4, 5, PHP_INT_MIN],
        'l8_bool' => [true, false, true, false, true, false, true, false], 'l7' => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 'l9' => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        'pair_rev' => [5.0, 1.0], 'pair_nan' => [NAN, 1.0], 'pair_inf' => [-INF, INF], 'pair_pct' => [-5.0, 150.0], 'pair_pct2' => [90.0, 10.0],
        'pair_huge' => [1e300, -1e300], 'pair_eq' => [2.0, 2.0], 'pair_bool' => [true, false], 'pair_int' => [3, 1], 'triple' => [1.0, 2.0, 3.0],
        'pair_2' => [[1.0, 2.0], [3.0, 4.0]], 'pair_2rev' => [[5.0, 1.0], [4.0, 0.0]],
        't2_neg' => [[-1, 2], [3, 4]], 't2_zero' => [[0, 0], [0, 0]], 't2_nan' => [[NAN, 1.0], [2.0, 3.0]], 't2_huge' => [[1e300, 1.0], [1.0, 1e300]],
        't2_bigc' => [[1000000, 2], [3, 4]], 't2_frac' => [[0.5, 1.5], [2.5, 3.5]], 't2_row0' => [[0, 0], [3, 4]], 't2_col0' => [[0, 2], [0, 4]],
        't3x3_neg' => [[1, -2, 3], [4, 5, 6], [7, 8, 9]], 't2x5' => [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], 't1x8' => [[1, 2, 3, 4, 5, 6, 7, 8]],
        't8x1' => [[1], [2], [3], [4], [5], [6], [7], [8]], 't8x3' => array_map(fn($i) => [($i * 3) % 7 + 0.5, ($i * 5) % 7 + 0.25, $i + 0.125], range(0, 7)),
        't8x3_ties' => array_fill(0, 8, [1.0, 1.0, 1.0]), 't2x2x2' => [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
    ];
    $SCALARS = [];
    $STRINGS = [];
    $AXES = [];
}
function realize(mixed $v): mixed { return $v instanceof \Closure ? $v() : $v; }

function typeOk(\ReflectionParameter $p, mixed $v): bool
{
    $t = $p->getType();
    if ($t === null) return true;
    $names = $t instanceof \ReflectionUnionType ? array_map(fn($x) => $x->getName(), $t->getTypes()) : [$t->getName()];
    if ($t->allowsNull()) $names[] = 'null';
    $ty = match (true) {
        $v === null => 'null', is_bool($v) => 'bool', is_int($v) => 'int', is_float($v) => 'float', is_string($v) => 'string',
        is_array($v) => 'array', $v instanceof NDArray => 'Tessero\\NDArray', $v instanceof \Closure => 'Tessero\\NDArray', default => 'object',
    };
    if (in_array('mixed', $names, true) || in_array($ty, $names, true)) return true;
    if ($ty === 'int' && in_array('float', $names, true)) return true;
    if ($ty === 'Tessero\\NDArray' && in_array('array', $names, true)) return false;
    return false;
}

// ------------------------------------------------------------------ case generation
$cases = [];   // list of [desc, closure]

function paramCases(callable $call, array $params, array $base, string $fname): array
{
    global $ARRAYS, $SCALARS, $STRINGS, $AXES;
    $cases = [];
    $names = array_map(fn($p) => $p->getName(), $params);
    $add = function (string $d, array $args) use (&$cases, $call) {
        $cases[] = [$d, function () use ($call, $args) { return $call(array_map('realize', $args)); }];
    };
    $add('baseline', $base);
    foreach ($params as $i => $p) {
        $n = $p->getName();
        $pool = [];
        foreach ($ARRAYS as $k => $v) $pool["A:$k"] = $v;
        foreach ($SCALARS as $v) $pool['S:' . desc($v)] = $v;
        foreach ($STRINGS as $v) $pool['T:' . substr($v, 0, 12)] = $v;
        if (stripos($n, 'axis') !== false || stripos($n, 'axes') !== false) foreach ($AXES as $v) $pool['X:' . desc($v)] = $v;
        foreach ($pool as $k => $v) {
            if (! typeOk($p, $v)) continue;
            $args = $base; $args[$i] = $v;
            $add("$n=$k", $args);
        }
    }
    // all array-like baseline params simultaneously replaced
    $arrIdx = [];
    foreach ($base as $i => $v) if (is_array($v) || $v instanceof NDArray) $arrIdx[] = $i;
    if (count($arrIdx) > 1) {
        foreach ($ARRAYS as $k => $v) {
            $args = $base; $ok = true;
            foreach ($arrIdx as $i) { if (! typeOk($params[$i], $v)) $ok = false; $args[$i] = $v; }
            if ($ok) $add('ALLARR=' . $k, $args);
        }
    }
    if (getenv('EXTRA')) return $cases;
    // array param x axis combos
    $axi = array_search('axis', $names, true);
    if ($axi !== false && $arrIdx) {
        foreach (['2d', '2d_row', '2d_col', '2d_0x3', '2d_3x0', '3d', '3d_0', '2d_1x1', 'empty', 'one', 'nan'] as $k) {
            foreach ([0, 1, 2, -1, -2, -3, null, 3, 5, -5] as $ax) {
                $args = $base;
                foreach ($arrIdx as $i) $args[$i] = $ARRAYS[$k];
                if (! typeOk($params[$axi], $ax)) continue;
                $args[$axi] = $ax;
                $add("ALLARR=$k,axis=" . desc($ax), $args);
                $args = $base; $args[$arrIdx[0]] = $ARRAYS[$k]; $args[$axi] = $ax;
                $add("A0=$k,axis=" . desc($ax), $args);
            }
        }
    }
    // numeric params combined with small/edge arrays
    foreach ($params as $i => $p) {
        if (in_array($i, $arrIdx, true) || $i === $axi) continue;
        if (! $arrIdx) break;
        foreach (['one', 'two', 'empty', 'ties', '2d'] as $k) {
            foreach ([-1, 0, 0.5, 1, 2, 3, 10, NAN, 1e300, PHP_INT_MAX, -1e300] as $v) {
                if (! typeOk($p, $v)) continue;
                $args = $base; $args[$arrIdx[0]] = $ARRAYS[$k]; $args[$i] = $v;
                $add("A0=$k," . $p->getName() . '=' . desc($v), $args);
            }
        }
    }
    return $cases;
}

function findBase(callable $call, array $params): array
{
    global $SAMPLE;
    $base = [];
    $req = [];
    foreach ($params as $i => $p) {
        if ($p->isDefaultValueAvailable()) { $base[$i] = $p->getDefaultValue(); } else { $base[$i] = $SAMPLE; $req[] = $i; }
    }
    // NDArray candidates are closures (a fresh array per call): the in-place writers (put, copyto, ...) need one
    $cands = [$SAMPLE, 2.0, 0.5, 3, [[3.0, 1.0, 2.0], [1.0, 4.0, 2.0], [2.0, 2.0, 5.0]], [$SAMPLE, [2.1, 0.4, 3.3, 1.8, 2.9]], 'norm', [0.1, 0.02, 0.3, 0.04, 0.5], [2, 5, 3, 1, 4], 0.05, true,
              fn() => NDArray::array($SAMPLE), fn() => NDArray::array([[3.0, 1.0, 2.0], [1.0, 4.0, 2.0], [2.0, 2.0, 5.0]]), fn() => NDArray::array([2, 5, 3, 1, 4])];
    $ok = fn($args) => (function () use ($call, $args) { try { $call(array_map('realize', $args)); return true; } catch (\Throwable $e) { return false; } })();
    if ($req && ! $ok($base)) {
        for ($pass = 0; $pass < 2; $pass++) {
            foreach ($req as $i) {
                foreach ($cands as $c) {
                    if (! typeOk($params[$i], $c)) continue;
                    $t = $base; $t[$i] = $c;
                    if ($ok($t)) { $base = $t; break; }
                }
            }
            if ($ok($base)) break;
        }
    }
    return $base;
}

if ($kind === 'stats' || $kind === 'np' || $kind === 'special') {
    $cls = ['stats' => Stats::class, 'np' => Np::class, 'special' => Special::class][$kind];
    $rm = new \ReflectionMethod($cls, $method);
    $params = array_values(array_filter($rm->getParameters(), fn($p) => $p->getName() !== 'out'));
    $call = fn(array $a) => $cls::$method(...$a);
    if ($kind === 'special') {
        // vectorised extreme-value sweep: each arg an array over EXT, broadcast to the full product
        $EXT = [-INF, -1e300, -1e10, -100.5, -3.0, -1.0, -0.5, -1e-300, 0.0, 1e-300, 0.5, 1.0, 2.0, 3.0, 10.5, 100.0, 1e5, 1e10, 1e300, INF, NAN, 9.2e18, -2147483648.0, 170.5];
        $SMALL = [-INF, -1e300, -3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0, 100.0, 1e300, INF, NAN];
        $nin = count($params);
        $pool = $nin >= 3 ? $SMALL : $EXT;
        foreach ($pool as $v0) {
            $args = [];
            for ($k = 0; $k < $nin; $k++) {
                if ($k === 0) { $args[] = $v0; continue; }
                $sh = array_fill(0, $nin - 1, 1); $sh[$k - 1] = count($pool);
                $args[] = NDArray::array($pool)->reshape($sh);
            }
            $cases[] = ['arg0=' . desc($v0) . ' x product', fn() => $call($args)];
        }
        foreach ([[1, 2, 3], [], [[1.0]], NDArray::zeros([0, 3])] as $a) {
            $cases[] = ['all=' . desc($a), fn() => $call(array_fill(0, $nin, $a))];
        }
        $ints = [PHP_INT_MAX, PHP_INT_MIN, -1, 0, 1, 2147483648];
        foreach ($ints as $iv) $cases[] = ['allint=' . $iv, fn() => $call(array_fill(0, $nin, $iv))];
    } else {
        $base = findBase($call, $params);
        $cases = paramCases($call, $params, $base, $method);
    }
} elseif ($kind === 'dist') {
    $rm = new \ReflectionMethod(Stats::class, $method);
    $params = $rm->getParameters();
    $mk = fn(array $a) => Stats::$method(...$a);
    // baseline shape params: search for a combo where pdf/pmf is finite
    $base = [];
    foreach ($params as $i => $p) $base[$i] = $p->isDefaultValueAvailable() ? $p->getDefaultValue() : 1.5;
    $isDisc = false;
    try { $d = $mk($base); $d->pmf(1.0); $isDisc = true; } catch (\Throwable $e) {}
    $good = function ($b) use ($mk, $isDisc) { try { $d = $mk($b); $v = $isDisc ? $d->pmf(1.0) : $d->pdf(0.5); return is_float($v) && ! is_nan($v); } catch (\Throwable $e) { return false; } };
    if (! $good($base)) {
        foreach ([0.5, 5, 2.5, 0.3, 10, 1] as $c) {
            $t = $base; foreach ($params as $i => $p) if (! $p->isDefaultValueAvailable()) $t[$i] = $c;
            if ($good($t)) { $base = $t; break; }
        }
    }
    $XS = [-INF, -1e300, -10.0, -1.0, -0.5, 0.0, 1e-300, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 10.0, 100.0, 1e300, INF, NAN, 9.2e18];
    $QS = [-1.0, 0.0, 1e-300, 1e-17, 0.25, 0.5, 0.999999, 1.0 - 1e-16, 1.0, 1.5, NAN, INF, -INF];
    $PV = [-1.0, -0.5, 0.0, 1e-300, 1e-10, 0.5, 1.0, 1.5, 2.5, 10.0, 1000.0, 1e10, 1e300, INF, -INF, NAN, 9.2e18, -1e300, 7.0];
    $methods = function (string $tag, array $args) use (&$cases, $mk, $XS, $QS, $isDisc) {
        $xcol = NDArray::array($XS)->reshape([count($XS), 1]);
        $qcol = NDArray::array($QS)->reshape([count($QS), 1]);
        $m = $isDisc ? ['pmf', 'logpmf', 'cdf', 'logcdf', 'sf', 'logsf'] : ['pdf', 'logpdf', 'cdf', 'logcdf', 'sf', 'logsf'];
        foreach ($m as $f) $cases[] = ["$tag.$f(XS)", fn() => $mk($args)->$f($xcol)];
        foreach (['ppf', 'isf'] as $f) $cases[] = ["$tag.$f(QS)", fn() => $mk($args)->$f($qcol)];
        $cases[] = ["$tag.stats(mvsk)", fn() => $mk($args)->stats('mvsk')];
        $cases[] = ["$tag.entropy", fn() => $mk($args)->entropy()];
        $cases[] = ["$tag.support", fn() => $mk($args)->support()];
        $cases[] = ["$tag.median", fn() => $mk($args)->median()];
        foreach ([1, 3, 5] as $o) $cases[] = ["$tag.moment($o)", fn() => $mk($args)->moment($o)];
        $cases[] = ["$tag.interval(0.9)", fn() => $mk($args)->interval(0.9)];
        $cases[] = ["$tag.rvs(7)", fn() => $mk($args)->rvs(7, Generator::defaultRng(42))];
    };
    $methods('base', $base);
    foreach ($params as $i => $p) {
        $a = $base; $a[$i] = NDArray::array($PV);            // vectorised over the adversarial parameter values
        $methods($p->getName() . '=PV', $a);
        foreach ([[], [[1.0, 2.0]], NDArray::zeros([0, 2])] as $arr) {
            $a = $base; $a[$i] = $arr;
            $methods($p->getName() . '=' . desc($arr), $a);
        }
    }
    // scalar sweeps for rvs/moment/entropy/stats (non-vectorised code paths may differ)
    foreach ($params as $i => $p) foreach ([-1.0, 0.0, NAN, INF, 1e300, 0.5, 1e-300, 9.2e18] as $v) {
        $a = $base; $a[$i] = $v; $t = $p->getName() . '=' . desc($v);
        $cases[] = ["$t.rvs(5)", fn() => $mk($a)->rvs(5, Generator::defaultRng(42))];
        $cases[] = ["$t.moment(4)", fn() => $mk($a)->moment(4)];
        $cases[] = ["$t.entropy", fn() => $mk($a)->entropy()];
        $cases[] = ["$t.stats", fn() => $mk($a)->stats('mvsk')];
        $cases[] = ["$t.ppf", fn() => $mk($a)->ppf([0.0, 0.3, 1.0])];
    }
    foreach ([-1, 0, 2, 100, 1000, PHP_INT_MAX, PHP_INT_MIN, -2147483649] as $o) $cases[] = ["base.moment($o)", fn() => $mk($base)->moment($o)];
    foreach ([-1.0, 0.0, 1.0, 2.0, NAN, 1e-300] as $c) $cases[] = ['base.interval(' . desc($c) . ')', fn() => $mk($base)->interval($c)];
    foreach ([0, [0, 3], [3, 0], -1, [3, -1], [], 100000, [2, 3, 4]] as $s) $cases[] = ['base.rvs(' . desc($s) . ')', fn() => $mk($base)->rvs($s, Generator::defaultRng(1))];
    $cases[] = ['base.stats(bad)', fn() => $mk($base)->stats('mq')];
    $cases[] = ['base.stats()', fn() => $mk($base)->stats('')];
} elseif ($kind === 'gen') {
    $rm = new \ReflectionMethod(Generator::class, $method);
    $params = $rm->getParameters();
    $call = fn(array $a) => Generator::defaultRng(7)->$method(...$a);
    $base = findBase($call, $params);
    foreach ($params as $i => $p) if ($p->getName() === 'size' && $base[$i] === null) $base[$i] = 5;
    $cases = paramCases($call, $params, $base, $method);
    $PV = [-1.0, -0.5, 0.0, 1e-300, 1e-10, 0.5, 1.0, 1.5, 2.5, 10.0, 1000.0, 1e10, 1e300, INF, -INF, NAN, 9.2e18, -1e300, 7.0, 1e18, 4294967296.0];
    foreach ($params as $i => $p) {
        if ($p->getName() === 'size' || ! typeOk($p, [1.0])) continue;
        $a = $base; $a[$i] = $PV;
        foreach ($params as $j => $q) if ($q->getName() === 'size') $a[$j] = null;
        $cases[] = [$p->getName() . '=PV', fn() => $call($a)];
        foreach ($params as $j => $q) if ($q->getName() === 'size') { $a[$j] = [4, count($PV)]; $cases[] = [$p->getName() . '=PV,size=[4,n]', fn() => $call($a)]; }
    }
    foreach ($params as $j => $q) if ($q->getName() === 'size') foreach ([0, [0, 3], [3, 0], [], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 1000000, [2, 3], PHP_INT_MAX, [PHP_INT_MAX, 2], [1 << 32, 1 << 32]] as $s) {
        $a = $base; $a[$j] = $s; $cases[] = ['size=' . desc($s), fn() => $call($a)];
    }
}

if ($listOnly) { echo count($cases), "\n"; foreach ($cases as $i => $c) echo "$i {$c[0]}\n"; exit(0); }

$n = count($cases);
E("@@NCASES $n");
for ($i = $start; $i < $n; $i++) {
    [$d, $f] = $cases[$i];
    E("@@CASE $i $d");
    $t0 = microtime(true);
    try {
        $r = $f();
        $acc = []; flat($r, $acc);
        if (poisoned($acc)) E("@@POISON $i $d");
        $r2 = $f();
        $acc2 = []; flat($r2, $acc2);
        if ($acc !== $acc2) E("@@NONDET $i $d");
        $res = 'ok';
    } catch (\Throwable $e) {
        $res = 'exc ' . get_class($e) . ': ' . substr(str_replace("\n", ' ', $e->getMessage()), 0, 150);
    }
    $dt = microtime(true) - $t0;
    E(sprintf("@@END %d %.3f %s", $i, $dt, $res));
}
E("@@DONE $n");
