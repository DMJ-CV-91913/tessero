<?php

// Composed-pipeline ("scenario") parity: run each multi-call NumPy/SciPy workflow mirrored in Tessero and
// compare every checkpoint to the recorded scipy/numpy result. Runs on one backend (FFI by default; the
// native extension when it is loaded and --backend=ext is passed). Exits non-zero on any mismatch.
//
//   php -d ffi.enable=1 tools/parity/run-scenarios.php --backend=ffi
//   php -d extension=.../tessero.so tools/parity/run-scenarios.php --backend=ext

declare(strict_types=1);

require __DIR__ . '/../../tests/autoload.php';

$backend = 'ffi';
foreach ($argv as $a) {
    if (preg_match('/^--backend=(ffi|ext)$/', $a, $m)) {
        $backend = $m[1];
    }
}
$ext = $backend === 'ext';
$ND  = $ext ? 'Tessero\\Ext\\NDArray' : 'Tessero\\NDArray';
$LA  = $ext ? 'Tessero\\Ext\\Linalg' : 'Tessero\\Linalg\\Linalg';
$NP  = $ext ? 'Tessero\\Ext\\Np' : 'Tessero\\Np';
$SIG = $ext ? 'Tessero\\Ext\\Signal' : 'Tessero\\Signal';

$data = json_decode((string) file_get_contents(__DIR__ . '/../../tests/fixtures/scenarios.json'), true, 512, JSON_THROW_ON_ERROR);

/** Coerce a scalar-ish result (float, int, or 0-d NDArray) to float. */
function sc(mixed $v): float
{
    if (is_int($v) || is_float($v)) {
        return (float) $v;
    }
    if (is_object($v)) {
        if (method_exists($v, 'item')) {
            return (float) $v->item();
        }
        if (method_exists($v, 'toList')) {
            $l = $v->toList();
            while (is_array($l)) {
                $l = $l[0];
            }
            return (float) $l;
        }
    }
    if (is_array($v)) {
        while (is_array($v)) {
            $v = $v[0];
        }
        return (float) $v;
    }
    return (float) $v;
}

/** Coerce an array-ish result to a flat PHP list<float>. */
function lst(mixed $v): array
{
    if (is_object($v) && method_exists($v, 'toList')) {
        $v = $v->toList();
    }
    $out = [];
    array_walk_recursive($v, static function ($x) use (&$out): void {
        $out[] = (float) $x;
    });
    return $out;
}

function closeScalar(float $got, float $exp, float $rtol = 1e-9, float $atol = 1e-11): bool
{
    return abs($got - $exp) <= $atol + $rtol * abs($exp);
}

function closeVec(array $got, array $exp, float $rtol = 1e-9, float $atol = 1e-11): bool
{
    if (count($got) !== count($exp)) {
        return false;
    }
    foreach ($exp as $i => $e) {
        if (abs($got[$i] - (float) $e) > $atol + $rtol * abs((float) $e)) {
            return false;
        }
    }
    return true;
}

$fail = 0;
$checked = 0;

/** Run one scenario; return a map checkpoint=>[gotScalarOrVec, ok]. */
function runScenario(string $name, array $in, string $ND, string $LA, string $NP, string $SIG): array
{
    switch ($name) {
        case 'linalg_solve_residual':
            $A = $ND::array($in['A']);
            $b = $ND::array($in['b']);
            $x = $LA::solve($A, $b);
            return [
                'x_sum' => $x->sum(),
                'residual' => $LA::norm($A->matmul($x)->sub($b)),
                'x_head' => array_slice(lst($x), 0, 5),
            ];
        case 'pca_eigvals':
            $X = $ND::array($in['X']);
            $mean = $X->mean(0);
            $std = $X->std(0);
            $Xc = $X->sub($mean);
            $Xs = $Xc->div($std);
            $C = $NP::cov($Xs, null, false);
            $w = $LA::eigvalsh($C)->sort();
            return [
                'eigvals' => $w,
                'trace' => $C->trace(),
                'col_means_abs_max' => $Xc->mean(0)->abs()->max(),
            ];
        case 'polyfit_rmse':
            $x = $ND::array($in['x']);
            $y = $ND::array($in['y']);
            $coef = $NP::polyfit($x, $y, 3);
            $yhat = $NP::polyval($coef, $x);
            return [
                'coef' => $coef,
                'rmse' => sqrt(sc($yhat->sub($y)->square()->mean())),
            ];
        case 'reduction_chain':
            $a = $ND::arange(1.0, 201.0);
            $b = $a->cumsum();
            $d = $NP::diff($b);
            return ['sum_a' => $a->sum(), 'max_cumsum' => $b->max(), 'mean_diff' => $d->mean()];
        case 'transcendental_chain':
            $x = $ND::linspace(0.0, 2.0 * M_PI, 1000);
            $y = $x->sin()->mul($x->mul(2.0)->cos())->add(0.5);
            return ['sum' => $y->sum(), 'max' => $y->max(), 'argmax' => sc($y->argmax())];
        case 'lstsq_residual':
            $A = $ND::array($in['A']);
            $b = $ND::array($in['b']);
            $r = $LA::lstsq($A, $b);
            $x = is_array($r) ? ($r['x'] ?? $r[0]) : (is_object($r) && isset($r->x) ? $r->x : $r);
            return ['x' => $x, 'residual' => $LA::norm($A->matmul($x)->sub($b))];
        case 'fftconvolve_signal':
            $a = $ND::array($in['a']);
            $v = $ND::array($in['v']);
            $conv = $SIG::fftconvolve($a, $v, 'full');
            $cl = lst($conv);
            return [
                'len' => (float) count($cl),
                'sum' => $conv->sum(),
                'argmax_abs' => sc($conv->abs()->argmax()),
                'head' => array_slice($cl, 0, 5),
            ];
        default:
            throw new RuntimeException("unknown scenario {$name}");
    }
}

foreach ($data['scenario_cases'] as $case) {
    $name = $case['name'];
    try {
        $got = runScenario($name, $case['inputs'] ?? [], $ND, $LA, $NP, $SIG);
    } catch (\Throwable $e) {
        echo "FAIL {$name}: threw " . $e->getMessage() . "\n";
        $fail++;
        continue;
    }
    foreach ($case['expected'] as $key => $exp) {
        $checked++;
        $g = $got[$key] ?? null;
        $ok = is_array($exp)
            ? closeVec(lst($g), $exp)
            : closeScalar(sc($g), (float) $exp);
        if (! $ok) {
            $fail++;
            $gs = is_array($exp) ? json_encode(array_slice(lst($g), 0, 4)) : (string) sc($g);
            $es = is_array($exp) ? json_encode(array_slice($exp, 0, 4)) : (string) $exp;
            echo "FAIL {$name}.{$key}: got {$gs} vs {$es}\n";
        }
    }
}

echo "scenarios ({$backend}): {$checked} checkpoints, {$fail} fail\n";
exit($fail === 0 ? 0 : 1);
