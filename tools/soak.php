<?php

/**
 * Soak test: run a mixed Tessero workload in ONE long-lived process for hours
 * and fail if anything accumulates - native memory, mapped files, PHP heap,
 * resident memory (RSS), or open file descriptors. This is what a queue
 * worker, an Octane worker or a long CLI job does to the library.
 *
 *   php -d ffi.enable=1 tools/soak.php --backend=ffi --duration=3600
 *   php -d extension=tessero tools/soak.php --backend=ext --duration=86400 --log=soak.jsonl
 *
 * Options:
 *   --backend=ffi|ext   which binding to exercise (default: ext when loaded)
 *   --duration=SECONDS  wall-clock run time (default 60)
 *   --threads=N         OpenMP threads for the run (default 2)
 *   --log=FILE          write one JSON sample per interval (JSON lines)
 *   --interval=SECONDS  sampling interval (default 10)
 *
 * Every iteration frees everything it allocates, so after warm-up the native
 * allocation counter and mapped bytes must be back at their baseline at the
 * end of every iteration (checked exactly), and PHP heap, RSS and descriptor
 * count must not trend upwards (least-squares slope over the second half of
 * the run, with thresholds below). Exit code 0 = pass, 1 = fail.
 *
 * The workload function tessero_soak_iteration() can also be called from a
 * route or a job to soak a real Octane / Horizon deployment; see
 * docs/operations/soak-testing.md.
 */

declare(strict_types=1);

// standalone: the repository's autoloader; included from an application: Composer's is already active
if (! class_exists(\Tessero\NDArray::class) && is_file(dirname(__DIR__) . '/tests/autoload.php')) {
    require_once dirname(__DIR__) . '/tests/autoload.php';
}

const SOAK_RSS_SLOPE_MAX = 64 * 1024;      // bytes per minute of sustained RSS growth
const SOAK_HEAP_SLOPE_MAX = 16 * 1024;     // bytes per minute of PHP heap growth
const SOAK_FD_GROWTH_MAX = 2;

/** One round of the mixed workload. Allocates, computes, maps files, solves; frees everything. */
function tessero_soak_iteration(string $backend, int $i, string $dir): void
{
    $X = $backend === 'ext' ? \Tessero\Ext\NDArray::class : \Tessero\NDArray::class;
    $M = $backend === 'ext' ? \Tessero\Ext\Math::class : \Tessero\Math::class;
    $n = 200 + ($i % 7) * 50;

    // element-wise, broadcasting, ufuncs with out, views, reductions over several axes
    $a = $X::arange((float) ($n * 60))->reshape([$n, 60])->div(1000.0);
    $b = $X::linspace(-2.0, 2.0, 60);
    $c = $M::logaddexp($a, $b);
    $M::erf($c['::2'], $c['::2']);
    $M::cbrt($a, $a);
    $s = $c->sum([0, 1]);
    $m = $c->mean(0, true);
    $v = $a->transpose()->std([0]);
    $mask = $a->gt(0.5);
    $f = $a[$mask];
    $j = $c['1:10, ::3']->toJson();
    unset($a, $b, $c, $s, $m, $v, $mask, $f, $j);

    // sort, FFT (FFI only exposes the Fft class), matmul
    $r = $X::linspace(0.0, 1.0, 4096)->mul(3.7)->sin();
    $r->sort();
    $r->argsort();
    if ($backend === 'ffi') {
        \Tessero\Fft\Fft::irfft(\Tessero\Fft\Fft::rfft($r), 4096);
    }
    $p = $X::ones([64, 64])->matmul($X::eye(64));
    unset($r, $p);

    // files: .npy round trip and memory maps (all modes); descriptors and mappings must be released
    $path = "{$dir}/soak_{$i}.npy";
    $w = $X::openMemmap($path, 'w+', [128, 16], 'float32');
    $w->assign($X::ones([128, 16])->mul((float) $i));
    $w->flush();
    unset($w);
    $ro = $X::load($path, 'r');
    $ro->sum();
    $cow = $X::load($path, 'c');
    $cow['0'] = 0.0;
    $heap = $X::load($path);
    unset($ro, $cow, $heap);
    $raw = "{$dir}/soak_{$i}.f64";
    $mm = $X::memmap($raw, 'w+', [1000]);
    $mm->assign(1.0);
    $mm['::10']->flush(true);
    unset($mm);
    @unlink($path);
    @unlink($raw);

    // solvers (through the FFI classes or the extension's Engine)
    if ($backend === 'ffi') {
        \Tessero\Optimize\LinearProgramming::linprog([-3, -5], [[1, 0], [0, 2], [3, 2]], [4, 12, 18]);
        $P = [[[0.9, 0.1], [0.2, 0.8]], [[0.5, 0.5], [0.1, 0.9]]];
        \Tessero\Mdp\MarkovDecisionProcess::fromDense($P, [[1.0, 0.0], [0.0, 2.0]])->policyIteration(0.9);
    } else {
        \Tessero\Ext\Engine::linprog([-3, -5], [[1, 0], [0, 2], [3, 2]], [4, 12, 18]);
    }

    // error paths must not leak either
    try {
        $X::zeros([3])->add($X::zeros([4]));
    } catch (\Throwable) {
    }
    try {
        $X::load("{$dir}/missing.npy");
    } catch (\Throwable) {
    }
}

function soak_rss(): int
{
    $s = @file_get_contents('/proc/self/status');
    if ($s !== false && preg_match('/VmRSS:\s+(\d+) kB/', $s, $m)) {
        return (int) $m[1] * 1024;
    }

    return memory_get_usage(true);
}

function soak_fds(): int
{
    $l = @scandir('/proc/self/fd');

    return $l === false ? -1 : count($l) - 2;
}

/** @return array{int, int} native bytes in use, mapped bytes */
function soak_native(string $backend): array
{
    if ($backend === 'ext') {
        $i = \Tessero\Ext\Engine::info();

        return [(int) ($i['memory_in_use'] ?? $i['allocated'] ?? 0), (int) $i['memory_mapped']];
    }
    $i = \Tessero\Tessero::info();

    return [(int) $i['memory_in_use'], (int) $i['memory_mapped']];
}

/**
 * Streaming least-squares slope (constant memory: the harness itself must not grow the heap it measures).
 * $acc = [n, Σx, Σy, Σxy, Σx²].
 */
function soak_acc(array &$acc, float $x, float $y): void
{
    $acc[0]++;
    $acc[1] += $x;
    $acc[2] += $y;
    $acc[3] += $x * $y;
    $acc[4] += $x * $x;
}

function soak_slope(array $acc): float
{
    [$n, $sx, $sy, $sxy, $sxx] = $acc;
    $den = $n * $sxx - $sx * $sx;

    return $n >= 3 && $den > 0 ? ($n * $sxy - $sx * $sy) / $den : 0.0;
}

// ---------------------------------------------------------------- main
if (PHP_SAPI === 'cli' && realpath($_SERVER['SCRIPT_FILENAME'] ?? '') === __FILE__) {
    $opt = getopt('', ['backend::', 'duration::', 'threads::', 'log::', 'interval::']);
    $backend = $opt['backend'] ?? (extension_loaded('tessero') ? 'ext' : 'ffi');
    $duration = (int) ($opt['duration'] ?? 60);
    $interval = max(1, (int) ($opt['interval'] ?? 10));
    $threads = (int) ($opt['threads'] ?? 2);
    $log = isset($opt['log']) ? fopen($opt['log'], 'w') : null;
    if ($backend === 'ext' && ! extension_loaded('tessero')) {
        fwrite(STDERR, "ext-tessero is not loaded\n");
        exit(2);
    }
    if ($backend === 'ext') {
        \Tessero\Ext\Engine::setMaxThreads($threads);
    } else {
        \Tessero\Tessero::setThreads($threads);
    }
    $dir = sys_get_temp_dir() . '/tessero_soak_' . getmypid();
    @mkdir($dir);

    // warm-up: caches, JIT, OpenMP pool, allocator arenas
    for ($i = 0; $i < 20; $i++) {
        tessero_soak_iteration($backend, $i, $dir);
    }
    gc_collect_cycles();
    [$native0, $mapped0] = soak_native($backend);
    $fd0 = soak_fds();
    $t0 = microtime(true);
    $next = $t0;
    $samples = 0;
    $rssAcc = $heapAcc = [0, 0.0, 0.0, 0.0, 0.0];
    $iter = 0;
    $failures = [];
    while (($now = microtime(true)) - $t0 < $duration) {
        tessero_soak_iteration($backend, $iter++, $dir);
        [$native, $mapped] = soak_native($backend);
        if ($native !== $native0 || $mapped !== $mapped0) {
            gc_collect_cycles();
            [$native, $mapped] = soak_native($backend);
            if ($native !== $native0 || $mapped !== $mapped0) {
                $failures[] = "iteration {$iter}: native {$native} (baseline {$native0}), mapped {$mapped} (baseline {$mapped0})";
                break;
            }
        }
        if ($now >= $next) {
            $sample = ['t' => round($now - $t0, 1), 'iterations' => $iter, 'native' => $native, 'mapped' => $mapped,
                'heap' => memory_get_usage(), 'rss' => soak_rss(), 'fds' => soak_fds()];
            $samples++;
            if ($sample['t'] >= $duration / 2) {     // trends over the second half; the first absorbs warm-up
                soak_acc($rssAcc, $sample['t'] / 60, (float) $sample['rss']);
                soak_acc($heapAcc, $sample['t'] / 60, (float) $sample['heap']);
            }
            if ($log) {
                fwrite($log, json_encode($sample) . "\n");
            }
            $next += $interval;
        }
    }
    $rssSlope = soak_slope($rssAcc);
    $heapSlope = soak_slope($heapAcc);
    $fdGrowth = soak_fds() - $fd0;
    if ($rssAcc[0] >= 3 && $rssSlope > SOAK_RSS_SLOPE_MAX) {
        $failures[] = sprintf('RSS grows %.1f KiB/min', $rssSlope / 1024);
    }
    if ($heapAcc[0] >= 3 && $heapSlope > SOAK_HEAP_SLOPE_MAX) {
        $failures[] = sprintf('PHP heap grows %.1f KiB/min', $heapSlope / 1024);
    }
    if ($fd0 >= 0 && $fdGrowth > SOAK_FD_GROWTH_MAX) {
        $failures[] = "file descriptors grew by {$fdGrowth}";
    }
    @rmdir($dir);
    $summary = ['backend' => $backend, 'php' => PHP_VERSION, 'duration_s' => round(microtime(true) - $t0), 'iterations' => $iter,
        'samples' => $samples, 'native_baseline' => $native0, 'rss_slope_kib_per_min' => round($rssSlope / 1024, 2),
        'heap_slope_kib_per_min' => round($heapSlope / 1024, 2), 'fd_growth' => $fdGrowth,
        'rss_final_mib' => round(soak_rss() / 1048576, 1), 'result' => $failures ? 'FAIL' : 'PASS', 'failures' => $failures];
    echo json_encode($summary, JSON_PRETTY_PRINT), "\n";
    exit($failures ? 1 : 0);
}
