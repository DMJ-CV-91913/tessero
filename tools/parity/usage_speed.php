<?php
// Usage-weighted per-call speed: each timed function's per-call time (tier 3 of the speed sweep: the rows of
// time_report.py --json, Tessero through FFI and, optionally, the extension) weighted by how often real library code
// calls it (tools/parity/usage.json, the corpus of the usage-weighted coverage), and the time a million and a billion
// calls of that mix take on each side. Functions without a timed row, or without a usage count, are left out; the
// share of the corpus' calls the timed functions carry is printed first. Each side may be a comma-separated list
// of row files (e.g. tier 3 plus bench/usage_rows.php's rows for the most-called functions tier 3 does not time);
// a later file's row replaces an earlier one's for the same function.
//   php tools/parity/usage_speed.php tools/parity/usage.json t3_ffi.json[,usage_rows_ffi.json] [t3_ext.json[,...]]
[$_, $uf, $ff] = $argv;
$xf = $argv[3] ?? null;
$usage = json_decode(file_get_contents($uf), true)['symbols'];
$prefix = ['np' => 'numpy.', 'fft' => 'numpy.fft.', 'linalg' => 'numpy.linalg.', 'special' => 'scipy.special.',
    'stats' => 'scipy.stats.', 'slinalg' => 'scipy.linalg.', 'scipyfft' => 'scipy.fft.', 'signal' => 'scipy.signal.',
    'windows' => 'scipy.signal.windows.', 'ndimage' => 'scipy.ndimage.', 'sparse' => 'scipy.sparse.',
    'splinalg' => 'scipy.sparse.linalg.', 'csgraph' => 'scipy.sparse.csgraph.', 'distance' => 'scipy.spatial.distance.',
    'hierarchy' => 'scipy.cluster.hierarchy.', 'cluster' => 'scipy.cluster.vq.', 'interpolate' => 'scipy.interpolate.'];
$polyPre = ['cheb' => 'chebyshev', 'herme' => 'hermite_e', 'herm' => 'hermite', 'lag' => 'laguerre', 'leg' => 'legendre', 'poly' => 'polynomial'];
$canon = function (string $fn) use ($prefix, $polyPre, $usage): ?string {
    [$m, $name] = explode('.', $fn, 2);
    if ($m === 'npoly') {
        foreach ($polyPre as $p => $mod) {
            if (str_starts_with($name, $p)) {
                $s = "numpy.polynomial.$mod.$name";
                return isset($usage[$s]) ? $s : null;
            }
        }
        return null;
    }
    if (! isset($prefix[$m])) return null;
    $s = $prefix[$m] . $name;
    return isset($usage[$s]) ? $s : null;
};
$load = function (string $files) use ($canon): array {
    $rows = [];
    foreach (explode(',', $files) as $f) {
        foreach (json_decode(file_get_contents($f), true)['rows'] as $r) {
            $s = $canon($r['fn']);
            if ($s !== null) $rows[$s] = $r;      // one timed row per usage symbol (the last file's)
        }
    }
    return $rows;
};
$total = 0; foreach ($usage as $v) $total += $v['calls'];
foreach (array_filter(['FFI' => $ff, 'extension' => $xf]) as $label => $f) {
    $rows = $load($f);
    $calls = 0; $lw = 0.0; $fast = 0; $tt = 0.0; $tr = 0.0;
    foreach ($rows as $s => $r) {
        $c = $usage[$s]['calls'];
        $calls += $c;
        $lw += $c * log($r['ratio']);
        if ($r['ratio'] < 1) $fast += $c;
        $tt += $c * $r['tessero_us'];
        $tr += $c * $r['reference_us'];
    }
    printf("== %s: %d timed functions carry %d of %d corpus calls (%.1f%%)\n", $label, count($rows), $calls, $total, 100 * $calls / $total);
    printf("   usage-weighted geomean Tessero/reference %.2f; Tessero faster on %.1f%% of calls\n", exp($lw / $calls), 100 * $fast / $calls);
    $mt = $tt / $calls; $mr = $tr / $calls;   // us per call of the mix
    printf("   per call of the usage mix: Tessero %.2f us, NumPy/SciPy %.2f us (%.2fx)\n", $mt, $mr, $mr / $mt);
    printf("   1 million calls: Tessero %.2f s, NumPy/SciPy %.2f s;  1 billion calls: Tessero %.1f min, NumPy/SciPy %.1f min\n",
        $mt, $mr, $mt * 1e3 / 60, $mr * 1e3 / 60);
    $top = $rows; uksort($top, fn ($a, $b) => $usage[$b]['calls'] <=> $usage[$a]['calls']);
    echo "   most-called timed functions: calls, Tessero us, reference us, ratio\n";
    $i = 0;
    foreach ($top as $s => $r) {
        printf("   %-34s %6d %9.2f %9.2f %6.2f\n", $s, $usage[$s]['calls'], $r['tessero_us'], $r['reference_us'], $r['ratio']);
        if (++$i >= 15) break;
    }
}
