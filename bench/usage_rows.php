<?php

/**
 * bench/usage_bench.php and .py output as rows for tools/parity/usage_speed.php (the shape of time_report.py --json):
 *
 *   php bench/usage_rows.php usage_ffi.json usage_numpy.json > usage_rows_ffi.json
 */

declare(strict_types=1);

[$_, $tf, $rf] = $argv;
$t = json_decode((string) file_get_contents($tf), true)['calls'];
$r = json_decode((string) file_get_contents($rf), true)['calls'];
$rows = [];
foreach ($t as $fn => $us) {
    if (isset($r[$fn]) && $us > 0 && $r[$fn] > 0) {
        $rows[] = ['fn' => $fn, 'cases' => 1, 'tessero_us' => $us, 'reference_us' => $r[$fn], 'ratio' => $us / $r[$fn]];
    }
}
echo json_encode(['rows' => $rows], JSON_PRETTY_PRINT), "\n";
