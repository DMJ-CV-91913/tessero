<?php

/**
 * Record the outcome of the PHPUnit suites that verify functions older than the registry (scope.yaml
 * `legacy_tests`) in tools/parity/results.json, from PHPUnit's JUnit log:
 *
 *   vendor/bin/phpunit --log-junit build/junit.xml
 *   php tools/parity/record-suites.php build/junit.xml [more.xml ...]
 *
 * A suite file passes when every test case in it passed in every log given (the FFI run and the extension run).
 * tools/parity/report.py counts a legacy function as verified (M4) only when all its suites passed.
 */

declare(strict_types=1);

$root = dirname(__DIR__, 2);
$logs = array_slice($argv, 1);
if ($logs === []) {
    fwrite(STDERR, "usage: php tools/parity/record-suites.php <junit.xml> ...\n");
    exit(2);
}
$suites = [];
foreach ($logs as $log) {
    $xml = simplexml_load_file($log);
    if ($xml === false) {
        fwrite(STDERR, "cannot read {$log}\n");
        exit(2);
    }
    foreach ($xml->xpath('//testsuite[@file]') ?: [] as $ts) {
        $file = (string) $ts['file'];
        $rel = str_starts_with($file, $root . '/') ? substr($file, strlen($root) + 1) : $file;
        $bad = (int) $ts['failures'] + (int) $ts['errors'];
        $ok = (int) $ts['tests'] > 0 && $bad === 0;
        $suites[$rel] = ($suites[$rel] ?? 'pass') === 'pass' && $ok ? 'pass' : 'fail';
    }
}
$path = "{$root}/tools/parity/results.json";
$results = is_file($path) ? (json_decode((string) file_get_contents($path), true) ?: []) : [];
$results['suites'] = $suites + ($results['suites'] ?? []);
ksort($results['suites']);
file_put_contents($path, json_encode($results, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES) . "\n");
foreach ($suites as $f => $s) {
    echo "{$s} {$f}\n";
}
