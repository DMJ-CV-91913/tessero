<?php

/**
 * Run every examples/**.php on both backends (the CI gate for the cookbook).
 *
 *   php tools/run-examples.php                         # both backends (ext auto-detected)
 *   php tools/run-examples.php --backend=ffi           # one backend
 *   php tools/run-examples.php --ext-so=/path/tessero.so
 *   php tools/run-examples.php --filter=optimize       # only matching paths
 *
 * Each example is run with examples/_bootstrap.php auto-prepended and
 * TESSERO_EXAMPLE_BACKEND set. An example that prints "SKIP: ..." and exits 0 counts as a
 * skip (e.g. an FFI-only example under the extension); a non-zero exit is a failure. The
 * command exits non-zero if any example fails on any backend.
 */

declare(strict_types=1);

$root = dirname(__DIR__);
$opt = getopt('', ['backend::', 'ext-so::', 'filter::']);
$which = $opt['backend'] ?? 'both';
$filter = $opt['filter'] ?? '';
$extSo = $opt['ext-so'] ?? (getenv('EXT_SO') ?: '');

$bootstrap = "{$root}/examples/_bootstrap.php";
$files = [];
$walk = static function (string $dir) use (&$walk, &$files): void {
    foreach (scandir($dir) ?: [] as $e) {
        if ($e === '.' || $e === '..' || $e[0] === '_') {
            continue;
        }
        $p = "{$dir}/{$e}";
        if (is_dir($p)) {
            $walk($p);
        } elseif (str_ends_with($e, '.php')) {
            $files[] = $p;
        }
    }
};
if (is_dir("{$root}/examples")) {
    $walk("{$root}/examples");
}
sort($files);

$backends = $which === 'both' ? ['ffi', 'ext'] : [$which];
if (in_array('ext', $backends, true) && $extSo === '') {
    foreach (["{$root}/ext/modules/tessero.so", (getenv('HOME') ?: '') . '/ext-build/modules/tessero.so'] as $c) {
        if ($c !== '' && is_file($c)) {
            $extSo = $c;
            break;
        }
    }
}

function run_example(string $php, array $ini, string $file, string $backend): array
{
    $cmd = array_merge([$php], $ini, ['-d', 'auto_prepend_file=' . $GLOBALS['bootstrap'], $file]);
    $env = getenv();
    $env['TESSERO_EXAMPLE_BACKEND'] = $backend;
    $p = proc_open($cmd, [1 => ['pipe', 'w'], 2 => ['pipe', 'w']], $pipes, null, $env);
    $out = stream_get_contents($pipes[1]);
    $err = stream_get_contents($pipes[2]);
    foreach ($pipes as $pipe) {
        fclose($pipe);
    }
    $rc = proc_close($p);
    $text = $out . $err;
    if (str_contains($out, 'SKIP:')) {
        return ['skip', trim(explode('SKIP:', $out, 2)[1])];
    }

    return [$rc === 0 ? 'pass' : 'fail', trim($text)];
}

$php = PHP_BINARY;
$counts = ['pass' => 0, 'skip' => 0, 'fail' => 0];
$failed = [];
foreach ($files as $f) {
    $rel = substr($f, strlen($root) + 1);
    if ($filter !== '' && ! str_contains($rel, $filter)) {
        continue;
    }
    $line = str_pad($rel, 52);
    foreach ($backends as $b) {
        if ($b === 'ext' && $extSo === '') {
            $line .= '  ext:no-so';
            continue;
        }
        $ini = $b === 'ext' ? ['-d', 'ffi.enable=1', '-d', "extension={$extSo}"] : ['-d', 'ffi.enable=1'];
        [$status, $detail] = run_example($php, $ini, $f, $b);
        $counts[$status]++;
        $line .= '  ' . strtoupper($b) . ':' . $status;
        if ($status === 'fail') {
            $failed[] = "{$rel} [{$b}]\n" . $detail;
        }
    }
    echo $line . "\n";
}

echo "\n" . sprintf("examples: %d pass, %d skip, %d fail (backends: %s)\n",
    $counts['pass'], $counts['skip'], $counts['fail'], implode('+', $backends));
if ($failed !== []) {
    echo "\n--- failures ---\n" . implode("\n\n", $failed) . "\n";
    exit(1);
}
exit(0);
