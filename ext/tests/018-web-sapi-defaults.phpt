--TEST--
Web SAPI defaults: the native budget follows memory_limit (at least 256 MiB) and the BLAS pool is capped at 1
--EXTENSIONS--
tessero
--SKIPIF--
<?php
if (!function_exists('proc_open')) die('skip proc_open is disabled');
if (PHP_OS_FAMILY === 'Windows') die('skip POSIX test');
if (getenv('SKIP_SLOW_TESTS')) die('skip slow test (starts the built-in web server)');
?>
--FILE--
<?php
// The built-in web server (SAPI "cli-server") is a web SAPI: it gets the php-fpm defaults. Each case starts one
// with this extension and reads the settings a request sees.
$extdir = ini_get('extension_dir');
$dir = sys_get_temp_dir() . '/tsr-sapi-' . getmypid();
@mkdir($dir);
file_put_contents("$dir/r.php", '<?php use Tessero\Ext\Engine; echo json_encode(["budget" => Engine::memoryBudget(), '
    . '"blas" => Engine::feature("openblas_threads") ? Engine::getBlasThreads() : 1, "sapi" => PHP_SAPI]);');

function serve(string $dir, string $extdir, array $ini): ?array
{
    $sock = stream_socket_server('tcp://127.0.0.1:0', $errno, $errstr);
    if (!$sock) return null;
    $port = (int) substr(strrchr(stream_socket_get_name($sock, false), ':'), 1);
    fclose($sock);
    $cmd = [PHP_BINARY, '-n', '-d', "extension_dir=$extdir", '-d', 'extension=tessero'];
    foreach ($ini as $k => $v) { $cmd[] = '-d'; $cmd[] = "$k=$v"; }
    array_push($cmd, '-S', "127.0.0.1:$port", '-t', $dir);
    $proc = proc_open($cmd, [0 => ['pipe', 'r'], 1 => ['file', '/dev/null', 'w'], 2 => ['file', '/dev/null', 'w']], $pipes);
    if (!is_resource($proc)) return null;
    $out = null;
    for ($i = 0; $i < 100 && $out === null; $i++) {          // wait for the server to listen (up to 10 s)
        usleep(100000);
        $body = @file_get_contents("http://127.0.0.1:$port/r.php");
        if ($body !== false) $out = json_decode($body, true);
    }
    proc_terminate($proc);
    proc_close($proc);
    return $out;
}

foreach ([
    'memory_limit=512M' => ['memory_limit' => '512M'],
    'memory_limit=64M' => ['memory_limit' => '64M'],
    'memory_limit=-1' => ['memory_limit' => '-1'],
    'explicit 3M' => ['memory_limit' => '512M', 'tessero.memory_budget' => '3M'],
    'explicit unlimited' => ['memory_limit' => '512M', 'tessero.memory_budget' => '-1'],
    'blas 0 (leave alone)' => ['memory_limit' => '512M', 'tessero.blas_threads' => '0'],
] as $label => $ini) {
    $r = serve($dir, $extdir, $ini);
    if ($r === null) { echo "$label: server did not start\n"; continue; }
    echo "$label: sapi={$r['sapi']} budget={$r['budget']}", $label === 'blas 0 (leave alone)' ? '' : " blas={$r['blas']}", "\n";
}
@unlink("$dir/r.php");
@rmdir($dir);
?>
--EXPECT--
memory_limit=512M: sapi=cli-server budget=536870912 blas=1
memory_limit=64M: sapi=cli-server budget=268435456 blas=1
memory_limit=-1: sapi=cli-server budget=0 blas=1
explicit 3M: sapi=cli-server budget=3145728 blas=1
explicit unlimited: sapi=cli-server budget=0 blas=1
blas 0 (leave alone): sapi=cli-server budget=536870912
