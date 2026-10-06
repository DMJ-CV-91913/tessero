<?php

/**
 * Emit docs/examples/manifest.json: machine-readable metadata for every cookbook example, so a
 * consumer (e.g. the website app building Blade pages) can render example cards/pages without
 * parsing PHP. Mirrors tools/gen-example-docs.php.
 *
 *   php tools/gen-examples-manifest.php          # write the manifest
 *   php tools/gen-examples-manifest.php --check    # CI/gate: non-zero exit if stale
 */

declare(strict_types=1);

$root = dirname(__DIR__);
$check = in_array('--check', array_slice($argv, 1), true);
$exRoot = "{$root}/examples";

$titles = [
    'arrays' => 'Arrays and indexing', 'math' => 'Math and ufuncs', 'reductions' => 'Reductions and statistics',
    'sorting' => 'Sorting and searching', 'linalg' => 'Linear algebra (dense)', 'sparse' => 'Sparse matrices',
    'fft' => 'FFT', 'random' => 'Random', 'io' => '.npy / .npz and memory maps', 'special' => 'scipy.special',
    'stats' => 'scipy.stats', 'optimize' => 'Optimize', 'csgraph' => 'Graphs (csgraph)',
    'mdp' => 'Markov decision processes', 'laravel' => 'Laravel bridge',
];

/** The example's leading docblock as lines (no comment markers). */
function block(string $path): array
{
    $src = (string) file_get_contents($path);
    if (! preg_match('#/\*\*(.*?)\*/#s', $src, $m)) {
        return [];
    }
    $lines = array_map(static fn (string $l): string => trim((string) preg_replace('/^\s*\*\s?/', '', $l)), preg_split('/\R/', $m[1]) ?: []);

    return array_values(array_filter($lines, static fn (string $l): bool => $l !== ''));
}

$examples = [];
foreach (scandir($exRoot) ?: [] as $mod) {
    if ($mod === '.' || $mod === '..' || $mod[0] === '_' || ! is_dir("{$exRoot}/{$mod}")) {
        continue;
    }
    $files = array_values(array_filter(scandir("{$exRoot}/{$mod}") ?: [], static fn (string $f): bool => str_ends_with($f, '.php')));
    sort($files);
    foreach ($files as $f) {
        $path = "{$exRoot}/{$mod}/{$f}";
        $lines = block($path);
        $src = (string) file_get_contents($path);
        $equivalent = '';
        foreach ($lines as $l) {
            if (preg_match('/^(NumPy|SciPy|Tessero):/i', $l)) {
                $equivalent = $l;
                break;
            }
        }
        $examples[] = [
            'id' => "{$mod}/" . pathinfo($f, PATHINFO_FILENAME),
            'module' => $mod,
            'moduleTitle' => $titles[$mod] ?? ucfirst($mod),
            'name' => pathinfo($f, PATHINFO_FILENAME),
            'file' => "examples/{$mod}/{$f}",
            'summary' => $lines[0] ?? '',
            'equivalent' => $equivalent,
            'backends' => str_contains($src, 'ffi_only(') ? ['ffi'] : ['ffi', 'ext'],
        ];
    }
}

$byModule = [];
foreach ($examples as $e) {
    $byModule[$e['module']] = ($byModule[$e['module']] ?? 0) + 1;
}

$manifest = [
    'generated_by' => 'tools/gen-examples-manifest.php',
    'total' => count($examples),
    'modules' => $byModule,
    'examples' => $examples,
];
$json = json_encode($manifest, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES) . "\n";

$path = "{$root}/docs/examples/manifest.json";
if ($check) {
    if (is_file($path) && file_get_contents($path) === $json) {
        echo "docs/examples/manifest.json is current\n";
        exit(0);
    }
    fwrite(STDERR, "docs/examples/manifest.json is out of date: run `php tools/gen-examples-manifest.php`\n");
    exit(1);
}
file_put_contents($path, $json);
echo 'wrote docs/examples/manifest.json (' . count($examples) . " examples)\n";
