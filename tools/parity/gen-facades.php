<?php

/**
 * Generate the FFI façade classes (Tessero\Special, Tessero\Stats, ...) from libtessero's function registry
 * (ADR 0011). The native extension builds the same classes at start-up from the same registry, so the two
 * APIs cannot drift; these files exist so that IDEs, static analysis and reflection see real methods.
 *
 *   php -d ffi.enable=1 tools/parity/gen-facades.php            # write src/<Class>.php
 *   php -d ffi.enable=1 tools/parity/gen-facades.php --check    # fail if a file is out of date
 *   php -d ffi.enable=1 tools/parity/gen-facades.php --out=DIR  # write the classes into DIR instead (for a
 *                                                                work-in-progress library, TESSERO_LIB=...)
 */

declare(strict_types=1);

require dirname(__DIR__, 2) . '/tests/autoload.php';

use Tessero\Native\Registry;

const CLASSES = [
    'special' => ['Special', 'scipy.special: special functions (gamma, Bessel, error functions, orthogonal polynomials, ...). Element-wise: arguments broadcast as in NumPy, all-scalar calls return floats.'],
    'stats' => ['Stats', 'scipy.stats: probability distributions and statistical functions.'],
    'np' => ['Np', 'NumPy functions beyond the NDArray methods: statistics, NaN-aware reductions, quantiles, histograms, set operations, searching (numpy.*).'],
    'linalg' => ['Linalg', 'numpy.linalg on LAPACK, shared by both backends through the kernel: solve, inv, det, slogdet, cholesky, svd, eigh, qr, eig, ... on stacked (..., M, M) arrays.'],
    'slinalg' => ['ScipyLinalg', 'scipy.linalg on LAPACK, shared by both backends through the kernel: solve, inv, det, cholesky, lu, solve_triangular, svd, eigh, ...'],
    'distance' => ['Distance', 'scipy.spatial.distance: pdist, cdist and squareform over real float64 data, with the common metrics (euclidean, cityblock, cosine, correlation, chebyshev, minkowski, ...).'],
    'sparse' => ['Sparse', 'scipy.sparse constructors: csr/csc/coo/dia _array and _matrix from a dense matrix, returning the storage arrays (data, indices/indptr or row/col or offsets, shape) as scipy lays them out.'],
    'signal' => ['Signal', 'scipy.signal: 1-D convolve (full/same/valid) and lfilter over real float64 sequences.'],
    'windows' => ['SignalWindows', 'scipy.signal.windows: closed-form window functions (hann, hamming, blackman, blackmanharris, nuttall, flattop, boxcar, triang, bartlett, cosine, lanczos, bohman, barthann, parzen, general_cosine, general_hamming), shared by both backends through the kernel.'],
    'ndimage' => ['Ndimage', 'scipy.ndimage: generate_binary_structure, 2-D connected-component label, and 2-D convolve with the standard boundary modes.'],
    'csgraph' => ['Csgraph', 'scipy.sparse.csgraph: connected_components (weak) over a dense adjacency matrix.'],
    'interpolate' => ['Interpolate', 'scipy.interpolate: 1-D piecewise-cubic interpolation evaluated at query points (pchip_interpolate, PchipInterpolator, Akima1DInterpolator, CubicSpline not-a-knot, CubicHermiteSpline), shared by both backends through the kernel.'],
    'cluster' => ['Cluster', 'scipy.cluster.vq: observation whitening and vector quantization (whiten, vq), shared by both backends through the kernel.'],
    'npoly' => ['Polynomial', 'numpy.polynomial: evaluation and calculus in the power, Chebyshev, Legendre, Laguerre, Hermite and HermiteE bases, shared by both backends through the kernel.'],
    'random' => ['Random/GeneratorMethods', 'The distribution methods of numpy.random.Generator, drawn with NumPy\'s own distribution code from this generator\'s PCG64 stream, so every call matches NumPy bit for bit.'],
];

function camel(string $s): string
{
    $parts = explode('_', $s);
    $out = array_shift($parts);
    foreach ($parts as $p) {
        $out .= $p === '' ? '_' : ucfirst($p);
    }

    return $out;
}

function phpVar(string $s): string
{
    $s = camel(preg_replace('/[^A-Za-z0-9_]/', '_', $s));

    return preg_match('/^[0-9]/', $s) ? "_{$s}" : $s;
}

function phpDefault(string $d): string
{
    if (preg_match("/^'.*'$/", $d)) {
        return $d;
    }

    return match (true) {
        $d === 'None' => 'null',
        $d === 'True' => 'true',
        $d === 'False' => 'false',
        $d === 'inf' => 'INF',
        $d === '-inf' => '-INF',
        is_numeric($d) => $d,
        default => var_export($d, true),
    };
}

$byClass = [];
foreach (Registry::all() as $name => $info) {
    [$mod, $fn] = explode('.', $name, 2);
    $byClass[$mod][$fn] = $info;
}

$check = in_array('--check', $argv, true);
$outDir = null;
foreach ($argv as $a) {
    if (str_starts_with($a, '--out=')) {
        $outDir = substr($a, 6);
    }
}
$stale = [];
foreach ($byClass as $mod => $fns) {
    if (! isset(CLASSES[$mod])) {
        fwrite(STDERR, "no façade class for module '{$mod}'\n");
        exit(1);
    }
    [$class, $summary] = CLASSES[$mod];
    ksort($fns);
    $methods = [];
    $table = [];
    foreach ($fns as $fn => $info) {
        $name = "{$mod}.{$fn}";
        $defaults = [];
        foreach (($info['defaults'] ?? []) as $k => $v) {
            $defaults[phpVar($k)] = $v;
        }
        $method = camel($fn);
        $ref = match ($mod) { 'np' => 'numpy.', 'linalg' => 'numpy.linalg.', 'slinalg' => 'scipy.linalg.', 'distance' => 'scipy.spatial.distance.', 'csgraph' => 'scipy.sparse.csgraph.', 'windows' => 'scipy.signal.windows.', 'random' => 'numpy.random.Generator.', default => "scipy.{$mod}." } . $fn;  // sparse -> scipy.sparse.
        $doc = trim((string) ($info['doc'] ?? ''));
        $table[] = "    '{$fn}' => '{$method}',";
        if ($info['kind'] === 'ufunc') {
            $args = array_map('phpVar', $info['args']);
            $params = array_map(static fn (string $a): string => "mixed \${$a}", $args);
            $single = $info['nout'] === 1;
            $params[] = $single ? '?NDArray $out = null' : '?array $out = null';
            $ret = $single ? 'NDArray|float' : 'array';
            $retDoc = $single ? '' : "\n     *\n     * @return array{" . implode(', ', array_map(static fn (string $o): string => "{$o}: NDArray|float", $info['outs'])) . '}';
            $call = 'Registry::ufunc(' . var_export($name, true) . ', [' . implode(', ', array_map(static fn (string $a): string => "\${$a}", $args)) . '], $out)';
            $methods[] = "    /**\n     * {$doc}\n     *\n     * {$ref}{$retDoc}\n     */\n    public static function {$method}(" . implode(', ', $params) . "): {$ret}\n    {\n        return {$call};\n    }\n";
        } elseif ($info['kind'] === 'gufunc') {
            $arrays = array_map('phpVar', $info['args']);
            $params = array_map(static fn (string $a): string => "mixed \${$a}" . (isset($defaults[$a]) ? ' = ' . phpDefault($defaults[$a]) : ''), $arrays);
            $hidden = $info['axis'] === 'hidden';
            $keep = ! $hidden && ($info['keepdims'] ?? 1);
            if (! $hidden) {
                $params[] = (($info['axis_single'] ?? 0) ? '?int' : 'int|array|null') . ' $axis = ' . phpDefault($info['axis']);
            }
            $named = [];
            foreach ($info['params'] as $p) {
                $v = phpVar($p['name']);
                if (isset($p['enum'])) {
                    $params[] = "string \${$v} = " . var_export($p['enum'][0], true);
                } else {
                    $d = phpDefault($p['default'] ?? 'None');
                    $params[] = "int|float|bool|null \${$v} = {$d}";
                }
                $named[] = var_export($p['name'], true) . " => \${$v}";
            }
            if ($keep) {
                $params[] = 'bool $keepdims = false';
            }
            $single = $info['nout'] === 1;
            // a bool result: a bool output (isin), or numpy.quantile's discrete methods on a bool array (keep_param)
            $ret = $single ? 'NDArray|float|int' . (($info['out_bool'] & 1) || ($info['keep_param'] ?? -1) >= 0 ? '|bool' : '') : 'array';
            $call = 'Registry::gufunc(' . var_export($name, true) . ', [' . implode(', ', array_map(static fn (string $a): string => "\${$a}", $arrays)) . '], ' . ($hidden ? 'null, false' : ($keep ? '$axis, $keepdims' : '$axis, false')) . ', [' . implode(', ', $named) . '])';
            $methods[] = "    /**\n     * {$doc}\n     *\n     * {$ref}\n     */\n    public static function {$method}(" . implode(', ', $params) . "): {$ret}\n    {\n        return {$call};\n    }\n";
        } elseif ($info['kind'] === 'routine' && ($info['variadic'] ?? -1) >= 0) {
            // numpy's f(*xi, option=...): positional arguments form the sequence, the options are named
            $var = $info['variadic'];
            $opts = [];
            foreach ($info['args'] as $i => $a) {
                if ($i !== $var) {
                    $v = phpVar($a);
                    $opts[] = '$' . $v . (isset($defaults[$v]) ? ' = ' . phpDefault($defaults[$v]) : '');
                }
            }
            $vname = phpVar($info['args'][$var]);
            $optDoc = $opts === [] ? '' : "\n     *\n     * Named options: " . implode(', ', $opts) . '.';
            $methods[] = "    /**\n     * {$doc}\n     *\n     * {$ref}: the positional arguments are numpy's *{$vname}.{$optDoc}\n     */\n    public static function {$method}(mixed ...\${$vname}): mixed\n    {\n        return Registry::routineVariadic(" . var_export($name, true) . ", \${$vname});\n    }\n";
        } elseif ($info['kind'] === 'routine') {
            $args = array_map('phpVar', $info['args']);
            $params = array_map(static fn (string $a): string => "mixed \${$a}" . (isset($defaults[$a]) ? ' = ' . phpDefault($defaults[$a]) : ''), $args);
            $nout = count($info['outs']);
            $call = 'Registry::routine(' . var_export($name, true) . ', [' . implode(', ', array_map(static fn (string $a): string => "\${$a}", $args)) . '])';
            $retDoc = $nout > 1 ? "\n     *\n     * @return mixed one array, or an array keyed by " . implode(', ', $info['outs']) . ' when several results are requested' : '';
            $methods[] = "    /**\n     * {$doc}\n     *\n     * {$ref}{$retDoc}\n     */\n    public static function {$method}(" . implode(', ', $params) . "): mixed\n    {\n        return {$call};\n    }\n";
        } elseif ($info['kind'] === 'random') {
            $args = array_map('phpVar', $info['args']);
            $params = array_map(static fn (string $a): string => "mixed \${$a}" . (isset($defaults[$a]) ? ' = ' . phpDefault($defaults[$a]) : ''), $args);
            $params[] = 'int|array|null $size = null';
            $named = [];
            foreach ($info['params'] as $p) {
                $v = phpVar($p['name']);
                $params[] = "string \${$v} = " . var_export($p['enum'][0], true);
                $named[] = var_export($p['name'], true) . " => \${$v}";
            }
            $ret = $info['out_int'] ? 'NDArray|int' : 'NDArray|float';
            $call = 'Registry::random(' . var_export($name, true) . ', $this->state, [' . implode(', ', array_map(static fn (string $a): string => "\${$a}", $args)) . '], $size, [' . implode(', ', $named) . '])';
            $methods[] = "    /**\n     * {$doc}\n     *\n     * {$ref}\n     */\n    public function {$method}(" . implode(', ', $params) . "): {$ret}\n    {\n        return {$call};\n    }\n";
        } elseif ($info['kind'] === 'dist') {
            $shapes = array_map('phpVar', $info['shapes']);
            $params = array_map(static fn (string $a): string => "mixed \${$a}", $shapes);
            $params[] = 'mixed $loc = 0';
            if (! $info['discrete']) {
                $params[] = 'mixed $scale = 1';
            }
            $argList = array_map(static fn (string $a): string => "\${$a}", $shapes);
            $argList[] = '$loc';
            if (! $info['discrete']) {
                $argList[] = '$scale';
            }
            $methods[] = "    /**\n     * {$doc}\n     *\n     * {$ref} (frozen)\n     */\n    public static function {$method}(" . implode(', ', $params) . "): Distribution\n    {\n        return new Distribution(" . var_export($name, true) . ', [' . implode(', ', $argList) . "]);\n    }\n";
        }
    }
    if ($mod === 'stats') {
        // poisson_binom's shape parameter is a vector, so it is not a broadcast registry distribution; it is a
        // hand-written facade over the kernel's tsr_poisson_binom entry points (Tessero\Stats\PoissonBinom).
        $table[] = "    'poisson_binom' => 'poissonBinom',";
        $methods[] = "    /**\n     * The Poisson binomial distribution: the number of successes in independent Bernoulli trials whose\n     * success probabilities are the vector \$p.\n     *\n     * scipy.stats.poisson_binom (frozen)\n     */\n    public static function poissonBinom(mixed \$p, mixed \$loc = 0): PoissonBinom\n    {\n        return new PoissonBinom(\$p, (float) \$loc);\n    }\n";
        $methods[] = "    /**\n     * Probability mass of the Poisson binomial distribution with probabilities \$p, evaluated at \$k (a functional\n     * shortcut for poissonBinom(\$p)->pmf(\$k)).\n     *\n     * scipy.stats.poisson_binom\n     */\n    public static function poissonBinomPmf(mixed \$k, mixed \$p, mixed \$loc = 0): NDArray|float\n    {\n        return (new PoissonBinom(\$p, (float) \$loc))->pmf(\$k);\n    }\n";
    }
    $uses = "use Tessero\\Native\\Registry;\n";
    if ($mod === 'stats') {
        $uses .= "use Tessero\\Stats\\Distribution;\nuse Tessero\\Stats\\PoissonBinom;\n";
    }
    if ($mod === 'random') {
        $code = "<?php\n\n// Generated by tools/parity/gen-facades.php from libtessero's function registry. Do not edit.\n\ndeclare(strict_types=1);\n\nnamespace Tessero\\Random;\n\nuse Tessero\\NDArray;\nuse Tessero\\Native\\Registry;\n\n/**\n * {$summary}\n *\n * Array arguments broadcast as in NumPy; size fixes the output shape (null: the broadcast shape, a scalar for\n * scalar arguments). The native extension's Tessero\\Ext\\Random\\Generator has the same methods.\n *\n * @property \\FFI\\CData \$state\n */\ntrait GeneratorMethods\n{\n" . implode("\n", $methods) . "}\n";
    } else {
    $code = "<?php\n\n// Generated by tools/parity/gen-facades.php from libtessero's function registry. Do not edit.\n\ndeclare(strict_types=1);\n\nnamespace Tessero;\n\n{$uses}\n/**\n * {$summary}\n *\n * Names are the SciPy/NumPy names in camelCase (log_ndtr -> logNdtr); apply() also takes the original names.\n * The native extension provides the same class as Tessero\\Ext\\{$class}.\n */\nfinal class {$class}\n{\n    /** SciPy/NumPy name => method name */\n    public const FUNCTIONS = [\n" . implode("\n", $table) . "\n    ];\n\n    private function __construct()\n    {\n    }\n\n    /**\n     * Call a function by its SciPy/NumPy name: {$class}::apply('log_ndtr', \$x).\n     */\n    public static function apply(string \$name, mixed ...\$args): mixed\n    {\n        if (! isset(self::FUNCTIONS[\$name])) {\n            throw new \\ValueError(\"{$class}::apply(): unknown function '{\$name}'\");\n        }\n\n        return self::{self::FUNCTIONS[\$name]}(...\$args);\n    }\n\n" . implode("\n", $methods) . "}\n";
    }
    $path = $outDir !== null ? $outDir . '/' . basename($class) . '.php' : dirname(__DIR__, 2) . "/src/{$class}.php";
    $old = is_file($path) ? file_get_contents($path) : null;
    if ($old !== $code) {
        $stale[] = "src/{$class}.php";
        if (! $check) {
            file_put_contents($path, $code);
        }
    }
    echo "{$class}: " . count($fns) . " functions\n";
}
// the registry's function list, read by tools/parity/report.py (which cannot load the library itself)
$reg = [];
foreach (Registry::all() as $name => $info) {
    $reg[$name] = ['kind' => $info['kind'], 'nin' => $info['nin'] ?? null, 'nout' => $info['nout'] ?? null];
}
ksort($reg);
$regJson = json_encode(['functions' => $reg], JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES) . "\n";
$regPath = __DIR__ . '/registry.json';
if ($outDir === null && (! is_file($regPath) || file_get_contents($regPath) !== $regJson)) {
    $stale[] = 'tools/parity/registry.json';
    if (! $check) {
        file_put_contents($regPath, $regJson);
    }
}
if ($check && $stale !== []) {
    fwrite(STDERR, 'out of date: ' . implode(' ', $stale) . "\n");
    exit(1);
}
