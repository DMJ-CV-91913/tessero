<?php

/**
 * Dump every public method of Tessero's user-facing classes as JSON (used by
 * tools/gap_inventory.py to verify the NumPy/SciPy gap analysis against the
 * code, and by the developer-guide hub page).
 *
 *   php -d ffi.enable=1 [-d extension=tessero] tools/api-dump.php > api.json
 */

declare(strict_types=1);

$root = dirname(__DIR__);
require $root . '/tests/autoload.php';
spl_autoload_register(static function (string $c) use ($root): void {
    if (str_starts_with($c, 'Tessero\\Laravel\\')) {
        $f = $root . '/laravel/src/' . str_replace('\\', '/', substr($c, 16)) . '.php';
        if (is_file($f)) {
            require $f;
        }
    }
});
if (is_file($root . '/laravel/tests/stubs.php')) {
    require_once $root . '/laravel/tests/stubs.php';
}
if (! class_exists('Illuminate\\Support\\ServiceProvider')) {
    eval('namespace Illuminate\\Support; abstract class ServiceProvider { public function __construct(public $app = null) {} }');
}
if (! class_exists('Illuminate\\Support\\Facades\\Facade')) {
    eval('namespace Illuminate\\Support\\Facades; abstract class Facade {}');
}

$classes = ['Tessero\\Cluster\\Hierarchy\\ClusterNode', 'Tessero\\Cluster\\Hierarchy\\DisjointSet', 'Tessero\\Cluster\\Hierarchy\\Trees', 'Tessero\\NDArray', 'Tessero\\Math', 'Tessero\\Tessero', 'Tessero\\Linalg\\Linalg', 'Tessero\\Fft\\Fft',
    'Tessero\\ScipyLinalg', 'Tessero\\Constants', 'Tessero\\Distance', 'Tessero\\Signal', 'Tessero\\SignalWindows', 'Tessero\\Ndimage', 'Tessero\\Csgraph',
    'Tessero\\Interpolate', 'Tessero\\Cluster', 'Tessero\\Polynomial', 'Tessero\\Hierarchy', 'Tessero\\SparseLinalg', 'Tessero\\ScipyFft', 'Tessero\\Random\\Generator', 'Tessero\\Sparse\\CsrMatrix', 'Tessero\\Optimize\\Minimize', 'Tessero\\Optimize\\Root',
    'Tessero\\Optimize\\LeastSquares', 'Tessero\\Optimize\\LinearProgramming', 'Tessero\\Optimize\\Assignment', 'Tessero\\Optimize\\Bounds', 'Tessero\\Optimize\\LinearConstraint', 'Tessero\\Optimize\\NonlinearConstraint', 'Tessero\\Optimize\\RootResults', 'Tessero\\Optimize\\QuadraticAssignment', 'Tessero\\Mdp\\MarkovDecisionProcess',
    'Tessero\\Io\\Npy', 'Tessero\\Io\\MatrixMarket', 'Tessero\\Io\\Wav', 'Tessero\\Io\\FortranFile', 'Tessero\\Io\\MatIO', 'Tessero\\Io\\HarwellBoeing', 'Tessero\\Io\\NetCDF', 'Tessero\\Io\\IdlSave', 'Tessero\\Signal\\ClosestStftDualWindow', 'Tessero\\Interpolate\\UnivariateSpline', 'Tessero\\Io\\SparseNpz', 'Tessero\\Integrate\\Quad', 'Tessero\\Integrate\\Tanhsinh', 'Tessero\\Integrate\\QuadVec', 'Tessero\\Spatial\\KDTree', 'Tessero\\Spatial\\Rectangle', 'Tessero\\Spatial\\Hausdorff', 'Tessero\\Spatial\\ConvexHull', 'Tessero\\Spatial\\Delaunay', 'Tessero\\Interpolate\\Bisplev', 'Tessero\\Interpolate\\InterpolatedUnivariateSpline', 'Tessero\\Interpolate\\LSQUnivariateSpline', 'Tessero\\Interpolate\\RectBivariateSpline', 'Tessero\\Interpolate\\NearestNDInterpolator', 'Tessero\\Interpolate\\MakeSplrep', 'Tessero\\Interpolate\\MakeSplprep', 'Tessero\\Interpolate\\GenerateKnots', 'Tessero\\Interpolate\\ApproximateTaylorPolynomial', 'Tessero\\Interpolate\\Rbf', 'Tessero\\Spatial\\Voronoi', 'Tessero\\Spatial\\Transform\\Rotation', 'Tessero\\Spatial\\Transform\\Slerp', 'Tessero\\Spatial\\Transform\\RigidTransform', 'Tessero\\Spatial\\Transform\\RotationSpline', 'Tessero\\Interpolate\\CloughTocher2DInterpolator', 'Tessero\\Interpolate\\AAA', 'Tessero\\Datetime\\Datetime', 'Tessero\\ArrayOps', 'Tessero\\OrthoPoly', 'Tessero\\SpecialSeq', 'Tessero\\StatsFns', 'Tessero\\ValueIndices', 'Tessero\\Ndimage\\GeometricTransform', 'Tessero\\NdimageGeneric', 'Tessero\\LabeledComprehension', 'Tessero\\VectorizedFilter', 'Tessero\\Ndimage\\GenericFilter1d', 'Tessero\\NdimageFns', 'Tessero\\Stats\\Normal', 'Tessero\\Stats\\Uniform', 'Tessero\\Stats\\OrthoGroup', 'Tessero\\Stats\\UnitaryGroup', 'Tessero\\Stats\\UniformDirection', 'Tessero\\Stats\\Bootstrap', 'Tessero\\Stats\\MonteCarloTest', 'Tessero\\Stats\\Logistic', 'Tessero\\Stats\\Binomial', 'Tessero\\Stats\\Truncated', 'Tessero\\Stats\\OrderStatistic', 'Tessero\\Stats\\Mixture', 'Tessero\\Stats\\Transformed', 'Tessero\\Stats\\MatrixT', 'Tessero\\Stats\\NormalInverseGamma', 'Tessero\\Interpolate\\MakeSmoothingSpline', 'Tessero\\Interpolate\\LinearNDInterpolator', 'Tessero\\Optimize\\DifferentialEvolution', 'Tessero\\Optimize\\DualAnnealing', 'Tessero\\Optimize\\BasinHopping', 'Tessero\\Optimize\\Direct', 'Tessero\\Optimize\\Shgo', 'Tessero\\Optimize\\LineSearch', 'Tessero\\Integrate\\NestedQuad', 'Tessero\\Integrate\\Nsum', 'Tessero\\Optimize\\NonlinSolve', 'Tessero\\Optimize\\BroydenFirst', 'Tessero\\Optimize\\KrylovJacobian', 'Tessero\\Optimize\\InverseJacobian', 'Tessero\\Differentiate\\Derivative', 'Tessero\\Integrate\\SolveIvp', 'Tessero\\Integrate\\SolveBvp', 'Tessero\\Integrate\\Lebedev', 'Tessero\\Integrate\\Cubature', 'Tessero\\PermutationTest', 'Tessero\\StatsBarnard', 'Tessero\\Boschloo', 'Tessero\\LogRank', 'Tessero\\Spatial\\SphericalVoronoi', 'Tessero\\Spatial\\HalfspaceIntersection', 'Tessero\\Integrate\\Ode', 'Tessero\\StatsDunnett', 'Tessero\\Stats\\RvHistogram', 'Tessero\\Signal\\FindPeaksCwt', 'Tessero\\Signal\\PlacePoles', 'Tessero\\Interpolate\\LSQBivariateSpline', 'Tessero\\Signal\\Czt', 'Tessero\\Signal\\ZoomFft', 'Tessero\\Stats\\Covariance', 'Tessero\\Stats\\CensoredData', 'Tessero\\Optimize\\NoConvergence', 'Tessero\\Optimize\\HessianUpdateStrategy', 'Tessero\\Optimize\\Bfgs', 'Tessero\\Optimize\\Sr1', 'Tessero\\Spatial\\QhullError', 'Tessero\\Stats\\BootstrapMethod', 'Tessero\\Stats\\MonteCarloMethod', 'Tessero\\Stats\\PermutationMethod', 'Tessero\\Signal\\Lti', 'Tessero\\Signal\\Dlti', 'Tessero\\Signal\\StateSpace', 'Tessero\\Signal\\TransferFunction', 'Tessero\\Signal\\ZerosPolesGain', 'Tessero\\Signal\\ShortTimeFFT', 'Tessero\\Sparse\\SparseRandom', 'Tessero\\Stats\\GoodnessOfFit', 'Tessero\\Stats\\Power', 'Tessero\\Signal\\SplineFilter', 'Tessero\\Stats\\MakeDistribution', 'Tessero\\Stats\\MakeDistributionFactory', 'Tessero\\Stats\\RandomCorrelation', 'Tessero\\Stats\\RandomTable', 'Tessero\\Integrate\\QmcQuad', 'Tessero\\Linalg\\ClarksonWoodruff', 'Tessero\\Sparse\\Linalg\\SuperLU', 'Tessero\\Sparse\\Linalg\\Factorized', 'Tessero\\Sparse\\Linalg\\LinearOperator', 'Tessero\\Sparse\\Linalg\\LaplacianNd', 'Tessero\\Sparse\\Linalg\\Aslinearoperator', 'Tessero\\Random\\MT19937', 'Tessero\\Stats\\MultiscaleGraphcorr', 'Tessero\\Stats\\Qmc\\Sobol', 'Tessero\\Stats\\SobolIndices', 'Tessero\\Ext\\NDArray', 'Tessero\\Ext\\Math', 'Tessero\\Ext\\Engine', 'Tessero\\Ext\\Random\\Generator',
    'Tessero\\Np', 'Tessero\\Special', 'Tessero\\Stats', 'Tessero\\Stats\\Distribution', 'Tessero\\Ext\\Np', 'Tessero\\Ext\\Special',
    'Tessero\\Ext\\Stats', 'Tessero\\Ext\\Distribution', 'Tessero\\Ext\\Fft\\Fft',
    'Tessero\\Ext\\Linalg', 'Tessero\\Ext\\ScipyLinalg', 'Tessero\\Ext\\Signal', 'Tessero\\Ext\\SignalWindows',
    'Tessero\\Ext\\Distance', 'Tessero\\Ext\\Ndimage', 'Tessero\\Ext\\Csgraph', 'Tessero\\Ext\\Interpolate', 'Tessero\\Ext\\Cluster', 'Tessero\\Ext\\Polynomial', 'Tessero\\Ext\\Hierarchy', 'Tessero\\Ext\\SparseLinalg', 'Tessero\\Ext\\ScipyFft',
    'Tessero\\Laravel\\TesseroManager', 'Tessero\\Laravel\\Rules\\NumericArray'];

function sig(ReflectionMethod $m): string
{
    $p = [];
    foreach ($m->getParameters() as $x) {
        $t = $x->getType();
        $s = ($t ? (string) $t . ' ' : '') . ($x->isVariadic() ? '...' : '') . '$' . $x->getName();
        if ($x->isOptional() && ! $x->isVariadic()) {
            try {
                $v = $x->getDefaultValue();
                $s .= ' = ' . (is_array($v) ? '[]' : ($v === null ? 'null' : (is_object($v) ? '…' : var_export($v, true))));
            } catch (Throwable) {
                $s .= ' = …';
            }
        }
        $p[] = $s;
    }
    $r = $m->getReturnType();

    return $m->getName() . '(' . implode(', ', $p) . ')' . ($r ? ': ' . $r : '');
}

$out = [];
foreach ($classes as $c) {
    if (! class_exists($c)) {
        continue;                                   // e.g. Tessero\Ext\* without the extension
    }
    $r = new ReflectionClass($c);
    $ms = [];
    foreach ($r->getMethods(ReflectionMethod::IS_PUBLIC) as $m) {
        if ($m->getDeclaringClass()->getName() !== $c) {
            continue;
        }
        $n = $m->getName();
        if ((str_starts_with($n, '__') && $n !== '__construct') || str_starts_with($n, 'offset')) {
            continue;
        }
        $d = $m->getDocComment();
        $doc = '';
        if ($d) {
            foreach (preg_split('/\R/', $d) as $l) {
                $l = trim((string) preg_replace('#^\s*/\*\*\s?|\s*\*/\s*$|^\s*\*\s?#', '', $l));
                if ($l === '' || $l[0] === '@') {
                    if ($doc !== '') {
                        break;
                    }
                    continue;
                }
                $doc .= ($doc ? ' ' : '') . $l;
            }
        }
        if ($doc === '' && $c === 'Tessero\\Ext\\Math') {
            $doc = Tessero\Ext\Math::ufuncs()[$n]['summary'] ?? '';
        }
        $ms[] = ['name' => $n, 'static' => $m->isStatic(), 'sig' => sig($m), 'doc' => $doc];
    }
    $entry = ['class' => $c, 'methods' => $ms];
    $consts = array_keys($r->getConstants());
    if ($consts !== []) {
        $entry['constants'] = $consts;
    }
    $out[] = $entry;
}
echo json_encode($out, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES), "\n";
