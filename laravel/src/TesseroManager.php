<?php

declare(strict_types=1);

namespace Tessero\Laravel;

use RuntimeException;
use Tessero\Ext\Engine;
use Tessero\Ext\NDArray as ExtArray;
use Tessero\Mdp\MarkovDecisionProcess;
use Tessero\NDArray as FfiArray;
use Tessero\Native\Library;
use Tessero\Optimize\LinearProgramming;
use Tessero\Random\Generator;
use Tessero\Tessero as FfiTessero;

/**
 * One entry point for applications, whichever backend is installed.
 *
 * Arrays come back as Tessero\Ext\NDArray (native extension) or
 * Tessero\NDArray (FFI); both have the same core API (shape, dtype, slicing,
 * arithmetic, reductions, toArray, toJson, serialize). Solver results are
 * normalised to plain arrays so controllers and jobs do not care which
 * backend produced them.
 */
final class TesseroManager
{
    private ?string $backend = null;

    /** @param array<string, mixed> $config */
    public function __construct(private readonly array $config = [])
    {
    }

    /** 'ext' or 'ffi' */
    public function backend(): string
    {
        if ($this->backend !== null) {
            return $this->backend;
        }
        $want = strtolower((string) ($this->config['backend'] ?? 'auto'));
        $ext = extension_loaded('tessero') && class_exists(ExtArray::class);
        $ffi = class_exists(FfiArray::class) && Library::available();

        return $this->backend = match ($want) {
            'ext' => $ext ? 'ext' : throw new RuntimeException('TESSERO_BACKEND=ext but the tessero extension is not loaded (pie install tessero/tessero-ext).'),
            'ffi' => $ffi ? 'ffi' : throw new RuntimeException('TESSERO_BACKEND=ffi but libtessero could not be loaded (run vendor/bin/tessero doctor).'),
            default => $ext ? 'ext' : ($ffi ? 'ffi' : throw new RuntimeException('No Tessero backend: install ext-tessero or enable FFI (vendor/bin/tessero doctor).')),
        };
    }

    /** Push threads / budget / tolerance into the backend (called at boot and per Octane request). */
    public function applyRuntimeSettings(): void
    {
        $threads = max(1, (int) ($this->config['threads'] ?? 1));
        $budget = max(0, (int) ($this->config['memory_budget'] ?? 0));
        if (extension_loaded('tessero')) {
            Engine::setMaxThreads($threads);
            Engine::setMemoryBudget($budget);
            Engine::setEpsilon((float) ($this->config['epsilon'] ?? 1e-6));
        }
        if (class_exists(FfiArray::class) && Library::available()) {
            FfiTessero::setThreads($threads);
            FfiTessero::setMemoryBudget($budget);
            FfiTessero::setBlasThreads(max(1, (int) ($this->config['blas_threads'] ?? 1)));
        }
    }

    public function array(mixed $data, ?string $dtype = null): ExtArray|FfiArray
    {
        return $this->backend() === 'ext' ? ExtArray::array($data, $dtype) : FfiArray::array($data, $dtype);
    }

    /** @param list<int>|int $shape */
    public function zeros(array|int $shape, string $dtype = 'float64'): ExtArray|FfiArray
    {
        return $this->backend() === 'ext' ? ExtArray::zeros($shape, $dtype) : FfiArray::zeros($shape, $dtype);
    }

    /** @param list<int> $shape */
    public function fromBytes(string $bytes, string $dtype, array $shape): ExtArray|FfiArray
    {
        return $this->backend() === 'ext' ? ExtArray::fromBytes($bytes, $dtype, $shape) : FfiArray::fromBytes($bytes, $dtype, $shape);
    }

    /**
     * scipy.optimize.linprog. Returns x, fun, success, status, message, nit, ineqlin, eqlin (marginals).
     *
     * @return array<string, mixed>
     */
    public function linprog(mixed $c, mixed $A_ub = null, mixed $b_ub = null, mixed $A_eq = null, mixed $b_eq = null, ?array $bounds = null): array
    {
        if ($this->backend() === 'ext') {
            return Engine::linprog($c, $A_ub, $b_ub, $A_eq, $b_eq, $bounds);
        }
        $r = LinearProgramming::linprog($c, $A_ub, $b_ub, $A_eq, $b_eq, $bounds);

        return [
            'x' => $r->x, 'fun' => $r->fun, 'success' => $r->success, 'status' => $r->status, 'message' => $r->message,
            'nit' => $r->nit, 'ineqlin' => $r->ineqlinMarginals, 'eqlin' => $r->eqlinMarginals, 'reduced_costs' => $r->reducedCosts,
        ];
    }

    /**
     * scipy.optimize.milp. Returns x, fun, success, status, message, nodes, best_bound.
     *
     * @return array<string, mixed>
     */
    public function milp(mixed $c, bool|array $integrality = true, mixed $A_ub = null, mixed $b_ub = null, mixed $A_eq = null, mixed $b_eq = null, ?array $bounds = null): array
    {
        if ($this->backend() === 'ext') {
            return Engine::milp($c, $integrality, $A_ub, $b_ub, $A_eq, $b_eq, $bounds);
        }
        $r = LinearProgramming::milp($c, $integrality, $A_ub, $b_ub, $A_eq, $b_eq, $bounds);

        return ['x' => $r->x, 'fun' => $r->fun, 'success' => $r->success, 'status' => $r->status, 'message' => $r->message,
            'nodes' => $r->nodes, 'best_bound' => $r->bestBound];
    }

    /**
     * Solve a discounted MDP. $P: (actions, states, states), $R: (states, actions).
     * $method: 'policy_iteration' (exact) or 'value_iteration'.
     *
     * @return array{values: list<float>, policy: list<int>, iterations: int, converged: bool}
     */
    public function solveMdp(mixed $P, mixed $R, float $gamma, string $method = 'policy_iteration'): array
    {
        if ($this->backend() === 'ext') {
            $r = $method === 'value_iteration'
                ? Engine::mdpValueIteration($P, $R, $gamma, (float) ($this->config['epsilon'] ?? 1e-6))
                : Engine::mdpPolicyIteration($P, $R, $gamma);

            return ['values' => $r['values']->toList(), 'policy' => $r['policy']->toList(), 'iterations' => $r['iterations'], 'converged' => $r['converged']];
        }
        $mdp = MarkovDecisionProcess::fromDense($P, $R);
        $r = $method === 'value_iteration' ? $mdp->valueIteration($gamma, (float) ($this->config['epsilon'] ?? 1e-6)) : $mdp->policyIteration($gamma);

        return ['values' => $r->values->toList(), 'policy' => $r->policy->toList(), 'iterations' => $r->iterations, 'converged' => $r->converged];
    }

    /**
     * NumPy-identical random numbers: numpy.random.default_rng($seed).normal(...).
     *
     * @param list<int>|int $shape
     */
    public function normal(int $seed, array|int $shape, float $mean = 0.0, float $sd = 1.0): ExtArray|FfiArray
    {
        return $this->backend() === 'ext' ? Engine::normal($seed, $shape, $mean, $sd) : Generator::defaultRng($seed)->normal($mean, $sd, $shape);
    }

    /** @param list<int>|int $shape */
    public function random(int $seed, array|int $shape): ExtArray|FfiArray
    {
        return $this->backend() === 'ext' ? Engine::random($seed, $shape) : Generator::defaultRng($seed)->random($shape);
    }

    /**
     * A memory-mapped array: the file's pages are the array's memory, so files
     * larger than RAM can be processed. Modes as numpy.memmap: 'r', 'r+', 'w+',
     * 'c'. Mapped bytes are not counted against memory_budget; info() reports
     * them as memory_mapped. Both backends share libtessero's mapping code.
     *
     * @param list<int>|null $shape null = 1-D, inferred from the file size
     */
    public function memmap(string $filename, string $mode = 'r+', ?array $shape = null, string $dtype = 'float64', int $offset = 0): ExtArray|FfiArray
    {
        return $this->backend() === 'ext'
            ? ExtArray::memmap($filename, $mode, $shape, $dtype, $offset)
            : FfiArray::memmap($filename, $mode, $shape, $dtype, $offset);
    }

    /** Read a .npy file (numpy.load); with $mmapMode ('r', 'r+', 'c') it is memory-mapped instead. */
    public function load(string $path, ?string $mmapMode = null): ExtArray|FfiArray
    {
        return $this->backend() === 'ext' ? ExtArray::load($path, $mmapMode) : FfiArray::load($path, $mmapMode);
    }

    /**
     * A memory-mapped .npy file (numpy.lib.format.open_memmap); 'w+' creates it.
     *
     * @param list<int>|null $shape
     */
    public function openMemmap(string $path, string $mode = 'r+', ?array $shape = null, string $dtype = 'float64', bool $fortranOrder = false): ExtArray|FfiArray
    {
        return $this->backend() === 'ext'
            ? ExtArray::openMemmap($path, $mode, $shape, $dtype, $fortranOrder)
            : FfiArray::openMemmap($path, $mode, $shape, $dtype, $fortranOrder);
    }

    /** @return array<string, mixed> */
    public function info(): array
    {
        $info = ['backend' => $this->backend()];
        if (extension_loaded('tessero')) {
            $info['ext'] = Engine::info();
        }
        if (class_exists(FfiArray::class) && Library::available()) {
            $info['ffi'] = FfiTessero::info();
        }

        return $info;
    }

    /** @return array<string, mixed> */
    public function config(): array
    {
        return $this->config;
    }
}
