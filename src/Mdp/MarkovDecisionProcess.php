<?php

declare(strict_types=1);

namespace Tessero\Mdp;

use FFI\CData;
use Tessero\DType;
use Tessero\Exceptions\ShapeError;
use Tessero\Exceptions\TesseroException;
use Tessero\NDArray;
use Tessero\Native\Library;
use Tessero\Random\Generator;
use Tessero\Sparse\CsrMatrix;

/**
 * A finite Markov decision process with discounted rewards, solved natively.
 *
 * Transitions are held as one CSR matrix with A*S rows (row a*S + s is the
 * distribution of the next state after action a in state s), so dense models
 * and sparse models with millions of states use the same solvers and memory
 * is proportional to the non-zeros. Rewards are expected immediate rewards
 * R[s, a].
 *
 *   $mdp = MarkovDecisionProcess::fromDense($P, $R);     // P (A, S, S), R (S, A) or (A, S, S)
 *   $res = $mdp->valueIteration(gamma: 0.95);
 *   $res->policy;  $res->values;
 */
final class MarkovDecisionProcess
{
    private function __construct(
        private readonly int $states,
        private readonly int $actions,
        private readonly NDArray $indptr,
        private readonly NDArray $indices,
        private readonly NDArray $data,
        private readonly NDArray $rewards,  // (S, A) float64, contiguous
    ) {
    }

    /**
     * @param mixed $P transition tensor (A, S, S), rows summing to 1
     * @param mixed $R rewards (S, A), or per-transition rewards (A, S, S) reduced to expectations
     */
    public static function fromDense(mixed $P, mixed $R, bool $validate = true): self
    {
        $Pt = NDArray::asArray($P)->astype(DType::Float64)->contiguous();
        if (count($Pt->shape()) !== 3 || $Pt->shape()[1] !== $Pt->shape()[2]) {
            throw new ShapeError('P must have shape (actions, states, states).');
        }
        [$A, $S] = $Pt->shape();
        $flat = CsrMatrix::fromDense($Pt->reshape([$A * $S, $S]));

        return self::build($S, $A, $flat->indptr(), $flat->indices(), $flat->data(), self::expectedRewards($R, $Pt, $S, $A), $validate);
    }

    /**
     * @param list<CsrMatrix> $perAction one S x S transition matrix per action
     */
    public static function fromSparse(array $perAction, mixed $R, bool $validate = true): self
    {
        $perAction = array_values($perAction);
        $A = count($perAction);
        if ($A === 0) {
            throw new ShapeError('At least one action is required.');
        }
        $S = $perAction[0]->shape()[0];
        $indptr = [0];
        $indices = [];
        $data = [];
        foreach ($perAction as $a => $m) {
            if ($m->shape() !== [$S, $S]) {
                throw new ShapeError("Transition matrix of action {$a} must be {$S}x{$S}.");
            }
            $base = end($indptr);
            foreach (array_slice($m->indptr()->toList(), 1) as $p) {
                $indptr[] = $base + $p;
            }
            array_push($indices, ...$m->indices()->toList());
            array_push($data, ...$m->data()->toList());
        }
        $R3 = NDArray::asArray($R)->astype(DType::Float64);
        if (count($R3->shape()) === 3) {
            throw new ShapeError('With sparse transitions give expected rewards R of shape (states, actions).');
        }

        return self::build(
            $S, $A,
            NDArray::fromFlat($indptr, [count($indptr)], DType::Int64),
            NDArray::fromFlat($indices, [count($indices)], DType::Int64),
            NDArray::fromFlat($data, [count($data)]),
            self::expectedRewards($R3, null, $S, $A),
            $validate,
        );
    }

    /**
     * Build from transition triples, e.g. read from a database:
     *   [[state, action, nextState, probability], ...] and rewards [[state, action, reward], ...].
     *
     * @param iterable<array{0: int, 1: int, 2: int, 3: float}> $transitions
     * @param iterable<array{0: int, 1: int, 2: float}> $rewards
     */
    public static function fromTransitions(int $states, int $actions, iterable $transitions, iterable $rewards, bool $validate = true): self
    {
        $rows = [];
        $cols = [];
        $vals = [];
        foreach ($transitions as [$s, $a, $s2, $p]) {
            if ($s < 0 || $s >= $states || $s2 < 0 || $s2 >= $states || $a < 0 || $a >= $actions) {
                throw new ShapeError("Transition ({$s}, {$a}, {$s2}) is outside {$states} states x {$actions} actions.");
            }
            $rows[] = $a * $states + $s;
            $cols[] = $s2;
            $vals[] = (float) $p;
        }
        $m = CsrMatrix::fromTriplets($rows, $cols, $vals, [$actions * $states, $states]);
        $R = array_fill(0, $states * $actions, 0.0);
        foreach ($rewards as [$s, $a, $r]) {
            $R[$s * $actions + $a] = $R[$s * $actions + $a] + (float) $r;
        }

        return self::build($states, $actions, $m->indptr(), $m->indices(), $m->data(), NDArray::fromFlat($R, [$states, $actions]), $validate);
    }

    public function states(): int
    {
        return $this->states;
    }

    public function actions(): int
    {
        return $this->actions;
    }

    /** Expected immediate rewards (S, A). */
    public function rewards(): NDArray
    {
        return $this->rewards;
    }

    /** Value iteration to an epsilon-optimal policy. */
    public function valueIteration(float $gamma, float $epsilon = 1e-6, int $maxIter = 10_000, mixed $initial = null): MdpResult
    {
        $this->checkGamma($gamma, true);
        $ffi = Library::ffi();
        $V = $initial === null ? NDArray::zeros([$this->states]) : NDArray::asArray($initial)->astype(DType::Float64, true)->contiguous();
        $this->checkVector($V, 'initial values');
        $pol = NDArray::zeros([$this->states], DType::Int64);
        $it = $ffi->new('int64_t');
        $delta = $ffi->new('double');
        $rc = $ffi->tsr_mdp_value_iteration(
            $this->states, $this->actions, $this->i($this->indptr), $this->i($this->indices), $this->d($this->data), $this->d($this->rewards),
            $gamma, $epsilon, $maxIter, $this->d($V), $this->i($pol), \FFI::addr($it), \FFI::addr($delta),
        );
        if ($rc < 0) {
            Library::check($rc, 'value iteration');
        }

        return new MdpResult($V, $pol, $it->cdata, $rc === 0, 'value_iteration', $delta->cdata);
    }

    /**
     * Policy iteration. $evalSweeps = 0 evaluates each policy exactly (dense LU, O(S^3): fine to a few
     * thousand states); k > 0 is modified policy iteration with k evaluation sweeps (large models).
     */
    public function policyIteration(float $gamma, int $evalSweeps = 0, float $epsilon = 1e-6, int $maxIter = 1_000, mixed $initialPolicy = null): MdpResult
    {
        $this->checkGamma($gamma, $evalSweeps > 0);
        $ffi = Library::ffi();
        $pol = $initialPolicy === null ? NDArray::zeros([$this->states], DType::Int64) : NDArray::asArray($initialPolicy)->astype(DType::Int64, true)->contiguous();
        $this->checkVector($pol, 'initial policy');
        $V = NDArray::zeros([$this->states]);
        $it = $ffi->new('int64_t');
        $rc = $ffi->tsr_mdp_policy_iteration(
            $this->states, $this->actions, $this->i($this->indptr), $this->i($this->indices), $this->d($this->data), $this->d($this->rewards),
            $gamma, $evalSweeps, $epsilon, $maxIter, $this->d($V), $this->i($pol), \FFI::addr($it),
        );
        if ($rc < 0) {
            Library::check($rc, 'policy iteration');
        }

        return new MdpResult($V, $pol, $it->cdata, $rc === 0, $evalSweeps > 0 ? 'modified_policy_iteration' : 'policy_iteration');
    }

    /** Exact value of a fixed policy: V = (I - gamma P_pi)^-1 R_pi. */
    public function evaluate(mixed $policy, float $gamma): NDArray
    {
        $this->checkGamma($gamma, false);
        $pol = NDArray::asArray($policy)->astype(DType::Int64)->contiguous();
        $this->checkVector($pol, 'policy');
        $V = NDArray::zeros([$this->states]);
        Library::check(Library::ffi()->tsr_mdp_policy_eval(
            $this->states, $this->actions, $this->i($this->indptr), $this->i($this->indices), $this->d($this->data), $this->d($this->rewards),
            $gamma, $this->i($pol), $this->d($V),
        ), 'policy evaluation');

        return $V;
    }

    /** Action values Q (S, A) for state values V. */
    public function qValues(mixed $V, float $gamma): NDArray
    {
        $Vv = NDArray::asArray($V)->astype(DType::Float64)->contiguous();
        $this->checkVector($Vv, 'values');
        $Q = NDArray::zeros([$this->states, $this->actions]);
        $Vo = NDArray::zeros([$this->states]);
        $pol = NDArray::zeros([$this->states], DType::Int64);
        Library::check(Library::ffi()->tsr_mdp_bellman(
            $this->states, $this->actions, $this->i($this->indptr), $this->i($this->indices), $this->d($this->data), $this->d($this->rewards),
            $gamma, $this->d($Vv), $this->d($Q), $this->d($Vo), $this->i($pol),
        ), 'bellman backup');

        return $Q;
    }

    /**
     * Backward induction over $horizon decisions.
     *
     * @return array{values: NDArray, policy: NDArray} values (T+1, S) by stage, policy (T, S)
     */
    public function finiteHorizon(int $horizon, float $gamma = 1.0, mixed $terminal = null): array
    {
        if ($horizon < 1) {
            throw new TesseroException('horizon must be at least 1.');
        }
        $this->checkGamma($gamma, true);
        $V = NDArray::zeros([$horizon + 1, $this->states]);
        $pol = NDArray::zeros([$horizon, $this->states], DType::Int64);
        $term = null;
        if ($terminal !== null) {
            $term = NDArray::asArray($terminal)->astype(DType::Float64)->contiguous();
            $this->checkVector($term, 'terminal values');
        }
        Library::check(Library::ffi()->tsr_mdp_finite_horizon(
            $this->states, $this->actions, $this->i($this->indptr), $this->i($this->indices), $this->d($this->data), $this->d($this->rewards),
            $gamma, $horizon, $term === null ? null : $this->d($term), $this->d($V), $this->i($pol),
        ), 'finite horizon');

        return ['values' => $V, 'policy' => $pol];
    }

    /**
     * Sample a trajectory following $policy from $start.
     *
     * @return list<array{state: int, action: int, reward: float}>
     */
    public function simulate(mixed $policy, int $start, int $steps, ?Generator $rng = null): array
    {
        $pol = NDArray::asArray($policy)->astype(DType::Int64)->toList();
        $rng ??= Generator::defaultRng();
        $ptr = $this->indptr->toList();
        $idx = $this->indices->toList();
        $val = $this->data->toList();
        $R = $this->rewards->toList();
        $s = $start;
        $out = [];
        $u = $rng->random(max(1, $steps))->toList();
        for ($t = 0; $t < $steps; $t++) {
            $a = $pol[$s];
            $out[] = ['state' => $s, 'action' => $a, 'reward' => $R[$s * $this->actions + $a]];
            $row = $a * $this->states + $s;
            $acc = 0.0;
            $next = $s;
            for ($p = $ptr[$row]; $p < $ptr[$row + 1]; $p++) {
                $acc += $val[$p];
                $next = $idx[$p];
                if ($u[$t] < $acc) {
                    break;
                }
            }
            $s = $next;
        }

        return $out;
    }

    // ------------------------------------------------------------------ internals

    private static function build(int $S, int $A, NDArray $indptr, NDArray $indices, NDArray $data, NDArray $R, bool $validate): self
    {
        $indptr = $indptr->astype(DType::Int64)->contiguous();
        $indices = $indices->astype(DType::Int64)->contiguous();
        $data = $data->astype(DType::Float64)->contiguous();
        if ($indptr->size() !== $A * $S + 1) {
            throw new ShapeError('Transition matrix must have actions * states rows.');
        }
        if ($R->shape() !== [$S, $A]) {
            throw new ShapeError("Rewards must have shape ({$S}, {$A}).");
        }
        if ($indptr->item(0) !== 0 || $indptr->item($A * $S) !== $data->size() || $indices->size() !== $data->size()) {
            throw new ShapeError('CSR transitions: indptr must start at 0 and end at len(data) == len(indices).');
        }
        if (! $R->isfinite()->all()) {
            throw new TesseroException('Rewards must be finite (no NaN or INF).');
        }
        if ($validate && $data->size() > 0) {
            if (! $data->isfinite()->all() || $data->min() < 0) {
                throw new TesseroException('Transition probabilities must be finite and non-negative.');
            }
            // row sums via one CSR product with a ones vector
            $sums = CsrMatrix::fromArrays($indptr, $indices, $data, [$A * $S, $S])->dot(NDArray::ones([$S]));
            $bad = $sums->sub(1.0)->abs()->gt(1e-8);
            if ($bad->any()) {
                $row = (int) $bad->argmax();
                throw new TesseroException(sprintf('Transition probabilities of state %d under action %d sum to %.12g, not 1.', $row % $S, intdiv($row, $S), $sums->item($row)));
            }
        }

        return new self($S, $A, $indptr, $indices, $data, $R->astype(DType::Float64, true)->contiguous());
    }

    private static function expectedRewards(mixed $R, ?NDArray $P, int $S, int $A): NDArray
    {
        $Rv = NDArray::asArray($R)->astype(DType::Float64);
        if (count($Rv->shape()) === 3) {
            if ($P === null || $Rv->shape() !== [$A, $S, $S]) {
                throw new ShapeError('Per-transition rewards must have shape (actions, states, states).');
            }
            // E[r | s, a] = sum_s' P(s'|s,a) r(s,a,s'), laid out (S, A)
            return $P->mul($Rv)->sum(2)->t()->copy();
        }
        if (count($Rv->shape()) === 1 && $Rv->size() === $S) {
            return $Rv->reshape([$S, 1])->broadcastTo([$S, $A])->copy(); // state rewards, same for every action
        }
        if ($Rv->shape() !== [$S, $A]) {
            throw new ShapeError("Rewards must have shape ({$S}, {$A}), ({$S},) or ({$A}, {$S}, {$S}).");
        }

        return $Rv->contiguous();
    }

    private function checkGamma(float $gamma, bool $allowOne): void
    {
        if ($gamma < 0.0 || $gamma > 1.0 || (! $allowOne && $gamma >= 1.0)) {
            throw new TesseroException('gamma must be in [0, 1)' . ($allowOne ? ' (1 allowed here)' : '') . '.');
        }
    }

    private function checkVector(NDArray $v, string $name): void
    {
        if ($v->shape() !== [$this->states]) {
            throw new ShapeError("{$name} must have one entry per state ({$this->states}).");
        }
    }

    private function d(NDArray $a): CData
    {
        return Library::ffi()->cast('double*', $a->ptr());
    }

    private function i(NDArray $a): CData
    {
        return Library::ffi()->cast('int64_t*', $a->ptr());
    }
}
