# Markov decision processes

`Tessero\Mdp\MarkovDecisionProcess` solves finite MDPs: a set of states, a
set of actions, transition probabilities `P(s' | s, a)` and expected
immediate rewards `R(s, a)`. The result is the policy (the best action in
each state) that maximises the expected discounted reward, together with each
state's value. Every solver loop runs in C.

Typical uses are maintenance and replacement decisions, inventory and
ordering, asset dispatch under uncertain prices, and pricing and admission
control.

## Building a model

Three constructors, all producing the same internal form:

```php
use Tessero\Mdp\MarkovDecisionProcess as MDP;

// 1. Dense tensors: P[action][state][next], R[state][action]
$mdp = MDP::fromDense($P, $R);

// 2. One sparse S×S matrix per action (large models)
$mdp = MDP::fromSparse([$P0, $P1], $R);           // list<CsrMatrix>

// 3. Triples, e.g. straight from a database query
$mdp = MDP::fromTransitions(
    states: 4, actions: 4,
    transitions: [[0, 1, 1, 0.7], [0, 1, 0, 0.3], /* state, action, next, probability */],
    rewards:     [[0, 1, -2.0],                   /* state, action, expected reward */],
);
```

`R` may also be `(S,)` (a reward per state, the same for every action) or
`(A, S, S)` (a reward per transition, reduced to expectations).

The model is validated by default. Every row of `P` must be finite,
non-negative and sum to 1 (within 1e-8). The error names the offending state
and action, for example "Transition probabilities of state 0 under action 0
sum to 0.9, not 1" (`TesseroException`; `Tessero\Ext\Exception` on the
extension). Rewards must be finite, and CSR input must be well formed (`indptr`
starting at 0, non-decreasing, ending at the number of non-zeros; indices in
range). These two checks always run. Pass `validate: false` to skip only the
probability check, for models you have already checked.

Internally, transitions are held as **one CSR matrix with A·S rows** (row
`a·S + s` is the distribution after action `a` in state `s`). Memory is
proportional to the number of non-zero transitions, so a model with a
million states and a few successors per state fits comfortably.

## Solving

```php
$vi = $mdp->valueIteration(gamma: 0.95, epsilon: 1e-6);
$pi = $mdp->policyIteration(gamma: 0.95);                   // exact evaluation
$mpi = $mdp->policyIteration(gamma: 0.95, evalSweeps: 20);  // modified PI

$pi->policy->toList();   // best action per state
$pi->values->toList();   // expected discounted reward from each state
$pi->iterations; $pi->converged; $vi->delta;
json_encode($pi);        // {"values": [...], "policy": [...], ...}
```

| Method | When to use |
|---|---|
| `valueIteration` | Any size. Stops when the sup-norm change is below `ε(1−γ)/(2γ)`, which guarantees an ε-optimal policy. For γ = 1 it stops on the span of the change. |
| `policyIteration()` (exact) | Up to a few thousand states. Each policy is evaluated exactly by LU (O(S³)). Usually converges in 3–10 iterations. |
| `policyIteration(evalSweeps: k)` | Large models. Modified policy iteration, often several times faster than value iteration (20 000 states × 4 actions: 164 ms against 367 ms). |
| `finiteHorizon(T)` | A fixed number of decisions (a season, a contract term). Backward induction. |

Ties go to the lowest action index (as `numpy.argmax`). With OpenMP and
threads > 1 the per-state loops run in parallel, and the results are
identical for any thread count.

## Other tools

```php
$mdp->evaluate($policy, 0.95);          // exact value of a given policy
$mdp->qValues($values, 0.95);           // Q(s, a), shape (S, A)
$fh = $mdp->finiteHorizon(12, gamma: 1.0, terminal: $salvage);
$fh['values'];                          // (T+1, S): row t = value at stage t; row T = terminal values
$fh['policy'];                          // (T, S): row t = action to take at stage t
$mdp->simulate($policy, start: 0, steps: 100, rng: Generator::defaultRng(7));
// [['state' => 0, 'action' => 1, 'reward' => -2.0], ...]
```

## Worked example: inventory

A shop holds 0–3 units. Each period it orders 0–3 units (cost 2 each, up to
capacity), pays 1 per unit held, and sells up to demand (0, 1 or 2 units
with probabilities 0.3, 0.4, 0.3) at price 8.

```php
$cap = 3; $S = $cap + 1; $A = $cap + 1;
$demand = [0 => 0.3, 1 => 0.4, 2 => 0.3];
$T = []; $R = [];
for ($s = 0; $s < $S; $s++) {
    for ($a = 0; $a < $A; $a++) {
        $stock = min($cap, $s + $a);
        $expected = -2.0 * $a - 1.0 * $stock;
        foreach ($demand as $d => $p) {
            $sold = min($stock, $d);
            $expected += $p * 8.0 * $sold;
            $T[] = [$s, $a, $stock - $sold, $p];
        }
        $R[] = [$s, $a, $expected];
    }
}
$mdp = MDP::fromTransitions($S, $A, $T, $R);
$best = $mdp->policyIteration(0.9);
$best->policy->toList();     // units to order in each stock state
```

`tests/docs/examples.php` runs this example and checks that value iteration
and policy iteration agree.

## Extension and Laravel

```php
use Tessero\Ext\Engine;

$r = Engine::mdpPolicyIteration($P, $R, 0.95);          // dense P (A,S,S)
$r = Engine::mdpValueIteration(['indptr' => $ip, 'indices' => $ix, 'data' => $pd], $R, 0.95);  // CSR with A·S rows
$r['policy']; $r['values']; $r['iterations']; $r['converged'];
Engine::mdpFiniteHorizon($P, $R, 12);

Tessero::solveMdp($P, $R, 0.95);                        // Laravel: plain arrays on either backend
```

## Performance

Measured on a 2-core VM: a 20 000-state, 4-action model with 5 successors per
transition takes 367 ms with value iteration (343 sweeps, γ = 0.95) and 164 ms
with modified policy iteration (18 iterations).
