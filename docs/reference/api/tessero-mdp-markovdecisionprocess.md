# MarkovDecisionProcess

`Tessero\Mdp\MarkovDecisionProcess`

A finite Markov decision process with discounted rewards, solved natively.

Transitions are held as one CSR matrix with A*S rows (row a*S + s is the
distribution of the next state after action a in state s), so dense models
and sparse models with millions of states use the same solvers and memory
is proportional to the non-zeros. Rewards are expected immediate rewards
R[s, a].

  $mdp = MarkovDecisionProcess::fromDense($P, $R);     // P (A, S, S), R (S, A) or (A, S, S)
  $res = $mdp->valueIteration(gamma: 0.95);
  $res->policy;  $res->values;

## Methods

### fromDense

```php
static fromDense(mixed $P, mixed $R, bool $validate = true): self
```

### fromSparse

```php
static fromSparse(array $perAction, mixed $R, bool $validate = true): self
```

### fromTransitions

```php
static fromTransitions(int $states, int $actions, iterable $transitions, iterable $rewards, bool $validate = true): self
```

Build from transition triples, e.g. read from a database:
  [[state, action, nextState, probability], ...] and rewards [[state, action, reward], ...].

### actions

```php
actions(): int
```

### evaluate

```php
evaluate(mixed $policy, float $gamma): Tessero\NDArray
```

Exact value of a fixed policy: V = (I - gamma P_pi)^-1 R_pi.

### finiteHorizon

```php
finiteHorizon(int $horizon, float $gamma = 1.0, mixed $terminal = null): array
```

Backward induction over $horizon decisions.

### policyIteration

```php
policyIteration(float $gamma, int $evalSweeps = 0, float $epsilon = 1.0E-6, int $maxIter = 1000, mixed $initialPolicy = null): Tessero\Mdp\MdpResult
```

Policy iteration. $evalSweeps = 0 evaluates each policy exactly (dense LU, O(S^3): fine to a few
thousand states); k > 0 is modified policy iteration with k evaluation sweeps (large models).

### qValues

```php
qValues(mixed $V, float $gamma): Tessero\NDArray
```

Action values Q (S, A) for state values V.

### rewards

```php
rewards(): Tessero\NDArray
```

Expected immediate rewards (S, A).

### simulate

```php
simulate(mixed $policy, int $start, int $steps, ?Tessero\Random\Generator $rng = null): array
```

Sample a trajectory following $policy from $start.

### states

```php
states(): int
```

### valueIteration

```php
valueIteration(float $gamma, float $epsilon = 1.0E-6, int $maxIter = 10000, mixed $initial = null): Tessero\Mdp\MdpResult
```

Value iteration to an epsilon-optimal policy.
