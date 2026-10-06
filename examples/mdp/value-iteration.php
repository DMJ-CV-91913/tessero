<?php

/**
 * Solve a small Markov decision process with value iteration.
 * (No direct SciPy equivalent; this is Tessero's Mdp\MarkovDecisionProcess.)
 *
 * Scenario: two states; two actions, 0 = "rest" (no reward) and 1 = "work" (reward 1). Each
 * action keeps the state. Working always pays, so the optimal policy is to work in every
 * state, and each state is worth 1/(1-gamma) = 10 at gamma = 0.9.
 * Dense form: P[action][state][next] (A, S, S), R[state][action] (S, A). FFI-only, skips under ext.
 */

declare(strict_types=1);

use Tessero\Mdp\MarkovDecisionProcess;

ffi_only('Markov decision processes');

$P = [
    [[1.0, 0.0], [0.0, 1.0]],   // action 0 (rest): stay put
    [[1.0, 0.0], [0.0, 1.0]],   // action 1 (work): stay put
];
$R = [
    [0.0, 1.0],                 // state 0: rest -> 0, work -> 1   (R is indexed [state][action])
    [0.0, 1.0],                 // state 1: rest -> 0, work -> 1
];

$mdp = MarkovDecisionProcess::fromDense($P, $R);
$res = $mdp->valueIteration(0.9);

say('policy = ' . json_encode($res->policy->toList()) . '  values = ' . json_encode($res->values->toList()) . '  on ' . backend());

check($res->converged, 'value iteration converged');
check_close($res->policy, [1, 1], 'optimal policy: work in both states', 0.0, 0.0);
check_close($res->values, [10.0, 10.0], 'each state worth 1/(1-gamma) = 10', 1e-4, 1e-6);
