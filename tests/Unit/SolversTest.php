<?php

declare(strict_types=1);

namespace Tessero\Tests\Unit;

use PHPUnit\Framework\TestCase;
use Tessero\Exceptions\ShapeError;
use Tessero\Exceptions\TesseroException;
use Tessero\Mdp\MarkovDecisionProcess;
use Tessero\NDArray;
use Tessero\Optimize\LinearProgramming;
use Tessero\Random\Generator;
use Tessero\Tessero;

final class SolversTest extends TestCase
{
    protected function tearDown(): void
    {
        Tessero::setThreads(1);
    }

    public function testProductionPlanningWithShadowPrices(): void
    {
        // maximise 40 chairs + 30 tables: 2c + t <= 100 (carpentry), c + t <= 80 (finishing), c <= 40
        $r = LinearProgramming::linprog([-40, -30], [[2, 1], [1, 1]], [100, 80], bounds: [[0, 40], [0, null]]);
        $this->assertTrue($r->success);
        $this->assertEqualsWithDelta(20.0, $r->x[0], 1e-9);
        $this->assertEqualsWithDelta(60.0, $r->x[1], 1e-9);
        $this->assertEqualsWithDelta(-2600.0, $r->fun, 1e-9);
        $this->assertEqualsWithDelta(-10.0, $r->ineqlinMarginals[0], 1e-9, 'one more carpentry hour is worth 10');
        $this->assertEqualsWithDelta(-20.0, $r->ineqlinMarginals[1], 1e-9);
        $this->assertEqualsWithDelta(0.0, $r->slack([[2, 1], [1, 1]], [100, 80])[0], 1e-9);
    }

    public function testSingleBoundPairAppliesToAll(): void
    {
        $r = LinearProgramming::linprog([1, 1], A_eq: [[1, 1]], b_eq: [3], bounds: [1, 2]);
        $this->assertEqualsWithDelta(3.0, $r->fun, 1e-12);
    }

    public function testShapeErrors(): void
    {
        $this->expectException(ShapeError::class);
        LinearProgramming::linprog([1, 2], [[1, 2, 3]], [1]);
    }

    public function testMilpFacilityChoice(): void
    {
        // open facilities (binary y) to cover demand 10 with capacities 6, 5, 8 at costs 10, 7, 12
        $r = LinearProgramming::milp([10, 7, 12], true, [[-6, -5, -8]], [-10], bounds: [0, 1]);
        $this->assertTrue($r->success);
        // {0,1}: capacity 11 for 17 beats {1,2}: 19 and {0,2}: 22
        $this->assertSame([1.0, 1.0, 0.0], array_map(static fn ($v) => round($v, 9) + 0.0, $r->x));
        $this->assertEqualsWithDelta(17.0, $r->fun, 1e-9);
        $this->assertSame(0.0, $r->gap());
    }

    public function testMdpValidation(): void
    {
        $this->expectException(TesseroException::class);
        MarkovDecisionProcess::fromDense([[[0.5, 0.4], [0.0, 1.0]]], [[0.0], [1.0]]);
    }

    public function testMdpFromTransitionsAndSimulation(): void
    {
        // machine maintenance: states good(0)/worn(1)/broken(2); actions run(0)/repair(1)
        $t = [
            [0, 0, 0, 0.8], [0, 0, 1, 0.2], [1, 0, 1, 0.6], [1, 0, 2, 0.4], [2, 0, 2, 1.0],
            [0, 1, 0, 1.0], [1, 1, 0, 1.0], [2, 1, 0, 1.0],
        ];
        $r = [[0, 0, 10], [1, 0, 6], [2, 0, 0], [0, 1, -4], [1, 1, -4], [2, 1, -15]];
        $mdp = MarkovDecisionProcess::fromTransitions(3, 2, $t, $r);
        $sol = $mdp->policyIteration(0.9);
        $this->assertSame(0, $sol->policy->item(0), 'run while good');
        $this->assertSame(1, $sol->policy->item(2), 'repair when broken');
        // Q-values confirm the policy is greedy with respect to its own values
        $q = $mdp->qValues($sol->values, 0.9);
        $this->assertSame($sol->policy->toList(), $q->argmax(1)->toList());
        $path = $mdp->simulate($sol->policy, 0, 50, Generator::defaultRng(7));
        $this->assertCount(50, $path);
        foreach ($path as $step) {
            $this->assertSame($sol->policy->item($step['state']), $step['action']);
        }
        $vi = $mdp->valueIteration(0.9, 1e-10);
        $this->assertTrue(NDArray::allclose($vi->values, $sol->values, 1e-8, 1e-8));
        $this->assertSame($sol->policy->toList(), $vi->policy->toList());
    }

    public function testThreadsGiveIdenticalResults(): void
    {
        $x = Generator::defaultRng(3)->normal(size: 400_000);
        Tessero::setThreads(1);
        $a = $x->exp()->toBytes();
        Tessero::setThreads(4);
        $this->assertSame(Tessero::info()['openmp'] ? 4 : 1, Tessero::threads());
        $this->assertSame($a, $x->exp()->toBytes());

        // MDP sweeps run in parallel above 4096 state-action pairs: results must not depend on threads
        $S = 1500;
        $A = 4;
        $rng = Generator::defaultRng(11);
        $T = [];
        $R = [];
        for ($s = 0; $s < $S; $s++) {
            for ($act = 0; $act < $A; $act++) {
                $next = $rng->integers(0, $S, 3)->toList();
                foreach ($next as $k => $n) {
                    $T[] = [$s, $act, $n, $k === 0 ? 0.5 : 0.25];
                }
                $R[] = [$s, $act, $rng->random()];
            }
        }
        $mdp = MarkovDecisionProcess::fromTransitions($S, $A, $T, $R);
        $solve = static function () use ($mdp): array {
            $v = $mdp->valueIteration(0.9, 1e-8);
            $p = $mdp->policyIteration(0.9, evalSweeps: 10);

            return [$v->values->toBytes(), $v->policy->toBytes(), $p->values->toBytes(), $v->iterations];
        };
        Tessero::setThreads(1);
        $one = $solve();
        Tessero::setThreads(4);
        $this->assertSame($one, $solve());
    }

    public function testToJsonHandlesNonFiniteAndDtypes(): void
    {
        $this->assertSame('[1.5,null,null]', NDArray::array([1.5, NAN, INF])->toJson());
        $this->assertSame('[[1,2],[3,4]]', NDArray::array([[1, 2], [3, 4]], 'int32')->toJson());
        $this->assertSame('[[1.0,2.0]]', NDArray::complex([1.0], [2.0])->toJson());
        $this->assertSame('[0.1]', NDArray::array([0.1], 'float32')->toJson());
    }

    public function testNonFiniteInputIsRejected(): void
    {
        foreach ([
            fn () => LinearProgramming::linprog([1.0, NAN]),
            fn () => LinearProgramming::linprog([1.0, 1.0], [[1.0, INF]], [1.0]),
            fn () => LinearProgramming::linprog([1.0], [[1.0]], [NAN]),
            fn () => LinearProgramming::milp([1.0, NAN]),
        ] as $i => $call) {
            try {
                $call();
                $this->fail("case {$i} accepted non-finite input");
            } catch (\InvalidArgumentException $e) {
                $this->assertStringContainsString('finite', $e->getMessage());
            }
        }
    }

    public function testMdpRejectsNonFiniteModels(): void
    {
        $P = [[[0.9, 0.1], [0.0, 1.0]], [[1.0, 0.0], [1.0, 0.0]]];
        foreach ([
            [[[[NAN, 1.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 0.0]]], [[1, 2], [3, 4]]],
            [$P, [[1, NAN], [3, 4]]],
            [$P, [[1, INF], [3, 4]]],
        ] as $i => [$p, $r]) {
            try {
                MarkovDecisionProcess::fromDense($p, $r);
                $this->fail("case {$i} accepted a non-finite model");
            } catch (TesseroException $e) {
                $this->assertStringContainsString('finite', $e->getMessage());
            }
        }
    }
}
