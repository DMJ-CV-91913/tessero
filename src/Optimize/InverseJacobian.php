<?php

declare(strict_types=1);

namespace Tessero\Optimize;

/**
 * scipy.optimize._nonlin.InverseJacobian: a thin wrapper that exposes a Jacobian
 * approximation ({@see JacobianApprox}) as the *inverse* operator. scipy binds
 *
 *     self.matvec = jacobian.solve
 *     self.update = jacobian.update
 *     self.setup  = jacobian.setup   (when present)
 *     self.rmatvec = jacobian.rsolve (when present)
 *
 * so InverseJacobian(jac).matvec(v) is exactly jac.solve(v), i.e. the action of
 * the approximate inverse Jacobian J^-1 on v. It carries no numerical logic of
 * its own; all math is delegated to the wrapped Jacobian (here {@see BroydenFirst}
 * and friends). Backend-neutral pure PHP, identical on both Tessero backends.
 *
 * @internal
 */
final class InverseJacobian
{
    private JacobianApprox $jacobian;

    public function __construct(JacobianApprox $jacobian)
    {
        $this->jacobian = $jacobian;
    }

    /** The wrapped Jacobian (scipy: InverseJacobian.jacobian). */
    public function jacobian(): JacobianApprox
    {
        return $this->jacobian;
    }

    /**
     * scipy: self.setup = jacobian.setup.
     *
     * @param list<float> $x
     * @param list<float> $F
     * @param callable(list<float>): list<float> $func
     */
    public function setup(array $x, array $F, callable $func): void
    {
        $this->jacobian->setup($x, $F, $func);
    }

    /**
     * scipy: self.matvec = jacobian.solve, i.e. J^-1 . v.
     *
     * @param list<float> $v
     * @return list<float>
     */
    public function matvec(array $v): array
    {
        return $this->jacobian->solve($v);
    }

    /**
     * scipy: self.update = jacobian.update.
     *
     * @param list<float> $x
     * @param list<float> $f
     */
    public function update(array $x, array $f): void
    {
        $this->jacobian->update($x, $f);
    }
}
