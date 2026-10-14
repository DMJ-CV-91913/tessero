<?php

declare(strict_types=1);

namespace Tessero\Optimize;

use Tessero\Exceptions\ConvergenceError;

/**
 * scipy.optimize.NoConvergence: raised when a nonlinear quasi-Newton solver
 * (broyden1/broyden2/anderson/diagbroyden) fails to converge within maxiter.
 *
 * Extends ConvergenceError so existing convergence-aware catch sites still match.
 * The best iterate reached is carried on ->x (scipy stores it as the exception arg).
 */
final class NoConvergence extends ConvergenceError
{
    /** @var list<float> best iterate reached before giving up */
    public array $x;

    /** @param list<float> $x */
    public function __construct(array $x, string $message = 'A solution was not found.')
    {
        $this->x = $x;
        parent::__construct($message);
    }
}
