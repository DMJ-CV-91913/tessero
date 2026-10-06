<?php

declare(strict_types=1);

namespace Tessero\Exceptions;

/** An iterative routine (SVD, eigen solver, Krylov, optimiser) did not converge. */
class ConvergenceError extends LinAlgError
{
}
