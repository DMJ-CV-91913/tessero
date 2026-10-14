<?php

declare(strict_types=1);

namespace Tessero\Optimize;

/**
 * scipy.optimize.NoConvergence: raised by the nonlinear (Broyden / Anderson /
 * quasi-Newton) solvers in scipy.optimize._nonlin when the iteration fails to
 * meet the tolerance within the allotted steps. scipy derives it directly from
 * the built-in Exception; this mirrors that so `catch (\Exception)` behaves the
 * same and the attribute surface (message / previous) matches.
 */
class NoConvergence extends \Exception
{
    /** Mirror scipy's Exception-derived signature (message, code, previous). */
    public function __construct(string $message = '', int $code = 0, ?\Throwable $previous = null)
    {
        parent::__construct($message, $code, $previous);
    }
}
