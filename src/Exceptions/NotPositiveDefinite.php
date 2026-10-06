<?php

declare(strict_types=1);

namespace Tessero\Exceptions;

/** Cholesky factorisation found a non-positive leading minor. */
class NotPositiveDefinite extends LinAlgError
{
}
