<?php

declare(strict_types=1);

namespace Tessero\Exceptions;

/** Native allocation failed or the memory budget (Tessero::setMemoryBudget) was exceeded. */
class MemoryError extends TesseroException
{
}
