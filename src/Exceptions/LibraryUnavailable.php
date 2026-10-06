<?php

declare(strict_types=1);

namespace Tessero\Exceptions;

/** libtessero (or BLAS/LAPACK for the requested routine) could not be loaded in this SAPI. */
class LibraryUnavailable extends TesseroException
{
}
