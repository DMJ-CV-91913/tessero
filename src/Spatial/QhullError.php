<?php

declare(strict_types=1);

namespace Tessero\Spatial;

/**
 * scipy.spatial.QhullError: raised by the Qhull-backed geometry classes
 * (ConvexHull / Delaunay / Voronoi / HalfspaceIntersection) when Qhull reports
 * a failure (e.g. degenerate / coplanar input). scipy derives it from the
 * built-in RuntimeError; this mirrors that (\RuntimeException is PHP's
 * RuntimeError equivalent) so the exception hierarchy and message surface match.
 */
class QhullError extends \RuntimeException
{
    /** Mirror scipy's RuntimeError-derived signature (message, code, previous). */
    public function __construct(string $message = '', int $code = 0, ?\Throwable $previous = null)
    {
        parent::__construct($message, $code, $previous);
    }
}
