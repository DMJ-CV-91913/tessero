<?php

/**
 * Reductions along axes: per-column and per-row statistics of a small dataset. Both backends.
 * NumPy: a.sum(); a.mean(axis=0); a.std(axis=0, ddof=1); a.max(axis=1)
 *
 * Scenario: four daily observations of three metrics (rows = days, cols = metrics). Column
 * means/standard deviations summarise each metric; the row max is the day's peak metric.
 */

declare(strict_types=1);

$data = ND::array([
    [10.0, 20.0, 30.0],
    [12.0, 18.0, 33.0],
    [11.0, 22.0, 29.0],
    [13.0, 20.0, 28.0],
]);

check_close($data->sum(), 246.0, 'grand total', 1e-12, 1e-12);        // sum of all 12 values

$colMean = $data->mean(0);                                           // mean of each column (axis 0)
check_close($colMean, [11.5, 20.0, 30.0], 'column means', 1e-12, 1e-12);

$colStd = $data->std(0, 1);                                          // sample std per column (ddof = 1)
check_close($colStd, [1.2909944487, 1.6329931619, 2.1602468995], 'column sample std', 1e-9, 1e-12);

$rowMax = $data->max(1);                                             // max of each row (axis 1)
check_close($rowMax, [30.0, 33.0, 29.0, 28.0], 'row maxima', 1e-12, 1e-12);

say('axis reductions ok on ' . backend());
