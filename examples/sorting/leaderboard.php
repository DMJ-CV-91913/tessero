<?php

/**
 * Sort, rank, and binary-search a 1-D array. Both backends.
 * NumPy: np.sort(a) == a.sort-copy; a.argsort(); np.searchsorted(sorted, v)
 * Tessero: NDArray::sort()/argsort() are instance methods; Np::searchsorted is static.
 *
 * Scenario: a leaderboard. Rank the scores (argsort), sort them, and find where a new
 * score of 80 would slot into the sorted list.
 */

declare(strict_types=1);

$scores = ND::array([42.0, 17.0, 99.0, 73.0, 5.0, 88.0]);

$order = $scores->argsort();                    // indices that sort the input: [4, 1, 0, 3, 5, 2]
check_close($order->toList(), [4, 1, 0, 3, 5, 2], 'argsort indices', 0.0, 0.0);

$sorted = $scores->sort();                      // ascending
check_close($sorted->toList(), [5.0, 17.0, 42.0, 73.0, 88.0, 99.0], 'sorted ascending', 0.0, 0.0);

$pos = NP::searchsorted($sorted, 80.0);         // insertion point to keep it sorted
check((int) $pos === 4, 'searchsorted(80) lands between 73 and 88 (index 4)');

say('leaderboard sorted, ranked and searched on ' . backend());
