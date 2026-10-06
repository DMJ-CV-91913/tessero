<?php

/**
 * Connected components of a graph from its adjacency matrix.
 * SciPy: scipy.sparse.csgraph.connected_components(graph, directed=False)  ->  (n, labels)
 * Tessero returns ['n_components' => int, 'labels' => NDArray].
 *
 * FFI-only today, so this skips under ext.
 */

declare(strict_types=1);

use Tessero\Csgraph;

ffi_only('scipy.sparse.csgraph');

// Undirected graph: 0-1 and 1-2 are linked (one cluster); 3-4 linked (another); no edge between.
$adjacency = [
    [0, 1, 0, 0, 0],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0],
];

$cc = Csgraph::connectedComponents($adjacency, false);

say('components = ' . $cc['n_components'] . '  labels = ' . json_encode($cc['labels']->toList()) . '  on ' . backend());

check($cc['n_components'] === 2, 'two connected components');
check_close($cc['labels'], [0, 0, 0, 1, 1], 'nodes 0-2 in one cluster, 3-4 in the other', 0.0, 0.0);
