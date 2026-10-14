#!/usr/bin/env python3
"""scipy.cluster.hierarchy data-structure fixtures: ClusterNode / to_tree tree structure, DisjointSet union-find
ops, dendrogram(no_plot=True) dict payload (default + custom palette) and set_link_color_palette effect. Built
over the already-landed linkage. Semantic comparators (tree struct, dendrogram dict, DisjointSet op log) are
emitted; run-tests.php replays them against the pure-PHP Tessero\\Cluster\\Hierarchy facades. Generated in the
pinned environment. Numeric arrays compared at rtol 1e-12; integers / structure / colours compared exactly."""
import json, os
import numpy as np
from scipy.cluster.hierarchy import (linkage, to_tree, DisjointSet, dendrogram, set_link_color_palette)
from scipy.spatial.distance import pdist
import fixtures_np as F

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   'tests', 'fixtures', 'parity', 'hierarchy_trees.json')
rng = np.random.default_rng(20261014)

def obs(n, f):
    return rng.normal(size=(n, f)) * rng.uniform(0.7, 3.0, f) + rng.uniform(-2, 2, f)

LINKS = [(8, 3, 'average', 'euclidean'), (10, 2, 'single', 'euclidean'), (12, 4, 'complete', 'cityblock'),
         (9, 3, 'ward', 'euclidean'), (14, 2, 'weighted', 'euclidean'), (11, 3, 'centroid', 'euclidean'),
         (7, 2, 'median', 'euclidean')]

def tree_nodes(root, nodes):
    """id/count/dist per node id + the leaf pre-order; structure compared exactly (dist at rtol)."""
    return {
        'ids': [int(nd.get_id()) for nd in nodes],
        'counts': [int(nd.get_count()) for nd in nodes],
        'dists': [float(nd.dist) for nd in nodes],
        'is_leaf': [bool(nd.is_leaf()) for nd in nodes],
        'left': [None if nd.get_left() is None else int(nd.get_left().get_id()) for nd in nodes],
        'right': [None if nd.get_right() is None else int(nd.get_right().get_id()) for nd in nodes],
        'preorder': [int(x) for x in root.pre_order()],
    }

# ---- to_tree / ClusterNode ----
totree_cases = []
for (n, f, meth, metric) in LINKS:
    X = obs(n, f); Z = linkage(pdist(X, metric=metric), method=meth)
    root, nodes = to_tree(Z, rd=True)
    totree_cases.append({'Z': F.enc(np.asarray(Z, float)), 'expect': tree_nodes(root, nodes)})

# ---- dendrogram (default + custom palette) ----
CUSTOM = ['m', 'g', 'y', 'k', 'b']
def dend_expect(d):
    return {
        'icoord': [[float(x) for x in row] for row in d['icoord']],
        'dcoord': [[float(x) for x in row] for row in d['dcoord']],
        'ivl': [str(x) for x in d['ivl']],
        'leaves': [int(x) for x in d['leaves']],
        'color_list': [str(x) for x in d['color_list']],
        'leaves_color_list': [None if x is None else str(x) for x in d['leaves_color_list']],
    }

dendro_cases = []
for (n, f, meth, metric) in LINKS:
    X = obs(n, f); Z = linkage(pdist(X, metric=metric), method=meth)
    set_link_color_palette(None)
    d0 = dendrogram(Z, no_plot=True)
    dendro_cases.append({'Z': F.enc(np.asarray(Z, float)), 'palette': None, 'expect': dend_expect(d0)})
    set_link_color_palette(list(CUSTOM))
    d1 = dendrogram(Z, no_plot=True)
    dendro_cases.append({'Z': F.enc(np.asarray(Z, float)), 'palette': list(CUSTOM), 'expect': dend_expect(d1)})
    set_link_color_palette(None)

# ---- DisjointSet op log ----
# The PHP replays the SAME op sequence on its DisjointSet and must match every recorded result.
disjoint_cases = []
for seed in range(6):
    r = np.random.default_rng(100 + seed)
    nel = int(r.integers(5, 12))
    els = list(range(nel))
    ds = DisjointSet(els)
    ops = []       # each op: {op, args, ret}
    merges = [(int(r.integers(0, nel)), int(r.integers(0, nel))) for _ in range(nel + 3)]
    for (a, b) in merges:
        ops.append({'op': 'merge', 'args': [a, b], 'ret': bool(ds.merge(a, b))})
    # queries after all merges
    for e in els:
        ops.append({'op': 'getitem', 'args': [e], 'ret': int(ds[e])})
        ops.append({'op': 'subset_size', 'args': [e], 'ret': int(ds.subset_size(e))})
        ops.append({'op': 'contains', 'args': [e], 'ret': bool(e in ds)})
        ops.append({'op': 'subset', 'args': [e], 'ret': sorted(int(x) for x in ds.subset(e))})
    for (a, b) in merges:
        ops.append({'op': 'connected', 'args': [a, b], 'ret': bool(ds.connected(a, b))})
    disjoint_cases.append({
        'elements': els,
        'ops': ops,
        'len': int(len(ds)),
        'n_subsets': int(ds.n_subsets),
        'iter': [int(x) for x in ds],
        'subsets': [sorted(int(x) for x in s) for s in ds.subsets()],
    })
# a couple of not-found KeyError cases (optional, deterministic): query an absent element
disjoint_err = []
ds2 = DisjointSet([0, 1, 2])
disjoint_err.append({'elements': [0, 1, 2], 'op': 'getitem', 'args': [99]})

out = {
    'module': 'hierarchy_trees',
    'scipy': __import__('scipy').__version__,
    'env': F.fixture_env.env(),
    'totree_cases': totree_cases,
    'dendro_cases': dendro_cases,
    'disjoint_cases': disjoint_cases,
    'disjoint_err_cases': disjoint_err,
}
with open(OUT, 'w') as fh:
    json.dump(out, fh, allow_nan=False, separators=(',', ':'))
print('hierarchy_trees: to_tree %d, dendrogram %d, disjointset %d (+%d err)'
      % (len(totree_cases), len(dendro_cases), len(disjoint_cases), len(disjoint_err)))
