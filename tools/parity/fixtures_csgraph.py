#!/usr/bin/env python3
"""scipy.sparse.csgraph.connected_components fixtures (weak connectivity) over a dense adjacency matrix. Pure
integer graph traversal, host-independent, so this generator does not require the pinned environment."""
import json, os
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components as cc
import fixtures_np as F  # enc / enc_result

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'tests', 'fixtures', 'parity', 'csgraph.json')
rng = np.random.default_rng(20261001)


def case(A, connection=None):
    kw, pyk = {}, {}
    if connection is not None:
        kw['connection'] = F.enc(connection); pyk['connection'] = connection
    return {'args': [F.enc(A)], 'kwargs': kw,
            'expect': F.enc_result(cc(sp.csr_array(A), **pyk), ['n_components', 'labels']), 'compare': 'tol'}


def undirected(n, p):
    A = (rng.uniform(0, 1, (n, n)) < p).astype(float)
    A = np.triu(A, 1)
    return A + A.T


two_plus_iso = np.array([[0., 1., 0., 0., 0.], [1., 0., 0., 0., 0.], [0., 0., 0., 1., 0.], [0., 0., 1., 0., 0.], [0., 0., 0., 0., 0.]])
path5 = np.diag(np.ones(4), 1); path5 = path5 + path5.T
MATS = [two_plus_iso, path5, np.zeros((4, 4)), np.ones((4, 4)) - np.eye(4), undirected(8, 0.25), undirected(10, 0.15), undirected(6, 0.5)]

cases = [case(A) for A in MATS] + [case(two_plus_iso, connection='weak'), case(undirected(7, 0.3), connection='weak')]

# ---- shortest paths (dense adjacency, 0 = no edge) ------------------------------------------------
from scipy.sparse.csgraph import shortest_path, dijkstra, bellman_ford, johnson, floyd_warshall


def dir_wgraph(n, p, lo=1, hi=9):
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and rng.uniform() < p:
                A[i, j] = float(rng.integers(lo, hi))
    return A


def undir_wgraph(n, p, lo=1, hi=9):
    A = dir_wgraph(n, p, lo, hi)
    A = np.triu(A, 1)
    return A + A.T


# explicit small graphs + random ones; some have unreachable nodes (inf in the result)
g_dir = np.array([[0., 2., 0., 4., 0.], [0., 0., 1., 0., 7.], [0., 0., 0., 3., 0.], [0., 0., 0., 0., 1.], [0., 0., 0., 0., 0.]])
g_iso = np.array([[0., 5., 0., 0.], [0., 0., 2., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]])   # node 3 unreachable
g_neg = np.array([[0., 3., 8., 0.], [0., 0., -2., 0.], [0., 0., 0., 2.], [0., 0., 0., 0.]])  # negative edge, no cycle
POS = [g_dir, g_iso, dir_wgraph(6, 0.4), dir_wgraph(8, 0.3), undir_wgraph(6, 0.4), undir_wgraph(7, 0.3)]


def dist_case(fname, A, directed, unweighted):
    pos = {'shortest_path': [F.enc(A), F.enc('auto'), F.enc(directed), F.enc(False), F.enc(unweighted)],
           'floyd_warshall': [F.enc(A), F.enc(directed), F.enc(False), F.enc(unweighted)],
           }.get(fname, [F.enc(A), F.enc(directed), F.enc(None), F.enc(False), F.enc(unweighted)])
    fn = {'shortest_path': shortest_path, 'dijkstra': dijkstra, 'bellman_ford': bellman_ford,
          'johnson': johnson, 'floyd_warshall': floyd_warshall}[fname]
    r = fn(A, directed=directed, unweighted=unweighted)
    return {'args': pos, 'kwargs': {}, 'expect': F.enc_result(r, []), 'compare': 'tol'}


calls = [{'fn': 'connected_components', 'cases': cases}]
for fname in ('shortest_path', 'dijkstra', 'floyd_warshall', 'johnson'):
    cs = []
    for A in POS:
        cs.append(dist_case(fname, A, True, False))
        cs.append(dist_case(fname, A, False, False))
        cs.append(dist_case(fname, A, True, True))        # unweighted
    calls.append({'fn': fname, 'cases': cs})
# bellman_ford also on the negative-weight graph
bf = []
for A in POS + [g_neg]:
    bf.append(dist_case('bellman_ford', A, True, False))
    bf.append(dist_case('bellman_ford', A, False, False) if not np.array_equal(A, g_neg) else dist_case('bellman_ford', A, True, True))
calls.append({'fn': 'bellman_ford', 'cases': bf})
calls.append({'fn': 'floyd_warshall', 'cases': [dist_case('floyd_warshall', g_neg, True, False)]})

# ---- traversal + structure -----------------------------------------------------------------------
from scipy.sparse.csgraph import laplacian, breadth_first_order, depth_first_order, structural_rank

GRAPHS = [dir_wgraph(6, 0.4), dir_wgraph(8, 0.3), undirected(6, 0.5), path5, two_plus_iso, dir_wgraph(5, 0.5)]
lap_c, bfo_c, dfo_c, sr_c = [], [], [], []
for A in GRAPHS:
    lap_c.append({'args': [F.enc(A)], 'kwargs': {}, 'expect': F.enc_result(np.asarray(laplacian(A), float), []), 'compare': 'tol'})
    lap_c.append({'args': [F.enc(A), F.enc(True)], 'kwargs': {}, 'expect': F.enc_result(np.asarray(laplacian(A, normed=True), float), []), 'compare': 'tol'})
    for directed in (True, False):
        no, pr = breadth_first_order(sp.csr_array(A), 0, directed=directed, return_predecessors=True)
        bfo_c.append({'args': [F.enc(A), F.enc(0), F.enc(bool(directed))], 'kwargs': {},
                      'expect': F.enc_result((np.asarray(no, dtype=np.int64), np.asarray(pr, dtype=np.int64)), ['node_array', 'predecessors']), 'compare': 'tol'})
        no, pr = depth_first_order(sp.csr_array(A), 0, directed=directed, return_predecessors=True)
        dfo_c.append({'args': [F.enc(A), F.enc(0), F.enc(bool(directed))], 'kwargs': {},
                      'expect': F.enc_result((np.asarray(no, dtype=np.int64), np.asarray(pr, dtype=np.int64)), ['node_array', 'predecessors']), 'compare': 'tol'})
    sr_c.append({'args': [F.enc(A)], 'kwargs': {}, 'expect': F.enc_result(int(structural_rank(sp.csr_array(A))), []), 'compare': 'tol'})
calls += [{'fn': 'laplacian', 'cases': lap_c}, {'fn': 'breadth_first_order', 'cases': bfo_c},
          {'fn': 'depth_first_order', 'cases': dfo_c}, {'fn': 'structural_rank', 'cases': sr_c}]

# minimum_spanning_tree: undirected graphs with DISTINCT weights (unique MST, order-independent)
from scipy.sparse.csgraph import minimum_spanning_tree as _mst
mst_c = []
for n in (6, 7, 8):
    A = np.zeros((n, n)); w = 1.0
    perm = rng.permutation(n * n)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            if rng.uniform() < 0.55:
                val = float(perm[k] + 1); k += 1          # distinct positive weights
                A[i, j] = val; A[j, i] = val
    mst_c.append({'args': [F.enc(A)], 'kwargs': {}, 'expect': F.enc_result(np.asarray(_mst(sp.csr_array(A)).toarray(), float), []), 'compare': 'tol'})
calls.append({'fn': 'minimum_spanning_tree', 'cases': mst_c})

# reconstruct_path + breadth/depth_first_tree (dense tree, [parent][child]=weight).
# Pairings avoid the asymmetric-graph-traversed-undirected case (where the stored edge weight is ambiguous):
# directed graphs use directed=True; symmetric graphs use directed=False.
from scipy.sparse.csgraph import reconstruct_path, breadth_first_tree, depth_first_tree, shortest_path
rp_c, bft_c, dft_c = [], [], []
def tw(A, i, j): return float(rng.integers(1, 20))
def sym_wgraph(n, p):
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.uniform() < p:
                w = float(rng.integers(1, 20)); A[i, j] = w; A[j, i] = w
    return A
for A, directed in [(dir_wgraph(6, 0.4), True), (dir_wgraph(7, 0.35), True), (sym_wgraph(6, 0.5), False), (sym_wgraph(7, 0.4), False)]:
    D, P = shortest_path(sp.csr_array(A), directed=directed, return_predecessors=True)
    rp_c.append({'args': [F.enc(A), F.enc(np.asarray(P[0], dtype=np.int64)), F.enc(bool(directed))], 'kwargs': {},
                 'expect': F.enc_result(np.asarray(reconstruct_path(sp.csr_array(A), P[0], directed=directed).toarray(), float), []), 'compare': 'tol'})
    bft_c.append({'args': [F.enc(A), F.enc(0), F.enc(bool(directed))], 'kwargs': {},
                  'expect': F.enc_result(np.asarray(breadth_first_tree(sp.csr_array(A), 0, directed=directed).toarray(), float), []), 'compare': 'tol'})
    dft_c.append({'args': [F.enc(A), F.enc(0), F.enc(bool(directed))], 'kwargs': {},
                  'expect': F.enc_result(np.asarray(depth_first_tree(sp.csr_array(A), 0, directed=directed).toarray(), float), []), 'compare': 'tol'})
calls += [{'fn': 'reconstruct_path', 'cases': rp_c}, {'fn': 'breadth_first_tree', 'cases': bft_c},
          {'fn': 'depth_first_tree', 'cases': dft_c}]

# construct_dist_matrix: distance matrix rebuilt from the predecessor tree
from scipy.sparse.csgraph import construct_dist_matrix
cdm_c = []
for A, directed in [(dir_wgraph(6, 0.4), True), (dir_wgraph(7, 0.35), True), (sym_wgraph(6, 0.5), False)]:
    D, P = shortest_path(sp.csr_array(A), directed=directed, return_predecessors=True)
    cdm_c.append({'args': [F.enc(A), F.enc(np.asarray(P, dtype=np.int64)), F.enc(bool(directed))], 'kwargs': {},
                  'expect': F.enc_result(np.asarray(construct_dist_matrix(A, P, directed=directed), float), []), 'compare': 'tol'})
calls.append({'fn': 'construct_dist_matrix', 'cases': cdm_c})

# min_weight_full_bipartite_matching: square, guaranteed full matching, distinct finite weights (unique optimum)
from scipy.sparse.csgraph import min_weight_full_bipartite_matching as _mwm
mwm_c = []
for n in (4, 5, 6):
    vals = rng.permutation(n * n) + 1.0                   # distinct positive weights
    A = vals.reshape(n, n).astype(float)
    mask = rng.uniform(size=(n, n)) < 0.4
    for i in range(n):
        mask[i, i] = False                                # keep the diagonal (ensures a full matching)
    A[mask] = 0.0
    ri, ci = _mwm(sp.csr_array(A))
    mwm_c.append({'args': [F.enc(A)], 'kwargs': {},
                  'expect': F.enc_result((np.asarray(ri, dtype=np.int64), np.asarray(ci, dtype=np.int64)), ['row_ind', 'col_ind']), 'compare': 'tol'})
calls.append({'fn': 'min_weight_full_bipartite_matching', 'cases': mwm_c})

out = {'module': 'csgraph', 'scipy': __import__('scipy').__version__, 'env': F.fixture_env.env(), 'calls': calls}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'csgraph: connected_components {len(cases)}, + shortest-path families ({sum(len(c["cases"]) for c in calls) - len(cases)} cases)')
