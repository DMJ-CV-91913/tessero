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

# maximum_flow: Dinic max-flow on dense integer capacity matrices. The .flow matrix is NON-UNIQUE,
# so the fixture stores only the (unique) flow_value and the parity harness validates .flow
# semantically (compare='maxflow'); capacity + source/sink are carried in args. Host-independent.
from scipy.sparse.csgraph import maximum_flow as _maxflow
_mf_cases = [
    (np.array([[0, 16, 13, 0, 0, 0], [0, 0, 10, 12, 0, 0], [0, 4, 0, 0, 14, 0],
               [0, 0, 9, 0, 0, 20], [0, 0, 0, 7, 0, 4], [0, 0, 0, 0, 0, 0]], np.int32), 0, 5),
    (np.array([[0, 5, 0, 5, 0], [0, 0, 4, 0, 0], [0, 0, 0, 0, 4],
               [0, 0, 3, 0, 2], [0, 0, 0, 0, 0]], np.int32), 0, 4),
    (np.array([[0, 3, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2], [0, 0, 0, 0]], np.int32), 0, 3),
    (np.array([[0, 0, 0], [0, 0, 5], [0, 0, 0]], np.int32), 0, 2),
    (np.array([[0, 7, 0, 0], [0, 0, 5, 2], [3, 0, 0, 6], [0, 0, 0, 0]], np.int32), 0, 3),
]
_mf_rng = np.random.default_rng(20261012)
for _ in range(8):
    _n = int(_mf_rng.integers(4, 9))
    _A = np.zeros((_n, _n), np.int32)
    for _i in range(_n):
        for _j in range(_n):
            if _i != _j and _mf_rng.uniform() < 0.45:
                _A[_i, _j] = int(_mf_rng.integers(0, 12))
    _mf_cases.append((_A, 0, _n - 1))
mf_c = []
for _A, _s, _t in _mf_cases:
    _r = _maxflow(sp.csr_array(_A), _s, _t)
    mf_c.append({'args': [{'shape': list(_A.shape), 'dtype': 'int64', 'data': [int(x) for x in _A.ravel()]}, _s, _t],
                 'kwargs': {}, 'expect': {'flow_value': int(_r.flow_value)}, 'compare': 'maxflow'})
calls.append({'fn': 'maximum_flow', 'cases': mf_c})


# ---- reverse_cuthill_mckee (RCM reordering; dense symmetric adjacency, 0 = no edge) --------------
# The RCM permutation is NON-UNIQUE under degree ties, so the fixture stores only scipy's permuted
# BANDWIDTH (int) and the parity harness validates the returned permutation semantically
# (compare='rcm_bandwidth'): it must be a valid permutation AND achieve the same bandwidth as scipy.
# Pure integer graph traversal -> host-independent (no pinned env needed). The adjacency is carried
# in args[0]; symmetric_mode=True is passed so the pattern is treated symmetric.
from scipy.sparse.csgraph import reverse_cuthill_mckee as _rcm


def _sym_adj(n, p, seed):
    _r = np.random.default_rng(seed)
    M = (_r.uniform(0, 1, (n, n)) < p).astype(float)
    M = np.triu(M, 1)
    return M + M.T


def _bandwidth(A, perm):
    n = A.shape[0]
    pos = np.empty(n, dtype=int)
    pos[np.asarray(perm, dtype=int)] = np.arange(n)
    bw = 0
    for i in range(n):
        for j in range(n):
            if i != j and (A[i, j] != 0.0 or A[j, i] != 0.0):
                bw = max(bw, abs(int(pos[i]) - int(pos[j])))
    return bw


_rcm_path = np.diag(np.ones(9), 1); _rcm_path = _rcm_path + _rcm_path.T
_rcm_ring = _rcm_path.copy(); _rcm_ring[0, 9] = _rcm_ring[9, 0] = 1.0
_rcm_block = np.zeros((11, 11))
_b1 = _sym_adj(5, 0.5, 6); _rcm_block[:5, :5] = _b1
_b2 = _sym_adj(5, 0.5, 7); _rcm_block[6:11, 6:11] = _b2
_rcm_mats = [
    ('path10', _rcm_path),
    ('ring10', _rcm_ring),
    ('full7', np.ones((7, 7)) - np.eye(7)),
    ('sym8', _sym_adj(8, 0.25, 1)),
    ('sym10', _sym_adj(10, 0.15, 2)),
    ('sym12', _sym_adj(12, 0.30, 3)),
    ('sym15', _sym_adj(15, 0.20, 4)),
    ('sym20', _sym_adj(20, 0.12, 5)),
    ('disconnected11', _rcm_block),
    ('empty6', np.zeros((6, 6))),
]
rcm_c = []
for _name, _A in _rcm_mats:
    _perm = np.asarray(_rcm(sp.csr_array(_A), symmetric_mode=True))
    _bw = _bandwidth(_A, _perm)
    rcm_c.append({'args': [F.enc(_A), F.enc(True)], 'kwargs': {},
                  'expect': int(_bw), 'compare': 'rcm_bandwidth'})
calls.append({'fn': 'reverse_cuthill_mckee', 'cases': rcm_c})

# maximum_bipartite_matching: maximum-cardinality matching of an m x n bipartite biadjacency.
# scipy ignores weights (structure only); the matching itself is NON-UNIQUE so the fixture stores
# only its (unique) cardinality and the harness validates the returned matching semantically
# (compare='bipartite_match'): a valid matching of the expected size against the pattern in args[0].
# perm_type='row' output has one entry per COLUMN (entry j = row matched to col j, or -1). Host-independent.
from scipy.sparse.csgraph import maximum_bipartite_matching as _mbm
_mbm_cases = [
    np.array([[1,0,1],[0,1,0],[1,1,0]], np.int64),
    np.array([[1,1,0,0],[0,1,1,0],[0,0,1,1]], np.int64),
    np.array([[1,0],[1,0],[0,1],[0,1]], np.int64),
    np.array([[0,0,0],[0,0,0]], np.int64),
    np.eye(5, dtype=np.int64),
    np.array([[1,1,0,0,0],[1,1,0,0,0],[0,1,1,0,0],[0,0,1,1,0],[0,0,0,1,1]], np.int64),  # chain, needs re-augmentation
    np.array([[1,1,1,1],[1,0,0,0],[1,0,0,0],[1,0,0,0]], np.int64),                      # re-augmentation stress
    np.array([[0,5,0,2],[7,0,0,0],[0,3,9,0]], np.int64),                                # weighted -> structure only
]
_mbm_rng = np.random.default_rng(20261012)
for _ in range(6):
    _m = int(_mbm_rng.integers(3, 8)); _n = int(_mbm_rng.integers(3, 8))
    _mbm_cases.append((_mbm_rng.uniform(size=(_m, _n)) < 0.4).astype(np.int64))
mbm_c = []
for _A in _mbm_cases:
    _r = _mbm(sp.csr_array(_A), perm_type='row')
    _sz = int(np.sum(_r != -1))
    mbm_c.append({'args': [{'shape': list(_A.shape), 'dtype': 'int64', 'data': [int(x) for x in _A.ravel()]}],
                  'kwargs': {}, 'expect': {'size': _sz}, 'compare': 'bipartite_match'})
calls.append({'fn': 'maximum_bipartite_matching', 'cases': mbm_c})

# yen: K loopless shortest-path DISTANCES between source and sink (Yen's algorithm) on a dense adjacency
# matrix (0 = no edge). SciPy returns ONLY the paths actually found (length min(#paths, K), NOT inf-padded);
# distances are unique even when paths tie, so the sorted distance array is host-independent. Pure graph
# arithmetic -> compare='tol' on the distance array. Positional args: (csgraph, source, sink, K, directed).
from scipy.sparse.csgraph import yen as _yen
yen_c = []


def _yen_case(A, s, t, K, directed):
    r = np.asarray(_yen(sp.csr_array(np.asarray(A, float)), s, t, K, directed=directed), float)
    return {'args': [F.enc(np.asarray(A, float)), F.enc(int(s)), F.enc(int(t)), F.enc(int(K)), F.enc(bool(directed))],
            'kwargs': {}, 'expect': F.enc_result(r, []), 'compare': 'tol'}


_yen_explicit = [
    (np.array([[0,3,2,0,0],[0,0,0,4,0],[0,1,0,2,0],[0,0,0,0,2],[0,0,0,0,0]], float), 0, 4, 3, True),
    (np.array([[0,1,0,3],[0,0,1,0],[0,0,0,1],[0,0,0,0]], float), 0, 3, 5, True),        # only 2 paths -> len 2
    (np.array([[0,1,0,3],[0,0,1,0],[0,0,0,1],[0,0,0,0]], float), 0, 3, 1, True),        # K=1
    (np.array([[0,7,9,0,0,14],[7,0,10,15,0,0],[9,10,0,11,0,2],
               [0,15,11,0,6,0],[0,0,0,6,0,9],[14,0,2,0,9,0]], float), 0, 4, 4, False),  # undirected classic
    (np.array([[0,7,9,0,0,14],[7,0,10,15,0,0],[9,10,0,11,0,2],
               [0,15,11,0,6,0],[0,0,0,6,0,9],[14,0,2,0,9,0]], float), 0, 3, 3, False),
]

_yen_rng = np.random.default_rng(20261012)


def _yen_rand(n, p, lo=1, hi=9):
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and _yen_rng.uniform() < p:
                A[i, j] = int(_yen_rng.integers(lo, hi + 1))
    return A


_yen_rand_specs = [(6, 0.5, 0, 5, 3, True), (7, 0.45, 0, 6, 4, True), (8, 0.4, 0, 7, 5, True),
                   (6, 0.6, 1, 4, 2, True), (7, 0.5, 0, 3, 6, False), (9, 0.35, 0, 8, 3, True),
                   (5, 0.7, 0, 4, 10, True), (7, 0.55, 0, 6, 4, False)]
for _n, _p, _s, _t, _K, _d in _yen_rand_specs:
    _yen_explicit.append((_yen_rand(_n, _p), _s, _t, _K, _d))

for _A, _s, _t, _K, _d in _yen_explicit:
    yen_c.append(_yen_case(_A, _s, _t, _K, _d))
calls.append({'fn': 'yen', 'cases': yen_c})


# ---- dense <-> CSR converters --------------------------------------------------------
from scipy.sparse.csgraph import csgraph_to_dense as _todense, csgraph_from_dense as _fromdense
_cg = np.random.default_rng(20261013)


def _to_dense_case(A, null_value):
    M = sp.csr_array(A)
    M.sort_indices()
    n = int(A.shape[0])
    exp = _todense(M, null_value=null_value)
    return {'args': [F.enc(np.asarray(M.data, float)), F.enc(np.asarray(M.indices, np.int64)),
                     F.enc(np.asarray(M.indptr, np.int64)), n, F.enc(float(null_value))],
            'kwargs': {}, 'expect': F.enc_result(np.asarray(exp, float), []), 'compare': 'tol'}


_td_mats = [
    np.array([[0, 1, 2, 0], [0, 0, 0, 1], [0, 0, 0, 3], [0, 0, 0, 0]], float),
    np.array([[0, 0, 0], [2, 0, 0], [0, 5, 0]], float),
    np.zeros((4, 4), float),
]
for _ in range(5):
    _n = int(_cg.integers(3, 7))
    _td_mats.append(_cg.integers(0, 4, size=(_n, _n)).astype(float))
_td_nvs = [0.0, 0.0, 0.0, -1.0, 99.0, 0.0, 0.0, 0.0]
_td_c = [_to_dense_case(_A, _td_nvs[_i]) for _i, _A in enumerate(_td_mats)]
# explicit duplicate-min (repeated (0,1) entry: min of 5 and 3 wins)
_dd = sp.csr_array((np.array([5.0, 3.0, 7.0]), np.array([1, 1, 2]), np.array([0, 2, 3, 3])), shape=(3, 3))
_dexp = _todense(_dd, null_value=0.0)
_td_c.append({'args': [F.enc(np.array([5.0, 3.0, 7.0])), F.enc(np.array([1, 1, 2], np.int64)),
                     F.enc(np.array([0, 2, 3, 3], np.int64)), 3, F.enc(0.0)], 'kwargs': {},
                     'expect': F.enc_result(np.asarray(_dexp, float), []), 'compare': 'tol'})
calls.append({'fn': 'csgraph_to_dense', 'cases': _td_c})


def _from_dense_case(A, null_value, nan_null, infinity_null):
    M = _fromdense(A, null_value=null_value, nan_null=nan_null, infinity_null=infinity_null).tocsr()
    M.sort_indices()
    kw = {}
    if null_value != 0.0:
        kw['null_value'] = F.enc(null_value)
    if not nan_null:
        kw['nan_null'] = F.enc(False)
    if not infinity_null:
        kw['infinity_null'] = F.enc(False)
    return {'args': [F.enc(np.asarray(A, float))], 'kwargs': kw,
            'expect': {'dict': {'indptr': F.enc(np.asarray(M.indptr, np.int64)),
                               'indices': F.enc(np.asarray(M.indices, np.int64)),
                               'data': F.enc(np.asarray(M.data, float))}}, 'compare': 'tol'}


_Ainf = np.array([[1, 0, np.nan], [np.inf, 2, 0], [-np.inf, 0, 3]], float)
_fd_c = [
    _from_dense_case(np.array([[0, 1, 2, 0], [0, 0, 0, 1], [0, 0, 0, 3], [0, 0, 0, 0]], float), 0.0, True, True),
    _from_dense_case(np.array([[0, 5], [7, 0]], float), 0.0, True, True),
    _from_dense_case(_Ainf, 0.0, True, True),
    _from_dense_case(_Ainf, 0.0, False, False),
    _from_dense_case(_Ainf, 0.0, True, False),
    _from_dense_case(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], float), 1.0, True, True),
]
for _ in range(3):
    _n = int(_cg.integers(3, 6))
    _fd_c.append(_from_dense_case(_cg.integers(0, 3, size=(_n, _n)).astype(float), 0.0, True, True))
calls.append({'fn': 'csgraph_from_dense', 'cases': _fd_c})

out = {'module': 'csgraph', 'scipy': __import__('scipy').__version__, 'env': F.fixture_env.env(), 'calls': calls}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'csgraph: connected_components {len(cases)}, + shortest-path families ({sum(len(c["cases"]) for c in calls) - len(cases)} cases)')
