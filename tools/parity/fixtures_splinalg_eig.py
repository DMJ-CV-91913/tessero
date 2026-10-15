#!/usr/bin/env python3
"""Generate parity fixtures for scipy.sparse.linalg.eigsh / eigs / svds (values only).

The C kernel reproduces SciPy's eigsh/eigs/svds VALUES via dense LAPACK + k/which
selection (ARPACK is not reproducible). We therefore generate expected arrays that
(a) match SciPy's own eigsh/eigs/svds to 1e-9 and (b) use the exact encoding the C
routine emits:
  - eigsh: float64 [k], selected eigenvalues ascending.
  - eigs : float64 [k,2] of (real, |imag|) sorted by (real, |imag|).
  - svds : float64 [k], top-k singular values ascending.
Every fixture is cross-checked against SciPy before being written; a mismatch aborts.
"""
import json, sys
import numpy as np
import scipy.sparse.linalg as sla
from scipy.sparse import csr_matrix

rng = np.random.default_rng(2026_10_13)
EPS = 1e-9

def select_eigsh(w, k, which):
    w = np.sort(w)
    if which == 'LA':   sel = w[-k:]
    elif which == 'SA': sel = w[:k]
    elif which == 'LM': sel = w[np.argsort(np.abs(w))[-k:]]
    elif which == 'SM': sel = w[np.argsort(np.abs(w))[:k]]
    elif which == 'BE':
        nhi = (k + 1)//2; nlo = k//2
        sel = np.concatenate([w[:nlo], w[-nhi:]]) if nhi else w[:nlo]
    return np.sort(sel)

def select_eigs_ri(w, k, which):
    if which == 'LM':   idx = np.argsort(np.abs(w))[-k:]
    elif which == 'LR': idx = np.argsort(w.real)[-k:]
    elif which == 'SR': idx = np.argsort(w.real)[:k]
    elif which == 'LI': idx = np.argsort(np.abs(w.imag))[-k:]
    sel = w[idx]
    ri = np.stack([sel.real, np.abs(sel.imag)], axis=1)
    order = np.lexsort((ri[:,1], ri[:,0]))
    return ri[order]

def mk_arg(a):
    return {"shape": list(a.shape), "dtype": "float64", "data": a.reshape(-1).tolist()}

def mk_arr(a):
    a = np.asarray(a, dtype=float)
    return {"shape": list(a.shape), "dtype": "float64", "data": a.reshape(-1).tolist()}

eigsh_cases, eigs_cases, svds_cases = [], [], []
fails = []

# ---------------- eigsh (symmetric) ----------------
sym = []
for i in range(9):
    n = int(rng.integers(6, 12)); M = rng.standard_normal((n, n)); sym.append((M + M.T)/2)
for which in ['LM','SM','LA','SA','BE']:
    for ci, A in enumerate(sym):
        n = A.shape[0]; k = int(rng.integers(2, n-1))
        mine = select_eigsh(np.linalg.eigvalsh(A), k, which)
        ref = np.sort(sla.eigsh(csr_matrix(A), k=k, which=which, return_eigenvectors=False))
        if not (mine.shape == ref.shape and np.allclose(mine, ref, atol=EPS, rtol=EPS)):
            fails.append(("eigsh", which, ci)); continue
        eigsh_cases.append({"args":[mk_arg(A), k],
                            "kwargs":{"which":{"s":which}},
                            "expect": mk_arr(mine),
                            "compare":"tol","tol":{"atol":1e-9,"rtol":1e-9}})

# ---------------- eigs (general) ----------------
gen = [rng.standard_normal((int(rng.integers(7,12)),)*2) for _ in range(9)]
for which in ['LM','LR','SR','LI']:
    for ci, A in enumerate(gen):
        n = A.shape[0]; k = int(rng.integers(2, n-2))
        mine = select_eigs_ri(np.linalg.eigvals(A), k, which)
        rvals = sla.eigs(csr_matrix(A), k=k, which=which, return_eigenvectors=False)
        ri = np.stack([rvals.real, np.abs(rvals.imag)], axis=1)
        ref = ri[np.lexsort((ri[:,1], ri[:,0]))]
        if not (mine.shape == ref.shape and np.allclose(mine, ref, atol=EPS, rtol=EPS)):
            fails.append(("eigs", which, ci)); continue
        eigs_cases.append({"args":[mk_arg(A), k],
                           "kwargs":{"which":{"s":which}},
                           "expect": mk_arr(mine),
                           "compare":"tol","tol":{"atol":1e-9,"rtol":1e-9}})

# ---------------- svds ----------------
for ci in range(10):
    m = int(rng.integers(6,12)); n = int(rng.integers(6,12))
    A = rng.standard_normal((m, n)); k = int(rng.integers(1, min(m,n)-1))
    mine = np.sort(np.linalg.svd(A, compute_uv=False))[-k:]
    _, s, _ = sla.svds(csr_matrix(A), k=k, which='LM'); ref = np.sort(s)
    if not (mine.shape == ref.shape and np.allclose(mine, ref, atol=EPS, rtol=EPS)):
        fails.append(("svds", "LM", ci)); continue
    svds_cases.append({"args":[mk_arg(A), k],
                       "kwargs":{"which":{"s":"LM"}},
                       "expect": mk_arr(mine),
                       "compare":"tol","tol":{"atol":1e-9,"rtol":1e-9}})

if fails:
    print("ABORT: scipy cross-check failures:", fails); sys.exit(1)

rows = [
    {"fn":"eigsh","cases":eigsh_cases},
    {"fn":"eigs", "cases":eigs_cases},
    {"fn":"svds", "cases":svds_cases},
]

# merge into a full splinalg.json ready to drop in
base = json.load(open("/home/tessero/tessero/tests/fixtures/parity/splinalg.json"))
base["calls"] = [c for c in base["calls"] if c["fn"] not in ("eigsh","eigs","svds")]
base["calls"].extend(rows)
base["calls"].sort(key=lambda c: c["fn"])
json.dump(base, open("/home/tessero/tessero/tests/fixtures/parity/splinalg.json","w"), indent=1)

print(f"eigsh {len(eigsh_cases)} cases, eigs {len(eigs_cases)} cases, svds {len(svds_cases)} cases")
print(f"TOTAL {len(eigsh_cases)+len(eigs_cases)+len(svds_cases)} cases; ALL cross-checked vs scipy @ 1e-9")
