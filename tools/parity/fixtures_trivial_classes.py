"""Deterministic exception/config classes credited by construction + attribute
inspection (no numeric math): scipy.optimize.NoConvergence, scipy.spatial.QhullError,
scipy.stats.BootstrapMethod / MonteCarloMethod / PermutationMethod. Each case names
the census module+symbol, how to construct the PHP class, and the scipy-observed
attributes to assert. Emitted as class_cases in trivial_classes.json."""
import json, os
import scipy

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "tests", "fixtures", "parity", "trivial_classes.json")


def cases():
    out = []
    import scipy.optimize as O
    e = O.NoConvergence("did not converge")
    out.append({"module": "optimize", "name": "NoConvergence", "kind": "exception",
                "ctor": ["did not converge"], "is_exception": isinstance(e, Exception),
                "message": str(e)})
    import scipy.spatial as SP
    q = SP.QhullError("qhull precision error")
    out.append({"module": "spatial", "name": "QhullError", "kind": "exception",
                "ctor": ["qhull precision error"], "is_runtime": isinstance(q, RuntimeError),
                "message": str(q)})
    import scipy.stats as ST
    bm = ST.BootstrapMethod()
    out.append({"module": "stats", "name": "BootstrapMethod", "kind": "config",
                "ctor": [], "fields": {"n_resamples": bm.n_resamples, "batch": bm.batch,
                                       "rng": bm.rng, "method": bm.method}})
    mm = ST.MonteCarloMethod()
    out.append({"module": "stats", "name": "MonteCarloMethod", "kind": "config",
                "ctor": [], "fields": {"n_resamples": mm.n_resamples, "batch": mm.batch,
                                       "rvs": mm.rvs, "rng": mm.rng}})
    pm = ST.PermutationMethod()
    out.append({"module": "stats", "name": "PermutationMethod", "kind": "config",
                "ctor": [], "fields": {"n_resamples": pm.n_resamples, "batch": pm.batch,
                                       "rng": pm.rng}})
    return out


if __name__ == "__main__":
    out = {"module": "trivial_classes", "scipy": scipy.__version__, "class_cases": cases()}
    with open(OUT, "w") as f:
        json.dump(out, f, separators=(",", ":"), allow_nan=False)
    print("wrote", OUT, "cases:", len(out["class_cases"]))
