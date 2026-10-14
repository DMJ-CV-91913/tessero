import json
import numpy as np
from scipy.integrate import ode, complex_ode


def make_rhs(kind, p):
    if kind == "exp_decay":
        k = p["k"]
        return lambda t, y: [-k * yi for yi in y]
    if kind == "harmonic":
        w = p["w"]
        return lambda t, y: [y[1], -w * w * y[0]]
    if kind == "lotka":
        a, b, c, d = p["a"], p["b"], p["c"], p["d"]
        return lambda t, y: [a * y[0] - b * y[0] * y[1], -c * y[1] + d * y[0] * y[1]]
    if kind == "logistic":
        r, K = p["r"], p["K"]
        return lambda t, y: [r * yi * (1.0 - yi / K) for yi in y]
    if kind == "linear2x2":
        M = np.asarray(p["M"], dtype=float)
        return lambda t, y: (M.dot(np.asarray(y))).tolist()
    raise ValueError(kind)


def make_crhs(kind, p):
    # complex RHS: y is a complex ndarray; returns a complex list
    if kind == "rot":            # y' = i*w*y  (rotation on the unit circle)
        w = p["w"]
        return lambda t, y: [1j * w * y[0]]
    if kind == "cdecay":         # y' = (a + i*b) y
        a, b = p["a"], p["b"]
        return lambda t, y: [(a + 1j * b) * y[0]]
    if kind == "cpair":          # 2-component complex linear system
        return lambda t, y: [1j * y[0] - 0.1 * y[1], 1j * 2.0 * y[1]]
    raise ValueError(kind)


OUT = "../../tests/fixtures/parity/integrate_ode.json"


def rcase(kind, params, y0, t0, ts, integrator="dopri5", rtol=1e-9, atol=1e-12):
    f = make_rhs(kind, params)
    r = ode(f).set_integrator(integrator, rtol=rtol, atol=atol)
    r.set_initial_value([float(v) for v in y0], float(t0))
    steps = []
    for t in ts:
        yv = r.integrate(float(t))
        assert r.successful(), (kind, integrator, t)
        steps.append([float(v) for v in np.atleast_1d(yv)])
    return {
        "complex": False,
        "kind": kind,
        "params": params,
        "y0": [float(v) for v in y0],
        "t0": float(t0),
        "ts": [float(t) for t in ts],
        "integrator": integrator,
        "rtol": rtol,
        "atol": atol,
        "expect": steps,   # list over ts of [components]
    }


def ccase(kind, params, y0, t0, ts, integrator="dop853", rtol=1e-10, atol=1e-12):
    f = make_crhs(kind, params)
    r = complex_ode(f).set_integrator(integrator, rtol=rtol, atol=atol)
    r.set_initial_value([complex(v) for v in y0], float(t0))
    steps = []
    for t in ts:
        yv = r.integrate(float(t))
        assert r.successful(), (kind, integrator, t)
        steps.append([[float(np.real(v)), float(np.imag(v))] for v in np.atleast_1d(yv)])
    return {
        "complex": True,
        "kind": kind,
        "params": params,
        "y0": [[float(np.real(v)), float(np.imag(v))] for v in y0],
        "t0": float(t0),
        "ts": [float(t) for t in ts],
        "integrator": integrator,
        "rtol": rtol,
        "atol": atol,
        "expect": steps,   # list over ts of [[re,im],...]
    }


cases = [
    # --- real, dopri5 (-> RK45) ---
    rcase("exp_decay", {"k": 0.7}, [2.0], 0.0, [1.0, 2.0, 3.0, 4.0, 5.0], "dopri5"),
    rcase("harmonic", {"w": 2.0}, [1.0, 0.0], 0.0, [1.0, 2.0, 3.0, 6.0], "dopri5"),
    rcase("lotka", {"a": 1.5, "b": 1.0, "c": 3.0, "d": 1.0}, [10.0, 5.0], 0.0, [2.0, 5.0, 10.0], "dopri5"),
    rcase("logistic", {"r": 1.0, "K": 10.0}, [1.0], 0.0, [2.0, 4.0, 6.0], "dopri5"),
    rcase("linear2x2", {"M": [[-0.5, 1.0], [-1.0, -0.5]]}, [1.0, 1.0], 0.0, [2.0, 4.0, 8.0], "dopri5"),
    # backward time
    rcase("harmonic", {"w": 1.5}, [0.0, 1.0], 0.0, [-1.0, -3.0, -5.0], "dopri5"),
    # --- real, dop853 (-> DOP853) ---
    rcase("exp_decay", {"k": 2.0}, [5.0], 0.0, [1.0, 2.0, 4.0], "dop853"),
    rcase("harmonic", {"w": 2.0}, [1.0, 0.0], 0.0, [1.0, 2.0, 3.0], "dop853"),
    rcase("lotka", {"a": 1.5, "b": 1.0, "c": 3.0, "d": 1.0}, [10.0, 5.0], 0.0, [5.0, 10.0], "dop853"),
    rcase("logistic", {"r": 1.0, "K": 10.0}, [1.0], 0.0, [3.0, 6.0], "dop853"),
    # --- complex_ode, dop853 ---
    ccase("rot", {"w": 1.0}, [1.0 + 0j], 0.0, [np.pi / 2, np.pi, 2 * np.pi], "dop853"),
    ccase("cdecay", {"a": -0.3, "b": 2.0}, [1.0 + 0j], 0.0, [1.0, 2.0, 4.0], "dop853"),
    ccase("cpair", {}, [1.0 + 0j, 0.5 + 0.5j], 0.0, [1.0, 2.0, 3.0], "dop853"),
    # --- complex_ode, dopri5 ---
    ccase("rot", {"w": 2.0}, [0.0 + 1.0j], 0.0, [0.5, 1.0, 1.5], "dopri5", rtol=1e-9),
]

out = {"module": "integrate", "scipy": __import__("scipy").__version__, "ode_cases": cases}
with open(OUT, "w") as fh:
    json.dump(out, fh, separators=(",", ":"))
print("integrate_ode: %d ode/complex_ode cases" % len(cases))
