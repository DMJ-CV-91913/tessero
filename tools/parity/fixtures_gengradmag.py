#!/usr/bin/env python3
"""scipy.ndimage.generic_gradient_magnitude fixtures (pure-PHP Tessero NdimageFns::genericGradientMagnitude).

Gradient magnitude from a caller-supplied per-axis derivative: sqrt(sum over axes of derivative(input,axis)^2).
The canonical derivative in these fixtures is ndimage.correlate1d with an explicit odd weight kernel (named
"correlate1d"); the PHP runner rebuilds that same derivative by calling Tessero Ndimage::correlate1d on the
active backend, so this exercises the pure-PHP orchestration (square/add/sqrt) on top of native correlate1d.

Pure correlation arithmetic (no BLAS), host-independent; the pinned env is not required to regenerate.
Writes tests/fixtures/parity/ndimage_gengradmag.json with key "gengradmag_cases".
"""
import json, os
import numpy as np
import scipy.ndimage as ndi
import fixtures_np as F  # enc: NDArray {shape,dtype,data} envelope (decoded by run-tests.php decodeValue)

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "tests", "fixtures", "parity", "ndimage_gengradmag.json",
)
rng = np.random.default_rng(20261012)

cases = []


def case(inp, weights, mode="reflect", cval=0.0, deriv="correlate1d"):
    """Record one generic_gradient_magnitude case.

    deriv names the derivative kind; "correlate1d" => ndi.correlate1d(input, weights, axis, mode, cval),
    which is exactly what the PHP runner reconstructs.
    """
    inp = np.asarray(inp, dtype=float)
    w = [float(x) for x in weights]
    dv = (lambda win: (lambda a, axis, output, m, c, *extra, **kw:
                       ndi.correlate1d(a, win, axis, output, m, c)))(w)
    ref = ndi.generic_gradient_magnitude(inp, dv, mode=mode, cval=cval)
    cases.append({
        "deriv": deriv,
        "weights": w,
        "input": F.enc(inp),
        "mode": mode,
        "cval": cval,
        "expect": F.enc(np.asarray(ref, dtype=float)),
    })


# Canonical central-difference derivative [-1,0,1] across all boundary modes (2-D).
img2 = rng.uniform(-5, 5, (6, 7))
for mode in ("reflect", "constant", "nearest", "mirror", "wrap"):
    case(img2, [-1.0, 0.0, 1.0], mode=mode)
case(img2, [-1.0, 0.0, 1.0], mode="constant", cval=2.5)   # non-zero cval

# Other central-difference kernels.
case(rng.uniform(-5, 5, (5, 5)), [-0.5, 0.0, 0.5], mode="reflect")
case(rng.uniform(-5, 5, (8, 4)), [1.0, 0.0, -1.0], mode="nearest")

# 1-D input (single axis): magnitude == |derivative|.
case(rng.uniform(-5, 5, 11), [-1.0, 0.0, 1.0], mode="reflect")
case(rng.uniform(-5, 5, 9), [-1.0, 0.0, 1.0], mode="mirror")

# 3-D input (three axes summed under the sqrt).
case(rng.uniform(-5, 5, (3, 4, 5)), [-1.0, 0.0, 1.0], mode="reflect")
case(rng.uniform(-5, 5, (3, 4, 5)), [-1.0, 0.0, 1.0], mode="constant", cval=-1.0)

# A longer 5-tap kernel (origin/width handling flows through correlate1d).
case(rng.uniform(-5, 5, (6, 6)), [1.0, -2.0, 0.0, 2.0, -1.0], mode="reflect")

out = {"module": "ndimage", "scipy": __import__("scipy").__version__, "gengradmag_cases": cases}
with open(OUT, "w") as f:
    json.dump(out, f, separators=(",", ":"))
print(f"ndimage_gengradmag: {len(cases)} generic_gradient_magnitude cases -> {OUT}")
