#!/usr/bin/env python3
"""Generate tests/fixtures/parity/ndimage_genfilter.json for scipy.ndimage.generic_filter.

A Python callable cannot cross into PHP, so each case names a REDUCER that the PHP
run-tests harness reconstructs as a PHP callable (mean/max/min/median/sum/ptp over the
flat C-order window). These reducers are order-insensitive, so scipy's footprint visit
order vs the PHP gather order is immaterial to the compared result.

Cases cover 1-D and 2-D inputs; size (scalar box) and explicit footprint; modes
reflect/constant/nearest/wrap/mirror; origin 0 (scipy default footprint centre).
"""
import json
import os

import numpy as np
from scipy import ndimage as ndi

REDUCERS = {
    "mean":   np.mean,
    "max":    np.max,
    "min":    np.min,
    "median": np.median,
    "sum":    np.sum,
    "ptp":    (lambda w: float(np.ptp(w))),
}

CASES = []


def add(name, reducer, arr, *, size=None, footprint=None, mode="reflect", cval=0.0, origin=0):
    a = np.asarray(arr, dtype=np.float64)
    out = ndi.generic_filter(
        a, REDUCERS[reducer],
        size=size, footprint=footprint, mode=mode, cval=cval, origin=origin,
    )
    CASES.append({
        "name": name,
        "reducer": reducer,
        "input": a.tolist(),
        "size": size,
        "footprint": (np.asarray(footprint, dtype=int).tolist() if footprint is not None else None),
        "mode": mode,
        "cval": float(cval),
        "origin": origin,
        "expect": out.tolist(),
    })


a1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
a2 = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]

add("1d_mean_s3_reflect",  "mean", a1, size=3, mode="reflect")
add("1d_max_s3_constant",  "max",  a1, size=3, mode="constant", cval=0.0)
add("1d_sum_s3_nearest",   "sum",  a1, size=3, mode="nearest")
add("1d_min_s3_wrap",      "min",  a1, size=3, mode="wrap")
add("1d_mean_s3_mirror",   "mean", a1, size=3, mode="mirror")
add("1d_ptp_s3_reflect",   "ptp",  a1, size=3, mode="reflect")

add("2d_min_s3_reflect",   "min",    a2, size=3, mode="reflect")
add("2d_sum_s3_constant",  "sum",    a2, size=3, mode="constant", cval=0.0)
add("2d_mean_s3_nearest",  "mean",   a2, size=3, mode="nearest")
add("2d_median_s3_wrap",   "median", a2, size=3, mode="wrap")
add("2d_ptp_s2_nearest",   "ptp",    a2, size=2, mode="nearest")

add("2d_median_cross_constant", "median", a2,
    footprint=[[0, 1, 0], [1, 1, 1], [0, 1, 0]], mode="constant", cval=0.0)
add("2d_max_cross_reflect", "max", a2,
    footprint=[[0, 1, 0], [1, 1, 1], [0, 1, 0]], mode="reflect")

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tests", "fixtures", "parity", "ndimage_genfilter.json")
with open(OUT, "w") as fh:
    json.dump({"genfilter_cases": CASES}, fh, indent=1)
print(f"wrote {OUT}: {len(CASES)} cases")
