#!/usr/bin/env python3
"""Generate tests/fixtures/parity/ndimage_labcomp.json for scipy.ndimage.labeled_comprehension.

A Python callable cannot cross into PHP, so each case names a REDUCER that the PHP run-tests
harness reconstructs as a PHP callable over the gathered group values (mean/max/min/sum/median/ptp),
plus a 'posmin' reducer that exercises pass_positions=True (minimum linear position).

Cases cover: scalar index, list index, index=None (all non-zero labels), labels=None (whole input),
a missing index hitting `default`, and pass_positions.
"""
import json
import os

import numpy as np
from scipy import ndimage as ndi

REDUCERS = {
    "mean":   np.mean,
    "max":    np.max,
    "min":    np.min,
    "sum":    np.sum,
    "median": np.median,
    "ptp":    (lambda v: float(np.ptp(v))),
}


def reducer(name):
    if name == "posmin":
        return (lambda vals, pos: float(np.min(pos))), True
    return REDUCERS[name], False


CASES = []


def add(name, rname, inp, labels, index, default, pass_positions=False):
    a = np.asarray(inp, dtype=np.float64)
    lab = None if labels is None else np.asarray(labels, dtype=np.intp)
    fn, needs_pos = reducer(rname)
    out = ndi.labeled_comprehension(a, lab, index, fn, np.float64, default, pass_positions)
    CASES.append({
        "name": name,
        "reducer": rname,
        "input": a.tolist(),
        "labels": (None if lab is None else lab.tolist()),
        "index": index,
        "default": (None if default is None else float(default)),
        "pass_positions": bool(pass_positions),
        "expect": (out.tolist() if isinstance(out, np.ndarray) else float(out)),
    })


inp = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
lab = [[1, 1, 0, 2], [1, 0, 2, 2], [3, 3, 3, 0]]

add("list_mean",        "mean",   inp, lab, [1, 2, 3], -1.0)
add("scalar_sum",       "sum",    inp, lab, 2, 0.0)
add("list_max_missing", "max",    inp, lab, [1, 9], -99.0)
add("list_min",         "min",    inp, lab, [1, 2, 3], 0.0)
add("list_median",      "median", inp, lab, [1, 2, 3], 0.0)
add("list_ptp",         "ptp",    inp, lab, [1, 2, 3], 0.0)
add("index_none_sum",   "sum",    inp, lab, None, 0.0)
add("index_none_mean",  "mean",   inp, lab, None, 0.0)
add("labels_none_sum",  "sum",    inp, None, None, 0.0)
add("labels_none_mean", "mean",   inp, None, None, 0.0)
add("pass_positions",   "posmin", inp, lab, [1, 2, 3], -1.0, pass_positions=True)
add("default_only",     "sum",    inp, lab, [7, 8], 42.0)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "tests", "fixtures", "parity", "ndimage_labcomp.json")
with open(OUT, "w") as fh:
    json.dump({"module": "ndimage", "labcomp_cases": CASES}, fh, indent=1)
print(f"wrote {OUT}: {len(CASES)} cases")
