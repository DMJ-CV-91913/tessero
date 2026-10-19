#!/usr/bin/env python3
"""Deterministic fixtures for scipy.stats.barnard_exact parity (Tessero\\StatsBarnard).

Emits stats_barnard.json with a 'statfn2_cases' list. Each case carries a 2x2 table,
the alternative / pooled / n parameters, and the expected {statistic, pvalue} from
scipy.stats.barnard_exact. The run-tests 'statfn2_cases' arm (see wiring.txt) feeds
the table + params to Tessero\\StatsBarnard::barnardExact and compares both outputs.

Run on the build server inside .venv-parity with the pinned env:
  NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR" OPENBLAS_CORETYPE=Haswell \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 fixtures_barnard.py > stats_barnard.json
"""
import json
import math

import numpy as np
import scipy
import scipy.stats as st

# Fixed, deterministic 2x2 tables spanning small/medium totals, a zero cell,
# a balanced table, and an asymmetric one.
TABLES = [
    [[7, 12], [8, 3]],
    [[10, 2], [3, 8]],
    [[2, 7], [8, 2]],
    [[20, 15], [10, 25]],
    [[5, 0], [1, 4]],
    [[0, 3], [4, 2]],
    [[6, 6], [6, 6]],
]

ALTERNATIVES = ["two-sided", "less", "greater"]
POOLED = [True, False]


def main():
    cases = []
    for table in TABLES:
        for alt in ALTERNATIVES:
            for pooled in POOLED:
                r = st.barnard_exact(table, alternative=alt, pooled=pooled, n=32)
                stat = r.statistic
                cases.append({
                    "fn": "barnard_exact",
                    "table": table,
                    "alternative": alt,
                    "pooled": pooled,
                    "n": 32,
                    "expect": {
                        "statistic": None if (isinstance(stat, float) and math.isnan(stat)) else float(stat),
                        "pvalue": float(r.pvalue),
                    },
                    "rtol": 1e-7,
                })
    out = {
        "module": "stats",
        "scipy": scipy.__version__,
        "statfn2_cases": cases,
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
