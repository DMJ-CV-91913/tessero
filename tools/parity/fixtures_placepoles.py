#!/usr/bin/env python3
"""scipy.signal.place_poles single-input pole placement by state feedback.

The pure-PHP Tessero\\Signal\\PlacePoles reproduces scipy.signal.place_poles for SINGLE-INPUT systems
(B has exactly one column). In that regime the state-feedback gain K placing the closed-loop poles of
(A - B K) is UNIQUE, so scipy's YT/KNV0 iterations converge to the same value Ackermann's formula
gives; the two agree to machine precision. Multi-input place_poles is intentionally out of scope: its
gain is non-unique and scipy's Kautsky-Nichols-Van Dooren iteration selects an implementation-specific
solution that is not bit-reproducible, so no multi-input cases are recorded here.

All data is embedded in the fixture, so the recorded scipy expectation is host-independent.

Writes tests/fixtures/parity/signal_placepoles.json with a `placepoles_cases` block:
  {fn:'place_poles', A:[[...]], B:[[...]], poles:[[re,im], ...], method,
   expect:{gain_matrix:[[...]]}, rtol}
"""
import json
import os

import numpy as np
from scipy.signal import place_poles

OUT = os.environ.get('SIGNALPLACEPOLES_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'signal_placepoles.json')

cases = []


def add(A, B, poles, method='YT', rtol=1e-6):
    A = np.array(A, float)
    B = np.array(B, float)
    assert B.shape[1] == 1, 'single-input only'
    pc = [complex(p[0], p[1]) if isinstance(p, (list, tuple)) else complex(p) for p in poles]
    r = place_poles(A, B, np.array(pc), method=method)
    cases.append({
        'fn': 'place_poles',
        'A': [[float(v) for v in row] for row in A],
        'B': [[float(v) for v in row] for row in B],
        'poles': [[float(p[0]), float(p[1])] if isinstance(p, (list, tuple)) else [float(p), 0.0]
                  for p in poles],
        'method': method,
        'expect': {'gain_matrix': [[float(v) for v in row] for row in r.gain_matrix]},
        'rtol': rtol,
    })


# n=2, distinct real poles (non-companion A).
add([[0., 1], [2, 3]], [[0.], [1]], [-1., -2])
# n=2, complex-conjugate pair.
add([[0., 1], [2, 3]], [[0.], [1]], [[-1, 2], [-1, -2]])
# n=3, controllable-canonical (companion) form.
add([[0., 1, 0], [0, 0, 1], [-6, -11, -6]], [[0.], [0], [1]], [-1., -2, -3])
# n=3, general (non-companion) A, B.
add([[1., 2, 0], [0, 1, 3], [1, 0, 2]], [[1.], [0], [1]], [-1., -2, -3])
# n=3, one real + complex-conjugate pair.
add([[1., 2, 0], [0, 1, 3], [1, 0, 2]], [[1.], [0], [1]], [-2., [-1, 1], [-1, -1]])
# n=4, distinct real poles (upper-bidiagonal A).
add([[1., 1, 0, 0], [0, 2, 1, 0], [0, 0, 3, 1], [0, 0, 0, 4]], [[0.], [0], [0], [1]], [-1., -2, -3, -4])
# n=4, two complex-conjugate pairs.
add([[1., 1, 0, 0], [0, 2, 1, 0], [0, 0, 3, 1], [0, 0, 0, 4]], [[0.], [0], [0], [1]],
    [[-1, 1], [-1, -1], [-2, 1], [-2, -1]])
# n=4, companion A, mixed real + complex-conjugate pair.
add([[0., 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -2, -3, -4]], [[0.], [0], [0], [1]],
    [-0.5, -1.5, [-2, 1], [-2, -1]])

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    json.dump({'placepoles_cases': cases}, f, allow_nan=False, indent=1)
print(f'wrote {len(cases)} cases to {OUT}')
