#!/usr/bin/env python3
"""scipy.spatial.Rectangle fixtures: the pure-PHP Tessero\\Spatial\\Rectangle geometry verified against
scipy. Deterministic (the rectangles/points are embedded), host-independent -- no pinned environment."""
import json
import os

import numpy as np
from scipy.spatial import Rectangle

OUT = os.environ.get('RECT_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'spatial_rect.json')

cases = []
rng = np.random.default_rng(0)
for _ in range(8):
    mn = rng.uniform(-3, 0, 3); mx = mn + rng.uniform(0.5, 4, 3)
    mn2 = rng.uniform(-3, 2, 3); mx2 = mn2 + rng.uniform(0.5, 4, 3)
    x = rng.uniform(-5, 7, 3)
    R = Rectangle(mx, mn); R2 = Rectangle(mx2, mn2)
    ml = [round(float(v), 10) for v in mx]; nl = [round(float(v), 10) for v in mn]
    ml2 = [round(float(v), 10) for v in mx2]; nl2 = [round(float(v), 10) for v in mn2]
    xl = [round(float(v), 10) for v in x]
    cases.append({'fn': 'volume', 'maxes': ml, 'mins': nl, 'expect': float(Rectangle(ml, nl).volume())})
    for p in (1.0, 2.0, 3.5):
        base = {'maxes': ml, 'mins': nl, 'p': p}
        cases.append({**base, 'fn': 'min_distance_point', 'x': xl, 'expect': float(Rectangle(ml, nl).min_distance_point(xl, p))})
        cases.append({**base, 'fn': 'max_distance_point', 'x': xl, 'expect': float(Rectangle(ml, nl).max_distance_point(xl, p))})
        cases.append({**base, 'fn': 'min_distance_rectangle', 'other': {'maxes': ml2, 'mins': nl2},
                      'expect': float(Rectangle(ml, nl).min_distance_rectangle(Rectangle(ml2, nl2), p))})
        cases.append({**base, 'fn': 'max_distance_rectangle', 'other': {'maxes': ml2, 'mins': nl2},
                      'expect': float(Rectangle(ml, nl).max_distance_rectangle(Rectangle(ml2, nl2), p))})

out = {'module': 'spatial', 'scipy': __import__('scipy').__version__, 'rectangle_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'spatial_rect: {len(cases)} cases')
