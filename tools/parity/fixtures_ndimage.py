#!/usr/bin/env python3
"""scipy.ndimage fixtures: generate_binary_structure, 2-D label, 2-D convolve. Integer/float arithmetic (no BLAS),
host-independent, so this generator does not require the pinned environment."""
import json, os
import numpy as np
import scipy.ndimage as nd
import fixtures_np as F  # enc / enc_result

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'tests', 'fixtures', 'parity', 'ndimage.json')
rng = np.random.default_rng(20261001)


def call_nd(fn, args, kwargs, names):
    f = getattr(nd, fn)
    r = f(*args, **(kwargs or {}))
    return {'args': [F.enc(a) for a in args], 'kwargs': {k: F.enc(v) for k, v in (kwargs or {}).items()},
            'expect': F.enc_result(r, names or []), 'compare': 'tol'}


gbs = [call_nd('generate_binary_structure', [r, c], {}, []) for r in (1, 2, 3) for c in range(1, r + 1)]

binmats = [
    (rng.uniform(0, 1, (5, 6)) > 0.5).astype(float),
    np.array([[1., 1., 0., 0., 1.], [0., 1., 0., 0., 1.], [0., 0., 0., 1., 0.], [1., 0., 0., 1., 1.]]),
    (rng.uniform(0, 1, (7, 7)) > 0.6).astype(float),
    (rng.uniform(0, 1, (4, 8)) > 0.55).astype(float),
]
lab = []
for B in binmats:
    lab.append(call_nd('label', [B], {}, ['labels', 'num']))
    lab.append(call_nd('label', [B], {'structure': nd.generate_binary_structure(2, 2)}, ['labels', 'num']))

conv = []
I, W = rng.uniform(-5, 5, (5, 6)), rng.uniform(-1, 1, (3, 3))
for mode in ('reflect', 'constant', 'nearest', 'mirror', 'wrap'):
    conv.append(call_nd('convolve', [I, W], {'mode': mode}, []))
conv.append(call_nd('convolve', [I, W], {'mode': 'constant', 'cval': 2.5}, []))
conv.append(call_nd('convolve', [rng.uniform(-5, 5, (6, 6)), rng.uniform(-1, 1, (3, 3))], {}, []))   # default reflect
conv.append(call_nd('convolve', [rng.uniform(-5, 5, (5, 5)), rng.uniform(-1, 1, (5, 5))], {'mode': 'nearest'}, []))
# n-D correlation with a weights array
cw = rng.uniform(-1, 1, (3, 3))
cor = [call_nd('correlate', [rng.uniform(-5, 5, (5, 6)), cw], {'mode': m}, []) for m in ('reflect', 'constant', 'nearest', 'mirror', 'wrap')] + [
    call_nd('correlate', [rng.uniform(-5, 5, (5, 6)), cw], {'mode': 'constant', 'cval': 1.5}, []),
    call_nd('correlate', [rng.uniform(-5, 5, (5, 6)), rng.uniform(-1, 1, (3, 3))], {'origin': 1}, []),
    call_nd('correlate', [rng.uniform(-5, 5, 9), np.array([0.5, 1.0, -0.5])], {}, [])]

full8 = nd.generate_binary_structure(2, 2)
dil = [call_nd('binary_dilation', [binmats[0]], {}, []), call_nd('binary_dilation', [binmats[2]], {}, []),
       call_nd('binary_dilation', [binmats[0]], {'structure': full8}, [])]
ero = [call_nd('binary_erosion', [np.ones((4, 5))], {}, []), call_nd('binary_erosion', [binmats[1]], {}, []),
       call_nd('binary_erosion', [(rng.uniform(0, 1, (6, 7)) > 0.25).astype(float)], {}, []),
       call_nd('binary_erosion', [np.ones((5, 5))], {'structure': full8}, [])]
mxf = [call_nd('maximum_filter', [np.arange(1.0, 13.0).reshape(3, 4)], {'size': 3}, []),
       call_nd('maximum_filter', [rng.uniform(-5, 5, (5, 6))], {'size': 3}, []),
       call_nd('maximum_filter', [rng.uniform(-5, 5, (6, 5))], {'size': 2}, []),
       call_nd('maximum_filter', [rng.uniform(-5, 5, (7, 7))], {'size': 5}, [])]
# binary morphology compositions (2-D)
cross = nd.generate_binary_structure(2, 1)
bop = [call_nd('binary_opening', [binmats[0]], {}, []), call_nd('binary_opening', [binmats[2]], {}, []),
       call_nd('binary_opening', [binmats[0]], {'structure': full8}, [])]
bcl = [call_nd('binary_closing', [binmats[0]], {}, []), call_nd('binary_closing', [binmats[2]], {}, []),
       call_nd('binary_closing', [binmats[1]], {'structure': full8}, [])]
hm_img = np.array([[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]], float)
bhm = [call_nd('binary_hit_or_miss', [hm_img], {}, []), call_nd('binary_hit_or_miss', [binmats[1]], {}, []),
       call_nd('binary_hit_or_miss', [binmats[2]], {'structure1': full8}, [])]
itr = [call_nd('iterate_structure', [cross, 2], {}, []), call_nd('iterate_structure', [cross, 3], {}, []),
       call_nd('iterate_structure', [full8, 2], {}, []), call_nd('iterate_structure', [cross, 1], {}, [])]
fill_img = np.array([[0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 0], [0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 1, 0], [0, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0]], float)
bfh = [call_nd('binary_fill_holes', [fill_img], {}, []), call_nd('binary_fill_holes', [fill_img], {'structure': full8}, []),
       call_nd('binary_fill_holes', [binmats[1]], {}, [])]
prop_seed = np.zeros((6, 6)); prop_seed[1, 1] = 1.0
prop_mask = np.array([[0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0], [0, 1, 1, 1, 0, 0], [0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0]], float)
bpr = [call_nd('binary_propagation', [prop_seed], {'mask': prop_mask}, []),
       call_nd('binary_propagation', [prop_seed], {'mask': prop_mask, 'structure': full8}, [])]

# 1-D filters along an axis
MODES = ('reflect', 'constant', 'nearest', 'mirror', 'wrap')
x1 = rng.uniform(-5, 5, 9)
x2 = rng.uniform(-5, 5, (4, 6))
w3 = np.array([0.25, 0.5, 0.25])
w4 = np.array([0.1, 0.2, 0.3, 0.4])
corr1 = [call_nd('correlate1d', [x1, w3], {'mode': m}, []) for m in MODES] + [
    call_nd('correlate1d', [x1, w4], {'mode': 'reflect'}, []),
    call_nd('correlate1d', [x1, w3], {'mode': 'constant', 'cval': 1.5}, []),
    call_nd('correlate1d', [x1, w3], {'origin': 1}, []), call_nd('correlate1d', [x1, w3], {'origin': -1}, []),
    call_nd('correlate1d', [x2, w3], {'axis': 0}, []), call_nd('correlate1d', [x2, w3], {'axis': 1, 'mode': 'wrap'}, [])]
conv1 = [call_nd('convolve1d', [x1, w4], {'mode': m}, []) for m in MODES] + [
    call_nd('convolve1d', [x1, w3], {}, []), call_nd('convolve1d', [x2, w4], {'axis': 0}, []),
    call_nd('convolve1d', [x1, w4], {'origin': 1}, [])]
unif1 = [call_nd('uniform_filter1d', [x1], {'size': s, 'mode': m}, []) for s in (3, 4) for m in ('reflect', 'nearest', 'wrap')] + [
    call_nd('uniform_filter1d', [x2], {'size': 3, 'axis': 0}, []), call_nd('uniform_filter1d', [x1], {'size': 3, 'origin': 1}, [])]
gauss1 = [call_nd('gaussian_filter1d', [x1], {'sigma': s, 'mode': m}, []) for s in (1.0, 2.0) for m in ('reflect', 'nearest', 'constant')] + [
    call_nd('gaussian_filter1d', [x2], {'sigma': 1.5, 'axis': 1}, []),
    call_nd('gaussian_filter1d', [x1], {'sigma': 1.0, 'truncate': 2.0}, []),
    call_nd('gaussian_filter1d', [x1], {'sigma': 2.0, 'radius': 3}, [])]
min1 = [call_nd('minimum_filter1d', [x1], {'size': s, 'mode': m}, []) for s in (3, 4) for m in ('reflect', 'nearest', 'wrap', 'mirror')] + [
    call_nd('minimum_filter1d', [x2], {'size': 3, 'axis': 0}, []), call_nd('minimum_filter1d', [x1], {'size': 3, 'mode': 'constant', 'cval': 0.0}, [])]
max1 = [call_nd('maximum_filter1d', [x1], {'size': s, 'mode': m}, []) for s in (3, 4) for m in ('reflect', 'nearest', 'wrap')] + [
    call_nd('maximum_filter1d', [x2], {'size': 2, 'axis': 1}, []), call_nd('maximum_filter1d', [x1], {'size': 3, 'origin': -1}, [])]

# n-D separable filters (gaussian_filter, uniform_filter)
gf = [call_nd('gaussian_filter', [x2], {'sigma': s, 'mode': m}, []) for s in (1.0, 2.0) for m in ('reflect', 'nearest')] + [
    call_nd('gaussian_filter', [x2], {'sigma': [1.0, 2.0]}, []),
    call_nd('gaussian_filter', [x2], {'sigma': 1.5, 'truncate': 2.0}, []),
    call_nd('gaussian_filter', [x1], {'sigma': 1.0}, [])]
uf = [call_nd('uniform_filter', [x2], {'size': 3, 'mode': m}, []) for m in ('reflect', 'nearest', 'wrap')] + [
    call_nd('uniform_filter', [x2], {'size': [2, 3]}, []),
    call_nd('uniform_filter', [x2], {'size': 3, 'origin': 1}, []),
    call_nd('uniform_filter', [x1], {'size': 3}, [])]
# n-D order-statistic filters (box footprint): minimum/median/rank/percentile (maximum generalized to n-D)
minf = [call_nd('minimum_filter', [x2], {'size': sz, 'mode': m}, []) for sz in (2, 3) for m in ('reflect', 'nearest', 'constant')] + [
    call_nd('minimum_filter', [x2], {'size': [2, 3]}, []), call_nd('minimum_filter', [x1], {'size': 3}, [])]
maxf = [call_nd('maximum_filter', [x2], {'size': sz, 'mode': m}, []) for sz in (2, 3) for m in ('reflect', 'wrap')] + [
    call_nd('maximum_filter', [x2], {'size': [3, 2]}, []), call_nd('maximum_filter', [x1], {'size': 3}, [])]
medf = [call_nd('median_filter', [x2], {'size': sz, 'mode': m}, []) for sz in (2, 3) for m in ('reflect', 'nearest')] + [
    call_nd('median_filter', [x2], {'size': 3, 'origin': 1}, []), call_nd('median_filter', [x1], {'size': 4}, [])]
rankf = [call_nd('rank_filter', [x2, r], {'size': 3}, []) for r in (0, 4, 8, -1)] + [
    call_nd('rank_filter', [x1, 1], {'size': 3, 'mode': 'wrap'}, []), call_nd('rank_filter', [x2, 2], {'size': [2, 3]}, [])]
pctf = [call_nd('percentile_filter', [x2, p], {'size': 3}, []) for p in (0, 25, 50, 75, 100)] + [
    call_nd('percentile_filter', [x1, 30], {'size': 4}, []), call_nd('percentile_filter', [x2, -20], {'size': 3}, [])]
# greyscale morphology (flat box SE): grey_erosion = min filter, grey_dilation = max filter with reflected footprint
greyf = [call_nd('grey_erosion', [x2], {'size': sz, 'mode': m}, []) for sz in (2, 3) for m in ('reflect', 'nearest')] + [
    call_nd('grey_erosion', [x2], {'size': [2, 3]}, []), call_nd('grey_erosion', [x1], {'size': 3}, [])]
greyd = [call_nd('grey_dilation', [x2], {'size': sz, 'mode': m}, []) for sz in (2, 3) for m in ('reflect', 'nearest')] + [
    call_nd('grey_dilation', [x2], {'size': [3, 2]}, []), call_nd('grey_dilation', [x1], {'size': 3}, [])]
greyo = [call_nd('grey_opening', [x2], {'size': sz, 'mode': m}, []) for sz in (2, 3) for m in ('reflect', 'nearest')] + [
    call_nd('grey_opening', [x1], {'size': 3}, [])]
greyc = [call_nd('grey_closing', [x2], {'size': sz, 'mode': m}, []) for sz in (2, 3) for m in ('reflect', 'nearest')] + [
    call_nd('grey_closing', [x1], {'size': 3}, [])]
# grey tophats + morphological gradient/laplace (compositions of grey erosion/dilation/opening/closing)
wth = [call_nd('white_tophat', [x2], {'size': sz, 'mode': m}, []) for sz in (2, 3) for m in ('reflect', 'nearest')] + [call_nd('white_tophat', [x1], {'size': 3}, [])]
bth = [call_nd('black_tophat', [x2], {'size': sz, 'mode': m}, []) for sz in (2, 3) for m in ('reflect', 'nearest')] + [call_nd('black_tophat', [x1], {'size': 3}, [])]
mgr = [call_nd('morphological_gradient', [x2], {'size': sz, 'mode': m}, []) for sz in (2, 3) for m in ('reflect', 'wrap')] + [call_nd('morphological_gradient', [x1], {'size': 3}, [])]
mla = [call_nd('morphological_laplace', [x2], {'size': sz, 'mode': m}, []) for sz in (2, 3) for m in ('reflect', 'nearest')] + [call_nd('morphological_laplace', [x1], {'size': 3}, [])]
# gradient / edge filters
sob = [call_nd('sobel', [x2], {'axis': ax, 'mode': m}, []) for ax in (0, 1) for m in ('reflect', 'nearest', 'constant')] + [
    call_nd('sobel', [x2], {}, [])]
pre = [call_nd('prewitt', [x2], {'axis': ax, 'mode': m}, []) for ax in (0, 1) for m in ('reflect', 'wrap')] + [
    call_nd('prewitt', [x2], {}, [])]
lapf = [call_nd('laplace', [x2], {'mode': m}, []) for m in ('reflect', 'nearest', 'wrap', 'constant')] + [
    call_nd('laplace', [x1], {}, [])]
# gaussian derivative edge filters: Laplacian and gradient magnitude built from Gaussian derivatives
glap = [call_nd('gaussian_laplace', [x2], {'sigma': s, 'mode': m}, []) for s in (1.0, 2.0) for m in ('reflect', 'nearest')] + [
    call_nd('gaussian_laplace', [x2], {'sigma': [1.0, 2.0]}, []),
    call_nd('gaussian_laplace', [x2], {'sigma': 1.5, 'truncate': 2.0}, []),
    call_nd('gaussian_laplace', [x1], {'sigma': 1.0}, [])]
ggm = [call_nd('gaussian_gradient_magnitude', [x2], {'sigma': s, 'mode': m}, []) for s in (1.0, 2.0) for m in ('reflect', 'nearest')] + [
    call_nd('gaussian_gradient_magnitude', [x2], {'sigma': [1.0, 2.0]}, []),
    call_nd('gaussian_gradient_magnitude', [x2], {'sigma': 1.5, 'truncate': 2.0}, []),
    call_nd('gaussian_gradient_magnitude', [x1], {'sigma': 1.0}, [])]
# labeled measurements: whole array, index=None (nonzero labels), scalar index, array index, background label 0
ma = rng.uniform(-5, 5, (3, 3))
mlab = np.array([[1, 1, 0], [2, 2, 0], [2, 0, 3]])
ma2 = rng.uniform(-5, 5, (4, 6))
mlab2 = np.array([[0, 1, 1, 2, 2, 0], [1, 1, 2, 2, 3, 3], [0, 0, 3, 3, 1, 1], [2, 2, 0, 0, 3, 3]])
def meas(fn, missing=True):
    cases = [call_nd(fn, [ma], {}, []), call_nd(fn, [ma, mlab], {}, []), call_nd(fn, [ma, mlab, 2], {}, []),
             call_nd(fn, [ma, mlab, [1, 2, 3]], {}, []), call_nd(fn, [ma2, mlab2, [1, 2, 3]], {}, []),
             call_nd(fn, [ma2, mlab2, 0], {}, [])]
    if missing:
        cases.append(call_nd(fn, [ma, mlab, [1, 2, 9]], {}, []))   # label 9 is absent -> empty-region behaviour
    return cases
msum, mmean, mvar = meas('sum_labels'), meas('mean'), meas('variance')
mstd, mmax, mmin = meas('standard_deviation'), meas('maximum'), meas('minimum')
mmed = meas('median', missing=False)   # scipy's missing-label median is an internal artifact; existing labels only
msm = meas('sum')                       # deprecated alias of sum_labels (same semantics)
# histogram: encode scipy's int counts as a float array (scalar/None index only)
def hist_case(a):
    r = np.asarray(nd.histogram(*a), dtype=float)
    return {'args': [F.enc(x) for x in a], 'kwargs': {}, 'expect': F.enc(r), 'compare': 'tol'}
hst = [hist_case([ma, -5.0, 5.0, 6]), hist_case([ma, -5.0, 5.0, 6, mlab]), hist_case([ma, -5.0, 5.0, 6, mlab, 2]),
       hist_case([ma2, -5.0, 5.0, 8])]
# extrema: (minimum, maximum, min_position, max_position); positions as float arrays
def extrema_case(a):
    mn, mx, mnp, mxp = nd.extrema(*a)
    exp = {'dict': {'minimum': F.enc_result_one(np.asarray(mn)), 'maximum': F.enc_result_one(np.asarray(mx)),
                    'min_position': F.enc(np.asarray(mnp, dtype=float)), 'max_position': F.enc(np.asarray(mxp, dtype=float))}}
    return {'args': [F.enc(x) for x in a], 'kwargs': {}, 'expect': exp, 'compare': 'tol'}
extr = [extrema_case([ma]), extrema_case([ma, mlab]), extrema_case([ma, mlab, 2]), extrema_case([ma, mlab, [1, 2, 3]])]
# find_objects: bounding box per dense label, encoded as an (L, ndim, 2) start/stop array
fo1 = np.array([[1, 1, 0, 2], [1, 0, 0, 2], [0, 0, 3, 3]])
fo2 = np.array([[0, 1, 1, 0, 0], [0, 1, 1, 0, 2], [3, 0, 0, 0, 2], [3, 3, 0, 0, 0]])
def find_objects_case(a):
    objs = nd.find_objects(*a)
    arr = np.zeros((len(objs), a[0].ndim, 2))
    for li, o in enumerate(objs):
        if o is None:
            continue
        for d, s in enumerate(o):
            arr[li, d, 0] = s.start
            arr[li, d, 1] = s.stop
    return {'args': [F.enc(x) for x in a], 'kwargs': {}, 'expect': F.enc(arr), 'compare': 'tol'}
fob = [find_objects_case([fo1]), find_objects_case([fo2]), find_objects_case([fo1, 3])]
# fourier filters: multiply an already-FFT'd (complex) input by the transform of a kernel
FX = np.fft.fftn(rng.uniform(-5, 5, (4, 6)))
fg = [call_nd('fourier_gaussian', [FX, s], {}, []) for s in (1.0, 2.0)] + [call_nd('fourier_gaussian', [FX, [1.0, 2.0]], {}, [])]
fu = [call_nd('fourier_uniform', [FX, s], {}, []) for s in (2.0, 3.0)] + [call_nd('fourier_uniform', [FX, [2.0, 3.0]], {}, [])]
fsh = [call_nd('fourier_shift', [FX, s], {}, []) for s in (1.0, 1.5)] + [call_nd('fourier_shift', [FX, [1.0, -2.0]], {}, [])]
FX1 = np.fft.fft(rng.uniform(-5, 5, 10))
FX3 = np.fft.fftn(rng.uniform(-5, 5, (3, 4, 5)))
fel = [call_nd('fourier_ellipsoid', [FX, s], {}, []) for s in (2.0, 3.0)] + [
    call_nd('fourier_ellipsoid', [FX, [2.0, 3.0]], {}, []), call_nd('fourier_ellipsoid', [FX1, 3.0], {}, []),
    call_nd('fourier_ellipsoid', [FX3, 2.0], {}, [])]
# exact Euclidean distance transform
de1 = np.array([[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]], float)
de2 = (rng.uniform(0, 1, (6, 7)) > 0.4).astype(float)
dte = [call_nd('distance_transform_edt', [de1], {}, []), call_nd('distance_transform_edt', [de2], {}, []),
       call_nd('distance_transform_edt', [de1], {'sampling': [2.0, 1.0]}, []),
       call_nd('distance_transform_edt', [np.array([0., 1, 1, 1, 1, 0, 1, 1])], {}, [])]
# map_coordinates (order 0/1): interpolate at given coordinates, various boundary modes
mc_in = rng.uniform(-5, 5, (5, 6))
mc_co = np.array([[0.5, 1.2, 3.0, 4.8, -1.3, 5.7], [0.5, 2.7, 1.0, 5.4, 2.0, 6.3]])  # avoid exact half-integer folds (order-0 ties)
mc = [call_nd('map_coordinates', [mc_in, mc_co], {'order': o, 'mode': m}, [])
      for o in (0, 1) for m in ('constant', 'nearest', 'reflect', 'mirror', 'wrap')] + [
    call_nd('map_coordinates', [mc_in, mc_co], {'order': 1, 'mode': 'constant', 'cval': 2.5}, []),
    call_nd('map_coordinates', [rng.uniform(-5, 5, 8), np.array([[0.3, 2.9, 7.4, -0.6]])], {'order': 1}, [])] + [
    call_nd('map_coordinates', [mc_in, mc_co], {'order': k, 'mode': 'mirror'}, []) for k in (2, 3, 4, 5)]
# B-spline prefilter (orders 2-5), mode mirror (the default; the fold modes are matched exactly)
sf_in = rng.uniform(-5, 5, (5, 6))
sfil1d = [call_nd('spline_filter1d', [sf_in], {'order': k}, []) for k in (2, 3, 4, 5)] + [
    call_nd('spline_filter1d', [sf_in], {'order': 3, 'axis': 0}, []), call_nd('spline_filter1d', [rng.uniform(-5, 5, 9)], {'order': 5}, [])]
sfil = [call_nd('spline_filter', [sf_in], {'order': k}, []) for k in (2, 3, 4, 5)]
# geometric transforms (order 0/1): shift, affine_transform, zoom
gt_in = rng.uniform(-5, 5, (5, 6))
gshift = [call_nd('shift', [gt_in, [1.3, -0.7]], {'order': o, 'mode': m}, []) for o in (0, 1) for m in ('constant', 'nearest', 'reflect', 'mirror', 'wrap')] + [
    call_nd('shift', [gt_in, 1.5], {'order': 1, 'cval': 2.0}, [])] + [
    call_nd('shift', [gt_in, [1.3, -0.7]], {'order': 3, 'mode': 'mirror'}, [])]
gaff = [call_nd('affine_transform', [gt_in, [[0.9, 0.1], [-0.2, 1.1]], [0.5, -0.3]], {'order': o, 'mode': m}, []) for o in (0, 1) for m in ('constant', 'nearest', 'reflect')] + [
    call_nd('affine_transform', [gt_in, [1.5, 0.5], [0.2, 0.1]], {'order': 1}, []),
    call_nd('affine_transform', [gt_in, [[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0]], {'order': 1, 'output_shape': [3, 4]}, [])] + [
    call_nd('affine_transform', [gt_in, [[0.9, 0.1], [-0.2, 1.1]], [0.5, -0.3]], {'order': 3, 'mode': 'mirror'}, [])]
gzoom = [call_nd('zoom', [gt_in, z], {'order': o, 'mode': m}, []) for z in (1.5, 2.0) for o in (0, 1) for m in ('reflect', 'nearest')] + [
    call_nd('zoom', [gt_in, [2.0, 1.5]], {'order': 1}, []), call_nd('zoom', [rng.uniform(-5, 5, 7), 2.0], {'order': 1}, [])] + [
    call_nd('zoom', [gt_in, z], {'order': 3, 'mode': 'mirror'}, []) for z in (1.5, 2.0)]
grot = [call_nd('rotate', [gt_in, ang], {'order': 1, 'mode': m}, []) for ang in (30.0, -45.0, 90.0) for m in ('constant', 'reflect', 'nearest')] + [
    call_nd('rotate', [gt_in, 90.0], {'order': 0}, []), call_nd('rotate', [gt_in, 180.0], {'order': 0}, []),
    call_nd('rotate', [gt_in, 30.0], {'order': 1, 'reshape': False}, []),
    call_nd('rotate', [gt_in, 30.0], {'order': 3, 'mode': 'mirror'}, [])]
def dist_case(fn, a, kw):
    r = np.asarray(getattr(nd, fn)(*a, **(kw or {})), dtype=float)
    return {'args': [F.enc(x) for x in a], 'kwargs': {k: F.enc(v) for k, v in (kw or {}).items()}, 'expect': F.enc(r), 'compare': 'tol'}
dbf = [dist_case('distance_transform_bf', [de1], {}), dist_case('distance_transform_bf', [de1], {'metric': 'taxicab'}),
       dist_case('distance_transform_bf', [de2], {'metric': 'chessboard'}), dist_case('distance_transform_bf', [de1], {'sampling': [2.0, 1.0]})]
dcdt = [dist_case('distance_transform_cdt', [de1], {}), dist_case('distance_transform_cdt', [de1], {'metric': 'taxicab'}),
        dist_case('distance_transform_cdt', [de2], {'metric': 'chessboard'})]
# coordinate measurements: center_of_mass + max/min position. Encode the coord tuple/list-of-tuples as a plain
# float array (np.asarray) so the fixture carries an array, not scipy's tuple. Existing labels only.
pa, pa2 = np.abs(ma) + 0.5, np.abs(ma2) + 0.5   # positive weights so the centroid (and argmax/argmin) are well-posed
def meas_coord(fn):
    f = getattr(nd, fn)
    def one(args, kwargs):
        r = np.asarray(f(*args, **(kwargs or {})), dtype=float)
        return {'args': [F.enc(a) for a in args], 'kwargs': {k: F.enc(v) for k, v in (kwargs or {}).items()},
                'expect': F.enc(r), 'compare': 'tol'}
    return [one([pa], {}), one([pa, mlab], {}), one([pa, mlab, 2], {}), one([pa, mlab, [1, 2, 3]], {}),
            one([pa2, mlab2, [1, 2, 3]], {}), one([np.abs(x1) + 0.5], {})]
com = meas_coord('center_of_mass')
maxp = meas_coord('maximum_position')
minp = meas_coord('minimum_position')

out = {'module': 'ndimage', 'scipy': __import__('scipy').__version__, 'env': F.fixture_env.env(),
       'calls': [{'fn': 'generate_binary_structure', 'cases': gbs}, {'fn': 'label', 'cases': lab}, {'fn': 'convolve', 'cases': conv},
                 {'fn': 'correlate', 'cases': cor}, {'fn': 'sum', 'cases': msm},
                 {'fn': 'binary_dilation', 'cases': dil}, {'fn': 'binary_erosion', 'cases': ero}, {'fn': 'maximum_filter', 'cases': mxf + maxf},
                 {'fn': 'binary_opening', 'cases': bop}, {'fn': 'binary_closing', 'cases': bcl},
                 {'fn': 'binary_hit_or_miss', 'cases': bhm}, {'fn': 'iterate_structure', 'cases': itr},
                 {'fn': 'binary_fill_holes', 'cases': bfh}, {'fn': 'binary_propagation', 'cases': bpr},
                 {'fn': 'correlate1d', 'cases': corr1}, {'fn': 'convolve1d', 'cases': conv1},
                 {'fn': 'uniform_filter1d', 'cases': unif1}, {'fn': 'gaussian_filter1d', 'cases': gauss1},
                 {'fn': 'minimum_filter1d', 'cases': min1}, {'fn': 'maximum_filter1d', 'cases': max1},
                 {'fn': 'gaussian_filter', 'cases': gf}, {'fn': 'uniform_filter', 'cases': uf},
                 {'fn': 'minimum_filter', 'cases': minf}, {'fn': 'median_filter', 'cases': medf},
                 {'fn': 'rank_filter', 'cases': rankf}, {'fn': 'percentile_filter', 'cases': pctf},
                 {'fn': 'grey_erosion', 'cases': greyf}, {'fn': 'grey_dilation', 'cases': greyd},
                 {'fn': 'grey_opening', 'cases': greyo}, {'fn': 'grey_closing', 'cases': greyc},
                 {'fn': 'white_tophat', 'cases': wth}, {'fn': 'black_tophat', 'cases': bth},
                 {'fn': 'morphological_gradient', 'cases': mgr}, {'fn': 'morphological_laplace', 'cases': mla},
                 {'fn': 'sobel', 'cases': sob}, {'fn': 'prewitt', 'cases': pre}, {'fn': 'laplace', 'cases': lapf},
                 {'fn': 'gaussian_laplace', 'cases': glap}, {'fn': 'gaussian_gradient_magnitude', 'cases': ggm},
                 {'fn': 'sum_labels', 'cases': msum}, {'fn': 'mean', 'cases': mmean}, {'fn': 'variance', 'cases': mvar},
                 {'fn': 'standard_deviation', 'cases': mstd}, {'fn': 'maximum', 'cases': mmax},
                 {'fn': 'minimum', 'cases': mmin}, {'fn': 'median', 'cases': mmed},
                 {'fn': 'center_of_mass', 'cases': com}, {'fn': 'maximum_position', 'cases': maxp},
                 {'fn': 'minimum_position', 'cases': minp},
                 {'fn': 'histogram', 'cases': hst}, {'fn': 'extrema', 'cases': extr},
                 {'fn': 'find_objects', 'cases': fob},
                 {'fn': 'fourier_gaussian', 'cases': fg}, {'fn': 'fourier_uniform', 'cases': fu},
                 {'fn': 'fourier_shift', 'cases': fsh}, {'fn': 'fourier_ellipsoid', 'cases': fel},
                 {'fn': 'distance_transform_edt', 'cases': dte},
                 {'fn': 'distance_transform_bf', 'cases': dbf}, {'fn': 'distance_transform_cdt', 'cases': dcdt},
                 {'fn': 'map_coordinates', 'cases': mc}, {'fn': 'shift', 'cases': gshift},
                 {'fn': 'affine_transform', 'cases': gaff}, {'fn': 'zoom', 'cases': gzoom},
                 {'fn': 'rotate', 'cases': grot}, {'fn': 'spline_filter1d', 'cases': sfil1d}, {'fn': 'spline_filter', 'cases': sfil}]}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'ndimage: order filters min {len(minf)} max {len(maxf)} med {len(medf)} rank {len(rankf)} pct {len(pctf)}')
