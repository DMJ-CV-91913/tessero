"""NumPy/SciPy timings for the cases of bench/tier2_bench.php (median of repeated runs, ms; same inputs, same seeds).

    OPENBLAS_NUM_THREADS=1 python3 bench/tier2_bench.py > scipy.json
"""
import json
import math
import os
import sys
import time

import numpy as np
import scipy.cluster.hierarchy as hi
import scipy.cluster.vq as vq
import scipy.fft as sf
import scipy.integrate as si
import scipy.interpolate as ip
import scipy.linalg as sl
import scipy.ndimage as nd
import scipy.optimize as so
import scipy.signal as sg
import scipy.sparse as sp
import scipy.sparse.csgraph as cg
import scipy.sparse.linalg as spl
import scipy.stats as st
from scipy.spatial import ConvexHull, Delaunay, KDTree


def med(f):
    s = time.perf_counter()
    f()
    first = (time.perf_counter() - s) * 1e3
    if first > 30000:
        return first  # over 30 s: one run says enough
    inner = int(max(1, min(500, 10.0 / max(first, 1e-4))))
    n = 3 if first > 500 else 9
    t = []
    for _ in range(n):
        s = time.perf_counter()
        for _ in range(inner):
            f()
        t.append((time.perf_counter() - s) * 1e3 / inner)
    t.sort()
    return t[n // 2]


def r(seed, size):
    return np.random.default_rng(seed).random(size)


x = r(10, 1000000) * 10.0
y = r(11, 1000000)
xi = np.floor(r(12, 1000000) * 1000.0)
s100k = np.sort(r(13, 100000))
t100k = r(14, 100000)
y100k = r(15, 100000)
k101 = r(16, 101)
s10k = r(17, 10000)
xp = np.linspace(0.0, 1.0, 1000)
fp = r(18, 1000)
C = r(19, (50, 100000))
A = r(20, (600, 600)) + np.eye(600) * 600.0
M3 = r(21, (300, 300))
M4 = r(22, (300, 300))
img1024 = r(23, (1024, 1024))
img1k = r(24, (1000, 1000))
img512 = r(25, (512, 512))
img500 = r(26, (500, 500))
k5x5 = r(27, (5, 5))
bin1k = img1k > 0.7
bin512 = img512 > 0.3
coords = r(28, (2, 100000)) * 999.0
xk = np.linspace(0.0, 100.0, 10000)
yk = np.sin(xk)
xnew = r(29, 1000000) * 100.0
rbfY = r(30, (500, 2))
rbfD = np.sin(rbfY[:, 0] * 3.0)
rbfX = r(31, (10000, 2))
sos = sg.tf2sos(*sg.butter(8, 0.1))  # as tier2_bench.php: tf2sos(butter(8, 0.1))
bb, ba = sg.butter(4, 0.1)
kde = r(32, 1000)
kpts = np.linspace(-0.5, 1.5, 1000)
ta = r(33, 100000)
tb = r(34, 100000) + 0.01
k10a = r(35, 10000)
k10b = r(36, 10000)
Q = r(37, (500, 500))
L1 = r(38, (1000, 100))
L2 = r(39, 1000)
S1k = r(40, (1000, 1000))
SPD = S1k @ S1k.T + np.eye(1000) * 1000.0
U = np.triu(r(41, (1000, 1000)) + np.eye(1000) * 10.0)
ub = r(42, 1000)
n = 100000
tri = sp.diags([np.full(n - 1, -1.0), np.full(n, 4.0), np.full(n - 1, -1.0)], [-1, 0, 1], shape=(n, n), format='csr')
tb100k = r(43, n)
# 2-D Poisson on a 100 x 100 grid: five diagonals, the +-1 ones cut at the grid rows
off1 = np.array([0.0 if (i + 1) % 100 == 0 else -1.0 for i in range(9999)])
poisson = sp.diags([np.full(9900, -1.0), off1, np.full(10000, 4.0), off1, np.full(9900, -1.0)], [-100, -1, 0, 1, 100], shape=(10000, 10000), format='csr')
pb = r(44, 10000)
gw = r(45, (500, 500))
graph = gw * (gw > 0.9)
pts3 = r(46, (100000, 3))
qpts = r(47, (100, 3))
tree = KDTree(pts3)
pts2 = r(48, (10000, 2))
km = r(49, (10000, 3))
kinit = r(50, (8, 3))
lk = r(51, (2000, 3))
x0_5 = [1.3, 0.7, 0.8, 1.9, 1.2]
x0_50 = [0.5] * 50
cfx = np.linspace(0.0, 4.0, 1000)
cfy = 2.5 * np.exp(-1.3 * cfx) + 0.5 + 0.01 * np.sin(37.0 * cfx)
teval = np.linspace(0.0, 15.0, 100)


def rosen(v):
    v = np.asarray(v)
    return float(np.sum(100.0 * (v[1:] - v[:-1] ** 2) ** 2 + (1 - v[:-1]) ** 2))


cases = {
    'numpy cumsum 1M': lambda: np.cumsum(x),
    'numpy argsort 1M': lambda: np.argsort(x),
    'numpy unique 1M (1000 values)': lambda: np.unique(xi),
    'numpy searchsorted 1M into 100k': lambda: np.searchsorted(s100k, y),
    'numpy percentile 1M': lambda: np.percentile(x, 90),
    'numpy median 1M': lambda: np.median(x),
    'numpy histogram 1M, 100 bins': lambda: np.histogram(x, 100),
    'numpy einsum ij,jk->ik 300x300': lambda: np.einsum('ij,jk->ik', M3, M4),
    'numpy polyfit deg 5, 100k': lambda: np.polyfit(t100k, y100k, 5),
    'numpy convolve 100k * 101': lambda: np.convolve(t100k, k101),
    'numpy interp 1M into 1000': lambda: np.interp(y, xp, fp),
    'numpy diff 1M': lambda: np.diff(x),
    'numpy clip 1M': lambda: np.clip(x, 2.0, 8.0),
    'numpy cov 50 x 100k': lambda: np.cov(C),
    'numpy linalg slogdet 600x600': lambda: np.linalg.slogdet(A),
    'numpy fft2 1024x1024': lambda: np.fft.fft2(img1024),
    'numpy rfft2 1024x1024': lambda: np.fft.rfft2(img1024),
    'scipy.fft dct 1M': lambda: sf.dct(x),
    'scipy.signal sosfilt order 8, 1M': lambda: sg.sosfilt(sos, x),
    'scipy.signal filtfilt order 4, 1M': lambda: sg.filtfilt(bb, ba, x),
    'scipy.signal welch 1M, nperseg 1024': lambda: sg.welch(x, nperseg=1024),
    'scipy.signal resample 100k -> 50k': lambda: sg.resample(t100k, 50000),
    'scipy.signal resample_poly 1M, 3/2': lambda: sg.resample_poly(x, 3, 2),
    'scipy.signal medfilt 100k, k=5': lambda: sg.medfilt(t100k, 5),
    'scipy.signal savgol_filter 1M (31, 3)': lambda: sg.savgol_filter(x, 31, 3),
    'scipy.signal correlate 10k * 101': lambda: sg.correlate(s10k, k101),
    'scipy.signal hilbert 1M': lambda: sg.hilbert(x),
    'scipy.signal find_peaks 1M': lambda: sg.find_peaks(y),
    'scipy.signal detrend 1M': lambda: sg.detrend(x),
    'scipy.signal decimate 1M by 10': lambda: sg.decimate(x, 10),
    'scipy.signal convolve2d 500x500 * 5x5': lambda: sg.convolve2d(img500, k5x5),
    'scipy.ndimage median_filter 512x512, 3': lambda: nd.median_filter(img512, 3),
    'scipy.ndimage sobel 1000x1000': lambda: nd.sobel(img1k),
    'scipy.ndimage laplace 1000x1000': lambda: nd.laplace(img1k),
    'scipy.ndimage maximum_filter 1000x1000, 5': lambda: nd.maximum_filter(img1k, 5),
    'scipy.ndimage label 1000x1000': lambda: nd.label(bin1k),
    'scipy.ndimage binary_erosion 1000x1000': lambda: nd.binary_erosion(bin1k),
    'scipy.ndimage distance_transform_edt 512x512': lambda: nd.distance_transform_edt(bin512),
    'scipy.ndimage zoom 500x500 x2': lambda: nd.zoom(img500, 2.0),
    'scipy.ndimage rotate 512x512, 30 deg': lambda: nd.rotate(img512, 30.0),
    'scipy.ndimage map_coordinates 1000x1000, 100k points': lambda: nd.map_coordinates(img1k, coords),
    'scipy.interpolate CubicSpline 10k knots, 1M points': lambda: ip.CubicSpline(xk, yk)(xnew),
    'scipy.interpolate make_interp_spline k=3, 1M points': lambda: ip.make_interp_spline(xk, yk, 3)(xnew),
    'scipy.interpolate PchipInterpolator 10k knots, 1M points': lambda: ip.PchipInterpolator(xk, yk)(xnew),
    'scipy.interpolate RBFInterpolator 500 -> 10k (2-D)': lambda: ip.RBFInterpolator(rbfY, rbfD)(rbfX),
    'scipy.integrate quad sin(x) exp(-x) on [0, 10]': lambda: si.quad(lambda t: math.sin(t) * math.exp(-t), 0.0, 10.0),
    'scipy.integrate solve_ivp Lotka-Volterra RK45': lambda: si.solve_ivp(
        lambda t, v: [1.5 * v[0] - v[0] * v[1], -3.0 * v[1] + v[0] * v[1]], [0.0, 15.0], [10.0, 5.0], t_eval=teval),
    'numpy trapezoid 1M': lambda: np.trapezoid(x),
    'scipy.optimize least_squares Rosenbrock residuals': lambda: so.least_squares(lambda v: [10.0 * (v[1] - v[0] ** 2), 1.0 - v[0]], [-1.2, 1.0]),
    'scipy.optimize curve_fit exp decay, 1000 points': lambda: so.curve_fit(lambda t, a, b, c: a * np.exp(-b * t) + c, cfx, cfy, p0=[1.0, 1.0, 1.0]),
    'scipy.optimize minimize Nelder-Mead rosen n=5': lambda: so.minimize(rosen, x0_5, method='Nelder-Mead'),
    'scipy.optimize minimize L-BFGS-B rosen n=50': lambda: so.minimize(rosen, x0_50, method='L-BFGS-B'),
    'scipy.optimize root (2 equations)': lambda: so.root(lambda v: [v[0] + 0.5 * (v[0] - v[1]) ** 3 - 1.0, 0.5 * (v[1] - v[0]) ** 3 + v[1]], [0.0, 0.0]),
    'scipy.stats spearmanr 100k': lambda: st.spearmanr(ta, tb),
    'scipy.stats kendalltau 10k': lambda: st.kendalltau(k10a, k10b),
    'scipy.stats mannwhitneyu 100k vs 100k': lambda: st.mannwhitneyu(ta, tb),
    'scipy.stats gaussian_kde 1000 points, 1000 evaluations': lambda: st.gaussian_kde(kde)(kpts),
    'scipy.stats rankdata 1M': lambda: st.rankdata(x),
    'scipy.stats ks_2samp 100k vs 100k': lambda: st.ks_2samp(ta, tb),
    'scipy.stats skew 1M': lambda: st.skew(x),
    'scipy.stats kurtosis 1M': lambda: st.kurtosis(x),
    'scipy.stats linregress 1M': lambda: st.linregress(x, y),
    'scipy.stats binned_statistic 1M, 100 bins': lambda: st.binned_statistic(x, y, 'mean', 100),
    'scipy.stats norm.ppf 1M': lambda: st.norm.ppf(y),
    'scipy.linalg qr 500x500': lambda: sl.qr(Q),
    'scipy.linalg lstsq 1000x100': lambda: sl.lstsq(L1, L2),
    'scipy.linalg cholesky 1000x1000': lambda: sl.cholesky(SPD),
    'scipy.linalg eig 300x300 (general)': lambda: sl.eig(M3),
    'scipy.linalg schur 300x300': lambda: sl.schur(M3),
    'scipy.linalg pinv 300x300': lambda: sl.pinv(M4),
    'scipy.linalg solve_triangular 1000x1000': lambda: sl.solve_triangular(U, ub),
    'scipy.linalg svdvals 300x300': lambda: sl.svdvals(M4),
    'scipy.sparse.linalg spsolve tridiagonal 100k': lambda: spl.spsolve(tri, tb100k),
    'scipy.sparse.linalg cg 2-D Poisson 100x100': lambda: spl.cg(poisson, pb),
    'scipy.sparse.linalg eigsh k=6, 2-D Poisson 100x100': lambda: spl.eigsh(poisson, 6),
    'scipy.sparse.csgraph shortest_path 500 nodes (Dijkstra)': lambda: cg.shortest_path(graph, 'D'),
    'scipy.sparse.csgraph minimum_spanning_tree 500 nodes': lambda: cg.minimum_spanning_tree(graph),
    'scipy.sparse.csgraph connected_components 500 nodes': lambda: cg.connected_components(graph),
    'scipy.spatial KDTree build 100k (3-D)': lambda: KDTree(pts3),
    'scipy.spatial KDTree query 100 points, k=5 (Tessero: exhaustive search)': lambda: tree.query(qpts, 5),
    'scipy.spatial ConvexHull 10k (2-D)': lambda: ConvexHull(pts2),
    'scipy.spatial Delaunay 10k (2-D)': lambda: Delaunay(pts2),
    'scipy.cluster.vq kmeans2 10k x 3, k=8': lambda: vq.kmeans2(km, kinit, minit='matrix'),
    'scipy.cluster.hierarchy linkage ward 2000x3': lambda: hi.linkage(lk, 'ward'),
}

only = sys.argv[1] if len(sys.argv) > 1 else None
out = {}
for k, f in cases.items():
    if only is not None and only not in k:
        continue
    try:
        out[k] = (f(), 0.0)[1] if os.environ.get('DRY') else med(f)   # DRY=1: call once, report errors only
    except Exception as e:  # noqa: BLE001 - recorded per case
        out[k] = f'ERR {type(e).__name__}: {str(e)[:160]}'
print(json.dumps(out, indent=4))
