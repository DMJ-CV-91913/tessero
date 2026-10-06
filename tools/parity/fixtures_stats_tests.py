#!/usr/bin/env python3
"""SciPy reference fixtures for the scipy.stats hypothesis tests (group "tests").

    python3 tools/parity/fixtures_stats_tests.py

Writes tests/fixtures/parity/stats_tests.json (see fixtures_stats_fn.py for the format).
"""
import numpy as np

import fixtures_stats_fn as S
from fixtures_stats_fn import call, data

TT = ['statistic', 'pvalue', 'df']
SP = ['statistic', 'pvalue']
ALTS = ['two-sided', 'less', 'greater']


def nanify(a, p=0.15):
    a = np.array(a, dtype=float)
    m = S.rng.uniform(size=a.shape) < p
    a[m] = np.nan
    return a


def axis_cases(fn, make, names, extra=None, nan=True, keep=True):
    """Common axis / keepdims / nan_policy cases for a decorated test; make(shape) -> positional args."""
    extra = extra or {}
    out = []
    out.append(call(fn, make((12,)), dict(extra), names))
    out.append(call(fn, make((10, 3)), dict(extra, axis=0), names))
    out.append(call(fn, make((3, 10)), dict(extra, axis=1), names))
    out.append(call(fn, make((3, 10)), dict(extra, axis=-1), names))
    out.append(call(fn, make((4, 9)), dict(extra, axis=None), names))
    out.append(call(fn, make((2, 9, 3)), dict(extra, axis=1), names))
    if keep:
        out.append(call(fn, make((9, 3)), dict(extra, axis=0, keepdims=True), names))
        out.append(call(fn, make((11,)), dict(extra, keepdims=True), names))
        out.append(call(fn, make((3, 9)), dict(extra, axis=None, keepdims=True), names))
    if nan:
        for pol in ['propagate', 'omit', 'raise']:
            args = [nanify(a) if isinstance(a, np.ndarray) and a.ndim >= 1 else a for a in make((14,))]
            out.append(call(fn, args, dict(extra, nan_policy=pol), names))
            args = [nanify(a, 0.1) if isinstance(a, np.ndarray) and a.ndim >= 1 else a for a in make((12, 4))]
            out.append(call(fn, args, dict(extra, axis=0, nan_policy=pol), names))
            args = [nanify(a, 0.1) if isinstance(a, np.ndarray) and a.ndim >= 1 else a for a in make((4, 12))]
            out.append(call(fn, args, dict(extra, axis=1, nan_policy=pol), names))
    return out


def ttests():
    B = {}
    c = []
    for alt in ALTS:
        c.append(call('ttest_1samp', [data((15,)), 0.5], {'alternative': alt}, TT))
    c += axis_cases('ttest_1samp', lambda s: [data(s), 0.3], TT)
    c.append(call('ttest_1samp', [data((10, 3)), np.array([0.1, 0.2, -0.3])], {'axis': 0}, TT))
    c.append(call('ttest_1samp', [data((3, 10)), np.array([[0.1], [0.2], [-0.3]])], {'axis': 1}, TT))
    c.append(call('ttest_1samp', [data((10, 3)), np.array([0.1, 0.2])], {'axis': 0}, TT))
    c.append(call('ttest_1samp', [data((7,), ints=True), 1], {}, TT))
    for n in (1, 2, 3):
        c.append(call('ttest_1samp', [data((n,)), 0.0], {}, TT))
    c.append(call('ttest_1samp', [data((1, 4)), 0.0], {'axis': 0}, TT))
    c.append(call('ttest_1samp', [np.array([2.0, 2.0, 2.0]), 2.0], {}, TT))
    c.append(call('ttest_1samp', [np.array([2.0, 2.0, 2.0]), 1.0], {}, TT))
    c.append(call('ttest_1samp', [data((5,)), 0.0], {'alternative': 'bogus'}, TT))
    c.append(call('ttest_1samp', [data((5,)), 0.0], {'nan_policy': 'bogus'}, TT))
    c.append(call('ttest_1samp', [data((5, 2)), 0.0], {'axis': 2}, TT))
    c.append(call('ttest_1samp', [np.array([np.nan, 1.0]), 0.0], {'nan_policy': 'omit'}, TT))
    c.append(call('ttest_1samp', [np.array([np.nan, np.nan]), 0.0], {'nan_policy': 'omit'}, TT))
    B['ttest_1samp'] = c

    c = []
    for ev in (True, False):
        for alt in ALTS:
            c.append(call('ttest_ind', [data((13,)), data((9,)) + 1], {'equal_var': ev, 'alternative': alt}, TT))
        c += axis_cases('ttest_ind', lambda s: [data(s), data(s) + 0.5], TT, {'equal_var': ev})
        c.append(call('ttest_ind', [data((8, 3)), data((5, 3))], {'axis': 0, 'equal_var': ev}, TT))
        c.append(call('ttest_ind', [data((3, 8)), data((1, 5))], {'axis': 1, 'equal_var': ev}, TT))
        c.append(call('ttest_ind', [data((1,)), data((4,))], {'equal_var': ev}, TT))
        c.append(call('ttest_ind', [data((2,)), data((2,))], {'equal_var': ev}, TT))
        c.append(call('ttest_ind', [data((1,)), data((1,))], {'equal_var': ev}, TT))
        c.append(call('ttest_ind', [np.array([1., 1., 1.]), np.array([1., 1.])], {'equal_var': ev}, TT))
        c.append(call('ttest_ind', [data((6,), ints=True), data((7,), ints=True)], {'equal_var': ev}, TT))
        for trim in (0.1, 0.2, 0.3):
            c.append(call('ttest_ind', [data((14,)), data((11,)) + 2], {'equal_var': ev, 'trim': trim}, TT))
            c.append(call('ttest_ind', [data((12, 3)), data((9, 3))], {'equal_var': ev, 'trim': trim, 'axis': 0}, TT))
    c.append(call('ttest_ind', [data((5,)), data((5,))], {'trim': 0.5}, TT))
    B['ttest_ind'] = c

    c = []
    for alt in ALTS:
        c.append(call('ttest_rel', [data((12,)), data((12,)) + 0.4], {'alternative': alt}, TT))
    c += axis_cases('ttest_rel', lambda s: [data(s), data(s) + 0.2], TT)
    c.append(call('ttest_rel', [data((6, 3)), data((6, 1))], {'axis': 0}, TT))
    c.append(call('ttest_rel', [data((5,)), data((4,))], {}, TT))
    c.append(call('ttest_rel', [data((2,)), data((2,))], {}, TT))
    c.append(call('ttest_rel', [data((6,), ints=True), data((6,), ints=True)], {}, TT))
    B['ttest_rel'] = c

    c = []
    for ev in (True, False):
        for alt in ALTS:
            c.append(call('ttest_ind_from_stats', [15.0, np.sqrt(87.5), 13, 12.0, np.sqrt(39.0), 11],
                          {'equal_var': ev, 'alternative': alt}, SP))
        c.append(call('ttest_ind_from_stats', [0.2, np.sqrt(0.161073), 150, 0.225, np.sqrt(0.175251), 200], {'equal_var': ev}, SP))
        c.append(call('ttest_ind_from_stats', [data((4,)), np.abs(data((4,))) + 1, 10, data((3, 1)), 2.0, 7],
                      {'equal_var': ev}, SP))
        c.append(call('ttest_ind_from_stats', [1.0, 2.0, 1, 3.0, 1.5, 5], {'equal_var': ev}, SP))
        c.append(call('ttest_ind_from_stats', [1.0, 0.0, 4, 1.0, 0.0, 5], {'equal_var': ev}, SP))
    B['ttest_ind_from_stats'] = c
    return B


def chisq():
    B = {}
    lam_names = ['pearson', 'log-likelihood', 'freeman-tukey', 'mod-log-likelihood', 'neyman', 'cressie-read']
    obs = np.array([16, 18, 16, 14, 12, 12])
    fexp = np.array([16, 16, 16, 16, 16, 8])
    counts = lambda s: [S.rng.integers(1, 30, size=s).astype(float)]
    c = []
    c.append(call('chisquare', [obs], {}, SP))
    c.append(call('chisquare', [obs, fexp], {}, SP))
    c.append(call('chisquare', [obs], {'ddof': 1}, SP))
    c.append(call('chisquare', [obs], {'ddof': 2}, SP))
    c.append(call('chisquare', [obs, np.array([16, 16, 16, 16, 16, 9])], {}, SP))
    c.append(call('chisquare', [obs, np.array([16, 16, 16, 16, 16, 9])], {'sum_check': False}, SP))
    c.append(call('chisquare', [obs, np.array([[16, 16, 16, 16, 16, 8], [8, 20, 20, 16, 12, 12]])], {'axis': 1}, SP))
    c.append(call('chisquare', [np.array([[16, 18], [16, 14], [12, 12]])], {}, SP))
    c.append(call('chisquare', [np.array([[16, 18], [16, 14], [12, 12]])], {'axis': None}, SP))
    c.append(call('chisquare', [np.array([[16, 18], [16, 14], [12, 12]])], {'axis': 1}, SP))
    c.append(call('chisquare', [np.array([5.0])], {}, SP))
    c.append(call('chisquare', [np.array([], dtype=float)], {}, SP))
    c += axis_cases('chisquare', counts, SP)
    c += axis_cases('chisquare', lambda s: [x := S.rng.integers(1, 30, size=s).astype(float), x[::-1].copy() if x.ndim == 1 else x], SP)
    B['chisquare'] = c

    c = []
    for lam in lam_names + [None, 1, 0, -1, 0.5, 2, 2.0 / 3, -0.5, 1.7, -2, 3]:
        # lambda_ positionally (f_obs, f_exp, ddof, axis, lambda_): run-tests.php's camelName('lambda_') is
        # 'lambda' while the façade parameter is $lambda_ (gen-facades.php keeps the trailing underscore)
        c.append(call('power_divergence', [obs, None, 0, 0, lam], {}, SP))
        c.append(call('power_divergence', [obs, fexp, 0, 0, lam], {}, SP))
        c.append(call('power_divergence', [counts((5, 3))[0], None, 0, 0, lam], {}, SP))
        c.append(call('power_divergence', [counts((3, 5))[0], None, 0, 1, lam], {'keepdims': True}, SP))
        c.append(call('power_divergence', [counts((3, 5))[0], None, 0, None, lam], {}, SP))
        c.append(call('power_divergence', [nanify(counts((12,))[0]), None, 0, 0, lam], {'nan_policy': 'omit'}, SP))
        c.append(call('power_divergence', [nanify(counts((10, 3))[0], 0.1), None, 0, 0, lam], {'nan_policy': 'omit'}, SP))
    c.append(call('power_divergence', [obs, None, 0, 0, 'bogus'], {}, SP))
    c.append(call('power_divergence', [np.array([4, 0, 3]), None, 0, 0, 'log-likelihood'], {}, SP))
    c.append(call('power_divergence', [np.array([4, 0, 3]), None, 0, 0, 'mod-log-likelihood'], {}, SP))
    c.append(call('power_divergence', [obs, None, 1, 0, 'cressie-read'], {}, SP))
    c.append(call('power_divergence', [obs], {'ddof': np.array([0, 1, 2])}, SP))
    c.append(call('power_divergence', [obs], {'ddof': np.array([[0], [1]]), 'keepdims': True}, SP))
    c.append(call('power_divergence', [counts((5, 3))[0]], {'ddof': np.array([0, 1, 2]), 'axis': 0}, SP))
    c.append(call('power_divergence', [counts((5, 3))[0]], {'ddof': np.array([[0], [1]]), 'axis': 0, 'keepdims': True}, SP))
    c.append(call('power_divergence', [counts((5, 3))[0]], {'ddof': np.array([0, 1]), 'axis': 0}, SP))
    c.append(call('power_divergence', [counts((3, 4))[0]], {'ddof': np.array([1, 2]), 'axis': None}, SP))
    c.append(call('power_divergence', [nanify(counts((8,))[0], 0.3)], {'ddof': np.array([0, 1]), 'nan_policy': 'omit'}, SP))
    c.append(call('power_divergence', [nanify(counts((8,))[0], 0.3)], {'ddof': np.array([0, 1])}, SP))
    c.append(call('chisquare', [obs], {'ddof': np.array([0, 1, 2])}, SP))
    c += axis_cases('power_divergence', counts, SP)
    B['power_divergence'] = c

    CC = ['statistic', 'pvalue', 'dof', 'expected_freq']
    c = []
    tables = [np.array([[10, 10, 20], [20, 20, 20]]), np.array([[12, 3], [5, 17]]), np.array([[3, 4], [5, 9]]),
              np.array([[1, 1], [1, 1]]), np.array([3, 4, 5]), np.array([[3, 4, 5]]),
              S.rng.integers(1, 20, size=(3, 4)), S.rng.integers(1, 20, size=(2, 3, 2)),
              np.array([[10.5, 3.2], [4.1, 7.7]]), np.array([[0, 2], [0, 3]]), np.array([[1, -2], [3, 4]]),
              np.array([[100, 1], [2, 90]]), np.array([[5, 5], [5, 5]])]
    for t in tables:
        for corr in (True, False):
            c.append(call('chi2_contingency', [t], {'correction': corr}, CC))
    for lam in lam_names + [0.5, 2]:
        c.append(call('chi2_contingency', [tables[0], True, lam], {}, CC))
        c.append(call('chi2_contingency', [tables[1], True, lam], {}, CC))
        c.append(call('chi2_contingency', [tables[1], False, lam], {}, CC))
    c.append(call('chi2_contingency', [np.zeros((0, 2))], {}, CC))
    B['chi2_contingency'] = c
    return B


def exact():
    B = {}
    c = []
    tables = [[[8, 2], [1, 5]], [[3, 4], [5, 9]], [[1, 9], [11, 3]], [[0, 5], [5, 0]], [[0, 0], [3, 4]],
              [[10, 10], [10, 10]], [[100, 2], [1000, 5]], [[2, 7], [8, 2]], [[12, 5], [3, 20]],
              [[5, 1], [0, 3]], [[1, 0], [0, 1]], [[0, 1], [2, 3]], [[190, 800], [200, 900]],
              [[6, 3], [1, 10]], [[4, 9], [9, 4]]]
    for t in tables:
        for alt in ALTS:
            c.append(call('fisher_exact', [np.array(t)], {'alternative': alt}, SP))
    c.append(call('fisher_exact', [np.array([[1, 2], [3, 4]])], {}, SP))
    c.append(call('fisher_exact', [np.array([[1, 2], [3, 4]])], {'alternative': 'bogus'}, SP))
    c.append(call('fisher_exact', [np.array([[1, -2], [3, 4]])], {}, SP))
    c.append(call('fisher_exact', [np.array([1, 2, 3, 4])], {}, SP))
    c.append(call('fisher_exact', [np.array([[1.7, 2.2], [3.9, 4.0]])], {}, SP))
    c.append(call('fisher_exact', [np.array([[-0.5, 2.0], [3.0, 4.0]])], {}, SP))
    c.append(call('fisher_exact', [np.array([[1, 2, 3]])], {}, SP))
    c.append(call('fisher_exact', [np.array([[1], [2], [3]])], {}, SP))
    c.append(call('fisher_exact', [np.zeros((2, 3), dtype=int)], {}, SP))
    c.append(call('fisher_exact', [np.array([[1, 2, 3]])], {'alternative': 'less'}, SP))
    c.append(call('fisher_exact', [np.zeros((0, 3), dtype=int)], {}, SP))
    B['fisher_exact'] = c

    BT = ['statistic', 'pvalue', 'k', 'n', 'proportion_estimate']
    c = []
    for k, n, p in [(3, 10, 0.5), (7, 10, 0.5), (0, 10, 0.3), (10, 10, 0.3), (5, 10, 0.5), (682, 925, 0.75),
                    (3, 15, 0.1), (14, 100, 0.1), (2, 100, 0.1), (60, 100, 0.55), (1, 1, 0.5), (0, 1, 0.5),
                    (48, 100, 0.48), (20, 30, 0.0), (0, 30, 0.0), (30, 30, 1.0), (7, 30, 1.0), (13, 50, 0.3333)]:
        for alt in ALTS:
            c.append(call('binomtest', [k, n], {'p': p, 'alternative': alt}, BT))
    c.append(call('binomtest', [11, 10], {}, BT))
    c.append(call('binomtest', [-1, 10], {}, BT))
    c.append(call('binomtest', [3, 0], {}, BT))
    c.append(call('binomtest', [3, 10], {'p': 1.5}, BT))
    c.append(call('binomtest', [3, 10], {'alternative': 'bogus'}, BT))
    B['binomtest'] = c
    return B


def ranks():
    B = {}
    c = []
    for meth in ['auto', 'asymptotic', 'exact']:
        for alt in ALTS:
            c.append(call('mannwhitneyu', [data((7,)), data((5,)) + 1], {'method': meth, 'alternative': alt}, SP))
            c.append(call('mannwhitneyu', [data((12,), ties=True), data((10,), ties=True)], {'method': meth, 'alternative': alt}, SP))
        c.append(call('mannwhitneyu', [data((20,)), data((15,))], {'method': meth, 'use_continuity': False}, SP))
        c.append(call('mannwhitneyu', [data((8,)), data((40,))], {'method': meth}, SP))
        c.append(call('mannwhitneyu', [data((3,)), data((2,))], {'method': meth}, SP))
        c.append(call('mannwhitneyu', [data((1,)), data((1,))], {'method': meth}, SP))
        c.append(call('mannwhitneyu', [data((6, 3)), data((5, 3))], {'method': meth, 'axis': 0}, SP))
        c.append(call('mannwhitneyu', [data((3, 6), ties=True), data((3, 5), ties=True)], {'method': meth, 'axis': 1}, SP))
    c.append(call('mannwhitneyu', [[19, 22, 16, 29, 24], [20, 11, 17, 12]], {'method': 'exact'}, SP))
    c.append(call('mannwhitneyu', [[19, 22, 16, 29, 24], [20, 11, 17, 12]], {'method': 'asymptotic'}, SP))
    c.append(call('mannwhitneyu', [data((5,)), data((5,))], {'method': 'bogus'}, SP))
    c.append(call('mannwhitneyu', [data((5,)), data((5,))], {'alternative': 'bogus'}, SP))
    c += axis_cases('mannwhitneyu', lambda s: [data(s), data(s) + 0.5], SP)
    B['mannwhitneyu'] = c

    WN = lambda kw: SP + (['zstatistic'] if kw.get('method') in ('asymptotic', 'approx') else [])
    c = []
    for zm in ['wilcox', 'pratt', 'zsplit']:
        for meth in ['auto', 'asymptotic', 'exact', 'approx']:
            for alt in ALTS:
                for corr in (False, True):
                    kw = {'zero_method': zm, 'method': meth, 'alternative': alt, 'correction': corr}
                    c.append(call('wilcoxon', [data((9,))], kw, WN(kw)))
                    c.append(call('wilcoxon', [data((11,), ties=True) - 1.5], kw, WN(kw)))
            kw = {'zero_method': zm, 'method': meth}
            c.append(call('wilcoxon', [data((20,), ties=True) - 1.5], kw, WN(kw)))
            c.append(call('wilcoxon', [data((60,))], kw, WN(kw)))
            c.append(call('wilcoxon', [data((10,)), data((10,))], kw, WN(kw)))
            c.append(call('wilcoxon', [data((8, 3), ties=True) - 1.5], dict(kw, axis=0), WN(kw)))
            c.append(call('wilcoxon', [data((3, 8))], dict(kw, axis=1), WN(kw)))
    c.append(call('wilcoxon', [np.array([0.0])], {}, SP))
    c.append(call('wilcoxon', [np.array([0.0, 0.0, 1.0])], {}, SP))
    c.append(call('wilcoxon', [np.array([1.0, 2.0, 3.0])], {}, SP))
    c.append(call('wilcoxon', [np.array([1.0])], {}, SP))
    c.append(call('wilcoxon', [data((5,))], {'zero_method': 'bogus'}, SP))
    c += axis_cases('wilcoxon', lambda s: [data(s), data(s) + 0.3], SP)
    c += axis_cases('wilcoxon', lambda s: [data(s)], SP + ['zstatistic'], {'method': 'asymptotic'}, nan=False)
    B['wilcoxon'] = c

    c = []
    c.append(call('kruskal', [data((7,)), data((5,)) + 1, data((9,)) - 1], {}, SP))
    c.append(call('kruskal', [data((7,), ties=True), data((5,), ties=True), data((9,), ties=True), data((3,), ties=True)], {}, SP))
    c.append(call('kruskal', [data((7,), ints=True), data((5,), ints=True)], {}, SP))
    c.append(call('kruskal', [data((7,))], {}, SP))
    c.append(call('kruskal', [np.array([1.0, 1.0]), np.array([1.0, 1.0])], {}, SP))
    c.append(call('kruskal', [data((1,)), data((1,))], {}, SP))
    c.append(call('kruskal', [data((6, 3)), data((4, 3)), data((5, 1))], {'axis': 0}, SP))
    c += axis_cases('kruskal', lambda s: [data(s), data(s) + 1, data(s) - 0.5], SP)
    B['kruskal'] = c

    c = []
    c.append(call('friedmanchisquare', [data((7,)), data((7,)) + 1, data((7,)) - 1], {}, SP))
    c.append(call('friedmanchisquare', [data((8,), ties=True), data((8,), ties=True), data((8,), ties=True), data((8,), ties=True)], {}, SP))
    c.append(call('friedmanchisquare', [data((7,)), data((7,))], {}, SP))
    c.append(call('friedmanchisquare', [data((1,)), data((1,)), data((1,))], {}, SP))
    c.append(call('friedmanchisquare', [data((5,), ints=True), data((5,), ints=True), data((5,), ints=True)], {}, SP))
    c += axis_cases('friedmanchisquare', lambda s: [data(s), data(s) + 1, data(s) - 0.5], SP)
    B['friedmanchisquare'] = c

    c = []
    for alt in ALTS:
        for dist in ['t', 'normal']:
            c.append(call('brunnermunzel', [data((9,)), data((7,)) + 1], {'alternative': alt, 'distribution': dist}, SP))
            c.append(call('brunnermunzel', [data((9,), ties=True), data((7,), ties=True)], {'alternative': alt, 'distribution': dist}, SP))
    c.append(call('brunnermunzel', [data((9,)), data((7,))], {'distribution': 'bogus'}, SP))
    c.append(call('brunnermunzel', [[1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 1, 1], [3, 3, 4, 3, 1, 2, 3, 1, 1, 5, 4]], {}, SP))
    c += axis_cases('brunnermunzel', lambda s: [data(s), data(s) + 1.0], SP)
    B['brunnermunzel'] = c

    c = []
    for alt in ALTS:
        c.append(call('ranksums', [data((9,)), data((7,)) + 1], {'alternative': alt}, SP))
        c.append(call('ranksums', [data((9,), ties=True), data((7,), ties=True)], {'alternative': alt}, SP))
    c += axis_cases('ranksums', lambda s: [data(s), data(s) + 1.0], SP)
    B['ranksums'] = c
    return B


def variances():
    B = {}
    c = []
    for ev in (True, False):
        c.append(call('f_oneway', [data((7,)), data((5,)) + 1, data((9,)) - 1], {'equal_var': ev}, SP))
        c.append(call('f_oneway', [data((7,), ints=True), data((5,), ints=True), data((4,), ints=True), data((6,), ints=True)], {'equal_var': ev}, SP))
        c.append(call('f_oneway', [np.array([3., 3, 3]), np.array([5., 5, 5, 5]), np.array([4., 4, 4])], {'equal_var': ev}, SP))
        c.append(call('f_oneway', [np.array([3., 3, 3]), np.array([3., 3, 3, 3])], {'equal_var': ev}, SP))
        c.append(call('f_oneway', [data((1,)), data((1,)), data((1,))], {'equal_var': ev}, SP))
        c.append(call('f_oneway', [data((1,)), data((4,))], {'equal_var': ev}, SP))
        c.append(call('f_oneway', [data((6, 3)), data((4, 3)), data((5, 1))], {'axis': 0, 'equal_var': ev}, SP))
        c += axis_cases('f_oneway', lambda s: [data(s), data(s) + 1, data(s) - 0.5], SP, {'equal_var': ev})
    c.append(call('f_oneway', [data((7,))], {}, SP))
    B['f_oneway'] = c

    for fn in ['levene', 'fligner']:
        c = []
        for center in ['median', 'mean', 'trimmed']:
            c.append(call(fn, [data((7,)), data((5,)) * 2, data((9,)) - 1], {'center': center}, SP))
            c.append(call(fn, [data((8,)), data((6,)) * 3], {'center': center}, SP))
            c.append(call(fn, [data((10,), ties=True), data((12,), ties=True)], {'center': center, 'proportiontocut': 0.1}, SP))
            c.append(call(fn, [data((20,)), data((24,)) * 2], {'center': center, 'proportiontocut': 0.25}, SP))
            c += axis_cases(fn, lambda s: [data(s), data(s) * 2, data(s) - 0.5], SP, {'center': center})
        c.append(call(fn, [data((7,)), data((5,))], {'center': 'bogus'}, SP))
        c.append(call(fn, [data((7,))], {}, SP))
        c.append(call(fn, [data((7,), ints=True), data((6,), ints=True)], {}, SP))
        c.append(call(fn, [data((1,)), data((1,))], {}, SP))
        B[fn] = c

    c = []
    c.append(call('bartlett', [data((7,)), data((5,)) * 2, data((9,)) - 1], {}, SP))
    c.append(call('bartlett', [data((7,), ints=True), data((6,), ints=True)], {}, SP))
    c.append(call('bartlett', [data((7,))], {}, SP))
    c.append(call('bartlett', [data((1,)), data((4,))], {}, SP))
    c.append(call('bartlett', [np.array([1., 1, 1]), np.array([2., 2, 2])], {}, SP))
    c.append(call('bartlett', *([data((5,)) for _ in range(9)],), {}, SP))
    c += axis_cases('bartlett', lambda s: [data(s), data(s) * 2, data(s) - 0.5], SP)
    B['bartlett'] = c

    c = []
    for alt in ALTS:
        c.append(call('ansari', [data((9,)), data((7,)) * 2], {'alternative': alt}, SP))
        c.append(call('ansari', [data((10,)), data((11,)) * 2], {'alternative': alt}, SP))
        c.append(call('ansari', [data((9,), ties=True), data((7,), ties=True)], {'alternative': alt}, SP))
        c.append(call('ansari', [data((10,), ties=True), data((12,), ties=True)], {'alternative': alt}, SP))
        c.append(call('ansari', [data((60,)), data((57,)) * 2], {'alternative': alt}, SP))
        c.append(call('ansari', [data((61,)), data((56,)) * 2], {'alternative': alt}, SP))
        for n, m in [(1, 5), (5, 1), (2, 6), (6, 2), (3, 4), (4, 3), (3, 6), (7, 3), (5, 5), (8, 9), (12, 7), (30, 40)]:
            c.append(call('ansari', [data((n,)), data((m,)) * 2], {'alternative': alt}, SP))
    c.append(call('ansari', [data((5,)), data((5,))], {'alternative': 'bogus'}, SP))
    c += axis_cases('ansari', lambda s: [data(s), data(s) * 2], SP)
    B['ansari'] = c

    c = []
    for alt in ALTS:
        c.append(call('mood', [data((9,)), data((7,)) * 2], {'alternative': alt}, SP))
        c.append(call('mood', [data((10,), ties=True), data((12,), ties=True)], {'alternative': alt}, SP))
        c.append(call('mood', [data((25,), ties=True), data((20,), ties=True)], {'alternative': alt}, SP))
    c.append(call('mood', [data((1,)), data((1,))], {}, SP))
    c.append(call('mood', [data((2,)), data((1,))], {}, SP))
    c.append(call('mood', [data((5, 3), ties=True), data((4, 3), ties=True)], {'axis': 0}, SP))
    c += axis_cases('mood', lambda s: [data(s), data(s) * 2], SP)
    B['mood'] = c

    MT = ['statistic', 'pvalue', 'median', 'table']
    c = []
    g1 = [10, 14, 14, 18, 20, 22, 24, 25, 31, 31, 32, 39, 43, 43, 48, 49]
    g2 = [28, 30, 31, 33, 34, 35, 36, 40, 44, 55, 57, 61, 91, 92, 99]
    g3 = [0, 3, 9, 22, 23, 25, 25, 33, 34, 34, 40, 45, 46, 48, 62, 67, 84]
    for ties in ['below', 'above', 'ignore']:
        for corr in (True, False):
            c.append(call('median_test', [g1, g2, g3], {'ties': ties, 'correction': corr}, MT))
            c.append(call('median_test', [g1, g2], {'ties': ties, 'correction': corr}, MT))
            c.append(call('median_test', [data((9,), ties=True), data((8,), ties=True)], {'ties': ties, 'correction': corr}, MT))
    for lam in ['log-likelihood', 'freeman-tukey', 0.5, 'cressie-read']:
        c.append(call('median_test', [g1, g2, g3], {'lambda_': lam}, MT))
    c.append(call('median_test', [g1], {}, MT))
    c.append(call('median_test', [g1, []], {}, MT))
    c.append(call('median_test', [g1, g2], {'ties': 'bogus'}, MT))
    c.append(call('median_test', [[1, 1, 1], [1, 1]], {}, MT))
    c.append(call('median_test', [[1, 2, 3], [5, 5, 5], [7, 8]], {'ties': 'ignore'}, MT))
    c.append(call('median_test', [nanify(data((10,))), data((8,))], {'nan_policy': 'omit'}, MT))
    c.append(call('median_test', [nanify(data((10,)), 0.3), data((8,))], {'nan_policy': 'raise'}, MT))
    c.append(call('median_test', [data((3, 2)), data((4,))], {}, MT))
    B['median_test'] = c
    return B


def normality():
    B = {}
    for fn, lo in [('skewtest', 8), ('kurtosistest', 5), ('normaltest', 8)]:
        c = []
        alts = ALTS if fn != 'normaltest' else [None]
        for alt in alts:
            kw = {} if alt is None else {'alternative': alt}
            c.append(call(fn, [data((30,))], kw, SP))
            c.append(call(fn, [data((30,)) ** 3], kw, SP))
            c.append(call(fn, [S.rng.exponential(size=40)], kw, SP))
        for n in (lo - 1, lo, lo + 1, 20):
            c.append(call(fn, [data((n,))], {}, SP))
        c.append(call(fn, [np.full(10, 3.0)], {}, SP))
        c.append(call(fn, [data((12,), ints=True)], {}, SP))
        c.append(call(fn, [data((6, 3))], {'axis': 0}, SP))
        c.append(call(fn, [data((3, 6))], {'axis': 1}, SP))
        c += axis_cases(fn, lambda s: [data(s)], SP)
        B[fn] = c
    c = []
    c.append(call('jarque_bera', [data((30,))], {}, SP))
    c.append(call('jarque_bera', [S.rng.exponential(size=50)], {}, SP))
    c.append(call('jarque_bera', [data((1,))], {}, SP))
    c.append(call('jarque_bera', [data((2,))], {}, SP))
    c.append(call('jarque_bera', [np.full(10, 3.0)], {}, SP))
    c.append(call('jarque_bera', [data((12,), ints=True)], {}, SP))
    c.append(call('jarque_bera', [data((6, 4))], {}, SP))
    c.append(call('jarque_bera', [data((6, 4))], {'axis': 0}, SP))
    c += axis_cases('jarque_bera', lambda s: [data(s)], SP)
    B['jarque_bera'] = c
    return B


def edges(blocks):
    """Empty inputs, mixed dimensions and axis errors through the _axis_nan_policy decorator."""
    e0, e03, e30 = np.zeros((0,)), np.zeros((0, 3)), np.zeros((3, 0))
    add = lambda fn, args, kw, names: blocks[fn].append(call(fn, args, kw, names))
    for arr, kw in [(e0, {}), (e03, {'axis': 0}), (e30, {'axis': 0}), (e30, {'axis': 1}), (e03, {'axis': 0, 'keepdims': True})]:
        add('ttest_1samp', [arr, 0.0], kw, TT)
        add('ttest_ind', [arr, arr], kw, TT)
        add('ttest_rel', [arr, arr], kw, TT)
        add('skewtest', [arr], kw, SP)
        add('jarque_bera', [arr], dict(kw, axis=kw.get('axis')), SP)
        add('kruskal', [arr, arr], kw, SP)
        add('mannwhitneyu', [arr, arr], kw, SP)
        add('wilcoxon', [arr], kw, SP)
        add('chisquare', [arr], kw, SP)
        add('f_oneway', [arr, arr], kw, SP)
        add('mood', [arr, arr], kw, SP)
    add('kruskal', [data((5,)), e0], {}, SP)
    add('ttest_ind', [data((3, 8)), data((8,))], {'axis': 1}, TT)
    add('ttest_ind', [data((8, 3)), data((6, 1))], {'axis': 0, 'keepdims': True}, TT)
    add('ttest_ind', [data((2, 3, 8)), data((3, 5))], {'axis': -1}, TT)
    add('ttest_ind', [data((3, 8)), data((2, 5))], {'axis': 1}, TT)
    add('ttest_ind', [data((3, 8)), data((3, 5))], {'axis': 2}, TT)
    add('ttest_ind', [data((3, 8)), data((3, 5))], {'axis': -3}, TT)
    add('ttest_1samp', [np.array([1.0, 2.0, 4.0]), np.nan], {}, TT)
    add('ttest_1samp', [np.array([1.0, 2.0, 4.0]), np.nan], {'nan_policy': 'omit'}, TT)
    add('ttest_1samp', [data((2, 3, 9)), 0.0], {'axis': 0}, TT)
    add('ttest_1samp', [data((2, 3, 9)), 0.0], {'axis': 2, 'keepdims': True}, TT)
    add('ttest_rel', [nanify(data((12, 3)), 0.2), data((12, 1))], {'axis': 0, 'nan_policy': 'omit'}, TT)
    add('wilcoxon', [nanify(data((12, 3)), 0.2), data((12, 3))], {'axis': 0, 'nan_policy': 'omit'}, SP)
    add('kruskal', [data((3, 8)), data((8,)), data((1, 4))], {'axis': 1}, SP)
    add('levene', [data((3, 8)), data((8,)), data((1, 4))], {'axis': 1}, SP)
    add('bartlett', [data((3, 8)), data((8,)), data((1, 4))], {'axis': 1, 'nan_policy': 'omit'}, SP)
    add('friedmanchisquare', [data((3, 8)), data((8,)), data((1, 8))], {'axis': 1}, SP)
    add('friedmanchisquare', [data((3, 8)), data((8,)), data((1, 7))], {'axis': 1}, SP)
    add('chisquare', [np.array([[16, 18, 16], [14, 12, 12]]), np.array([15, 15, 14])], {'axis': 1}, SP)
    add('chisquare', [np.array([[16, 18, 16], [14, 12, 12]]), np.array([15, 15, 14])], {'axis': None}, SP)


def main():
    blocks = {}
    for part in (ttests, chisq, exact, ranks, variances, normality):
        blocks.update(part())
    edges(blocks)
    # negative / NaN proportiontocut with center='trimmed' (SciPy raises: np.partition kth out of bounds, int(nan))
    for prop in (-0.2, float('nan')):
        for fn in ('levene', 'fligner'):
            blocks[fn].append(call(fn, [data(10), data(12)], {'center': 'trimmed', 'proportiontocut': prop}, SP))
    S.write('tests', blocks)


if __name__ == '__main__':
    main()
