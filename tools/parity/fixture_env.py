"""The numerical environment a fixture file was generated in (ADR 0012), recorded in every fixture file as "env".

NumPy dispatches float64 exp/log/pow/expm1/log1p/sinh/arcsin to AVX-512 SVML-style kernels on CPUs that have
AVX-512 (X86_V4, AVX512_ICL, AVX512_SPR); those differ from the C library by an ulp on some inputs, so the
reference would depend on the machine that generated it. tools/parity/gen-fixtures.sh disables those dispatch
targets, which makes NumPy's float64 loops the C library's (checked: 0 differences in 200 000 inputs per
function), and fixes OpenBLAS's kernel (Haswell) so reductions through BLAS use one summation order.

Re-baselined 2026-10-02 to the Haswell OpenBLAS kernel (AVX2): the canonical fixture host is an AVX2 machine,
on which the earlier SkylakeX (AVX-512) OpenBLAS kernel is an illegal instruction. Single-threaded BLAS
(OPENBLAS_NUM_THREADS=1) makes the summation order deterministic; Haswell kernels run on any AVX2+ CPU.
"""
import os
import numpy as np
import scipy

PINNED = {'NPY_DISABLE_CPU_FEATURES': 'X86_V4 AVX512_ICL AVX512_SPR', 'OPENBLAS_CORETYPE': 'Haswell',
          'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}


TRANSCENDENTAL = 'exp|log|power|expm1|log1p|sinh|cosh|tanh|arcsin|arccos|arctan|sin|cos|tan|exp2|log2|log10|cbrt|arcsinh|arccosh|arctanh'


def env():
    info = np.lib.introspect.opt_func_info(func_name=f'^({TRANSCENDENTAL})$', signature='float64')
    dispatch = {fn: next(iter(sigs.values()))['current'] for fn, sigs in sorted(info.items())}
    pinned = all(os.environ.get(k) == v for k, v in PINNED.items())
    return {'numpy': np.__version__, 'scipy': scipy.__version__, 'pinned': pinned,
            'vars': {k: os.environ.get(k) for k in PINNED}, 'float64_dispatch': dispatch}


def require_pinned():
    """Refuse to write a fixture file outside gen-fixtures.sh (set TSR_FIXTURE_UNPINNED=1 for a scratch run)."""
    if not env()['pinned'] and os.environ.get('TSR_FIXTURE_UNPINNED') != '1':
        raise SystemExit('fixtures must be generated through tools/parity/gen-fixtures.sh (pinned numerical environment); '
                         'TSR_FIXTURE_UNPINNED=1 for a scratch file')
