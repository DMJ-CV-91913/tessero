# 0012. Reference fixtures are generated in a pinned numerical environment

- Status: Accepted
- Date: 2026-09-28
- Amends: [0002](0002-numpy-as-specification.md) (how the reference values are produced)

## Context

Several distributions and random-number methods failed their fixtures by one ulp. Tessero's code was a
faithful port, and on inspection the difference was in the reference. On CPUs with AVX-512, NumPy 2.x
evaluates float64 `exp`, `log`, `power`, `expm1`, `log1p`, `sinh` and `arcsin` with its own vector kernels.
Those differ from the C library by an ulp on some inputs. The same NumPy therefore gives different reference
values on different machines. A port of SciPy that calls the C library cannot match all of them, and chasing
them would mean copying one machine's vector kernels.

## Decision

Fixtures are generated only by `tools/parity/gen-fixtures.sh`, which sets:

- `NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"`: NumPy's float64 loops fall back to the targets
  whose results equal the C library's. This was checked over 200 000 inputs per function, with no difference.
- `OPENBLAS_CORETYPE=SkylakeX`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`: one BLAS kernel and one
  summation order.

`tools/parity/fixture_env.py` records these settings and NumPy's per-function dispatch
(`numpy.lib.introspect.opt_func_info`) in every fixture file as `"env"`. The generators refuse to write a
fixture outside the pinned environment; `TSR_FIXTURE_UNPINNED=1` allows a scratch run.

## Consequences

- The reference is reproducible on any x86-64 machine with NumPy 2.4.4 and SciPy 1.17.1.
- Tolerances that only absorbed the machine difference were removed. Random draws are compared exactly, and
  `cov`, `corrcoef`, `correlate` and `convolve` use the default rtol 1e-12.
- Users on AVX-512 machines who compare Tessero with their own NumPy can still see one-ulp differences in
  those functions. The difference lies between NumPy's two code paths, not in Tessero.

## Alternatives considered

- *Tolerances per function*: that would hide real errors of the same size.
- *Tessero using NumPy's vector kernels*: that ties Tessero to one CPU family's results.
