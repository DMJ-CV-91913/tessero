#!/usr/bin/env bash
# Regenerate the NumPy/SciPy reference fixtures in the pinned numerical environment (ADR 0012).
#
#   tools/parity/gen-fixtures.sh                 # every fixture file
#   tools/parity/gen-fixtures.sh stats np        # some groups: special np shape numeric random stats desc corr tests tests2 legacy
#   tools/parity/gen-fixtures.sh stats=gamma,norm # some distributions (merged into stats.json)
#
# The environment (tools/parity/fixture_env.py) disables NumPy's AVX-512 float64 dispatch targets, so NumPy's
# transcendental loops are the C library's on every x86-64 machine, and fixes OpenBLAS's kernel and thread
# count; each fixture file records it under "env", and the generators refuse to write without it.
set -euo pipefail
cd "$(dirname "$0")"
export NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"
export OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
groups=("$@")
[ ${#groups[@]} -eq 0 ] && groups=(special special_complex np shape numeric linalg slinalg fft distance sparse signal windows ndimage csgraph optimize io integrate spatial datetime random stats desc corr tests tests2 legacy)
for g in "${groups[@]}"; do
    t=$SECONDS
    case $g in
        special) python3 fixtures.py ;;
        special_complex) python3 fixtures_special_complex.py ;;
        np)      python3 fixtures_np.py ;;
        shape)   python3 fixtures_np_shape.py ;;
        numeric) python3 fixtures_np_numeric.py ;;
        linalg)  python3 fixtures_linalg.py ;;
        slinalg) python3 fixtures_scipy_linalg.py ;;
        fft)     python3 fixtures_fft.py ;;
        distance) python3 fixtures_distance.py ;;
        sparse) python3 fixtures_sparse.py ;;
        signal) python3 fixtures_signal.py ;;
        windows) python3 fixtures_signal_windows.py ;;
        ndimage) python3 fixtures_ndimage.py ;;
        csgraph) python3 fixtures_csgraph.py ;;
        optimize) python3 fixtures_optimize.py ;;
        io) python3 fixtures_io.py ;;
        integrate) python3 fixtures_integrate.py ;;
        spatial) python3 fixtures_spatial.py ;;
        datetime) python3 fixtures_datetime.py ;;
        random)  python3 fixtures_random.py ;;
        stats)   python3 fixtures_stats.py ;;
        stats=*) python3 fixtures_stats.py $(echo "${g#stats=}" | tr , " ") ;;
        desc|corr|tests|tests2) python3 "fixtures_stats_$g.py" ;;
        legacy)  python3 ../../tests/fixtures/generate_parity.py && python3 ../../tests/fixtures/generate_ufunc.py ;;   # suites older than the registry
        *) echo "unknown group: $g" >&2; exit 2 ;;
    esac
    echo "== $g: $((SECONDS - t))s"
done
