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
[ ${#groups[@]} -eq 0 ] && groups=(special special_complex special_orthopoly special_seq stats_opt stats_normal stats_uniform stats_orthogroup stats_unifdir bootstrap stats_dists barnard boschloo logrank permtest ndimage_valindices ndimage_geomxform ndimage_genfilter ndimage_genfilter1d ndimage_genlaplace ndimage_gengradmag spatial_rect spatial_hausdorff spatial_hull spatial_delaunay interpolate_nn interpolate_griddata interpolate_bisplev interpolate_ius interpolate_lsq interpolate_rbs interpolate_splrep interpolate_splprep interpolate_generate_knots interpolate_atp interpolate_rbf spatial_voronoi spatial_transform spatial_slerp spatial_rigid spatial_rotspline interpolate_cloughtocher interpolate_aaa interpolate_mss interpolate_linearnd np shape numeric linalg slinalg fft distance sparse signal windows ndimage csgraph interpolate cluster npoly hierarchy splinalg splinalg_iter splinalg_lobpcg stats_unitarygroup scipyfft optimize qap de da bh linesearch io integrate integrate_tanhsinh integrate_quadvec integrate_solveivp integrate_nestedquad integrate_nsum differentiate nonlin stats_distinfra scipyio sparse_npz spatial datetime random stats desc corr tests tests2 legacy)
for g in "${groups[@]}"; do
    t=$SECONDS
    case $g in
        special) python3 fixtures.py ;;
        special_complex) python3 fixtures_special_complex.py ;;
        special_orthopoly) python3 fixtures_special_orthopoly.py ;;
        special_seq) python3 fixtures_special_seq.py ;;
        stats_opt) python3 fixtures_stats_opt.py ;;
        barnard) python3 fixtures_barnard.py > ../../tests/fixtures/parity/stats_barnard.json ;;
        boschloo) python3 fixtures_boschloo.py ;;
        logrank) python3 fixtures_logrank.py ;;
        permtest) python3 fixtures_permtest.py ;;
        stats_normal) python3 fixtures_normal.py ;;
        stats_uniform) python3 fixtures_uniform.py ;;
        stats_orthogroup) python3 fixtures_orthogroup.py ;;
        stats_unifdir) python3 fixtures_unifdir.py ;;
        bootstrap) python3 fixtures_bootstrap.py ;;
        stats_dists) python3 fixtures_dists.py ;;
        ndimage_valindices) python3 fixtures_valindices.py ;;
        ndimage_labcomp)   python3 fixtures_labcomp.py ;;
        ndimage_vecfilter) python3 fixtures_vecfilter.py ;;
        statsmvs)          python3 fixtures_statsmvs.py ;;
        ndimage_geomxform) python3 fixtures_geomxform.py ;;
        ndimage_genfilter) python3 fixtures_genfilter.py ;;
        ndimage_genfilter1d) python3 fixtures_genfilter1d.py ;;
        ndimage_genlaplace) python3 fixtures_genlaplace.py ;;
        ndimage_gengradmag) python3 fixtures_gengradmag.py ;;
        spatial_rect) python3 fixtures_spatial_rect.py ;;
        spatial_hausdorff) python3 fixtures_spatial_hausdorff.py ;;
        interpolate_nn) python3 fixtures_interpolate_nn.py ;;
        interpolate_griddata) python3 fixtures_interpolate_griddata.py ;;
        interpolate_bisplev) python3 fixtures_bisplev.py ;;
        interpolate_ius) python3 fixtures_ius.py ;;
        interpolate_splrep) python3 fixtures_splrep.py ;;
        interpolate_splprep) python3 fixtures_splprep.py ;;
        interpolate_generate_knots) python3 fixtures_generate_knots.py ;;
        interpolate_atp) python3 fixtures_atp.py ;;
        interpolate_rbf) python3 fixtures_rbf.py ;;
        spatial_voronoi) python3 fixtures_spatial_voronoi.py ;;
        spatial_transform) python3 fixtures_transform.py ;;
        spatial_slerp) python3 fixtures_slerp.py ;;
        spatial_rigid) python3 fixtures_rigid.py ;;
        spatial_rotspline) python3 fixtures_rotspline.py ;;
        interpolate_cloughtocher) python3 fixtures_cloughtocher.py ;;
        interpolate_aaa) python3 fixtures_aaa.py ;;
        interpolate_lsq) python3 fixtures_interpolate_lsq.py ;;
        interpolate_rbs) python3 fixtures_rbs.py ;;
        spatial_hull) python3 fixtures_hull.py ;;
        spatial_delaunay) python3 fixtures_delaunay.py ;;
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
        interpolate) python3 fixtures_interpolate.py ;;
        cluster) python3 fixtures_cluster.py ;;
        npoly) python3 fixtures_npoly.py ;;
        hierarchy) python3 fixtures_hierarchy.py ;;
        splinalg) python3 fixtures_splinalg.py && python3 fixtures_splinalg_eig.py ;;
        splinalg_iter) python3 fixtures_splinalg_iter.py ;;
        splinalg_lobpcg) python3 fixtures_splinalg_lobpcg.py ;;
        scipyfft) python3 fixtures_scipyfft.py ;;
        optimize) python3 fixtures_optimize.py ;;
        qap) python3 fixtures_qap.py ;;
        de) python3 fixtures_de.py ;;
        da) python3 fixtures_da.py ;;
        bh) python3 fixtures_bh.py ;;
        nonlin) python3 fixtures_nonlin.py ;;
        linesearch) python3 fixtures_linesearch.py ;;
        interpolate_mss) python3 fixtures_mss.py ;;
        interpolate_linearnd) python3 fixtures_linearnd.py ;;
        io) python3 fixtures_io.py ;;
        stats_distinfra) python3 fixtures_distinfra.py ;;
        scipyio) python3 fixtures_scipyio.py ;;
        sparse_npz) python3 fixtures_sparse_npz.py ;;
        integrate) python3 fixtures_integrate.py ;;
        integrate_tanhsinh) python3 fixtures_tanhsinh.py ;;
        integrate_quadvec) python3 fixtures_quadvec.py ;;
        integrate_solveivp) python3 fixtures_solveivp.py ;;
        spatial) python3 fixtures_spatial.py ;;
        datetime) python3 fixtures_datetime.py ;;
        random)  python3 fixtures_random.py ;;
        stats)   python3 fixtures_stats.py ;;
        stats=*) python3 fixtures_stats.py $(echo "${g#stats=}" | tr , " ") ;;
        desc|corr|tests|tests2) python3 "fixtures_stats_$g.py" ;;
        legacy)  python3 ../../tests/fixtures/generate_parity.py && python3 ../../tests/fixtures/generate_ufunc.py ;;   # suites older than the registry
        integrate_nestedquad) python3 fixtures_nestedquad.py ;;
        integrate_nsum) python3 fixtures_nsum.py ;;
        differentiate) python3 fixtures_differentiate.py ;;
        stats_unitarygroup) python3 fixtures_unitarygroup.py ;;
        *) echo "unknown group: $g" >&2; exit 2 ;;
    esac
    echo "== $g: $((SECONDS - t))s"
done
