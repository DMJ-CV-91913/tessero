dnl tessero: native Zend extension for the Tessero numeric platform.
dnl   phpize && ./configure --enable-tessero [--enable-tessero-openmp] && make && make test

PHP_ARG_ENABLE([tessero],
  [whether to enable the Tessero native extension],
  [AS_HELP_STRING([--enable-tessero], [Enable Tessero\Ext (NDArray, Engine, Operand)])],
  [no])

PHP_ARG_ENABLE([tessero-openmp],
  [whether to build libtessero with OpenMP threads],
  [AS_HELP_STRING([--enable-tessero-openmp], [Parallel element-wise kernels and MDP sweeps (tessero.threads)])],
  [yes], [no])

if test "$PHP_TESSERO" != "no"; then
  dnl BEGIN sources (tools/sync-ext.sh)
  TESSERO_KERNEL="libtessero/src/alloc.c libtessero/src/array.c libtessero/src/elementwise.c libtessero/src/fft.c libtessero/src/fn_core.c libtessero/src/fn_tables.c libtessero/src/gen_special_table.c libtessero/src/index.c libtessero/src/lp.c libtessero/src/matmul.c libtessero/src/mdp.c libtessero/src/mmap.c libtessero/src/np_io.c libtessero/src/np_linalg.c libtessero/src/np_numeric.c libtessero/src/np_poly.c libtessero/src/np_random.c libtessero/src/np_sets.c libtessero/src/np_shape.c libtessero/src/np_stats.c libtessero/src/reduce.c libtessero/src/rng.c libtessero/src/scipy_csgraph.c libtessero/src/scipy_interpolate.c libtessero/src/scipy_ndimage.c libtessero/src/scipy_signal.c libtessero/src/scipy_sparse.c libtessero/src/sort.c libtessero/src/sparse.c libtessero/src/spatial_distance.c libtessero/src/ufunc.c libtessero/third_party/scipy/cdflib.c libtessero/third_party/scipy/_cosine.c libtessero/third_party/scipy/quadpack/quadpack.c libtessero/third_party/scipy/zeros/brentq.c libtessero/third_party/numpy/random/src/distributions.c libtessero/third_party/numpy/random/src/logfactorial.c libtessero/third_party/numpy/random/src/random_hypergeometric.c libtessero/third_party/numpy/random/src/random_mvhg_count.c libtessero/third_party/numpy/random/src/random_mvhg_marginals.c"
  TESSERO_KERNEL_CXX="libtessero/cxx/dist_basic.cpp libtessero/cxx/dist_cont1.cpp libtessero/cxx/dist_cont2.cpp libtessero/cxx/dist_cont3.cpp libtessero/cxx/dist_cont4.cpp libtessero/cxx/dist_cont5.cpp libtessero/cxx/dist_cont6.cpp libtessero/cxx/dist_disc.cpp libtessero/cxx/fn_guard.cpp libtessero/cxx/gen_special.cpp libtessero/cxx/stats_dist.cpp libtessero/cxx/stats_fn_corr.cpp libtessero/cxx/stats_fn_desc.cpp libtessero/cxx/stats_fn_tests2.cpp libtessero/cxx/stats_fn_tests.cpp libtessero/third_party/scipy/wright.cc"
  dnl END sources
  dnl kernel symbols stay private to this module, so they never clash with the FFI copy of libtessero
  TESSERO_CFLAGS="-std=gnu11 -O3 -fno-math-errno -fno-trapping-math -fvisibility=hidden -DZEND_ENABLE_STATIC_TSRMLS_CACHE=1 -I@ext_srcdir@/libtessero/third_party/numpy/include"
  if test "$PHP_TESSERO_OPENMP" != "no"; then
    TESSERO_CFLAGS="$TESSERO_CFLAGS -fopenmp"
    PHP_ADD_LIBRARY(gomp, 1, TESSERO_SHARED_LIBADD)
  fi
  PHP_ADD_LIBRARY(m, 1, TESSERO_SHARED_LIBADD)
  dnl LAPACKE (row-major C LAPACK) backs linalg.* (libtessero/src/np_linalg.c); pulls in the system LAPACK/BLAS.
  dnl On macOS/Windows this needs the platform LAPACKE (Accelerate / OpenBLAS-for-MSVC): tracked cross-platform work.
  PHP_ADD_LIBRARY(lapacke, 1, TESSERO_SHARED_LIBADD)
  PHP_SUBST(TESSERO_SHARED_LIBADD)
  dnl the special-function libraries SciPy uses (xsf, Boost.Math, SciPy glue) are C++17 (ADR 0011)
  PHP_REQUIRE_CXX()
  TESSERO_INC="-I@ext_srcdir@/libtessero/third_party -I@ext_srcdir@/libtessero/third_party/xsf -I@ext_srcdir@/libtessero/third_party/scipy -I@ext_srcdir@/libtessero/third_party/boost -I@ext_srcdir@/libtessero/third_party/numpy/include"
  PHP_NEW_EXTENSION(tessero, tessero.c tessero_ndarray.c tessero_solvers.c tessero_ufunc.c tessero_memmap.c tessero_fn.c tessero_dist.c tessero_random.c $TESSERO_KERNEL, $ext_shared,, $TESSERO_CFLAGS, cxx)
  TESSERO_INC=`echo "$TESSERO_INC" | sed "s|@ext_srcdir@|$ext_srcdir|g"`
  TESSERO_CXXFLAGS="-std=c++17 -O2 -fno-math-errno -fvisibility=hidden -DBOOST_MATH_STANDALONE $TESSERO_INC -Wno-deprecated-declarations"
  if test "$PHP_TESSERO_OPENMP" != "no"; then
    TESSERO_CXXFLAGS="$TESSERO_CXXFLAGS -fopenmp"
  fi
  PHP_ADD_SOURCES_X(PHP_EXT_DIR(tessero), $TESSERO_KERNEL_CXX, $TESSERO_CXXFLAGS, shared_objects_tessero, yes)
  PHP_ADD_BUILD_DIR([$ext_builddir/libtessero/src])
  PHP_ADD_BUILD_DIR([$ext_builddir/libtessero/cxx])
  PHP_ADD_BUILD_DIR([$ext_builddir/libtessero/third_party/scipy])
  PHP_ADD_BUILD_DIR([$ext_builddir/libtessero/third_party/numpy/random/src])
  PHP_ADD_BUILD_DIR([$ext_builddir/libtessero/third_party/scipy/quadpack])
  PHP_ADD_BUILD_DIR([$ext_builddir/libtessero/third_party/scipy/zeros])
  dnl GCC's target_clones ifunc symbols ignore -fvisibility=hidden; a version script keeps them local.
  dnl ($ext_srcdir is set by PHP_NEW_EXTENSION) (GNU ld, lld, gold)
  case $host_os in
    linux*|*gnu*|freebsd*)
      TESSERO_SHARED_LIBADD="$TESSERO_SHARED_LIBADD -Wl,--version-script=$ext_srcdir/tessero.map"
      ;;
  esac

  PHP_ADD_EXTENSION_DEP(tessero, json)
  PHP_ADD_EXTENSION_DEP(tessero, spl)
fi
