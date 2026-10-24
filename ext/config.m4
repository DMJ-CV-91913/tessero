dnl tessero: native Zend extension for the Tessero numeric platform.
dnl   phpize && ./configure --enable-tessero [--enable-tessero-openmp] && make && make test

PHP_ARG_ENABLE([tessero],
  [whether to enable the Tessero native extension],
  [AS_HELP_STRING([--enable-tessero], [Enable Tessero\Ext (NDArray, Engine, Operand)])],
  [no])

PHP_ARG_ENABLE([tessero-harden],
  [whether to apply the probed compiler and linker hardening flags],
  [AS_HELP_STRING([--disable-tessero-harden], [Do not add the OpenSSF hardening flags (sanitizer builds pass this)])],
  [yes], [no])

PHP_ARG_ENABLE([tessero-openmp],
  [whether to build libtessero with OpenMP threads],
  [AS_HELP_STRING([--enable-tessero-openmp], [Parallel element-wise kernels and MDP sweeps (tessero.threads)])],
  [yes], [no])

if test "$PHP_TESSERO" != "no"; then
  dnl BEGIN sources (tools/sync-ext.sh)
  TESSERO_KERNEL="libtessero/src/alloc.c libtessero/src/array.c libtessero/src/elementwise.c libtessero/src/fft.c libtessero/src/fn_core.c libtessero/src/fn_tables.c libtessero/src/fpenv.c libtessero/src/gen_special_table.c libtessero/src/index.c libtessero/src/lp.c libtessero/src/matmul.c libtessero/src/mdp.c libtessero/src/mmap.c libtessero/src/np_io.c libtessero/src/np_linalg.c libtessero/src/np_numeric.c libtessero/src/np_poly.c libtessero/src/np_polynomial.c libtessero/src/np_random.c libtessero/src/np_sets.c libtessero/src/np_shape.c libtessero/src/np_stats.c libtessero/src/reduce.c libtessero/src/rng.c libtessero/src/scipy_cluster.c libtessero/src/scipy_cluster_hier.c libtessero/src/scipy_csgraph.c libtessero/src/scipy_fft.c libtessero/src/scipy_interpolate.c libtessero/src/scipy_ndimage.c libtessero/src/scipy_signal.c libtessero/src/scipy_sparse.c libtessero/src/scipy_sparse_linalg.c libtessero/src/selfcheck.c libtessero/src/sort.c libtessero/src/sparse.c libtessero/src/spatial_distance.c libtessero/src/ufunc.c libtessero/third_party/scipy/cdflib.c libtessero/third_party/scipy/_cosine.c libtessero/third_party/scipy/quadpack/quadpack.c libtessero/third_party/scipy/zeros/brentq.c libtessero/third_party/scipy/fitpack/bispeu.c libtessero/third_party/scipy/fitpack/bispev.c libtessero/third_party/scipy/fitpack/clocur.c libtessero/third_party/scipy/fitpack/cocosp.c libtessero/third_party/scipy/fitpack/concon.c libtessero/third_party/scipy/fitpack/concur.c libtessero/third_party/scipy/fitpack/cualde.c libtessero/third_party/scipy/fitpack/curev.c libtessero/third_party/scipy/fitpack/curfit.c libtessero/third_party/scipy/fitpack/dblint.c libtessero/third_party/scipy/fitpack/evapol.c libtessero/third_party/scipy/fitpack/fitpack_support.c libtessero/third_party/scipy/fitpack/fourco.c libtessero/third_party/scipy/fitpack/fpader.c libtessero/third_party/scipy/fitpack/fpadno.c libtessero/third_party/scipy/fitpack/fpadpo.c libtessero/third_party/scipy/fitpack/fpback.c libtessero/third_party/scipy/fitpack/fpbacp.c libtessero/third_party/scipy/fitpack/fpbfout.c libtessero/third_party/scipy/fitpack/fpbisp.c libtessero/third_party/scipy/fitpack/fpbspl.c libtessero/third_party/scipy/fitpack/fpchec.c libtessero/third_party/scipy/fitpack/fpched.c libtessero/third_party/scipy/fitpack/fpchep.c libtessero/third_party/scipy/fitpack/fpclos.c libtessero/third_party/scipy/fitpack/fpcoco.c libtessero/third_party/scipy/fitpack/fpcons.c libtessero/third_party/scipy/fitpack/fpcosp.c libtessero/third_party/scipy/fitpack/fpcsin.c libtessero/third_party/scipy/fitpack/fpcurf.c libtessero/third_party/scipy/fitpack/fpcuro.c libtessero/third_party/scipy/fitpack/fpcyt1.c libtessero/third_party/scipy/fitpack/fpcyt2.c libtessero/third_party/scipy/fitpack/fpdeno.c libtessero/third_party/scipy/fitpack/fpdisc.c libtessero/third_party/scipy/fitpack/fpfrno.c libtessero/third_party/scipy/fitpack/fpgivs.c libtessero/third_party/scipy/fitpack/fpgrdi.c libtessero/third_party/scipy/fitpack/fpgrpa.c libtessero/third_party/scipy/fitpack/fpgrre.c libtessero/third_party/scipy/fitpack/fpgrsp.c libtessero/third_party/scipy/fitpack/fpinst.c libtessero/third_party/scipy/fitpack/fpintb.c libtessero/third_party/scipy/fitpack/fpknot.c libtessero/third_party/scipy/fitpack/fpopdi.c libtessero/third_party/scipy/fitpack/fpopsp.c libtessero/third_party/scipy/fitpack/fporde.c libtessero/third_party/scipy/fitpack/fppara.c libtessero/third_party/scipy/fitpack/fppasu.c libtessero/third_party/scipy/fitpack/fpperi.c libtessero/third_party/scipy/fitpack/fppocu.c libtessero/third_party/scipy/fitpack/fppogr.c libtessero/third_party/scipy/fitpack/fppola.c libtessero/third_party/scipy/fitpack/fprank.c libtessero/third_party/scipy/fitpack/fprati.c libtessero/third_party/scipy/fitpack/fpregr.c libtessero/third_party/scipy/fitpack/fprota.c libtessero/third_party/scipy/fitpack/fprppo.c libtessero/third_party/scipy/fitpack/fprpsp.c libtessero/third_party/scipy/fitpack/fpseno.c libtessero/third_party/scipy/fitpack/fpspgr.c libtessero/third_party/scipy/fitpack/fpsphe.c libtessero/third_party/scipy/fitpack/fpsuev.c libtessero/third_party/scipy/fitpack/fpsurf.c libtessero/third_party/scipy/fitpack/fpsysy.c libtessero/third_party/scipy/fitpack/fptrnp.c libtessero/third_party/scipy/fitpack/fptrpe.c libtessero/third_party/scipy/fitpack/insert.c libtessero/third_party/scipy/fitpack/parcur.c libtessero/third_party/scipy/fitpack/parder.c libtessero/third_party/scipy/fitpack/pardeu.c libtessero/third_party/scipy/fitpack/pardtc.c libtessero/third_party/scipy/fitpack/parsur.c libtessero/third_party/scipy/fitpack/percur.c libtessero/third_party/scipy/fitpack/pogrid.c libtessero/third_party/scipy/fitpack/polar.c libtessero/third_party/scipy/fitpack/profil.c libtessero/third_party/scipy/fitpack/regrid.c libtessero/third_party/scipy/fitpack/spalde.c libtessero/third_party/scipy/fitpack/spgrid.c libtessero/third_party/scipy/fitpack/sphere.c libtessero/third_party/scipy/fitpack/splder.c libtessero/third_party/scipy/fitpack/splev.c libtessero/third_party/scipy/fitpack/splint.c libtessero/third_party/scipy/fitpack/sproot.c libtessero/third_party/scipy/fitpack/surev.c libtessero/third_party/scipy/fitpack/surfit.c"
  TESSERO_KERNEL_CXX="libtessero/cxx/dist_basic.cpp libtessero/cxx/dist_cont1.cpp libtessero/cxx/dist_cont2.cpp libtessero/cxx/dist_cont3.cpp libtessero/cxx/dist_cont4.cpp libtessero/cxx/dist_cont5.cpp libtessero/cxx/dist_cont6.cpp libtessero/cxx/dist_disc.cpp libtessero/cxx/fn_guard.cpp libtessero/cxx/gen_special.cpp libtessero/cxx/stats_dist.cpp libtessero/cxx/stats_fn_corr.cpp libtessero/cxx/stats_fn_desc.cpp libtessero/cxx/stats_fn_tests.cpp libtessero/cxx/stats_fn_tests2.cpp libtessero/third_party/scipy/wright.cc"
  TESSERO_KERNEL_REPRO="libtessero/src/repro_f32.c"
  TESSERO_KERNEL_NPRAND="libtessero/third_party/numpy/random/src/distributions.c libtessero/third_party/numpy/random/src/logfactorial.c libtessero/third_party/numpy/random/src/random_hypergeometric.c libtessero/third_party/numpy/random/src/random_mvhg_count.c libtessero/third_party/numpy/random/src/random_mvhg_marginals.c"
  dnl END sources
  dnl kernel symbols stay private to this module, so they never clash with the FFI copy of libtessero
  TESSERO_CFLAGS="-std=gnu11 -O3 -fno-math-errno -fno-trapping-math -fvisibility=hidden -DZEND_ENABLE_STATIC_TSRMLS_CACHE=1 -I@ext_srcdir@/libtessero/third_party/numpy/include"
  dnl reproducible builds: phpize compiles with absolute paths, so __FILE__ (assert messages) would carry the
  dnl build directory into tessero.so. Map it to "." (probed: GCC >= 8, clang >= 10), as csrc/Makefile does.
  dnl The sources are compiled by absolute path under the source directory; PHP_NEW_EXTENSION replaces
  dnl @ext_srcdir@ in these flags, and the other flag lists below are sed'ed the same way.
  TESSERO_PREFIX_MAP=""
  tessero_save_CFLAGS=$CFLAGS; CFLAGS="$CFLAGS -Werror -ffile-prefix-map=/tessero-probe=."
  AC_COMPILE_IFELSE([AC_LANG_PROGRAM([], [])], [TESSERO_PREFIX_MAP=" -ffile-prefix-map=@ext_srcdir@=."])
  CFLAGS=$tessero_save_CFLAGS
  TESSERO_CFLAGS="$TESSERO_CFLAGS$TESSERO_PREFIX_MAP"
  if test "$PHP_TESSERO_OPENMP" != "no"; then
    TESSERO_CFLAGS="$TESSERO_CFLAGS -fopenmp"
    PHP_ADD_LIBRARY(gomp, 1, TESSERO_SHARED_LIBADD)
  fi
  dnl OpenSSF compiler-options hardening, each flag probed so an older toolchain simply drops it. None changes
  dnl floating-point semantics (the parity gate checks results stay bit-exact). Mirrors HARDEN in csrc/Makefile.
  TESSERO_HARDEN=""
  if test "$PHP_TESSERO_HARDEN" != "no"; then
    for f in "-U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=3" -fstack-protector-strong -fstack-clash-protection              -fcf-protection=full -mbranch-protection=standard -ftrivial-auto-var-init=zero              -fno-delete-null-pointer-checks; do
      dnl not on macOS: the arm64 (non-arm64e) unwinder does not follow pac-ret, so C++ exceptions could not
      dnl unwind (csrc/Makefile has the details)
      case "$host_os:$f" in darwin*:-mbranch-protection*) continue ;; esac
      tessero_save_CFLAGS=$CFLAGS; CFLAGS="$CFLAGS -O2 -Werror $f"
      AC_LINK_IFELSE([AC_LANG_PROGRAM([], [])], [TESSERO_HARDEN="$TESSERO_HARDEN $f"])
      CFLAGS=$tessero_save_CFLAGS
    done
    for f in -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack; do
      tessero_save_LDFLAGS=$LDFLAGS; LDFLAGS="$LDFLAGS -Werror $f"
      AC_LINK_IFELSE([AC_LANG_PROGRAM([], [])], [TESSERO_SHARED_LIBADD="$TESSERO_SHARED_LIBADD $f"])
      LDFLAGS=$tessero_save_LDFLAGS
    done
    AC_MSG_NOTICE([tessero hardening flags:$TESSERO_HARDEN])
    TESSERO_CFLAGS="$TESSERO_CFLAGS $TESSERO_HARDEN"
  fi
  PHP_ADD_LIBRARY(m, 1, TESSERO_SHARED_LIBADD)
  dnl LAPACKE (row-major C LAPACK) backs linalg.* (libtessero/src/np_linalg.c); pulls in the system LAPACK/BLAS.
  dnl On macOS/Windows this needs the platform LAPACKE (Accelerate / OpenBLAS-for-MSVC): tracked cross-platform work.
  PHP_ADD_LIBRARY(lapacke, 1, TESSERO_SHARED_LIBADD)
  dnl matmul.c calls cblas_dgemm/cblas_sgemm directly: link BLAS explicitly instead of relying on lapacke pulling it in
  PHP_ADD_LIBRARY(blas, 1, TESSERO_SHARED_LIBADD)
  PHP_SUBST(TESSERO_SHARED_LIBADD)
  dnl the special-function libraries SciPy uses (xsf, Boost.Math, SciPy glue) are C++17 (ADR 0011)
  PHP_REQUIRE_CXX()
  TESSERO_INC="-I@ext_srcdir@/libtessero/third_party -I@ext_srcdir@/libtessero/third_party/xsf -I@ext_srcdir@/libtessero/third_party/scipy -I@ext_srcdir@/libtessero/third_party/boost -I@ext_srcdir@/libtessero/third_party/numpy/include"
  PHP_NEW_EXTENSION(tessero, tessero.c tessero_ndarray.c tessero_solvers.c tessero_ufunc.c tessero_memmap.c tessero_fn.c tessero_dist.c tessero_random.c $TESSERO_KERNEL, $ext_shared,, $TESSERO_CFLAGS, cxx)
  TESSERO_INC=`echo "$TESSERO_INC" | sed "s|@ext_srcdir@|$ext_srcdir|g"`
  TESSERO_CXXFLAGS="-std=c++17 -O2 -fno-math-errno -fvisibility=hidden -DBOOST_MATH_STANDALONE $TESSERO_INC -Wno-deprecated-declarations $TESSERO_HARDEN$TESSERO_PREFIX_MAP"
  TESSERO_CXXFLAGS=`echo "$TESSERO_CXXFLAGS" | sed "s|@ext_srcdir@|$ext_srcdir|g"`
  if test "$PHP_TESSERO_OPENMP" != "no"; then
    TESSERO_CXXFLAGS="$TESSERO_CXXFLAGS -fopenmp"
  fi
  PHP_ADD_SOURCES_X(PHP_EXT_DIR(tessero), $TESSERO_KERNEL_CXX, $TESSERO_CXXFLAGS, shared_objects_tessero, yes)
  dnl the reproducible-f32 primitives MUST build with contraction and fast-math off, as csrc/Makefile does:
  dnl otherwise the compiler may fuse their a*b+c and the results stop being bit-identical across CPUs
  if test -n "$TESSERO_KERNEL_REPRO"; then
    TESSERO_REPRO_CFLAGS=`echo "$TESSERO_CFLAGS -fno-fast-math -ffp-contract=off" | sed "s|@ext_srcdir@|$ext_srcdir|g"`
    PHP_ADD_SOURCES_X(PHP_EXT_DIR(tessero), $TESSERO_KERNEL_REPRO, $TESSERO_REPRO_CFLAGS, shared_objects_tessero, yes)
  fi
  dnl numpy/random builds with contraction off, as in csrc/Makefile: aarch64 GCC and clang otherwise fuse
  dnl lower + range*x and friends, and the draws move 1 ulp off NumPy's x86-64 reference
  TESSERO_NPRAND_CFLAGS=`echo "$TESSERO_CFLAGS -ffp-contract=off" | sed "s|@ext_srcdir@|$ext_srcdir|g"`
  PHP_ADD_SOURCES_X(PHP_EXT_DIR(tessero), $TESSERO_KERNEL_NPRAND, $TESSERO_NPRAND_CFLAGS, shared_objects_tessero, yes)
  PHP_ADD_BUILD_DIR([$ext_builddir/libtessero/src])
  PHP_ADD_BUILD_DIR([$ext_builddir/libtessero/cxx])
  PHP_ADD_BUILD_DIR([$ext_builddir/libtessero/third_party/scipy])
  PHP_ADD_BUILD_DIR([$ext_builddir/libtessero/third_party/numpy/random/src])
  PHP_ADD_BUILD_DIR([$ext_builddir/libtessero/third_party/scipy/quadpack])
  PHP_ADD_BUILD_DIR([$ext_builddir/libtessero/third_party/scipy/zeros])
  PHP_ADD_BUILD_DIR([$ext_builddir/libtessero/third_party/scipy/fitpack])
  dnl GCC's target_clones ifunc symbols ignore -fvisibility=hidden; a version script keeps them local.
  dnl ($ext_srcdir is set by PHP_NEW_EXTENSION) (GNU ld, lld, gold)
  case $host_os in
    linux*|*gnu*|freebsd*)
      TESSERO_SHARED_LIBADD="$TESSERO_SHARED_LIBADD -Wl,--version-script=$ext_srcdir/tessero.map"
      ;;
  esac
  dnl -z nodelete (probed): the module is never unmapped by dlclose() at PHP shutdown, so OpenMP worker threads
  dnl of libgomp can finish without executing unmapped code. A preventive measure, not a reproduction-tested fix
  dnl (docs/architecture/adr/0014-abi-stability.md). Engine::feature('nodelete') reports whether it was applied.
  tessero_save_LDFLAGS=$LDFLAGS; LDFLAGS="$LDFLAGS -Werror -Wl,-z,nodelete"
  AC_LINK_IFELSE([AC_LANG_PROGRAM([], [])], [TESSERO_SHARED_LIBADD="$TESSERO_SHARED_LIBADD -Wl,-z,nodelete"])
  LDFLAGS=$tessero_save_LDFLAGS

  PHP_ADD_EXTENSION_DEP(tessero, json)
  PHP_ADD_EXTENSION_DEP(tessero, spl)
fi
