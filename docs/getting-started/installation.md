# Installation

Tessero needs PHP 8.2 or newer on a 64-bit platform. It reaches its C kernel in one of two ways: the native
extension, or the FFI package, which will need no compiler once prebuilt binaries are released (not yet; see
Option B).

**Use the native extension.** It calls the kernel without the FFI layer. In the
[speed sweep](../project/speed-sweep.md) a small call takes 0.074 of NumPy/SciPy's time on average (geomean over
1,022 functions) and is faster on 966 of them. Through FFI each call carries a fixed cost of a few microseconds,
which makes small calls to NumPy's core functions (`zeros`, `asarray`, `dot`, `sqrt`) slower than in NumPy. Both
backends give the same numbers.

The extension's classes (`Tessero\Ext\NDArray`, `Np`, `Linalg`, `Special`, `Stats`, `Signal`, ... ) cover the same
NumPy and SciPy functions, each checked against NumPy/SciPy on the extension. Install the PHP package (Option B)
alongside it for the PHP-level APIs and the command-line tools. Where the extension is loaded, the Laravel package
and the classes that look for it (`KDTree`, `ConvexHull`, `Delaunay`, `curve_fit`, `least_squares`) run on it.
The PHP package's other classes still go through FFI, so keep Option B's FFI settings if you use them. Use the FFI
package alone where you cannot compile or load an extension, for example on shared hosting.

## Option A: the native extension (recommended)

From a source checkout:

```bash
sudo apt install php8.4-dev g++ liblapacke-dev libopenblas-dev   # Debian/Ubuntu; match your PHP version
cd ext
phpize
./configure --enable-tessero            # --disable-tessero-openmp to drop libgomp
make -j
make test                               # phpt suite
sudo make install
echo "extension=tessero" | sudo tee "$(php-config --ini-dir)/30-tessero.ini"
php --ri tessero
```

Build requirements: the PHP headers (`phpize`), a C compiler and a C++17 compiler (the special-function libraries
SciPy uses are C++), and LAPACKE with BLAS, which the build links for linear algebra and matrix products. The
extension compiles its own copy of the kernel (`ext/libtessero/`), so it needs neither `ext-ffi` nor a prebuilt
`libtessero`. Build it on Linux. On macOS and Windows it needs a LAPACKE the build can find, which is still open
cross-platform work. There, use the FFI package with a library built by `make -C csrc`.

Once the package is registered with PIE (not yet: tracked as TSR-011), `pie install tessero/tessero-ext` will build
it for the running PHP. Build details, INI settings and fleet packaging are in
[Native extension](../operations/native-extension.md).

## Option B: the FFI package

```bash
composer require tessero/tessero
vendor/bin/tessero doctor
```

Not on Packagist yet, and no prebuilt binaries are published yet (both tracked as TSR-011). Until then, add the
repository as a Composer path or VCS repository and build the library yourself (`vendor/bin/tessero build` or
`make -C csrc`, which needs a C compiler). Once released, the package carries prebuilt binaries and needs no
compiler.

Requirements:

- `ext-ffi` (bundled with PHP; on Debian/Ubuntu it is part of `php8.x-common`; on
  the official Docker images run `docker-php-ext-install ffi`).
- `ffi.enable=1` for the CLI, or `ffi.enable=preload` plus the preload script
  for PHP-FPM ([Deployment](../operations/deployment.md)).
- A `libtessero` for your platform in `vendor/tessero/tessero/lib/<os>-<arch>/`.
  Release packages will carry linux-x86_64, linux-aarch64, linux-musl-x86_64,
  darwin-arm64, darwin-x86_64 and windows-x86_64. Until then, or on anything else, run
  `vendor/bin/tessero build` (needs a C compiler) or set `TESSERO_LIB` to a
  library you built.
- For linear algebra only: an LP64 OpenBLAS with LAPACKE
  (`apt install libopenblas0 liblapacke`, `apk add openblas lapack`,
  `brew install openblas`). The copy bundled with `pip install scipy` is found
  automatically. Set `TESSERO_BLAS` to use another build. ILP64 builds are refused
  because they would silently truncate indices.
- Supported LAPACK: 3.11 or newer (Ubuntu 24.04, current Debian, Homebrew, the SciPy wheels). On LAPACK 3.10
  (Ubuntu 22.04) Tessero runs safely, with workarounds for two LAPACKE 3.10 defects, but four functions may
  return different though valid values: `signal.savgol_filter` and `linalg.eig_banded` (eigenvector signs),
  `linalg.subspace_angles` (a tiny angle returned as 0) and `linalg.ordqz` (sign and last digits). CI checks
  that 3.10 runs without a crash or any other difference.

## Option C: Laravel

```bash
composer require tessero/laravel                   # pulls tessero/tessero
php artisan vendor:publish --tag=tessero-config    # optional: config/tessero.php
php artisan tessero:doctor
```

The service provider is auto-discovered. See the [Laravel guide](../guide/laravel.md).

## Check the installation

```bash
vendor/bin/tessero doctor
```

`doctor` reports the PHP version, whether `ext-ffi` is loaded and what
`ffi.enable` is set to, whether a binary exists for this platform and loads in
this SAPI, the SIMD level the kernel selected (baseline, AVX2, AVX-512, NEON),
OpenMP support, and whether an LP64 OpenBLAS with LAPACKE was found. Each
failing line says what to install or set.

`vendor/bin/tessero info` prints the same facts as JSON (for monitoring), and
`vendor/bin/tessero php-ini` prints the preload lines for your paths.

## Build the kernel yourself

```bash
make -C csrc                 # lib/<os>-<arch>/libtessero.{so,dylib,dll}
make -C csrc test            # 174 C checks
make -C csrc sanitize        # the same under ASan + UBSan
make -C csrc OPENMP=         # without OpenMP
sudo make -C csrc install    # library + public headers under PREFIX (default /usr/local)
```

`make install` puts the library in `$PREFIX/lib` (on Windows the DLLs go to `$PREFIX/bin`) and `tessero.h`,
`tessero_version.h`, `tessero_native.h` and `tessero_repro_f32.h` in `$PREFIX/include`, which is the layout a
native consumer such as Talario expects from `--with-tessero=$PREFIX`. `DESTDIR` stages a package build.

The library records the commit it was built from: `vendor/bin/tessero info` and `tessero:doctor --json` show it
as `commit`, so you can check a deployed library against the reference in `composer.lock`. A build outside a git
checkout reports `unknown`; pass `make -C csrc TSR_BUILD_COMMIT=<sha>` there.

On Windows (MSYS2 MINGW64), `make -C csrc` also copies the MinGW runtime DLLs that `tessero.dll` needs
(OpenBLAS, gfortran, winpthread, ...) next to it, and the FFI loader searches that directory, so `PATH` needs no
change. Keep those DLLs with `tessero.dll` if you move it.

GCC 9+ or Clang 12+ on Linux; Apple Clang on macOS (built without OpenMP);
MinGW-w64 on Windows.
