# Installation

Tessero needs PHP 8.2 or newer on a 64-bit platform. Pick one backend (or
install both; the Laravel package uses the extension when it is loaded and
falls back to FFI otherwise).

## Option A: the FFI package (no compiler needed)

```bash
composer require tessero/tessero
vendor/bin/tessero doctor
```

Requirements:

- `ext-ffi` (bundled with PHP; on Debian/Ubuntu it is part of `php8.x-common`; on
  the official Docker images run `docker-php-ext-install ffi`).
- `ffi.enable=1` for the CLI, or `ffi.enable=preload` plus the preload script
  for PHP-FPM ([Deployment](../operations/deployment.md)).
- A prebuilt `libtessero` for your platform in `vendor/tessero/tessero/lib/<os>-<arch>/`.
  Release packages carry linux-x86_64, linux-aarch64, linux-musl-x86_64,
  darwin-arm64, darwin-x86_64 and windows-x86_64. On anything else run
  `vendor/bin/tessero build` (needs a C compiler) or set `TESSERO_LIB` to a
  library you built.
- For linear algebra only: an LP64 OpenBLAS with LAPACKE
  (`apt install libopenblas0 liblapacke`, `apk add openblas lapack`,
  `brew install openblas`). The copy bundled with `pip install scipy` is found
  automatically. Set `TESSERO_BLAS` to use another build. ILP64 builds are refused
  because they would silently truncate indices.

## Option B: the native extension

```bash
pie install tessero/tessero-ext
```

or from a source checkout:

```bash
cd ext
phpize
./configure --enable-tessero            # --disable-tessero-openmp to drop libgomp
make -j
make test                               # 12 phpt files
sudo make install
echo "extension=tessero" | sudo tee "$(php-config --ini-dir)/30-tessero.ini"
php --ri tessero
```

The extension compiles its own copy of the kernel (`ext/libtessero/`), so it
needs neither `ext-ffi` nor the FFI package. It does not include LAPACK-based
linear algebra, multi-axis reductions or `.npy` I/O yet; install the FFI
package alongside it when you need those ([Choosing a backend](choosing-a-backend.md)).

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
```

GCC 9+ or Clang 12+ on Linux; Apple Clang on macOS (built without OpenMP);
MinGW-w64 on Windows.
