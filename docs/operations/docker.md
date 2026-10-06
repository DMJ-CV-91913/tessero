# Docker

The repository ships a reference image, `docker/Dockerfile`: PHP-FPM with
ext-tessero, the FFI package, OpenBLAS/LAPACKE and production INI settings
(`docker/tessero.ini`).

```bash
docker build -f docker/Dockerfile -t tessero-fpm .
docker build -f docker/Dockerfile --build-arg PHP_VERSION=8.3 -t tessero-fpm:8.3 .
docker run --rm tessero-fpm php -d ffi.enable=1 /opt/tessero/bin/tessero doctor
```

The build has two stages:

1. **build**: compiles `libtessero` (and runs its C tests), then compiles
   the extension against the image's PHP and runs the phpt suite. A failing
   test fails the image build.
2. **runtime**: `php:*-fpm-bookworm`, plus `libgomp1` (OpenMP),
   `libopenblas0-pthread` and `liblapacke` (linear algebra), `ffi` and
   `opcache`, the compiled extension, and the FFI package under
   `/opt/tessero`. It ends with a smoke check that both backends load.

In an application image you would normally get the FFI package through
Composer (`vendor/tessero/tessero`) and copy only the extension from the build
stage. Point `opcache.preload` at `vendor/tessero/tessero/resources/preload.php`.

## Alpine

Alpine uses musl. Use the `linux-musl-x86_64` binary for the FFI package, and
build the extension in an Alpine stage:

```dockerfile
FROM php:8.4-fpm-alpine AS build
RUN apk add --no-cache $PHPIZE_DEPS linux-headers libgomp
COPY ext/ /src/ext/
RUN cd /src/ext && phpize && ./configure --enable-tessero && make -j"$(nproc)" && make install

FROM php:8.4-fpm-alpine
RUN apk add --no-cache libgomp openblas lapack libffi \
 && apk add --no-cache --virtual .deps libffi-dev $PHPIZE_DEPS \
 && docker-php-ext-install ffi opcache && apk del .deps
COPY --from=build /usr/local/lib/php/extensions/ /usr/local/lib/php/extensions/
RUN docker-php-ext-enable tessero
```

Alpine's `lapack` package does not include the LAPACKE C interface on every
release. If `tessero doctor` reports no LAPACKE, use the Debian image for
linear algebra workloads. The extension does not need LAPACK.

## Resource limits in containers

- Set `tessero.memory_budget` (or `TESSERO_MEMORY_BUDGET`) so that
  `pm.max_children × (memory_limit + memory_budget)` fits inside the
  container's memory limit ([Memory and threads](memory-and-threads.md)).
- OpenMP detects the host's CPUs, not the container's CPU quota. Keep
  `tessero.threads` at 1 under FPM, and set it explicitly (not "all cores") for
  worker containers.
- OpenBLAS likewise starts one thread per host CPU by default. The FFI package
  sets it to 1 in non-CLI SAPIs; set `TESSERO_BLAS_THREADS` for CLI workers.

## Status

The Dockerfile is built by CI (`docker` job). It has not been built in the
reference environment, which has no container daemon.
