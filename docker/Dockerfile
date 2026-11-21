# Reference production image: PHP-FPM with ext-tessero (native backend) and the
# FFI package's LAPACK support. Build from the repository root:
#
#   docker build -f docker/Dockerfile -t tessero-fpm .
#   docker run --rm tessero-fpm php /opt/tessero/bin/tessero doctor
#
# CI builds this image and runs the checks at the end on every commit.

# Base images are pinned by digest (reproducible builds; Dependabot's docker ecosystem proposes digest updates).
# To move to another PHP version, change both tags and their digests together.

# ---------------------------------------------------------------- build stage
FROM php:8.4.28-cli-bookworm@sha256:85da896b3eeb8bcd83fe612e51a529b7d5983a1fab92ef5f63c7ac6e2842fc46 AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends $PHPIZE_DEPS libffi-dev liblapacke-dev libopenblas-dev \
 && rm -rf /var/lib/apt/lists/*
COPY csrc/ /src/csrc/
COPY ext/ /src/ext/
# the kernel for the FFI package ...
RUN make -C /src/csrc -j"$(nproc)" && make -C /src/csrc test
# ... and the extension (its own copy of the kernel)
RUN cd /src/ext && phpize && ./configure --enable-tessero && make -j"$(nproc)" \
 && NO_INTERACTION=1 make test TESTS=tests \
 && make install

# ---------------------------------------------------------------- runtime stage
FROM php:8.4.28-fpm-bookworm@sha256:e4b1eec29a5be22428aa93fb43da75a0e39006e403da0c3e93631dde92f067df
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libopenblas0-pthread liblapacke libffi-dev \
 && docker-php-ext-install ffi opcache \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /usr/local/lib/php/extensions/ /usr/local/lib/php/extensions/
RUN docker-php-ext-enable tessero
COPY docker/tessero.ini /usr/local/etc/php/conf.d/zz-tessero.ini
# the FFI package (normally vendor/tessero/tessero from composer)
COPY src/ /opt/tessero/src/
# the FFI binding reads the C declarations from the header at run time (Native\Library)
COPY csrc/include/ /opt/tessero/csrc/include/
COPY bin/ /opt/tessero/bin/
COPY resources/ /opt/tessero/resources/
COPY tests/autoload.php /opt/tessero/tests/autoload.php
COPY --from=build /src/lib/ /opt/tessero/lib/

# smoke checks: both backends load, LAPACK is found, operators work
RUN php -d ffi.enable=1 /opt/tessero/bin/tessero doctor \
 && php -r '$a = Tessero\Ext\NDArray::array([1.0, 2.0, 3.0]); exit(($a * 2)->sum() === 12.0 ? 0 : 1);'
