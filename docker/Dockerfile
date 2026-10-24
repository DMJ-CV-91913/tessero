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
FROM php:8.4.27-cli-bookworm@sha256:a2d10c137d6ec9a41331e8e03e9d5468c85bfa8b1971436ffb2ad4f19d890e27 AS build
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
FROM php:8.4.27-fpm-bookworm@sha256:e0005a213080e1c580c961e47bc6fdd9bebd3b7bb484e80479507af41547cffa
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
