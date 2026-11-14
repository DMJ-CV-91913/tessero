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
FROM php:8.5.12-cli-bookworm@sha256:050713ce008e2360ea9dc96e1f9a075690132388d917fc3552806f7a7be6f2a0 AS build
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
FROM php:8.5.12-fpm-bookworm@sha256:eec275cd99580e0a6e103dd343a7c160c86917003f6b9bba2d4c75d82419c683
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
