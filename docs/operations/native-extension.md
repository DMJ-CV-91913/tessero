# Native extension (ext-tessero)

## Building

```bash
cd ext
phpize
./configure --enable-tessero [--disable-tessero-openmp]
make -j"$(nproc)"
make test                         # phpt suite
make install                      # into $(php-config --extension-dir)
```

| Requirement | Notes |
|---|---|
| PHP 8.2–8.5 headers (`php-dev` / `php8.x-dev`) | the extension uses the stable Zend API; one build per PHP minor version and thread-safety mode |
| GCC ≥ 9 or Clang ≥ 12 | C11 atomics; GCC `target_clones` gives AVX2/AVX-512 paths on x86-64 |
| libgomp | only with OpenMP (the default). `config.m4` links `-lgomp`; when building with Clang, install libgomp or pass `--disable-tessero-openmp` |

The build compiles `ext/libtessero/*.c` (a copy of `csrc/`) into the module.
Kernel symbols are compiled with hidden visibility, and a linker version
script (`ext/tessero.map`) keeps the ifunc symbols that GCC's `target_clones`
would otherwise export. The module exports only `get_module` (CI checks this
with `nm`), so it can share a process with the FFI library.

Windows: `config.w32` is provided (`--enable-tessero`). OpenMP is not
enabled on Windows builds.

## Installing with PIE

```bash
pie install tessero/tessero-ext
```

PIE (the PHP Foundation's installer) builds the release source for the
running PHP and writes the INI file. Pin a version in production:
`pie install tessero/tessero-ext:0.2.0`.

## Packaging for a fleet

Build once per (PHP minor, NTS/ZTS, architecture, libc) and ship the `.so`:

```dockerfile
FROM php:8.4-cli AS ext
RUN apt-get update && apt-get install -y $PHPIZE_DEPS
COPY ext/ /src/ext/
RUN cd /src/ext && phpize && ./configure --enable-tessero && make -j"$(nproc)" && make test

FROM php:8.4-fpm
COPY --from=ext /src/ext/modules/tessero.so /usr/local/lib/php/extensions/no-debug-non-zts-20240924/
RUN docker-php-ext-enable tessero
```

A module built for one PHP minor version will not load in another. PHP
refuses it with "Module compiled with module API=…". Rebuild per version.

## Configuration

| INI | Default | Changeable | Meaning |
|---|---|---|---|
| `tessero.threads` | 1 | anywhere (`ini_set`) | OpenMP threads for large element-wise kernels and MDP sweeps; capped at 4 × cores |
| `tessero.memory_budget` | 0 (unlimited) | anywhere | native memory cap per process; accepts `256M`, `2G` |
| `tessero.epsilon` | 1e-6 | anywhere | default MDP tolerance |

Settings are re-applied from INI at the start of every request (RINIT), so a
value changed at runtime with `Engine::setMaxThreads()` lasts for the current
request only. That makes FPM and Octane workers safe by default.

## Verifying a host

```bash
php --ri tessero
php -r 'var_dump(Tessero\Ext\Engine::info());'
```

`Engine::info()` returns `extension`, `kernel`, `simd` (`baseline`, `avx2`,
`avx512`, `neon`), `openmp`, `threads`, `memory_in_use`, `memory_peak`,
`memory_budget` and `zts`.

## Sanitizer build (for contributors and incident analysis)

```bash
./configure --enable-tessero CFLAGS="-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer" LDFLAGS="-fsanitize=address,undefined"
make -j && USE_ZEND_ALLOC=0 LD_PRELOAD=$(gcc -print-file-name=libasan.so) NO_INTERACTION=1 make test
```

CI runs the phpt suite this way on every commit. The one expected report is a
small leak inside libgomp's initialisation, which is benign and suppressed in CI.

## ZTS (thread-safe PHP)

The extension builds and runs under ZTS. The kernel's thread count, memory
counter and budget are process-wide rather than per PHP thread, because they
live in the kernel. Under ZTS servers (FrankenPHP in thread mode, Apache
`mpm_event` with mod_php ZTS) every PHP thread shares one budget. Size it for
the whole process.
