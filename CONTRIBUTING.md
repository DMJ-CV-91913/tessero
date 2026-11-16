# Contributing to Tessero

Tessero is open source under the BSD-3-Clause licence: anyone may read, use,
modify and redistribute it under that licence. It is **not open to outside
contributions**. Douglas Johnson writes and maintains all of the code, and the
project does not accept pull requests or patches, and has no public issue
tracker.

You are welcome to fork the project under the licence.

## Reporting a bug

Email **support@tessero.org**. The most valuable report is a wrong number.
Include:

- the input (export it with `Tessero\Io\Npy::save` or `serialize()`);
- what NumPy/SciPy return for the same input;
- the output of `vendor/bin/tessero info --json`.

For an installation problem, include the output of `tessero doctor` and
`php -v`, and your OS and architecture.

Security problems: do **not** send them to the support address; follow
`SECURITY.md` (security@tessero.org).

## Building from source

Requirements: PHP 8.2+ with `ext-ffi` and headers (`php-dev`), GCC or Clang,
Python 3 with NumPy and SciPy (for fixtures), Composer.

```bash
make -C csrc                         # build libtessero into lib/<platform>/
make -C csrc test                    # C unit tests
make -C csrc sanitize                # C tests under ASan + UBSan
composer install
vendor/bin/phpunit                   # unit + parity + fuzz (+ ext when loaded)

cd ext && phpize && ./configure --enable-tessero && make && make test && cd ..
php -d extension=ext/modules/tessero.so vendor/bin/phpunit --testsuite ext

php -d ffi.enable=1 tests/docs/examples.php     # doc examples
bash tests/e2e/preload.sh                       # FPM-style preload
php -d ffi.enable=1 bench/run.php               # benchmarks with gates
```

After changing anything in `csrc/`, run `bash tools/sync-ext.sh` to update
the extension's copy. CI fails if they differ.

## Engineering rules (how the maintainer works)

1. **Tests come from the reference.** New numerical behaviour gets a case
   computed by NumPy/SciPy, not a hand-typed expected value.
2. **Both backends.** If a feature exists in the FFI package and the extension,
   the extension parity suite must cover it.
3. **No PHP loops over elements.** Loops over data belong in `csrc/`.
4. **The ABI header stays FFI-parseable**: declarations only in
   `csrc/include/tessero.h`, no macros or includes. New op codes go in
   `csrc/src/ops.h` and `Native\Abi` together (a unit test checks them).
5. **Memory:** array-sized allocations use `tsr_alloc`/`tsr_free`, so they count
   against the budget. Every error path frees what it allocated. ASan must stay clean.
6. **No undefined behaviour:** signed integer arithmetic on data uses the
   `TSR_W*` wrap macros.
7. **Errors are typed:** C returns a negative code; bindings map it to an
   exception with a message that names shapes, sizes or the offending index.
8. **Documentation is part of the change.** Guide page, API docblock,
   regenerated API reference and a changelog entry under "Unreleased".
9. **Architecture changes get an ADR** in `docs/architecture/adr/`.

## Style

- PHP: PSR-12, `declare(strict_types=1)`, typed signatures, docblocks for
  anything public. In hot loops, avoid compound assignment on array elements,
  list destructuring, and element-by-element reads of `CData` (copy out with
  `FFI::string` + `unpack`). All three triggered PHP JIT miscompilations; see
  `docs/project/upstream-bugs.md`.
- C: C11, 4-space indent, `tsr_` prefix for public symbols, `static` for
  everything else, comments that explain *why*.
- Commit messages: imperative subject of 72 characters or fewer, body explaining
  motivation.
