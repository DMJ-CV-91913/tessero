# Contributing to Tessero

Thank you for helping. Tessero is numerical infrastructure, so the bar is
correctness first, then clarity, then speed.

## Ways to help

- **Report a wrong number.** This is the most valuable contribution. Include
  the input (export it with `Tessero\Io\Npy::save` or `serialize()`), what
  NumPy/SciPy return for the same input, and `vendor/bin/tessero info --json`.
- **Report an installation problem** with the output of `tessero doctor` and
  `php -v`, and your OS/architecture.
- **Improve the documentation.** The docs are Markdown under `docs/`
  (`mkdocs serve` to preview).
- **Pick an item from the roadmap** (`docs/project/roadmap.md`). Comment on the
  issue first so work is not duplicated.

Security problems: do **not** open an issue; see `SECURITY.md`.

## Development setup

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

## Rules for changes

1. **Tests come from the reference.** New numerical behaviour gets a case in
   `tests/fixtures/generate_parity.py` computed by NumPy/SciPy, not a
   hand-typed expected value. Regenerate with
   `python tests/fixtures/generate_parity.py` and commit the fixture.
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
8. **Documentation is part of the change.** Update the guide page and the API
   docblock, regenerate the API reference
   (`php -d ffi.enable=1 -d extension=tessero tools/gen-api-docs.php`), and add
   a changelog entry under "Unreleased".
9. **Architecture changes need an ADR** in `docs/architecture/adr/`.

## Style

- PHP: PSR-12, `declare(strict_types=1)`, typed signatures, docblocks for
  anything public. In hot loops, avoid compound assignment on array elements,
  list destructuring, and element-by-element reads of `CData` (copy out with
  `FFI::string` + `unpack`). All three triggered PHP JIT miscompilations; see
  `docs/project/upstream-bugs.md`.
- C: C11, 4-space indent, `tsr_` prefix for public symbols, `static` for
  everything else, comments that explain *why*.
- Commit messages: imperative subject ≤ 72 characters, body explaining
  motivation.

## Pull requests

- One topic per pull request, with tests.
- CI must be green. For numerical changes, say in the description which parity
  cases cover the change.
- Maintainers review within a week. Two approvals are needed for changes to
  `csrc/` or `ext/`, one elsewhere (see `GOVERNANCE.md`).

By contributing you agree that your contribution is licensed under the
project's BSD-3-Clause licence and that you follow the Code of Conduct.
