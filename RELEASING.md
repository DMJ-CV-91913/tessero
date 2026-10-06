# Releasing

## Versioning

Semantic Versioning. `libtessero` (`tsr_version()`), the FFI package
(`Tessero::VERSION`), the extension (`PHP_TESSERO_VERSION`), and the
`tessero/laravel` constraint move together and always carry the same version.

## Checklist

1. **Changelog**: move "Unreleased" entries under the new version and date.
   Breaking changes go under their own heading, and each is listed in
   `docs/operations/upgrading.md`.
2. **Version bump** in `csrc/src/alloc.c` (`tsr_version`), `src/Tessero.php`
   (`VERSION`), `ext/php_tessero.h` (`PHP_TESSERO_VERSION`) and
   `laravel/composer.json` (`tessero/tessero` constraint).
3. **Sync and regenerate**:
   ```bash
   bash tools/sync-ext.sh
   php -d ffi.enable=1 -d extension=ext/modules/tessero.so tools/gen-api-docs.php
   python tests/fixtures/generate_parity.py     # only if NumPy/SciPy were updated
   python tests/fixtures/generate_ufunc.py
   python tests/fixtures/generate_npy.py
   ```
4. **Full verification**: all CI jobs green on the release commit (native ×6,
   PHP matrix, sanitizers, fuzz, extension + ASan, Laravel app, preload, docs,
   Docker, benchmarks) and the last nightly run green (30-minute fuzzing per
   target, 5.5-hour soaks, property tests ×50).
   **Long soak**: `tools/soak.php` for at least 24 hours per backend on
   production-like hardware (see `docs/operations/soak-testing.md`);
   attach the JSON summaries to the release.
5. **Tag** `vX.Y.Z` (signed). The `release` CI job builds the six binaries,
   packages them, and attaches the tarball and a `SHA256SUMS` file to the GitHub
   release.
6. **Publish**: Packagist picks up the tag for `tessero/tessero`. The
   `laravel/` and `ext/` directories are published as read-only subtree splits
   (`tessero/laravel`, `tessero/tessero-ext`) tagged with the same version. `ext/`
   is self-contained (it carries its own `libtessero/` copy), so PIE can build
   the split directly.
7. **Docs**: the docs job deploys the site for the tag.
8. **Announce** with a summary, upgrade notes and checksums.

## Hotfixes

Branch from the tag (`release/X.Y`), fix with a test, bump the patch version,
and follow the checklist. Forward-port to `main`.

## Yanking

If a release ships a correctness or security defect, publish a patch release
first, then mark the bad version as yanked on the GitHub release and in the
changelog, with the reason.
