## What this changes

<!-- A short description, and the issue it closes (Closes #NNN). -->

## Checklist

- [ ] `QUICK=1 bash tools/parity/gate.sh` is green (or a note on why it can't run in this environment)
- [ ] New or changed numerical behaviour is covered by generated fixtures that pass on **both** backends (FFI + extension)
- [ ] No tolerances loosened and no generated files hand-edited (coverage is measured, not typed)
- [ ] Runnable `examples/` and/or docs updated if the public API changed (`php tools/run-examples.php`)
- [ ] Changelog entry added under "Unreleased" if the change is user-facing
- [ ] `vendor/bin/tessero doctor` still passes on a built checkout

## Notes for reviewers

<!-- Anything that needs attention: tricky memory handling, a deliberate deviation, a benchmark result. -->
