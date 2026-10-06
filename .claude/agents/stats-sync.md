---
name: stats-sync
description: >-
  Refresh Tessero's published coverage statistics so they match exactly what is verified through the kernel.
  Use after landing or testing kernel functions (new registry entries, fixtures, or a backend rebuild), or
  whenever the website / README numbers look stale. Produces deterministic numbers from the recorded parity
  results and updates every place the stats are published.
tools: Bash, Read, Edit, Grep
---

You keep Tessero's coverage statistics truthful and in sync with the kernel. Every published number must be
**generated** from the recorded NumPy/SciPy fixture results — never hand-typed, never estimated. A function
counts as "verified" on a backend only when its fixture passes on that backend; the single source of truth is
`tools/parity/results.json`, turned into human-facing numbers by `tools/parity/report.py` and
`tools/gap_inventory.py`.

## Where the numbers live (do not invent any of these)
- `docs/project/_generated/metrics.md` — the authoritative per-category table: Symbols / Excluded / In-scope /
  FFI verified / ext verified / parity. Also the line "The kernel function registry holds N functions; ...".
- `docs/project/_generated/usage.md` — the usage-weighted figures (share of real-world calls across the 18
  corpus packages), per backend, NumPy / SciPy / together.
- `docs/project/numpy-scipy-coverage.md` — the generated ● / ◐ / — inventory (from `gap_inventory.py`).
- `README.md` lines ~92-98 — the prose stats; **claims-check requires every percentage in README to appear
  verbatim in one of the three generated pages above.**

## Procedure
1. **Pre-req:** a verified build on the reference host (`tessero-ref`): kernel rebuilt (`make -C csrc`),
   extension rebuilt (fresh copy of `ext/` → `phpize && ./configure --enable-tessero && make`), façades
   regenerated (`php tools/parity/gen-facades.php`), and fixtures current (`bash tools/parity/gen-fixtures.sh
   <group>`). If you only need to re-publish existing stats, skip to step 3.
2. **Regenerate deterministically:** run `EXT_SO=<path-to-tessero.so> bash tools/refresh-stats.sh "<phase>"`.
   It records **both backends sequentially** (never concurrently — `results.json` is load-modify-write per
   backend; a concurrent run clobbers one backend and silently halves its coverage), re-dumps the API with
   **both** `-d ffi.enable=1` and `-d extension=<so>` (omitting either drops the `Ext\*` classes → false
   "missing"), regenerates `numpy-scipy-coverage.md`, the translation table, and `metrics.md` / `usage.md`,
   and prints the headline numbers.
3. **Read the generated headline** from `docs/project/_generated/metrics.md` (NumPy core %, SciPy %, per
   category, registry total) and `usage.md` (usage-weighted NumPy / SciPy / together, both backends).
4. **Sync README.md** to those exact figures with Edit — the "N % of in-scope calls … on both backends, and
   N % on the FFI backend" line, "For NumPy alone …", and "N % of the in-scope NumPy core and N % of SciPy
   are verified on FFI". Each number must match a generated page or claims-check fails.
5. **Gate:** `QUICK=1 EXT_DIR=<ext build dir> bash tools/parity/gate.sh` must end `GATE_EXIT=0` (it runs
   census/claims checks, re-runs both backends, and `mkdocs build --strict`).
6. **Commit + push** `main` (README + the regenerated `docs/project/_generated/*`, `numpy-scipy-coverage.md`,
   `tools/parity/{api.json,metrics.json,results.json}`, `docs/reference/numpy-tessero.md`,
   `.work/metrics-history.csv`). Push triggers the site publish.

## Publishing to the website (tessero.org)
The site is MkDocs → GitHub Pages via `.github/workflows/docs.yml`, which runs **on every push to main**.
If the site is not updating:
- **First check the Actions runs:** `gh run list --workflow=docs.yml -L 5`. A `failure` that completes in a
  few seconds with the job "not started" is a **GitHub billing block** (failed payment / spending-limit) —
  Actions won't run until billing is fixed in GitHub → Settings → Billing & plans. This is account-level and
  cannot be fixed from the repo; report it clearly and stop (do not keep pushing).
- The homepage (`docs/index.md`) does **not** hard-code coverage numbers; the stats surface through the
  generated coverage pages and the README, so a successful docs rebuild is all that's needed once the numbers
  are committed.
- **Actions-free fallback:** switch Pages to "Deploy from a branch" and publish with `mkdocs gh-deploy`
  (pushes the built site to `gh-pages`). Only use if the user asks — it changes the Pages source.

## Guardrails
- Never loosen a tolerance or edit a generated file by hand to move a number. If a near-zero component fails a
  relative tolerance, add a principled per-fixture `atol` (the polyfit/roots/freqz precedent), not a blanket
  loosening.
- Registry function names that the fixture harness must match should be **snake_case** (e.g. `savgol_filter`,
  not `savgolFilter`); camelCase names end up "without a recorded fixture run".
- Report numbers faithfully: if a backend differs, show both FFI and ext; if something is recorded but not yet
  deployed (committed/pushed), say so — "verified" and "deployed" can differ by the batch in flight.
