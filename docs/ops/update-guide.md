# Tessero: Update & Development Guide

How to change Tessero safely and get the change onto the ref server. The governing
rule mirrors the parity program (see `CLAUDE.md`): **covered means verified -- a
change ships only when the parity gate is green on BOTH backends (FFI + ext).
Never loosen a tolerance or edit generated files to get green.**

> **Scope.** This guide governs the **Tessero repo** only
> (`github.com/DMJ-CV-91913/tessero`, ref server `tessero-ref`). It builds the
> native numerical core `libtessero` and its two PHP bindings. The **Talario engine**
> (`github.com/DMJ-CV-91913/talario`) is a separate repo that only *consumes* a built
> `libtessero` via `--with-tessero` -- see §4 and Talario's own
> `docs/ops/update-guide.md`. The `tesserophp-org` / `talariophp-org` websites are
> separate repos again.

## 0. Shell & user conventions (used throughout)
Every command block below is tagged with where it runs, because the two
environments use different shells and users:
- **[dev / PowerShell]** -- your Windows dev box, in PowerShell, from the local
  clone `C:\xampp\htdocs\tessero\tessero` (note: the real repo is this **inner**
  folder; the outer `tessero\` is a scratch area).
- **[ref / bash]** -- the Ubuntu build server (delta1cx), in bash, as the **`tessero`**
  user (the repo's `tessero-ref` remote resolves here), from `~/tessero`. Note: the
  clone that feeds Talario is a *separate* checkout under user `kinetix` (§4), and the
  website deploy is a *third* user, `snoopy72` (§7).

Git is the only thing that crosses between the two. The built `libtessero` is a
per-host native artifact and is **never** synced over git -- you build it on the ref
server (and any box that runs the extension), from pushed source.

## 1. The loop (source of truth = the public repo)
Edit on the dev box -> commit -> push -> the ref server pulls, builds, and
**gates on the parity suite**.

**[dev / PowerShell]** -- commit + push (end commit messages with the
Co-Authored-By/Claude-Session trailers):
```powershell
git add -A
git commit -m "..."
git push origin main
```

**[ref / bash]** -- pull, build the core, run the gate:
```bash
ssh tessero-ref
cd ~/tessero
git fetch --prune origin && git pull --ff-only origin main
make -C csrc clean && make -C csrc          # -> lib/<platform>/libtessero.so
bash tools/sync-ext.sh                       # refresh ext/libtessero from csrc (kernel changed)
QUICK=1 bash tools/parity/gate.sh            # fast gate; drop QUICK=1 for the full sanitize+fuzz+soak+bench
```
`QUICK=1` skips sanitizers, fuzzing, soak and benchmarks (per `CLAUDE.md`); run the
full `bash tools/parity/gate.sh` before a release or anything ABI-affecting. The gate
runs BOTH backends (`run-tests.php --backend=ffi` and `--backend=ext`). Only after
green do you let a consumer relink (§4).

## 1a. Build prerequisites (ref server, once)
**[ref / bash]** -- the Linux linalg path links LAPACKE from OpenBLAS
(`csrc/Makefile` adds `-llapacke -lblas` on Linux). Install the dev packages:
```bash
sudo apt-get update && sudo apt-get install -y build-essential liblapacke-dev libopenblas-dev
```
`make portable` (adds `-DTSR_NO_CLONES`) is the build for musl / cross / non-native
targets; the default `make` uses `target_clones` for runtime SIMD dispatch.

## 2. Build outputs (the consumption surface)
`make -C csrc` produces, relative to the repo root:
- **Library:** `lib/<platform>/libtessero.<ext>` -- on the ref server that is
  `lib/linux-x86_64/libtessero.so` (`linux-musl-x86_64` under musl; `.dylib` on
  macOS, `tessero.dll` on Windows).
- **Header / ABI:** `csrc/include/tessero.h` -- the whole public C ABI (this is what
  `tsr_gemm` and friends are declared in; Talario's adapter binds against it).

These two paths are the entire surface anything native (Talario) links against.
`make -C csrc clean` removes the `build/` objects; the Makefile keys off timestamps,
so clean before a build that must include recent edits.

## 3. The two PHP bindings (same kernel)
Tessero ships one kernel behind two bindings (ADR 0003, "one kernel, two bindings"):
- **FFI** (`src/*.php` facades): loads the prebuilt `libtessero` at runtime, no
  compiler needed. The facades are **generated** by
  `php tools/parity/gen-facades.php` after a kernel build -- do not hand-edit them.
- **Zend extension** (`ext/`): the kernel compiled in. `ext/libtessero` is a **synced
  copy** of `csrc` -- run `bash tools/sync-ext.sh` after any kernel change, then build
  the ext: copy `ext/` to a scratch dir, `phpize && ./configure --enable-tessero && make`.

Both read the same runtime function registry (`csrc/src/fn_core.c`); the gate proves
they agree.

## 4. Feeds Talario (the `--with-tessero` hand-off)
Talario is a **separate repo** that links `libtessero` natively. On a box that runs
both, after a green Tessero gate here, you rebuild Talario against the just-built core
(this is Talario's §1a: `--with-tessero=$TESSERO_PREFIX`).

Talario links `libtessero` natively from a **self-contained prefix**: one directory
holding both `include/tessero.h` and `lib/libtessero.so`. It points `--with-tessero`
there and bakes `<prefix>/lib` into `talario.so` as RUNPATH, so there is **no**
`LD_LIBRARY_PATH` / `TALARIO_TESSERO_LIB` needed at runtime.

On delta1cx the Talario deploy user (`kinetix`) keeps its **own** clone of this repo at
`/home/kinetix/tessero`, built so that `csrc/` is that prefix -- `csrc/include/tessero.h`
+ `csrc/lib/libtessero.so` both present (verified: `readelf -d .../talario.so` shows
`RUNPATH [/home/kinetix/tessero/csrc/lib]`). So the Talario-side one-shot (run in the
**Talario** repo, as its deploy user, not here) is:
```bash
# [talario / bash as kinetix on delta1cx] in ~/talario, after rebuilding libtessero in that clone
# use $HOME, NOT ~ -- tilde does not expand after = in --with-tessero=...
scripts/update.sh --test --no-deploy --with-tessero="$HOME/tessero/csrc"
```
Caveats:
- That is a **separate clone** from this repo's `tessero-ref` checkout -- on delta1cx
  they are even different users (`kinetix` vs `tessero`) at different commits. Both pull
  from the same `origin`, so keep them on compatible commits: a Talario link is only as
  current as the Tessero clone it points at.
- This repo's `csrc/Makefile` emits the lib to `lib/<platform>/`, but the consumer
  prefix needs it under `csrc/lib/`. The kinetix clone has `libtessero.so` directly in
  `csrc/lib/`; if a build emits to `lib/<platform>/` instead, copy or symlink it into
  `csrc/lib/` (or point `--with-tessero` at a prefix laid out with both subdirs).

## 5. Rollback
- Code: **[dev / PowerShell]** `git revert <sha>` (or reset a branch) -> `git push`,
  then **[ref / bash]** re-pull + `make -C csrc clean && make -C csrc` +
  `QUICK=1 bash tools/parity/gate.sh` to re-gate the rolled-back tree.
- A consumer (Talario) keeps linking the **old** `libtessero` until it is rebuilt
  against the new one; reverting here does not change an already-linked `talario.so`
  until that rebuild. Safe: gate here, then relink the consumer.

## 6. Coverage / metrics hand-off
Coverage numbers are **generated**, never typed by hand (`CLAUDE.md`). After a parity
run, record suites and regenerate the coverage doc:
```bash
# [ref / bash] -- after the gate
php tools/parity/record-suites.php <junit.xml>...
python3 tools/parity/report.py --record "<phase>"
php tools/api-dump.php > tools/parity/api.json
python3 tools/gap_inventory.py tools/parity/api.json > docs/project/numpy-scipy-coverage.md
```
The `tesserophp-org` site renders the generated docs; never re-estimate numbers into
prose.

## 7. Updating the website (tessero.org)
The public site is a **separate Laravel repo** (`github.com/DMJ-CV-91913/tesserophp-org`),
served on delta1cx behind a Cloudflare tunnel. It renders docs **live** from this
library via `php artisan docs:sync`, so most content updates flow from the library
(the generated coverage/metrics of §6), not by hand-editing the site.

- **Content from this repo:** regenerate coverage/metrics here (§6) and push this repo's
  `main`; the site's `docs:sync` picks it up on deploy. No site code change needed.
- **Site code / copy changes:** done in the `tesserophp-org` repo; read its `AGENTS.md`
  / `CLAUDE.md` house rules first (notably: NO em/en dashes anywhere; honest,
  source-backed messaging; verify routes + `npm run build` before shipping).
- **Deploy (owner runs it; only `main` goes live):** on delta1cx as user **`snoopy72`**
  (a different user than the `tessero` build user and the `kinetix` Talario user), from
  `/var/www/tesserophp-org`, under `sudo`:
```bash
# [prod / bash as snoopy72 on delta1cx]
cd /var/www/tesserophp-org
sudo COMPOSER_ALLOW_SUPERUSER=1 ./deploy.sh   # pull main -> composer --no-dev -> npm build -> migrate -> caches -> docs:sync -> opcache reset
```
Do NOT push to the site's `main` or run `deploy.sh` without explicit owner go-ahead.
