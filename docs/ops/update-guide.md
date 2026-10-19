# Tessero: Update & Development Guide

How to change Tessero safely and get the change onto the ref server. The governing
rule mirrors the parity program (see `CLAUDE.md`): **covered means verified -- a
change ships only when the parity gate is green on BOTH backends (FFI + ext).
Never loosen a tolerance or edit generated files to get green.**

> **Scope.** This guide governs the **Tessero repo** only
> (`github.com/DMJ-CV-91913/tessero`). It builds the native numerical core
> `libtessero` and its two PHP bindings. The **Talario engine**
> (`github.com/DMJ-CV-91913/talario`) is a separate repo that only *consumes* a built
> `libtessero` via `--with-tessero` -- see §4 and Talario's own
> `docs/ops/update-guide.md`. The `tesserophp-org` / `talariophp-org` websites are
> separate repos again.

## 0. Shell & environment conventions (used throughout)
Every command block below is tagged with where it runs, because the dev box and the
build server use different shells:
- **[dev / PowerShell]** -- your Windows dev box, in PowerShell, from the local
  clone (note: the real repo is the **inner** `tessero/` folder; the outer `tessero/`
  is a scratch area).
- **[server / bash]** -- a Linux build server, in bash, as your deploy user, from the
  server-side checkout of this repo. Reach it over SSH via a host alias (`<server>`).
  A box that runs both projects may keep a *separate* checkout for the Talario link
  (§4); the website is deployed from its own separate repo on the server (§7).

Git is the only thing that crosses between dev and server. The built `libtessero` is a
per-host native artifact and is **never** synced over git -- you build it on the server
(and any box that runs the extension), from pushed source. Run `scp`/`ssh` from the dev
box, not from the server, so one shell drives the whole loop.

## 1. The loop (source of truth = the public repo)
Edit on the dev box -> commit -> push -> the build server pulls, builds, and
**gates on the parity suite**.

**[dev / PowerShell]** -- commit + push (end commit messages with the
Co-Authored-By/Claude-Session trailers):
```powershell
git add -A
git commit -m "..."
git push origin main
```

**[server / bash]** -- pull, build the core, run the gate. Chain with `&&` so a
failure halts the loop (`<server>` = your SSH host alias):
```bash
ssh <server>
cd <server checkout> \
  && git fetch --prune origin && git pull --ff-only origin main \
  && make -C csrc clean && make -C csrc \
  && bash tools/sync-ext.sh \
  && QUICK=1 bash tools/parity/gate.sh
# make -C csrc -> lib/<platform>/libtessero.so
# sync-ext.sh refreshes ext/libtessero from csrc (run after a kernel change)
# QUICK=1 = fast gate; drop it for the full sanitize+fuzz+soak+bench
```
`QUICK=1` skips sanitizers, fuzzing, soak and benchmarks (per `CLAUDE.md`); run the
full `bash tools/parity/gate.sh` before a release or anything ABI-affecting. The gate
runs BOTH backends (`run-tests.php --backend=ffi` and `--backend=ext`). Only after
green do you let a consumer relink (§4).

Notes that have bitten before, all generic:
- Run `scp`/`ssh` from the dev box, not the server, so one shell drives the loop.
- If this repo is **private**, the server's `git pull` needs working auth -- configure
  `gh` or an HTTPS credential helper on the server (an SSH deploy key may be read-only).

## 1a. Build prerequisites (build server, once)
**[server / bash]** -- the Linux linalg path links LAPACKE from OpenBLAS
(`csrc/Makefile` adds `-llapacke -lblas` on Linux). Install the dev packages:
```bash
sudo apt-get update && sudo apt-get install -y build-essential liblapacke-dev libopenblas-dev
```
`make portable` (adds `-DTSR_NO_CLONES`) is the build for musl / cross / non-native
targets; the default `make` uses `target_clones` for runtime SIMD dispatch.

## 2. Build outputs (the consumption surface)
`make -C csrc` produces, relative to the repo root:
- **Library:** `lib/<platform>/libtessero.<ext>` -- on a Linux x86-64 build that is
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

A box that runs both projects keeps its **own** clone of this repo for the Talario link,
built so that `csrc/` is that prefix -- `csrc/include/tessero.h` + `csrc/lib/libtessero.so`
both present (`readelf -d .../talario.so` then shows a `RUNPATH` pointing at that
`csrc/lib`). So the Talario-side one-shot (run in the **Talario** repo, as its deploy
user, not here) is:
```bash
# [server / bash] in the Talario checkout, after rebuilding libtessero in that clone
# use $HOME, NOT ~ -- tilde does not expand after = in --with-tessero=...
./build.sh --with-tessero="$HOME/tessero/csrc" && scripts/test.sh
```
Caveats:
- That may be a **separate clone** from this repo's server-side checkout, possibly at a
  different commit. Both pull from the same `origin`, so keep them on compatible commits:
  a Talario link is only as current as the Tessero clone it points at.
- This repo's `csrc/Makefile` emits the lib to `lib/<platform>/`, but the consumer
  prefix needs it under `csrc/lib/`. Arrange for `libtessero.so` to be directly in
  `csrc/lib/`; if a build emits to `lib/<platform>/` instead, copy or symlink it into
  `csrc/lib/` (or point `--with-tessero` at a prefix laid out with both subdirs).
- To load for any user and SAPI, install `libtessero` into a system prefix like
  `/usr/local/lib`; do not depend on a specific user's home directory.

## 5. Rollback
- Code: **[dev / PowerShell]** `git revert <sha>` (or reset a branch) -> `git push`,
  then **[server / bash]** re-pull + `make -C csrc clean && make -C csrc` +
  `QUICK=1 bash tools/parity/gate.sh` to re-gate the rolled-back tree.
- A consumer (Talario) keeps linking the **old** `libtessero` until it is rebuilt
  against the new one; reverting here does not change an already-linked `talario.so`
  until that rebuild. Safe: gate here, then relink the consumer.

## 6. Coverage / metrics hand-off
Coverage numbers are **generated**, never typed by hand (`CLAUDE.md`). After a parity
run, record suites and regenerate the coverage doc:
```bash
# [server / bash] -- after the gate
php tools/parity/record-suites.php <junit.xml>...
python3 tools/parity/report.py --record "<phase>"
php tools/api-dump.php > tools/parity/api.json
python3 tools/gap_inventory.py tools/parity/api.json > docs/project/numpy-scipy-coverage.md
```
The `tesserophp-org` site renders the generated docs; never re-estimate numbers into
prose.

## 7. Updating the website (tessero.org)
The public site is a **separate Laravel repo** (`github.com/DMJ-CV-91913/tesserophp-org`),
served on a Linux box behind a reverse-proxy tunnel. It renders docs **live** from this
library via `php artisan docs:sync`, so most content updates flow from the library
(the generated coverage/metrics of §6), not by hand-editing the site.

- **Content from this repo:** regenerate coverage/metrics here (§6) and push this repo's
  `main`; the site's `docs:sync` picks it up on deploy. No site code change needed.
- **Site code / copy changes:** done in the `tesserophp-org` repo; read its `AGENTS.md`
  / `CLAUDE.md` house rules first (notably: NO em/en dashes anywhere; honest,
  source-backed messaging; verify routes + `npm run build` before shipping).
- **Deploy (owner runs it; only `main` goes live):** the site is a separate repo with
  its **own** `deploy.sh`, run from its own server-side checkout (a different checkout
  than this build):
```bash
# [server / bash] -- in the site's checkout
git pull --ff-only && ./deploy.sh   # pull main -> composer --no-dev -> npm build -> migrate -> caches -> docs:sync -> opcache reset
```
Do NOT push to the site's `main` or run `deploy.sh` without explicit owner go-ahead.
