# Tessera → Tessero cutover (library side)

Status and remaining steps for the rename. This file lives in the repo, so it survives the folder move
and a relaunched session can resume from here.

## Final naming scheme

| Thing | From | To |
|---|---|---|
| GitHub repo | `DMJ-CV-91913/tesseraphp` | `DMJ-CV-91913/tessero` |
| Library folder | `C:\xampp\htdocs\tesseraphp\tessera` | `C:\xampp\htdocs\tessero` |
| Web app folder | `C:\xampp\htdocs\tesseraphp-org` | `C:\xampp\htdocs\tesserophp-org` |
| Server user / home | `tessera` / `/home/tessera` | `tessero` / `/home/tessero` |
| Server clone dir | `/home/tessera/tessera` | `/home/tessero/tessero` |
| SSH alias (Windows) | `tessera-ref` (User tessera) | `tessero-ref` (User tessero) |

Unchanged on purpose: internal C ABI prefix `tsr_`/`TSR_`; the `.work/*` historical logs.

## Done (on branch `rename/tessero`, gate-green QUICK)

- Content rename across 343 files (namespace `Tessero\`, packages `tessero/*`, `libtessero`,
  extension `tessero`, `TESSERO_LIB`, `tessero.*` ini, `bin/tessero`, docs) + 77 file renames.
- `ext/libtessera` mirror → `ext/libtessero` (393 files); `config.m4` → `TESSERO_KERNEL`/`libtessero`.
- Infra refs updated: CI `../tessera`→`../tessero`, issue-template GitHub URLs → `tessero`,
  `website-handoff.md` → `tesserophp-org` / `../tessero`.
- QUICK gate on the reference host: both backends 777/0, PHPUnit 244+5, docs examples 95/0,
  cookbook 44 pass/12 skip/0 fail, claims-check ok.

## Remaining steps (in order)

1. **GitHub repo rename** (user, web UI): repo Settings → rename `tesseraphp` → `tessero`.
2. **Re-point Windows remote** (agent): `git remote set-url origin https://github.com/DMJ-CV-91913/tessero.git`
3. **Server rename** (user, run as a sudo account e.g. `snoopy72`, while `tessera` is idle):
   ```bash
   sudo usermod -l tessero tessera
   sudo groupmod -n tessero tessera
   sudo usermod -d /home/tessero -m tessero
   sudo -u tessero bash -lc 'mv ~/tessera ~/tessero'
   sudo -u tessero bash -lc 'git -C ~/tessero remote set-url origin git@github.com:DMJ-CV-91913/tessero.git'
   sudo -u tessero bash -lc 'cd ~/tessero && python3 -m venv --clear .venv-parity && . .venv-parity/bin/activate && pip -q install -r tools/parity/requirements.txt && pip -q install mkdocs-material'
   ```
   (This box is python3.12 — not `python3.11`. The venv needs the full `requirements.txt` — numpy 2.4.4,
   scipy 1.17.1, PyYAML — plus `mkdocs-material`, not just numpy/scipy.)
4. **Update Windows SSH config** (agent): alias `tessero-ref` → HostName <ref-host-ip>, User tessero,
   IdentityFile unchanged.
5. **Merge + push** (agent): `git checkout main && git merge --ff-only rename/tessero && git push origin main`
6. **Final gate** (agent, on server): `ssh tessero-ref 'cd ~/tessero && . .venv-parity/bin/activate && QUICK=1 bash tools/parity/gate.sh'`
7. **Local folder move** (user, LAST — breaks the live session's cwd): close the session, then
   rename `C:\xampp\htdocs\tesseraphp\tessera` → `C:\xampp\htdocs\tessero`.
   Leave `C:\xampp\htdocs\tesseraphp` (scaffolding + tessera.zip) for the user to review.
8. **Relaunch** in `C:\xampp\htdocs\tessero`; hand the web agent the new repo slug `tessero` and
   `library_path: ../tessero`.

## Notes
- The old untracked cookbook copies on the server are parked in a recoverable `git stash`
  ("server-local pre-rename"); drop it once main is confirmed.
- Composer `tessero/tessero` + `tessero/laravel` are path repos (not on Packagist, TSR-011).
