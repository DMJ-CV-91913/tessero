# Website integration handoff (for the tesserophp-org app/site)

This document is for the agent building the Laravel app + marketing/docs site under
`C:\xampp\htdocs\tesserophp-org`. The library lives next to it at `C:\xampp\htdocs\tessero`
(referred to below as `../tessero`). The app should **consume** these library files; the library agent
does not edit the app tree. Everything listed here is generated and gate-checked in the library, so
regenerate from the tools rather than copying by hand.

> **Rename note (read first):** the project was renamed Tessera → **Tessero**. The PHP namespace is now
> `Tessero\`, Composer packages are `tessero/tessero` and `tessero/laravel`, the CLI is `bin/tessero`, the
> binary is `libtessero.{so,dll,dylib}`, the env var is `TESSERO_LIB`, the GitHub repo is `tessero`, and the
> library folder is `C:\xampp\htdocs\tessero` (sibling path `../tessero`). The internal C ABI prefix
> (`tsr_`/`TSR_`) is intentionally unchanged.

## 1. Depend on the library (Composer)

Tessero is not on Packagist yet (issue TSR-011), so wire it as local path repositories:

```jsonc
// tesserophp-org/composer.json
"repositories": [
  { "type": "path", "url": "../tessero" },         // package: tessero/tessero
  { "type": "path", "url": "../tessero/laravel" }   // package: tessero/laravel  (the bridge)
],
"require": {
  "tessero/tessero": "*",
  "tessero/laravel": "*"
}
```

- `Tessero\Laravel\TesseroServiceProvider` is **auto-discovered**; a `Tessero` facade alias is registered.
- Public API the Blades/controllers call: `Tessero\Laravel\TesseroManager` (array/zeros/ones/random/load/memmap/linprog/milp/solveMdp/info/backend), the `AsNDArray` Eloquent cast, the `NumericArray` validation rule.

## 2. The backend binary (design constraint — read this)

The FFI package needs `libtessero.{so,dll,dylib}` for the host, built only for **linux-x86_64** today
(Windows DLL is TSR-002, not done). So:

- **Linux server**: works. Build `make -C vendor/tessero/tessero/csrc`, or load the native extension.
- **Windows / xampp**: no binary, so any Tessero call throws `Tessero\Exceptions\LibraryUnavailable`.
  Guard with `vendor/bin/tessero doctor` (it reports ffi.enable, the located binary and BLAS) and degrade
  gracefully, or point the app at the Linux server. Do not assume Tessero is callable in local dev.

## 3. Files to build the site content from

All paths relative to `../tessero`. These are the source of truth; adapt their content into Blade pages.

### Machine-readable (best for generating pages/cards)
- `tools/parity/api.json` — every public class and method (name, static/instance, full signature, doc). Drive the **API reference** pages from this.
- `tools/parity/registry.json` — the 750 kernel functions (module.fn, kind, arity).
- `docs/reference/numpy-tessero.md` — the **NumPy/SciPy → Tessero** translation table (750 rows), grouped by module; good for a "coming from NumPy?" page.
- `docs/examples/manifest.json` — one entry per cookbook example: `id, module, moduleTitle, name, file, summary, equivalent, backends`. Iterate this to render **example cards**; read `file` for the code.

### Prose content (adapt into marketing/docs pages)
- `README.md` — the pitch, the measured coverage figures, quickstart.
- `composer.json` — `name`, `description`, `keywords` (hero copy).
- `mkdocs.yml` — `site_name`, `site_description`, and the **`nav:` tree** = the information architecture to mirror in the site navbar.
- `docs/getting-started/*` — installation, quickstart, choosing a backend.
- `docs/guide/*` — one page per surface (arrays, linear algebra, FFT, random, statistics, sparse, optimisation, LP, MDP, io, memmap, Laravel).
- `docs/examples/*.md` + `examples/<module>/*.php` — the runnable cookbook (code + what it asserts).
- `docs/operations/*` — deployment, native extension, docker, performance, security.
- `docs/project/{roadmap,open-issues,numpy-scipy-coverage}.md` — roadmap and status for a "project/status" section.

### The bridge classes the Blades will call
- `laravel/src/TesseroManager.php`, `laravel/src/Casts/AsNDArray.php`, `laravel/src/Rules/NumericArray.php`, `laravel/src/Facades/Tessero.php`, `laravel/src/Console/DoctorCommand.php`.

## 4. Suggested site information architecture (mirror of `mkdocs.yml` nav)

Navbar: **Home · Getting started · Guide · Examples · API reference · Project**

- **Home** — hero (name/description from `composer.json`), the coverage numbers from `README.md`, CTAs to install + examples.
- **Getting started** — from `docs/getting-started/*`.
- **Guide** — from `docs/guide/*` (one page per surface).
- **Examples** — generated from `docs/examples/manifest.json`: a grid of cards per module (title, `summary`, `equivalent`, backend badges from `backends`), each opening the embedded code from `file`.
- **API reference** — generated from `tools/parity/api.json`; include the translation table (`docs/reference/numpy-tessero.md`).
- **Project** — roadmap + open issues from `docs/project/*`.

## 5. Suggested Blade components
- `layouts/app.blade.php` with the navbar above (sections driven by the IA).
- `components/example-card.blade.php` — props from a `manifest.json` entry; shows summary, the matching NumPy/SciPy call (the `equivalent` field), backend badges, and a copyable `<pre>` of the example source.
- `components/code.blade.php` — syntax-highlighted PHP block with a copy button (used by example and guide pages).
- `components/api-method.blade.php` — renders one `api.json` method (signature + doc).
- `components/doctor-note.blade.php` — a banner shown when `TesseroManager::info()` / `doctor` reports the backend is unavailable (the Windows-dev case).

## 6. Keeping in sync
These generated artifacts are refreshed (and gate-checked) by:
`php tools/api-dump.php > tools/parity/api.json`, `php tools/gen-translation-table.php`,
`php tools/gen-examples-manifest.php`, `php tools/gen-example-docs.php`. If the library API changes,
re-run those and the app re-reads the new files; nothing in the app needs hand-editing to track the API.
