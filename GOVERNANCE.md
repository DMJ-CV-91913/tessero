# Governance

Tessero is open source (BSD-3-Clause) with a single maintainer and closed
contribution: the code is public, but only the maintainer changes it.

## Maintainer

Douglas Johnson is the sole author and maintainer (see `MAINTAINERS.md`). He
decides the roadmap, reviews and merges every change, and makes every release.
The project does not accept pull requests or patches from outside, and does not
add committers or maintainers from outside contributions.

## How decisions are made

- **Everyday changes** (fixes, docs, tests, features within the existing
  design) must pass the CI and parity gates on both backends.
- **Architecture changes** are recorded as an ADR in `docs/architecture/adr/`.
- **Breaking API changes** after 1.0 need an ADR and a deprecation period of at
  least one minor release.

## Bug reports and security

Bugs are reported by email to support@tessero.org (see `CONTRIBUTING.md`).
Vulnerabilities go to security@tessero.org under `SECURITY.md`. There is no
public issue tracker.

## Releases

Releases follow `RELEASING.md`.

## Continuity

The source, build, tests, reference fixtures and release process are public and
documented, so anyone can build, verify and fork Tessero under the licence if the
maintainer stops working on it.

## Funding and independence

The project may seek grants, sponsorship and fiscal sponsorship. Money is spent
on maintenance, CI, security review and documentation, as decided by the
maintainer. Sponsors do not get decision rights.

## Code of Conduct

All interaction with the project is governed by `CODE_OF_CONDUCT.md`.

## Changing this document

The maintainer changes this document and records the change in the changelog.
