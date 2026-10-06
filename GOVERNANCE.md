# Governance

Tessero is an open-source project run in the open. This document says who
decides what and how that changes.

## Roles

| Role | Who | Can |
|---|---|---|
| **Contributor** | anyone who opens an issue or pull request | propose changes, review, discuss |
| **Committer** | contributors with a record of good reviews and merged work, invited by maintainers | approve pull requests, triage issues |
| **Maintainer** | committers responsible for the project as a whole | merge, release, set the roadmap, invite committers and maintainers, enforce the Code of Conduct |

The current maintainers are listed in `MAINTAINERS.md`. The project's goal is at
least two active maintainers from different organisations. Until then, the
founding maintainer acts alone, and every decision below that needs two
maintainers is published as a public proposal with a 7-day comment period
instead.

## Decisions

- **Everyday changes** (fixes, docs, tests, features within the existing
  design) are decided by review: one approval, or **two** for changes to the
  C kernel (`csrc/`), the extension (`ext/`) or the ABI header.
- **Architecture changes** need an ADR (`docs/architecture/adr/`) merged with
  approval from two maintainers after at least 7 days of public discussion.
- **Breaking API changes** after 1.0 need an ADR and a deprecation period of at
  least one minor release.
- **Disagreements** are resolved by discussion aiming at consensus. If that
  fails, maintainers vote. A simple majority decides, and ties keep the status quo.

## Releases

Any maintainer can release by following `RELEASING.md`. Releases after 1.0
need two maintainers: one to prepare, one to verify and publish.

## Becoming a committer or maintainer

Maintainers invite contributors who have shown sustained, careful work (code,
reviews, docs, support). Invitations are agreed by the maintainers and
announced publicly. Inactive maintainers (12 months) move to emeritus status
and can return on request.

## Funding and independence

The project may seek fiscal sponsorship (for example NumFOCUS affiliation) and
grants. Money is spent transparently on maintenance, CI, security review and
documentation, as decided by the maintainers and reported publicly. Sponsors
do not get decision rights.

## Code of Conduct

All participation is governed by `CODE_OF_CONDUCT.md`. Maintainers enforce it.

## Changing this document

By pull request, with approval from two maintainers after 14 days of public
comment (or, while there is a single maintainer, after 14 days of public
comment).
