# Security policy

## Supported versions

| Version | Security fixes |
|---|---|
| 0.2.x | yes |
| < 0.2 | no |

After 1.0, the latest minor release and the one before it receive fixes.

## Reporting a vulnerability

Please report privately. Do not open a public issue.

- Use the repository's **private vulnerability reporting** ("Report a
  vulnerability" under the Security tab), or
- contact the maintainers listed in `MAINTAINERS.md` directly.

Include the affected version and backend (FFI package or extension), a
reproducer if possible, and the impact you see (memory corruption, denial of
service, information disclosure).

What to expect:

1. Acknowledgement within 3 working days.
2. An initial assessment within 10 working days.
3. A fix and a coordinated disclosure date agreed with you. The target is
   within 90 days, sooner for actively exploited issues.
4. Credit in the advisory and changelog, unless you prefer otherwise.

## Scope

In scope: memory-safety bugs in `csrc/` or `ext/`, ways to bypass the native
memory budget, unsafe handling of `.npy`, serialised or cast data, and anything
that lets untrusted input execute code.

Out of scope: denial of service by workloads the operator chose to accept
without size limits (see the security guide, `docs/operations/security.md`),
and issues in PHP, OpenBLAS or the OS (please report those upstream; we are
happy to help).
