# Security policy

## Supported versions

No release has been cut yet. The 0.2.x line is a pre-release: fixes land on `main` and in the next 0.2.x tag,
and there are no backports to earlier commits.

| Version | Security fixes |
|---|---|
| 0.2.x (pre-release, `main`) | yes |
| < 0.2 | no |

From 1.0 on, the latest minor release and the one before it receive security fixes.

## Reporting a vulnerability

Please report privately. Do not open a public issue, discussion or pull request.

- Use the repository's **private vulnerability reporting** ("Report a vulnerability" under the Security tab).
  This is the only channel; it reaches the maintainers listed in `MAINTAINERS.md`.

Include the affected version or commit and the backend (FFI package or extension), a reproducer if possible,
and the impact you see (memory corruption, denial of service, information disclosure, code execution).

## Coordinated disclosure timeline

1. Acknowledgement within 3 working days.
2. Initial assessment (confirmed or not, severity, affected versions) within 10 working days.
3. A fix and a disclosure date agreed with you. The default embargo is 90 days from the report, shorter for
   issues that are actively exploited or already public, and extended only by mutual agreement.
4. A GitHub security advisory (with a CVE requested through GitHub when the issue warrants one) is published
   together with the fixed release; the changelog links it.
5. Credit in the advisory and changelog, unless you prefer otherwise.

If we do not respond within the times above, you may remind us through the same channel; after 90 days
without a fix you are free to disclose.

## Scope

In scope:

- the C kernel (`csrc/`, including the vendored code in `csrc/third_party/`) and its copy in the extension:
  memory-safety bugs, undefined behaviour reachable from PHP, integer overflows in shape and size arithmetic;
- the Zend extension (`ext/`): object lifecycle, reference counting, bypasses of the native memory budget
  (`tessero.memory_budget`);
- the FFI package (`src/`): unchecked pointers or lengths passed to native code, ways to make the binding load an
  unexpected library;
- file and data readers: `.npy` and sparse `.npz`, memory-mapped files, `unserialize` and `fromBytes` of arrays, Matrix
  Market, Harwell-Boeing, MATLAB `.mat`, NetCDF, IDL `.sav`, WAV and Fortran unformatted files (`src/Io/`);
- anything that lets untrusted input execute code, read or write outside the intended buffers, or read files
  other than the ones the caller named.

Out of scope: denial of service by workloads the operator chose to accept without size limits (see the
security guide, `docs/operations/security.md`); numerical differences from NumPy or SciPy (report those as
ordinary issues); and issues in PHP, OpenBLAS, LAPACK or the operating system (please report those upstream;
we are happy to help).

## Verifying releases

Release assets carry SLSA build provenance and an SBOM attestation (`gh attestation verify`), and `SHA256SUMS`
is signed with Sigstore (`cosign verify-blob`). The commands are in `RELEASING.md`.
