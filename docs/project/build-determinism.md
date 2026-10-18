# Determinism and deployment-robustness design: closing every leak from arithmetic to host

Status: proposed. Owner posture: hard-fail conformance gate. Dev-first: build and verify on delta1cx.
Style: ASCII only, no em dashes (owner preference). This is the authoritative spec; Talario points here.

## Origin

This came out of Talario GPU attestation. A real-hardware run (NVIDIA H100, Lambda, 2026-10-18, Talario
commit b213eeb, scale 1024, real Tessero) exposed a build-level determinism leak in Tessero's CPU reductions:
the canonical matmul attestation produced f4b5bd0c... on that host instead of the golden 4f984242... . The
divergence also hit `linear` and `conv2d` (the GEMM family); `bmm`, `add`, `gather`, `embedding`, `segment_sum`
were bit-identical. Reran byte-for-byte identical, so it is deterministic, not a flake. The transcendental repro
primitives were bit-identical across all three GPU generations and every CPU; the GEMM/axpy/dot reductions were
not.

Proof the fix matters (same H100 run): forcing the FMA path turned the invalid run (matmul CPU f4b5bd0c) into a
clean pass (4f984242). That is the concrete payoff of this work. Once Layer 1 (explicit fma) + Layer 4
(build-time self-test) land, a plain `make CC=gcc-12` produces the canonical 4f984242 result on any CPU, with no
-march / -DTSR_NO_CLONES gymnastics - the determinism stops depending on which clone the runtime happens to pick.
The valid canonical-FMA H100 attestation is banked in Talario (test(attest), commit b79071d).

That one bug is the tip of a larger surface. A determinism library's promise - the same answer to the bit,
everywhere - can be broken at four stages: the arithmetic, the build, the packaging, and the host it loads on.
This document closes all four. Part I hardens the library; Part II hardens deployment; Part III is the
failure-mode matrix; Part IV is the phased plan to build and verify on delta1cx.

## Core principle

Determinism is a property of the arithmetic, and must not leak into the build or the deployment. For every
guarantee we make, it must be impossible for the same source to produce different bits - or to ship/load a binary
that produces different bits - because of a compiler, flag, CPU clone, thread count, FPU mode, libc, OpenBLAS
version, or packaging choice. The whole solution is one idea applied at every stage: close every place where the
result depends on a build-time, link-time, load-time, or run-time choice, and make the system prove conformance
rather than hope for it.

---

## Scope taxonomy: two guarantees, not one (read this first)

Over-scoping this work would be a mistake. Tessero makes two distinct guarantees, and the hardening differs.

Bit-exact / attestation surface: the repro-f32 primitives (rsqrt/exp2/exp/erf) and the attested exact-op ops
(matmul forward + grads, axpy, dot). Guarantee: 0-ULP, identical bits on every SIMD width, thread count, CPU, and
GPU. It may use only `+ - * /` and sqrt (IEEE correctly-rounded) and fma (one rounding); no libm, no
reassociation. Strategy: Layers 1, 1b, 2, 4 - eliminate every build/run choice, prove with a golden gate.

Tolerance parity surface: most of NumPy/SciPy (sin, exp, special functions, solvers via LAPACK). Guarantee:
matches the pinned reference (NumPy 2.4.4 / SciPy 1.17.1, ADR 0012) within a per-call tolerance. It may use libm,
Boost.Math/xsf, OpenBLAS/LAPACK. Strategy: stays tolerance-bounded; libm ~1-ULP drift across versions is accepted
by design (ADR 0012), not a regression.

Consequences that keep the effort honest:
- The FMA-contraction fix (Layer 1) is required for the bit-exact surface and for any op that claims cross-SIMD /
  thread bit-identity. For the parity surface it is cheap, good hygiene, and makes cross-SIMD bit-identity
  achievable - but a cos/sin/exp that calls libm can never be bit-identical across libm versions, and we do not
  pretend otherwise.
- The audit found many `acc += a*b` sites and libm trig/exp sites across scipy_* and np_* (inventory below). The
  rule: explicit fma everywhere it is cheap (removes a whole class of cross-SIMD surprises), but a site only moves
  onto the bit-exact surface if it is attested or feeds an attested result; otherwise it stays on the tolerance
  surface and libm use is fine.

---

# PART I - The library (arithmetic determinism)

## Layer 1 - Make FMA explicit in source [REQUIRED, P0]

Eliminate the leak at the root. Every determinism-critical multiply-add uses single-rounded fmaf (float) / fma
(double), like the repro primitives already do. fmaf/fma are single-rounded on every target (hardware FMA, or
libm soft-FMA on no-FMA CPUs), so all target_clones variants and all CPUs converge to the canonical bits.

Verified core sites (csrc/src/matmul.c):
- mm_* kernels route multiply-add through a macro seam already: `#define FFMA(T,c,a,b) ((c)+(a)*(b))` (line 39),
  applied at line 34. This is a clean one-point swap:

      /* single-rounded multiply-add, by type - the determinism-critical seam */
      static inline double tsr_fma_f64(double c, double a, double b) { return fma(a, b, c); }
      static inline float  tsr_fma_f32(float  c, float  a, float  b) { return fmaf(a, b, c); }
      #define FFMA(T, c, a, b) _Generic((T)0, double: tsr_fma_f64, float: tsr_fma_f32)((c), (a), (b))

  (IFMA, the integer path, stays TSR_WADD/TSR_WMUL - integer arithmetic has no rounding, no leak.)
- gemm_*: `ci[j] += s * b[j*ldb+p]` (lines 81-82, both transposed and bp[j] branches) -> fma.
- axpy_*: `y[i] += alpha * x[i]` (lines 107, 109) -> fma.
- dot_f64: `s[j] += x[i+j]*y[i+j]` and `r += ...` (lines 131, 134) -> fma. The final tree reduction
  `r = (s[0]+s[1])+(s[2]+s[3])` (line 133) is already fixed-order; keep it exactly (Layer 2 -ffp-contract=off
  prevents the compiler reassociating it).

Audit-identified additional `acc += a*b` sites (sweep across csrc/src; verify exact line numbers against current
source at implementation time - only the matmul.c lines above were read directly for this doc): np_linalg.c
(dot / matvec / matmul, QR+SVD reconstruction), spatial_distance.c (L2 / cosine), scipy_fft.c (DCT/DST),
scipy_signal.c (convolution, filters), scipy_sparse_linalg.c (sparse matvec, CG residual), scipy_cluster.c
(k-means), np_polynomial.c (Vandermonde scaling), np_numeric.c (quadrature). Apply the same _Generic helper.
Prioritize any op on the bit-exact surface; the rest is hygiene.

Attestation detail: the H100 run showed `linear` and `conv2d` diverge with matmul (GEMM family), while bmm, add,
gather, embedding, segment_sum were bit-identical. Confirm whether those already fuse or were simply not
contraction-sensitive at this size, and make them explicit too for safety.

Must not change the goldens. FMA-contracted `+=` and explicit fma are the same bits; the current goldens were
generated on a fused build. This change only makes the divergent (baseline no-FMA clone) path converge.
Regenerate the goldens and diff - if any golden moves, stop and investigate.

Note on the attested path: tsr_gemm (matmul.c:96-97) always uses the native gemm_* macro and never routes to
OpenBLAS; the autograd / attestation grads go through it, which is why the grads expose the leak. The F32/F64
tsr_matmul does route to cblas_{s,d}gemm when dims fit INT_MAX (matmul.c:50-60) - a separate determinism story
(Layer 3, OpenBLAS).

## Layer 1b - Runtime floating-point-environment guard [REQUIRED, P0]

Even a perfectly built binary gives different bits if the process FPU state is wrong. The audit confirmed the
kernel assumes round-to-nearest with no flush-to-zero but never checks or sets it (no fe*/MXCSR calls outside a
non-C99 stub). Hazard: a co-loaded library compiled with -ffast-math sets FTZ/DAZ (denormals -> zero), or a host
calls fesetround(FE_UPWARD) - now every `+ - * / fma` in the bit-exact surface diverges, with no error.

Fix - a tiny, one-time guard the kernel runs at init and the attestation entrypoints assert:
- Provide `int tsr_fpenv_canonical(void)` returning true iff rounding is FE_TONEAREST and FTZ/DAZ are off.
- Provide `void tsr_fpenv_set_canonical(void)` to force round-to-nearest and clear FTZ/DAZ (MXCSR bits 15/6 off),
  callable at library init. Document that Tessero assumes and enforces this.
- The Layer 4 self-test and the per-call attestation path assert tsr_fpenv_canonical() and hard-fail with a clear
  message if a host has corrupted the FPU state.
- doctor reports the FPU mode (Part II) so a deployment catches a -ffast-math co-tenant before production.

## Layer 2 - Pin the FP flags, in BOTH build systems [REQUIRED, P0]

The precedent exists: repro_f32.c already compiles with `-fno-fast-math -ffp-contract=off` (csrc/Makefile:84).
Generalize it, and mirror it in the extension, which has an entirely separate flag set.

- csrc/Makefile:57 (`CFLAGS = -O3 -std=gnu11 -fPIC -fno-math-errno -fno-trapping-math ...`): add
  -ffp-contract=off (safe once Layer 1 makes the fusions explicit) and -fno-fast-math.
- ext/config.m4:20 (verified: `TESSERO_CFLAGS = -std=gnu11 -O3 -fno-math-errno -fno-trapping-math
  -fvisibility=hidden ...`): add the same -ffp-contract=off -fno-fast-math, and TESSERO_CXXFLAGS (config.m4:35)
  likewise. The extension compiles matmul.c and every reduction file itself (config.m4:16, 33); without this it
  keeps the contraction leak even after Layer 1, because -std=gnu11's default -ffp-contract=fast would re-fuse
  any residual `+*`.
- Refuse-to-build guard (both systems): if CFLAGS / TESSERO_CFLAGS contains -ffast-math,
  -funsafe-math-optimizations, or -Ofast, error out. These reassociate sums -> instant nondeterminism.
- Gap found: the repro-f32 primitives (tsr_repro_*_f32) are absent from the extension source list
  (config.m4:16), so that bit-exact surface exists only on the FFI backend. Decide: add repro_f32.c to the
  extension sources (with the pinned flags) so both backends carry the attested surface, or document it as
  FFI-only. For a determinism library, add it - the two backends should not differ in what they can attest.

## Layer 3 - Toolchain and OpenBLAS policy [P1]

- Compiler multiversioning (macro side already done). TSR_CLONES is guarded on `__GNUC__ >= 12`
  (internal.h:38; Tessero commit 2c304aa): GCC 11 degrades to a clean single-version build, GCC 12+ gets FMV.
  After Layer 1, target_clones is pure performance with zero correctness risk - the baseline clone produces
  identical bits. The internal.h:32-34 comment ("results are IDENTICAL ... determinism is by construction")
  becomes true instead of aspirational. Optional: Makefile auto-prefers gcc-12/gcc-13.
- OpenBLAS is a real determinism variable on the F32/F64 tsr_matmul fast path. OpenBLAS sums the k-dimension
  across threads and across version-specific kernels, so its bits depend on thread count, CPU-tuned kernel, and
  OpenBLAS version. The audit found Blas.php already forces openblas_set_num_threads(1) outside CLI, but
  OPENBLAS_NUM_THREADS / OPENBLAS_CORETYPE can still override. Policy for the bit-exact matmul guarantee:
  - Attested matmul goes through tsr_gemm (native, deterministic) - keep it that way; do not let the attested
    path route to cblas.
  - For the tsr_matmul fast path, document it as tolerance-parity, not bit-exact, OR offer a
    TESSERO_DETERMINISTIC=1 mode that forces the native gemm_* kernel and single-thread OpenBLAS.
  - Pin OpenBLAS: minimum version plus OPENBLAS_NUM_THREADS=1 for any reproducible run; doctor validates a
    cblas_ddot golden to catch an ABI/kernel-swapped system BLAS (Part II).

## Layer 4 - Build-time conformance self-test [REQUIRED - the "no issues" guarantee, P0]

Reuse the existing machinery rather than reinvent it. csrc/repro/determinism_test.c plus run_determinism.sh plus
GOLDENS.md already do exactly the raw-bytes -> SHA-256, build-3-ways (scalar / AVX2 / AVX-512), OMP 1 vs 4
comparison, with full 64-char goldens embedded. Clone that pattern:

- New csrc/repro/determinism_verify.c: links the built kernel, computes the canonical exact-op results (repro-f32
  four primitives and matmul forward + grad_a + grad_b on the fixed attestation seed), SHA-256s them, hard-fails
  on any mismatch with a message naming the likely cause (toolchain / flag / clone / FPU).
- New `make check-determinism` target depending on the built library; make it a prerequisite of `all` (hard-fail
  posture - a determinism library must not be producible in a non-canonical state), and run it in CI.
- Mirror it for the extension build - the same goldens, invoked against tessero.so, so a mis-flagged extension
  (Layer 2 gap) cannot ship either.
- Embed goldens as full 64-char strings in a header csrc/include/attestation_goldens.h (no truncation at build
  time; the full values are in the Reference values section below). Remaining dependency: the matmul forward and
  grad goldens are now recorded here, but the fixed attestation seed/inputs (dimensions and the deterministic
  input pattern) still live in the Talario harness tests/attest/repro_gpu_validate.c - pull the exact seed
  generator before wiring the matmul part of the gate so the recomputed bytes match. Until then, the gate can run
  on the four in-repo repro-f32 goldens.
- Assert tsr_fpenv_canonical() (Layer 1b) at the top of the verifier.

Posture: hard-fail, not warn. This catches any future toolchain / flag / clone / CPU / FPU regression at build
time, on the spot.

## Layer 5 - Documented support and determinism matrix [REQUIRED, P1]

One page: supported compilers (GCC >= 12; clang >= validated), required flags (-ffp-contract=off, no fast-math),
the GCC-11 single-version degrade, the FPU-mode requirement (Layer 1b), the OpenBLAS pin plus
OPENBLAS_NUM_THREADS=1 rule, the glibc/libm pin (ADR 0012), and the statement that the Layer 4 self-test gates
conformance. Cross-link ADR 0006 (process-wide settings) and ADR 0012 (pinned reference environment).

---

# PART II - Deployment (the binary reaching, and loading on, a host)

A bit-perfect library is worthless if packaging corrupts it, the wrong binary ships, or it loads silently-wrong
on a host. The guiding rule here: fail loud and early, never silent and late. Every mismatch should surface as a
clear error at load or in doctor, not as a wrong number or a SIGILL three requests later.

## D1 - Build and packaging integrity [P0/P1]

- csrc <-> ext flag parity [P0]. The two build systems are independent (csrc/Makefile vs ext/config.m4). Any FP
  flag change must land in both (Layer 2). Add a CI check that greps both for the required flags and the banned
  ones. Also keep `tools/sync-ext.sh --check` a blocking gate (a sync failure must stop the extension build, not
  just warn).
- Reproducible archive [P1]. Pack release tarballs deterministically:
  `tar --sort=name --owner=0 --group=0 --mtime=@$SOURCE_DATE_EPOCH -czf ...` with
  `export SOURCE_DATE_EPOCH=$(git log -1 --format=%ct)`. Without this, identical builds yield different archive
  checksums and defeat audit/attestation.
- glibc/libc floor tagging [P0]. Prebuilt .so files carry an implicit minimum glibc (symbol versions). Deploying
  a glibc-2.35-built binary on an older host (CentOS 7 / glibc 2.17) dlopens and then crashes on first call with
  a cryptic "GLIBC_2.35 not found". Fix: in release CI, record each binary's max required GLIBC_ version
  (objdump -T) into a BINARY_MANIFEST.json, fail the release if it exceeds the documented floor, and have
  doctor/load check it (D3).
- Baseline / portable variant [P1]. The ifunc path needs glibc; on musl or -static, TSR_CLONES is empty and the
  single build's -march decides the ISA. A v3 (-march=haswell) binary on a v2 host -> SIGILL. Ship a baseline
  (x86-64-v2) binary alongside the tuned one (the `make portable` / -DTSR_NO_CLONES target exists at
  Makefile:112), and select by CPUID at load.
- OpenBLAS pinning [P1]. Pin a minimum OpenBLAS version and OPENBLAS_DYNAMIC_ARCH=1; document it in ADR 0012 and
  deployment docs. Prefer a vendored/known OpenBLAS over whatever `apt install` provides.

## D2 - Load-time handshake: version + ABI + registry + SIMD + libc [P0]

Verified gap: Library::checkVersion() (src/Native/Library.php:169-176) compares only the semver string from
tsr_version(). Its own comment admits "a mismatch would corrupt memory rather than fail". A same-version binary
with a changed tsr_array struct layout or a changed function registry passes and then corrupts memory or returns
wrong bits. Harden the handshake into four checks, all fail-hard:

1. Semver - keep the existing check.
2. ABI tag [P0]. At build, hash the ABI-defining headers (internal.h struct layouts + tessero.h + the registry
   schema) into `const char tsr_abi_tag[]` embedded in the .so; generate the matching constant into
   src/Native/AbiTag.php during facade generation. At load, compare. Mismatch -> clear error ("binary and PHP
   layer are out of sync; rebuild / reinstall matching versions"). This is the real guard against silent memory
   corruption from struct/registry drift.
3. Registry freshness [P1]. The FFI facades are generated from the kernel registry
   (tools/parity/gen-facades.php). Hash the runtime registry (names, arities, defaults) in Registry::all() and
   compare to a stored build-time hash; mismatch -> "regenerate facades / rebuild". Add to CI as a gate.
4. SIMD-level and libc match [P1]. Embed `const char tsr_build_simd_level[]` and the target libc; at load,
   compare to tsr_simd_level() / detected libc. If the binary needs a higher ISA than the CPU (SIGILL risk) or a
   different libc (glibc vs musl), fail with a precise message before the first kernel call rather than crashing
   inside it. Library::platform() already distinguishes linux-musl-* (Library.php:96-98) - extend that to an ELF
   DT_NEEDED libc check.

Always run the handshake, including on the preloaded (FFI::scope) path - it already does (Library.php:130); keep
it that way and add checks 2-4 there too.

## D3 - Backend selection and fail-fast [P1]

- Version-match both backends. If the extension and the FFI package are both present, assert their versions (and
  ABI tags) match; a composer update that bumps the FFI package while the manually-installed extension lags is a
  silent divergence. Fail with "reinstall the extension to match".
- No silent capability fallback. `backend: auto` picks the extension if loaded, but the extension lacks parts of
  the API (LAPACK, multi-axis reductions, .npy I/O). Code written for FFI then dies on a missing method at
  runtime. Fix: when a config explicitly names a backend, enforce it (fatal if absent); under auto, surface a
  clear capability error naming the missing feature and the backend, not a generic method-not-found.

## D4 - PHP/host environment hazards [P1]

- Preload failure detection. Under ffi.enable=preload, if opcache.preload is misconfigured (wrong path, parse
  error, permissions), the scope never binds. Library.php:138 correctly throws, but the message always says "was
  not preloaded", masking the real cause. Fix: in resources/preload.php, verify each FFI::load produced a scope
  and fail hard (E_ERROR) with the specific stage that failed; have doctor report preload status.
- open_basedir / rpath / LD_LIBRARY_PATH. A located .so outside open_basedir, or one with unresolved dynamic deps
  (libgomp, custom BLAS) not on the loader path, fails dlopen with a cryptic message. Fix: prefer self-contained
  linking; in locate(), if open_basedir is set, check the path is inside it and emit a precise error; document the
  LD_LIBRARY_PATH requirement for dynamic deps in the FPM service file.
- SAPI drift. doctor runs per-SAPI; a CLI run can look healthy while FPM is misconfigured. Add a warning when CLI
  is in preload mode, and recommend running doctor under the production SAPI.

## D5 - doctor as the deploy-time gate [P1]

Make `vendor/bin/tessero doctor` the single deploy-time conformance check, reporting and hard-checking: load mode
(scope/cdef); resolved .so path + glibc floor + libc match + SIMD level vs CPU; semver + ABI tag + registry
freshness; FPU mode canonical (Layer 1b); OpenBLAS version + thread count + a cblas_ddot golden; and a mini
determinism self-test (recompute one repro-f32 primitive + matmul forward, compare to golden - the Layer 4 gate,
at deploy time). Report both backends' versions if both are present. This turns "it loaded" into "it loaded and
proved it computes the canonical bits on this host".

---

# PART III - Deployment failure-mode matrix

| # | Scenario | Today | Hazard | Fail-safe fix | Pri |
|---|---|---|---|---|---|
| 1 | No-FMA clone on cloud CPU | different bits (f4b5bd0c) | wrong result, silent | Layer 1 explicit fma | P0 |
| 2 | Host/co-tenant sets FTZ or non-RN rounding | different bits on bit-exact surface | silent | Layer 1b FPU guard + assert | P0 |
| 3 | Extension built without pinned FP flags | ext bits != FFI bits | silent divergence between backends | Layer 2 in config.m4 | P0 |
| 4 | -ffast-math/-Ofast in CFLAGS | reassociated sums | nondeterminism | Layer 2 refuse-to-build | P0 |
| 5 | Same-version binary, changed struct/registry | checkVersion passes | memory corruption | D2 ABI tag | P0 |
| 6 | glibc-2.35 binary on glibc-2.17 host | dlopen ok, crash on call | cryptic GLIBC_ error | D1 floor tag + D3 load check | P0 |
| 7 | v3 binary on v2 CPU (musl/static) | SIGILL | crash | D1 baseline variant + D2 SIMD check | P1 |
| 8 | OpenBLAS version/thread change | matmul fast-path bits change | drift on parity surface | Layer 3 pin + TESSERO_DETERMINISTIC + doctor golden | P1 |
| 9 | musl vs glibc binary mismatch | missing libc symbols | crash | D2 libc match check | P1 |
| 10 | Preload misconfigured (FPM) | throws "not preloaded" | misleading, hard to debug | D4 preload stage detection | P1 |
| 11 | backend: auto picks ext, code needs LAPACK | method-not-found at runtime | late failure | D3 capability error | P1 |
| 12 | FFI pkg updated, ext not rebuilt | version skew | silent divergence | D3 both-backend version match | P1 |
| 13 | Non-reproducible release tarball | different checksums | breaks audit/attestation | D1 reproducible tar + SOURCE_DATE_EPOCH | P1 |
| 14 | open_basedir/rpath blocks .so deps | cryptic dlopen failure | hard to diagnose | D4 path checks + self-contained link | P2 |

---

# PART IV - Phased implementation and verification (on delta1cx, dev-first)

Per the dev-first workflow: local edit -> scp to tessero-ref -> build/verify -> commit/push from local. Each
phase builds and re-verifies against goldens before the next.

Phase 0 - Seed and FPU guard scaffolding. Pull the fixed attestation seed/input generator from the Talario harness
(the goldens themselves are already recorded below). Add tsr_fpenv_canonical/set_canonical (Layer 1b).

Phase 1 - Layer 1 (P0). _Generic fma/fmaf helper + FFMA swap; convert gemm_*, axpy_*, dot_f64; grep-audit and
convert the other `+= a*b` sites (verify each line against current source). tools/sync-ext.sh to mirror into
ext/libtessero.

Phase 2 - Layer 4 (P0). determinism_verify.c + check-determinism target, hard-fail, prerequisite of all, mirrored
for the extension, wired into CI. Assert FPU-canonical.

Phase 3 - Layer 2 (P0). Pin -ffp-contract=off -fno-fast-math in both csrc/Makefile:57 and ext/config.m4:20/35;
add the refuse-to-build guard; decide and implement adding repro_f32.c to the extension.

Phase 4 - D1/D2 (P0/P1). ABI tag embed + AbiTag.php + load check; glibc-floor manifest + load check;
SIMD-level/libc load check; reproducible tar + SOURCE_DATE_EPOCH.

Phase 5 - D3/D4/D5 + Layers 3/5 (P1). Backend version/capability fail-fast; preload/open_basedir diagnostics;
doctor as the deploy gate; OpenBLAS pin + TESSERO_DETERMINISTIC mode; the support matrix doc.

Verification each phase:
1. `make clean && make CC=gcc-12 -j"$(nproc)"` clean; extension builds too.
2. Regenerate exact-op goldens -> confirm unchanged (matmul forward still 4f984242...; repro-f32 four unchanged).
   Any movement = stop.
3. Force the non-FMA baseline (`make noclones` / a -march without mfma) -> confirm it now yields 4f984242...
   (was f4b5bd0c...).
4. Full CPU gate: scalar == AVX2 == AVX-512 == baseline(no-FMA) == N-thread, byte-identical, whole exact-op
   surface; FFI backend == extension backend.
5. Negative tests: inject -ffast-math -> build fails; corrupt the FPU mode -> self-test fails; bump the ABI tag ->
   load fails; stale facades -> registry check fails.

---

## Reference values

Canonical (correct) attestation goldens (from the H100 run GPU column, bit-identical to the frozen CPU goldens on
a correct build):

| op | SHA-256 |
|---|---|
| matmul forward | 4f98424294fa325c40932c385e8185a4ea3c676aee3a001f0c96159a61562601 |
| matmul grad_a | 736577e0945f25a9d756b337622aa29126f41037aee04c91eb056e4bf3f4b0f6 |
| matmul grad_b | 828c11df7eb8b8f5b9581631a009707eb9d8521f78dbbca4390a53d1cde1182f |

Repro-f32 goldens (from csrc/repro/GOLDENS.md, in-repo):

| primitive | SHA-256 |
|---|---|
| rsqrt | a55f20e5934645f1a554a0fddfa094531b13aa8b95b360ed8a44a24dc220e53f |
| exp2 | d7b57bb03e2c26b20642f0697daf64a366bedd9d7a3ee5edd79d8dae502aabba |
| exp | 7c0748f832cf0c6ea64d0382af76615033014602c90e882437dac5108d0aa336 |
| erf | aebea57c4f009254b1dfd67953c209441bad55acb561ef3ca5f1f3b75e5f8797 |

Non-canonical (wrong, non-FMA) values the CPU baseline clone produced on the H100 box:

| op | SHA-256 |
|---|---|
| matmul forward | f4b5bd0c818ce0bc44ef5753e033324c4a1f54fe44b56de4240c391426d993bd (== llvmpipe no-FMA) |
| matmul grad_a | 0ebdb399d2f97578e1485ab893fbe8bc3ec6a44e1e768bebd358d1de667768df |
| matmul grad_b | 0821b6caeb01e5a0fecac983099a8a4a113412dbef9e8ce29a2fdeb6ffa9f1be |

## Why this is the right shape

The keystones remain Layer 1 (explicit FMA) + Layer 4 (build-time self-test): together they make the arithmetic
leak impossible to reintroduce and impossible to ship. Layer 1b closes the run-time FPU-state leak; Layer 2
closes it in both build systems; Part II closes the packaging and load-time leaks and converts every silent
deployment failure into a loud, early, specific error - with doctor proving on each host that it computes the
canonical bits. The whole design extends the discipline the repro primitives already proved - explicit fma, fixed
order, don't trust the compiler, prove it - from one surface to the entire path from arithmetic to host.

## Audit provenance

Synthesized from five parallel source audits (build/packaging, runtime loading and FFI, threading and env,
attestation machinery, kernel-wide FP sweep). Directly verified against source for this doc: csrc/src/matmul.c,
csrc/src/internal.h, csrc/Makefile, ext/config.m4, src/Native/Library.php. Other file:line references (scipy_* /
np_* reduction sites, Blas.php, CI workflow, resources/preload.php) are audit-reported and should be re-confirmed
against current source at implementation time. Evidence hashes and the H100 finding originate from Talario
attestation (commit b213eeb; see Talario docs/papers/attestation-results-log.md and
docs/project/lambda-gpu-attestation-playbook.md).
