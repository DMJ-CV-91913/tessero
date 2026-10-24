# libtessero build. `make` -> ../lib/<platform>/libtessero.<ext>
#   make test       C unit tests (optimised build)
#   make sanitize   C unit tests under AddressSanitizer + UBSan
#   make portable   build without target_clones (musl, clang, cross builds)

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
LIBNAME := libtessero
# LAPACKE (row-major C LAPACK) and cblas back numpy.linalg / scipy.linalg and the float matmul in the kernel.
#   Linux:   liblapacke + the system BLAS (OpenBLAS via the alternatives system).
#   macOS:   Homebrew OpenBLAS, which ships LAPACK, LAPACKE (lapacke.h) and cblas in one library (Apple's
#            Accelerate has LAPACK but no LAPACKE). `brew install openblas`; OPENBLAS_PREFIX overrides the path.
#   Windows: MSYS2 MINGW64 OpenBLAS (`pacman -S mingw-w64-x86_64-openblas`), which also carries LAPACKE.
# Set LAPACK (link flags) and LAPACK_INC (include flags) to use any other LAPACKE provider.
LAPACK ?=
LAPACK_INC ?=
ifeq ($(UNAME_S),Darwin)
  PLATFORM := darwin-$(UNAME_M)
  EXT := dylib
  SHARED := -dynamiclib -install_name @rpath/libtessero.dylib
  OPENMP ?=
  ifeq ($(LAPACK),)
    OPENBLAS_PREFIX ?= $(shell brew --prefix openblas 2>/dev/null)
    ifneq ($(OPENBLAS_PREFIX),)
      LAPACK_INC := -I$(OPENBLAS_PREFIX)/include
      LAPACK := -L$(OPENBLAS_PREFIX)/lib -lopenblas -Wl,-rpath,$(OPENBLAS_PREFIX)/lib
    endif
  endif
else ifneq (,$(findstring MINGW,$(UNAME_S))$(findstring MSYS,$(UNAME_S)))
  PLATFORM := windows-$(UNAME_M)
  EXT := dll
  LIBNAME := tessero
  SHARED := -shared -static-libgcc
  OPENMP ?=
  ifeq ($(LAPACK),)
    MINGW_PREFIX ?= /mingw64
    LAPACK := -lopenblas
    LAPACK_INC := $(if $(wildcard $(MINGW_PREFIX)/include/openblas/lapacke.h),-I$(MINGW_PREFIX)/include/openblas)
  endif
else
  MUSL := $(shell ldd --version 2>&1 | grep -qi musl && echo -musl)
  PLATFORM := linux$(MUSL)-$(UNAME_M)
  EXT := so
  SHARED := -shared -Wl,-soname,libtessero.so
  LAPACK := -llapacke -lblas
endif

CC ?= cc
CXX ?= c++
BUILD ?= ./build
OUT := ../lib/$(PLATFORM)
# numpy/random distributions (vendored, for numpy.random.Generator and scipy.stats rvs)
NPRAND := third_party/numpy/random/src/distributions.c third_party/numpy/random/src/logfactorial.c \
  third_party/numpy/random/src/random_hypergeometric.c third_party/numpy/random/src/random_mvhg_count.c \
  third_party/numpy/random/src/random_mvhg_marginals.c
# SciPy's QUADPACK translation and brentq (scipy.stats generic methods)
FITPACKC := $(wildcard third_party/scipy/fitpack/*.c)
SCIPYC := third_party/scipy/quadpack/quadpack.c third_party/scipy/zeros/brentq.c third_party/scipy/_cosine.c $(FITPACKC)
SRC := $(wildcard src/*.c) third_party/scipy/cdflib.c $(NPRAND) $(SCIPYC)
# C++: the special-function libraries SciPy uses (xsf, Boost.Math, SciPy's glue), ADR 0011
SRC_CXX := $(wildcard cxx/*.cpp) third_party/scipy/wright.cc $(EXTRA_CXX)
# EXTRA_CXX: work-in-progress sources in cxx/wip/ (make EXTRA_CXX=cxx/wip/dist_x.cpp), built only when named
TP_INC := -Ithird_party -Ithird_party/xsf -Ithird_party/scipy -Ithird_party/boost -Ithird_party/numpy/include
CXXFLAGS ?= -O2 -std=c++17 -fPIC -fno-math-errno -fvisibility=hidden -DBOOST_MATH_STANDALONE $(OPENMP) -Wall -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable -Wno-sign-compare -Wno-maybe-uninitialized -Wno-unused-but-set-variable -Wno-strict-aliasing -Wno-deprecated-declarations
OBJ_CXX = $(patsubst %,$(BUILD)/cxx/%.o,$(notdir $(SRC_CXX)))
ifeq ($(UNAME_S),Darwin)
  CXXLINK :=
else
  CXXLINK := -static-libstdc++ -static-libgcc
endif
WARN := -Wall -Wextra -Wno-unused-parameter -Wshadow -Wno-sign-compare -Wno-type-limits -Wno-missing-field-initializers
OPENMP ?= -fopenmp
CFLAGS ?= -O3 -std=gnu11 -fPIC -fno-math-errno -fno-trapping-math -fvisibility=default $(OPENMP) $(WARN)
LDLIBS := -lm $(LAPACK)
CFLAGS += $(LAPACK_INC)
CXXFLAGS += $(LAPACK_INC)

# ---- compiler and linker hardening (OpenSSF "Compiler Options Hardening Guide for C and C++"). Every flag is
#      probed, so a toolchain that lacks one drops it instead of failing. None changes floating-point semantics;
#      the parity gate verifies results stay bit-exact. HARDEN=0 turns it off. The sanitizer and fuzz builds keep
#      their own flags (FORTIFY conflicts with the ASan interceptors, zero-init would hide uninitialised reads).
HARDEN ?= 1
comma := ,
try_cc = $(shell printf 'int main(void){return 0;}\n' | $(CC) -Werror -O2 $(1) -x c - -o /dev/null >/dev/null 2>&1 && echo "$(1)")
try_cxx = $(shell printf 'int main(void){return 0;}\n' | $(CXX) -Werror -O2 $(1) -x c++ - -o /dev/null >/dev/null 2>&1 && echo "$(1)")
try_so = $(shell printf 'int tsr_probe(void){return 0;}\n' | $(CC) -Werror -fPIC -shared $(1) -x c - -o /dev/null >/dev/null 2>&1 && echo "$(1)")
ifeq ($(HARDEN),1)
  HARDEN_CC := $(call try_cc,-U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=3) $(call try_cc,-fstack-protector-strong) \
    $(call try_cc,-fstack-clash-protection) $(call try_cc,-fcf-protection=full) \
    $(call try_cc,-mbranch-protection=standard) $(call try_cc,-ftrivial-auto-var-init=zero) \
    $(call try_cc,-fno-delete-null-pointer-checks)
  HARDEN_CXX := $(HARDEN_CC) $(call try_cxx,-D_GLIBCXX_ASSERTIONS)
  HARDEN_LD := $(call try_so,-Wl$(comma)-z$(comma)relro) $(call try_so,-Wl$(comma)-z$(comma)now) \
    $(call try_so,-Wl$(comma)-z$(comma)noexecstack)
  CFLAGS += $(HARDEN_CC) $(call try_cc,-Wformat=2) $(call try_cc,-Werror=format-security)
  CXXFLAGS += $(HARDEN_CXX)
  LDLIBS += $(HARDEN_LD)
endif

all: $(OUT)/$(LIBNAME).$(EXT)

$(BUILD)/cxx/%.cpp.o: cxx/%.cpp cxx/*.hpp src/fn.h
	@mkdir -p $(BUILD)/cxx
	$(CXX) $(CXXFLAGS) $(TP_INC) -MMD -MP -c -o $@ $<

$(BUILD)/cxx/%.cpp.o: cxx/wip/%.cpp cxx/*.hpp src/fn.h
	@mkdir -p $(BUILD)/cxx
	$(CXX) $(CXXFLAGS) $(TP_INC) -Icxx -MMD -MP -c -o $@ $<

$(BUILD)/cxx/wright.cc.o: third_party/scipy/wright.cc
	@mkdir -p $(BUILD)/cxx
	$(CXX) $(CXXFLAGS) $(TP_INC) -MMD -MP -c -o $@ $<

OBJ_C = $(patsubst %.c,$(BUILD)/c/%.o,$(notdir $(SRC)))
$(BUILD)/c/%.o: src/%.c src/*.h include/tessero.h
	@mkdir -p $(BUILD)/c
	$(CC) $(CFLAGS) -Ithird_party/numpy/include -c -o $@ $<
# repro_f32.c: the reproducible-f32 primitives MUST compile with contraction and fast-math OFF so
# the compiler cannot contract or reassociate the non-fma arithmetic. This is what keeps the result
# bit-identical across SIMD level, thread count and compiler (see csrc/repro/GOLDENS.md). The flags
# follow $(CFLAGS) so they override anything inherited. This explicit target beats the generic rule.
$(BUILD)/c/repro_f32.o: src/repro_f32.c src/*.h include/tessero.h include/tessero_repro_f32.h
	@mkdir -p $(BUILD)/c
	$(CC) $(CFLAGS) -fno-fast-math -ffp-contract=off -Ithird_party/numpy/include -c -o $@ $<
$(BUILD)/c/%.o: third_party/numpy/random/src/%.c
	@mkdir -p $(BUILD)/c
	$(CC) $(CFLAGS) -Ithird_party/numpy/include -w -MMD -MP -c -o $@ $<
$(BUILD)/c/%.o: third_party/scipy/quadpack/%.c
	@mkdir -p $(BUILD)/c
	$(CC) $(CFLAGS) -w -MMD -MP -c -o $@ $<
$(BUILD)/c/%.o: third_party/scipy/fitpack/%.c
	@mkdir -p $(BUILD)/c
	$(CC) $(CFLAGS) -w -MMD -MP -c -o $@ $<
$(BUILD)/c/%.o: third_party/scipy/zeros/%.c
	@mkdir -p $(BUILD)/c
	$(CC) $(CFLAGS) -w -MMD -MP -c -o $@ $<
$(BUILD)/c/_cosine.o: third_party/scipy/_cosine.c
	@mkdir -p $(BUILD)/c
	$(CC) $(CFLAGS) -w -MMD -MP -c -o $@ $<
$(BUILD)/c/cdflib.o: third_party/scipy/cdflib.c
	@mkdir -p $(BUILD)/c
	$(CC) $(CFLAGS) -w -MMD -MP -c -o $@ $<

$(OUT)/$(LIBNAME).$(EXT): $(OBJ_C) $(OBJ_CXX)
	@mkdir -p $(OUT)
	$(CXX) $(CFLAGS) $(OBJ_C) $(OBJ_CXX) $(SHARED) $(CXXLINK) -o $@ $(LDLIBS)

# header dependencies of the C++ and vendored objects (-MMD)
-include $(wildcard $(BUILD)/cxx/*.d $(BUILD)/c/*.d)

portable:
	$(MAKE) CFLAGS="$(CFLAGS) -DTSR_NO_CLONES"

$(BUILD)/test_opt: $(OBJ_C) tests/test_main.c $(OBJ_CXX)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -c -o $(BUILD)/test_main.o tests/test_main.c
	$(CXX) $(CFLAGS) $(OBJ_C) $(BUILD)/test_main.o $(OBJ_CXX) $(CXXLINK) -o $@ $(LDLIBS)

SAN := -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer
OBJ_CXX_SAN = $(patsubst %,$(BUILD)/cxx_san/%.o,$(notdir $(SRC_CXX)))

$(BUILD)/cxx_san/%.cpp.o: cxx/%.cpp cxx/*.hpp src/fn.h
	@mkdir -p $(BUILD)/cxx_san
	$(CXX) -O1 -g -std=c++17 -DBOOST_MATH_STANDALONE $(OPENMP) $(SAN) $(TP_INC) -c -o $@ $<

$(BUILD)/cxx_san/wright.cc.o: third_party/scipy/wright.cc
	@mkdir -p $(BUILD)/cxx_san
	$(CXX) -O1 -g -std=c++17 $(OPENMP) $(SAN) $(TP_INC) -c -o $@ $<

$(BUILD)/test_san: $(SRC) tests/test_main.c $(OBJ_CXX_SAN)
	@mkdir -p $(BUILD)/c_san
	cd $(BUILD)/c_san && $(CC) -O1 -g -std=gnu11 -fno-math-errno -fno-trapping-math $(OPENMP) $(SAN) \
	    -DTSR_NO_CLONES $(WARN) -w -I$(abspath third_party/numpy/include) -c $(abspath $(SRC)) $(abspath tests/test_main.c)
	$(CXX) $(SAN) $(OPENMP) $(BUILD)/c_san/*.o $(OBJ_CXX_SAN) -o $@ $(LDLIBS)

test: $(BUILD)/test_opt
	$(BUILD)/test_opt

sanitize: $(BUILD)/test_san
	ASAN_OPTIONS=detect_leaks=1 $(BUILD)/test_san

clean:
	rm -rf $(BUILD)

.PHONY: all portable test sanitize clean fuzz fuzz-standalone

# ---- fuzzing (see csrc/fuzz/): coverage-guided with libFuzzer (clang + compiler-rt), or the standalone
#      mutation driver under ASan + UBSan anywhere. Corpora live in fuzz/corpus/<target>.
FUZZ_TARGETS := slice npy mmap_plan json
# the fuzzed parsers do not need the function registry (C++): build without it
FUZZ_SRC := $(filter-out src/gen_%_table.c src/np_%.c src/scipy_%.c third_party/scipy/cdflib.c $(NPRAND) $(SCIPYC),$(SRC))
FUZZ_TIME ?= 60
FUZZ_RUNS ?= 200000
FUZZ_FLAGS := -O1 -g -std=gnu11 -Ithird_party/numpy/include -fno-math-errno -fno-trapping-math -fno-omit-frame-pointer -DTSR_NO_CLONES $(WARN)

fuzz:
	@mkdir -p $(BUILD)
	@for t in $(FUZZ_TARGETS); do \
	    clang $(FUZZ_FLAGS) -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all \
	        -DTSR_NO_REGISTRY -o $(BUILD)/fuzz_$$t $(FUZZ_SRC) fuzz/fuzz_$$t.c $(LDLIBS) || exit 1; \
	    mkdir -p $(BUILD)/corpus_$$t; \
	    $(BUILD)/fuzz_$$t -max_total_time=$(FUZZ_TIME) -max_len=4096 $(BUILD)/corpus_$$t fuzz/corpus/$$t || exit 1; \
	done

fuzz-standalone:
	@mkdir -p $(BUILD)
	@for t in $(FUZZ_TARGETS); do \
	    $(CC) $(FUZZ_FLAGS) -fsanitize=address,undefined -fno-sanitize-recover=all \
	        -DTSR_NO_REGISTRY -o $(BUILD)/fuzzsa_$$t $(FUZZ_SRC) fuzz/fuzz_$$t.c fuzz/driver.c $(LDLIBS) || exit 1; \
	    ASAN_OPTIONS=detect_leaks=1 $(BUILD)/fuzzsa_$$t -runs=$(FUZZ_RUNS) -seed=$${FUZZ_SEED:-1} fuzz/corpus/$$t || exit 1; \
	done
