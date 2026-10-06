# libtessero build. `make` -> ../lib/<platform>/libtessero.<ext>
#   make test       C unit tests (optimised build)
#   make sanitize   C unit tests under AddressSanitizer + UBSan
#   make portable   build without target_clones (musl, clang, cross builds)

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
LIBNAME := libtessero
# LAPACKE (row-major C LAPACK) backs numpy.linalg / scipy.linalg in the kernel (src/np_linalg.c). On glibc/musl
# Linux liblapacke pulls in the system LAPACK/BLAS (OpenBLAS via the alternatives system). macOS and Windows
# linalg linkage is tracked cross-platform work (Accelerate / OpenBLAS-for-MSVC): set LAPACK on those targets.
LAPACK ?=
ifeq ($(UNAME_S),Darwin)
  PLATFORM := darwin-$(UNAME_M)
  EXT := dylib
  SHARED := -dynamiclib -install_name @rpath/libtessero.dylib
  OPENMP ?=
else ifneq (,$(findstring MINGW,$(UNAME_S))$(findstring MSYS,$(UNAME_S)))
  PLATFORM := windows-$(UNAME_M)
  EXT := dll
  LIBNAME := tessero
  SHARED := -shared -static-libgcc
  OPENMP ?=
else
  MUSL := $(shell ldd --version 2>&1 | grep -qi musl && echo -musl)
  PLATFORM := linux$(MUSL)-$(UNAME_M)
  EXT := so
  SHARED := -shared -Wl,-soname,libtessero.so
  LAPACK := -llapacke
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
SCIPYC := third_party/scipy/quadpack/quadpack.c third_party/scipy/zeros/brentq.c third_party/scipy/_cosine.c
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
$(BUILD)/c/%.o: third_party/numpy/random/src/%.c
	@mkdir -p $(BUILD)/c
	$(CC) $(CFLAGS) -Ithird_party/numpy/include -w -MMD -MP -c -o $@ $<
$(BUILD)/c/%.o: third_party/scipy/quadpack/%.c
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
FUZZ_SRC := $(filter-out src/gen_%_table.c src/np_%.c third_party/scipy/cdflib.c $(NPRAND) $(SCIPYC),$(SRC))
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
