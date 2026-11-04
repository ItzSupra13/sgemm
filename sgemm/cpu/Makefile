.RECIPEPREFIX := >

# One compiler for everything. ":=" (not "?=") because make predefines CC=cc,
# which "?=" would never override. Override on the command line if you must: make CC=clang
CC    := gcc
WARN  := -std=gnu11 -Wall -Wextra
INC   := -I.
LIBS  := -lopenblas -lm

# Add each new stage here: put kernels/03_xxx.c in place and append 03_xxx.
KERNELS := 01_naive 02_ikj
COMPARE_KERNEL := 02_ikj

HARNESS := bench.c bench.h util.h

# Flag sets. -O0 and -O2 are the plain baselines; -O3 is "everything the compiler can do for free".
FLAGS_O0 := -O0
FLAGS_O1 := -O1
FLAGS_O2 := -O2
# -fno-finite-math-only: -ffast-math alone lets the compiler assume no NaNs, which folds away the
# harness's NaN check (a kernel that never writes C then "passes" with rel_err=nan). Keep it.
FLAGS_O3 := -O3 -march=native -ffast-math -fno-finite-math-only

BINS := bin/ref_openblas bin/peak $(addprefix bin/,$(KERNELS)) \
        bin/$(COMPARE_KERNEL)_O0 bin/$(COMPARE_KERNEL)_O1 bin/$(COMPARE_KERNEL)_O2 bin/$(COMPARE_KERNEL)_O3
# To test all 4 compiler optimization levels for all kernels:
#   $(foreach k,$(KERNELS),bin/$(k)_O0 bin/$(k)_O1 bin/$(k)_O2 bin/$(k)_O3)
.PHONY: all clean info
all: $(BINS)

# Results and plots are NOT created here: the Python runner owns sgemm/cpu/results/<tag>/
# and sgemm/cpu/plots/<tag>/.
bin:
> mkdir -p bin

bin/%: kernels/%.c $(HARNESS) | bin
> $(CC) $(WARN) $(INC) $(FLAGS_O3) -DBUILD_TAG='"$(FLAGS_O3)"' $< bench.c -o $@ $(LIBS)

bin/$(COMPARE_KERNEL)_O0: kernels/$(COMPARE_KERNEL).c $(HARNESS) | bin
> $(CC) $(WARN) $(INC) $(FLAGS_O0) -DBUILD_TAG='"$(FLAGS_O0)"' $< bench.c -o $@ $(LIBS)

bin/$(COMPARE_KERNEL)_O1: kernels/$(COMPARE_KERNEL).c $(HARNESS) | bin
> $(CC) $(WARN) $(INC) $(FLAGS_O1) -DBUILD_TAG='"$(FLAGS_O1)"' $< bench.c -o $@ $(LIBS)

bin/$(COMPARE_KERNEL)_O2: kernels/$(COMPARE_KERNEL).c $(HARNESS) | bin
> $(CC) $(WARN) $(INC) $(FLAGS_O2) -DBUILD_TAG='"$(FLAGS_O2)"' $< bench.c -o $@ $(LIBS)

bin/$(COMPARE_KERNEL)_O3: kernels/$(COMPARE_KERNEL).c $(HARNESS) | bin
> $(CC) $(WARN) $(INC) $(FLAGS_O3) -DBUILD_TAG='"$(FLAGS_O3)"' $< bench.c -o $@ $(LIBS)

bin/ref_openblas: kernels/ref_openblas.c $(HARNESS) | bin
> $(CC) $(WARN) $(INC) $(FLAGS) -DBUILD_TAG='"$(FLAGS)"' $< bench.c -o $@ $(LIBS)

bin/peak: peak.c util.h | bin
> $(CC) $(WARN) $(INC) $(FLAGS_O3) peak.c -o $@

info:
> @$(CC) --version | head -1

clean:
> rm -rf bin
