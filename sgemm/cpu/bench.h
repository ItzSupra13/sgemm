// Each stage lives in its own .c file and defines these two symbols.
// Row-major, square n x n, single precision. The kernel OVERWRITES C
// (C = A * B, i.e. beta = 0), so zeroing C, if needed, is part of the kernel's cost.
#ifndef BENCH_H
#define BENCH_H

#include <stddef.h> // size_t, so every kernel file can use it for index math

void gemm(int n, const float *A, const float *B, float *C);
extern const char *kernel_name;

#endif
