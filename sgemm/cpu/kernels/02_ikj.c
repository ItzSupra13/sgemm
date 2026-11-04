#include <string.h>
#include "bench.h"

const char *kernel_name = "02_ikj";

void gemm(int n, const float *A, const float *B, float *C) {
    memset(C, 0, (size_t)n * n * sizeof(float)); // zero C (all-bits-zero == 0.0f) before accumulating
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            const float a = A[i*n + k];
            for (int j = 0; j < n; j++)
                C[i*n + j] += a * B[k*n + j];
        }
    }
}
// Same arithmetic as stage 1, only the loop order changes. The inner loop now runs
// over j, so B[k][j] and C[i][j] are both walked along a row (stride 1 float)
// instead of B being walked down a column (stride n floats). A[i][k] does not
// depend on j, so it becomes a single scalar reused across the whole inner loop.
//
// What this does NOT fix: every FMA still loads C, adds into it and stores it
// back (C is read-modified-written n times per row instead of being accumulated
// in a register). That memory traffic is what stage 3 removes.
