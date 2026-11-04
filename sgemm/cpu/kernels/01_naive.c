#include "bench.h"

const char *kernel_name = "01_naive";

void gemm(int n, const float *A, const float *B, float *C) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++)
                sum += A[i*n + k] * B[k*n + j];
            C[i*n + j] = sum; // assignment, not accumulation, so no memset needed
        }
    }
}
// Inner loop walks B down a column (stride n floats): nearly every access to B
// touches a new cache line, which is why this is slow once B no longer fits in cache.
