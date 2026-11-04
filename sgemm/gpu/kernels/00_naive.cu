#include <cuda_runtime.h>

#define BLOCK 16

__global__ void sgemm_naive(int n, const float *A, const float *B, float *C) {
    // threadIdx.x -> ROW, threadIdx.y -> COLUMN. This is deliberately the "obvious" mapping.
    const int i = blockIdx.x * BLOCK + threadIdx.x;   // row of C
    const int j = blockIdx.y * BLOCK + threadIdx.y;   // column of C
    if (i < n && j < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++)
            sum += A[i * n + k] * B[k * n + j];
        C[i * n + j] = sum;   // assignment, so C needs no zeroing
    }
}

extern "C" int launch(int n, const float *A, const float *B, float *C) {
    dim3 block(BLOCK, BLOCK);
    dim3 grid((n + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);
    sgemm_naive<<<grid, block>>>(n, A, B, C);
    return (int)cudaGetLastError();
}

// Why it is slow: consecutive threads in a warp have consecutive threadIdx.x, i.e. consecutive
// ROWS i. So a warp reads A[i*n+k] and writes C[i*n+j] with a stride of n floats: every lane hits
// a different 128-byte line (uncoalesced), and the same happens on every iteration of k.
// B[k*n+j] has the same address for all lanes sharing j (a broadcast), which is cheap.
// Nothing is reused through shared memory either: each A and B element is re-read from global
// memory n times. Stage 1 swaps the two index mappings so a warp walks along a row of C.
