// Acts as our baseline
#include <cuda_runtime.h>

#define BX 32    // exactly one warp
#define BY 32    // rows per block

__global__ void coalesced(int n, const float *A, const float *B, float *C) {
    const int j = blockIdx.x * BX + threadIdx.x;
    const int i = blockIdx.y * BY + threadIdx.y;
    if (i < n && j < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++)
            sum += A[i * n + k] * B[k * n + j];
        C[i * n + j] = sum;
    }
}

extern "C" int launch(int n, const float *A, const float *B, float *C) {
    dim3 block(BX, BY);
    dim3 grid((n + BX - 1) / BX, (n + BY - 1) / BY);
    coalesced<<<grid, block>>>(n, A, B, C);
    return (int)cudaGetLastError();
}