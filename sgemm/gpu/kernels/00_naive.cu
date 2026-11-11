#include <cuda_runtime.h>

#define BLOCK 32

__global__ void naive(int n, const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
    const int i = blockIdx.x * BLOCK + threadIdx.x; // row of C
    const int j = blockIdx.y * BLOCK + threadIdx.y; // column of C
    if (i < n && j < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++)
            sum += A[i * n + k] * B[k * n + j];
        C[i * n + j] = sum;
    }
}

extern "C" int launch(int n, const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
    dim3 block(BLOCK, BLOCK);
    dim3 grid((n + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);
    naive<<<grid, block>>>(n, A, B, C);
    return (int)cudaGetLastError();
}