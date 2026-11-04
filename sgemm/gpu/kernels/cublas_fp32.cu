// Reference: cuBLAS SGEMM through the same launch() interface as the hand-written kernels, so it
// pays exactly the same ctypes/launch overhead and is the fair "100%" for the fp32 ladder.
#include <cublas_v2.h>
#include <cuda_runtime.h>

static cublasHandle_t handle = nullptr;

extern "C" int launch(int n, const float *A, const float *B, float *C) {
    if (!handle) {
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return -1;
        // Plain fp32 FMA path, no tensor-op math. (A T4 has no TF32 anyway; this keeps the
        // reference honest if the code is ever run on an Ampere or newer GPU.)
        if (cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH) != CUBLAS_STATUS_SUCCESS) return -2;
    }
    const float alpha = 1.0f, beta = 0.0f;
    // cuBLAS is column-major, our matrices are row-major. A row-major matrix read as column-major
    // is its transpose, so C^T = B^T * A^T: pass B first and A second, no explicit transposes.
    cublasStatus_t s = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                   &alpha, B, n, A, n, &beta, C, n);
    return s == CUBLAS_STATUS_SUCCESS ? (int)cudaGetLastError() : 1000 + (int)s;
}
