#include <cblas.h>
#include "bench.h"

const char *kernel_name = "openblas";

void gemm(int n, const float *A, const float *B, float *C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0f, A, n, B, n, 0.0f, C, n);
}
// Reference: OpenBLAS sgemm
// Run with OPENBLAS_NUM_THREADS=1 and pinned to one core (run.sh does both).
