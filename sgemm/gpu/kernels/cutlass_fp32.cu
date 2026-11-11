// Reference: CUTLASS fp32 SIMT GEMM (CUDA cores, no tensor cores) through the same launch() interface.
// Default configuration for Sm75 + OpClassSimt: 128x128x8 threadblock tile, 32x64x8 warp tile.
// Header-only; build.py fetches the pinned CUTLASS tag into sgemm/gpu/third_party/ on first build.
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cuda_runtime.h>

using RowMajor = cutlass::layout::RowMajor;
using Gemm = cutlass::gemm::device::Gemm<
    float, RowMajor,            // A
    float, RowMajor,            // B
    float, RowMajor,            // C / D
    float,                      // accumulator
    cutlass::arch::OpClassSimt, cutlass::arch::Sm75>;

extern "C" int launch(int n, const float *A, const float *B, float *C) {
    static Gemm gemm_op;
    static int configured_n = -1;

    typename Gemm::Arguments args({n, n, n},          // M, N, K
                                  {A, n}, {B, n},     // (pointer, leading dimension)
                                  {C, n}, {C, n},     // source C (unused: beta = 0) and destination D
                                  typename Gemm::EpilogueOutputOp::Params(1.0f, 0.0f));  // alpha, beta

    cutlass::Status s;
    if (n != configured_n) {              // full setup only when the problem size changes ...
        s = gemm_op.initialize(args);
        configured_n = n;
    } else {                              // ... otherwise just refresh the pointers (cheap)
        s = gemm_op.update(args);
    }
    if (s != cutlass::Status::kSuccess) return 2000 + (int)s;
    s = gemm_op.run();                    // default stream, same as the other kernels
    if (s != cutlass::Status::kSuccess) return 3000 + (int)s;
    return (int)cudaGetLastError();
}