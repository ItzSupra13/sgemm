// Stage 2: shared-memory cache blocking + 1D block tiling.
// Each block computes a BM x BN tile of C, marching over K in steps of BK. Per step the block
// copies an A tile (BM x BK) and a B tile (BK x BN) into shared memory once, and every thread
// then computes TM outputs in one column of the tile, so each value read from shared memory is
// reused across TM FMAs. Contract: export launch() with C linkage, overwrite C, return cudaGetLastError().
#include <cuda_runtime.h>

#define BM 64
#define BN 64
#define BK 8
#define TM 8                          // outputs per thread (a 1D strip down one column)
#define NTHREADS ((BM * BN) / TM)     // 512

static_assert(NTHREADS == BM * BK && NTHREADS == BK * BN,
              "one thread loads exactly one element of each tile; change the loaders if you retune");

__global__ void __launch_bounds__(NTHREADS)
sgemm_smem_1d(int n, const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int tid = threadIdx.x;
    const int tx = tid % BN;               // this thread's column inside the tile
    const int ty = tid / BN;               // which strip of TM rows inside the tile
    const int row0 = blockIdx.y * BM;      // top-left corner of the tile in C
    const int col0 = blockIdx.x * BN;

    const int a_r = tid / BK, a_c = tid % BK;   // where this thread writes in As (BM x BK)
    const int b_r = tid / BN, b_c = tid % BN;   // where this thread writes in Bs (BK x BN)

    float acc[TM] = {0.0f};

    for (int k0 = 0; k0 < n; k0 += BK) {
        // Cooperative load; out-of-range elements become 0 so any n works.
        const int ar = row0 + a_r, ac = k0 + a_c;
        As[a_r][a_c] = (ar < n && ac < n) ? A[ar * n + ac] : 0.0f;
        const int br = k0 + b_r, bc = col0 + b_c;
        Bs[b_r][b_c] = (br < n && bc < n) ? B[br * n + bc] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            const float b = Bs[kk][tx];            // one shared load, reused TM times from a register
            #pragma unroll
            for (int m = 0; m < TM; m++)
                acc[m] += As[ty * TM + m][kk] * b;
        }
        __syncthreads();                            // do not overwrite the tiles while others still read
    }

    #pragma unroll
    for (int m = 0; m < TM; m++) {
        const int r = row0 + ty * TM + m, c = col0 + tx;
        if (r < n && c < n) C[r * n + c] = acc[m];
    }
}

extern "C" int launch(int n, const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
    dim3 grid((n + BN - 1) / BN, (n + BM - 1) / BM);
    sgemm_smem_1d<<<grid, NTHREADS>>>(n, A, B, C);
    return (int)cudaGetLastError();
}

// Shared-memory traffic per FMA: each inner step does TM loads of As plus 1 of Bs for TM FMAs,
// about (TM+1)/TM = 1.1 loads per FMA, versus 2 per FMA for plain shared-memory tiling.
// Bank conflicts: a warp is 32 consecutive tx with the same ty, so As[..][kk] is one address
// (broadcast) and Bs[kk][tx] hits 32 different banks. Neither conflicts.
// What it still does not fix: each thread's TM accumulators are one column wide, so every FMA
// still needs a fresh shared load. Stage 3 (2D tiling) computes a TMxTN block per thread.