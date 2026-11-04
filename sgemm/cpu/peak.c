// Peak-FMA microbenchmark (AVX2, single thread).
//
// Measures the best GFLOPS this core actually sustains right now, using N
// independent FMA dependency chains. Sweeping N shows the latency/throughput
// story: one chain is limited by FMA latency, and you need roughly
// latency x throughput (about 4-5 cycles x 2 per cycle = ~8-10) chains in
// flight to saturate the two FMA ports.
//
// "Implied GHz" assumes 32 FLOP/cycle (2 FMA ports x 8 floats x 2 FLOP) and
// is a cross-check against the cpufreq reading.
#define _GNU_SOURCE // must precede every include: sched_getcpu() is a GNU extension
#include <immintrin.h>
#include <sched.h>
#include "util.h"

static volatile float sink;

#define DEFINE_PEAK(N)                                                          \
static __attribute__((noinline)) double peak_##N(long iters) {                  \
    volatile float va = 1.0f, vb = 1e-3f; /* runtime values: no folding */      \
    __m256 a = _mm256_set1_ps(va), b = _mm256_set1_ps(vb);                      \
    __m256 acc[N];                                                              \
    for (int i = 0; i < N; i++) acc[i] = _mm256_set1_ps((float)i);              \
    double t0 = now();                                                          \
    for (long it = 0; it < iters; it++) {                                       \
        _Pragma("GCC unroll 16")                                                \
        for (int i = 0; i < N; i++) acc[i] = _mm256_fmadd_ps(a, b, acc[i]);     \
    }                                                                           \
    double t1 = now();                                                          \
    float s = 0.0f;                                                             \
    for (int i = 0; i < N; i++) {                                               \
        float tmp[8];                                                           \
        _mm256_storeu_ps(tmp, acc[i]);                                          \
        s += tmp[0];                                                            \
    }                                                                           \
    sink = s;                                                                   \
    return t1 - t0;                                                             \
}

DEFINE_PEAK(1)
DEFINE_PEAK(2)
DEFINE_PEAK(4)
DEFINE_PEAK(6)
DEFINE_PEAK(8)
DEFINE_PEAK(10)
DEFINE_PEAK(12)

typedef struct { int chains; double (*fn)(long); } Variant;

int main(void) {
    Variant v[] = {{1, peak_1}, {2, peak_2}, {4, peak_4}, {6, peak_6},
                   {8, peak_8}, {10, peak_10}, {12, peak_12}};
    const double total_fma = 1.6e9; // FMA instructions per timed run, whatever N is

    printf("chains,gflops,implied_ghz,cpufreq_mhz\n");
    for (size_t k = 0; k < sizeof v / sizeof v[0]; k++) {
        long iters = (long)(total_fma / v[k].chains);
        v[k].fn(iters / 20); // warmup: let the core ramp up
        double best = 1e30;
        for (int r = 0; r < 3; r++) {
            double t = v[k].fn(iters);
            if (t < best) best = t;
        }
        double gflops = 2.0 * 8.0 * v[k].chains * (double)iters / best * 1e-9;
        printf("%d,%.1f,%.2f,%.0f\n", v[k].chains, gflops, gflops / 32.0, cur_mhz(sched_getcpu()));
        fflush(stdout);
    }
    return 0;
}
