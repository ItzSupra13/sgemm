// Shared benchmark driver. Link with exactly one kernel file.
//
// ./bench [-n 64,128,...] [-r reps] [-m max_n] [-H]
//
// stdout: CSV (one row per size).  stderr: human-readable environment info.
#define _GNU_SOURCE // must precede every include: sched_getcpu() is a GNU extension
#include <cblas.h>
#include <math.h>
#include <sched.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "bench.h"
#include "util.h"

// OpenBLAS extras; not declared by the generic cblas.h that Arch ships
extern char *openblas_get_corename(void);
extern int openblas_get_num_threads(void);

#ifndef BUILD_TAG
#define BUILD_TAG "unknown"
#endif

#define TOL 1e-4          // max |C - Cref| / max |Cref|
#define MIN_SAMPLE_S 0.05 // each timed sample should last at least this long

static uint32_t rng_state = 12345u; // fixed seed -> identical inputs every run
static float rnd(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return (float)(rng_state >> 8) * (2.0f / 16777216.0f) - 1.0f; // [-1, 1)
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static float *alloc_matrix(int n) {
    size_t bytes = (size_t)n * n * sizeof(float);
    bytes = (bytes + 63) & ~(size_t)63; // aligned_alloc needs a multiple of alignment
    float *p = aligned_alloc(64, bytes);
    if (!p) { perror("aligned_alloc"); exit(1); }
    return p;
}

int main(int argc, char **argv) {
    int reps = 7, max_n = 1 << 30, header = 1;
    const char *sizes = "64,128,256,512,1024,2048,4096,8192";
    int opt;
    while ((opt = getopt(argc, argv, "n:r:m:H")) != -1) {
        switch (opt) {
        case 'n': sizes = optarg; break;
        case 'r': reps = atoi(optarg); break;
        case 'm': max_n = atoi(optarg); break;
        case 'H': header = 0; break;
        default:
            fprintf(stderr, "usage: %s [-n sizes] [-r reps] [-m max_n] [-H]\n", argv[0]);
            return 2;
        }
    }

    fprintf(stderr, "# kernel=%s build=[%s]\n", kernel_name, BUILD_TAG);
    fprintf(stderr, "# openblas core=%s threads=%d\n",
            openblas_get_corename(), openblas_get_num_threads());
    fprintf(stderr, "# pinned cpu=%d (start) mhz=%.0f\n", sched_getcpu(), cur_mhz(sched_getcpu()));

    if (header)
        printf("kernel,build,n,median_gflops,best_gflops,median_ms,max_rel_err,ok,cpu,mhz_end\n");

    char *list = strdup(sizes);
    for (char *tok = strtok(list, ","); tok; tok = strtok(NULL, ",")) {
        int n = atoi(tok);
        if (n <= 0 || n > max_n) continue;

        float *A = alloc_matrix(n), *B = alloc_matrix(n);
        float *C = alloc_matrix(n), *R = alloc_matrix(n);
        for (size_t i = 0; i < (size_t)n * n; i++) A[i] = rnd();
        for (size_t i = 0; i < (size_t)n * n; i++) B[i] = rnd();

        // Reference result (OpenBLAS), computed once, outside any timing.
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n,
                    1.0f, A, n, B, n, 0.0f, R, n);

        // Fill C with NaN so a kernel that forgets to write an entry fails the check.
        memset(C, 0xFF, (size_t)n * n * sizeof(float));
        double t_first = now();
        gemm(n, A, B, C); // doubles as the warmup run
        t_first = now() - t_first;

        // Tolerance check, not exact equality: summation order differs between kernels.
        double max_diff = 0.0, max_ref = 0.0;
        int bad = 0;
        for (size_t i = 0; i < (size_t)n * n; i++) {
            double d = fabs((double)C[i] - (double)R[i]);
            if (!(d == d)) bad = 1; // NaN
            if (d > max_diff) max_diff = d;
            if (fabs((double)R[i]) > max_ref) max_ref = fabs((double)R[i]);
        }
        double rel = bad ? INFINITY : max_diff / max_ref;
        int ok = !bad && rel <= TOL;

        // Several calls per sample for small n, so timer resolution never matters.
        int inner = (int)ceil(MIN_SAMPLE_S / (t_first > 1e-9 ? t_first : 1e-9));
        if (inner < 1) inner = 1;
        if (inner > 100000) inner = 100000;

        double *t = malloc(sizeof(double) * reps);
        for (int r = 0; r < reps; r++) {
            double t0 = now();
            for (int k = 0; k < inner; k++) gemm(n, A, B, C);
            t[r] = (now() - t0) / inner;
        }
        int cpu = sched_getcpu();
        double mhz = cur_mhz(cpu);

        qsort(t, reps, sizeof(double), cmp_double);
        double med = t[reps / 2], best = t[0];
        double flops = 2.0 * n * (double)n * n;
        printf("%s,\"%s\",%d,%.2f,%.2f,%.3f,%.2e,%d,%d,%.0f\n", kernel_name, BUILD_TAG, n,
               flops / med * 1e-9, flops / best * 1e-9, med * 1e3, rel, ok, cpu, mhz);
        fflush(stdout);
        if (!ok) fprintf(stderr, "# WARNING: n=%d FAILED correctness (rel err %.2e)\n", n, rel);

        free(t); free(A); free(B); free(C); free(R);
    }
    free(list);
    return 0;
}
