#ifndef UTIL_H
#define UTIL_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <time.h>

// Monotonic wall clock in seconds. CLOCK_MONOTONIC is immune to NTP jumps.
static inline double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// Current frequency of a logical CPU in MHz (from cpufreq), or -1 if unavailable.
// We log this next to every result: GFLOPS without the clock it ran at is not
// interpretable on a 15 W laptop chip that boosts and throttles.
// (On Colab VMs cpufreq is not exposed, so this reads -1: use the peak benchmark's
// "implied GHz" as the clock estimate there.)
static inline double cur_mhz(int cpu) {
    char path[128];
    snprintf(path, sizeof path,
             "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", cpu);
    FILE *f = fopen(path, "r");
    if (!f) return -1.0;
    long khz = 0;
    if (fscanf(f, "%ld", &khz) != 1) khz = 0;
    fclose(f);
    return khz > 0 ? khz / 1000.0 : -1.0;
}

#endif
