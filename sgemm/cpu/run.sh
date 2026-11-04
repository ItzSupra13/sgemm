#!/usr/bin/env bash
# Run any binary single-threaded, pinned to one hardware thread.
#
#   bash run.sh bin/peak
#   bash run.sh bin/ref_openblas > results/ref.csv
#   bash run.sh bin/01_naive -n 64,128,256,512,1024 -r 5 > results/01.csv
#
# CPU=3 bash run.sh ...   to pick another logical CPU (default 2; run.py picks one
# that exists on the machine automatically, so this matters only for manual runs).
#
# On the i7-1355U laptop: logical CPUs 0-3 are the two P-cores (0,1 = core 0;
# 2,3 = core 1), CPUs 4-11 are E-cores. We default to CPU 2 because CPU 0 tends to
# service interrupts, and we leave its SMT sibling (CPU 3) idle so we own the core.
# On a 2-vCPU Colab VM there is no CPU 2, so run.py passes CPU=0.

set -euo pipefail

CPU=${CPU:-2}
base=/sys/devices/system/cpu/cpu$CPU/cpufreq
gov=$(cat "$base/scaling_governor" 2>/dev/null || echo unknown)
epp=$(cat "$base/energy_performance_preference" 2>/dev/null || echo unknown)

echo "# pin=cpu$CPU governor=$gov epp=$epp" >&2

if [[ $gov != performance && $gov != unknown ]]; then
    echo "# WARNING: governor is '$gov'. For stable numbers run:" >&2
    echo "#   sudo cpupower frequency-set -g performance" >&2
fi

if upower -e >/dev/null 2>&1 && ! cat /sys/class/power_supply/AC*/online 2>/dev/null | grep -q 1; then
    echo "# WARNING: laptop looks like it is on battery. Plug in." >&2
fi

# One thread everywhere: the single-thread ladder must not be helped by BLAS threading.
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1

exec taskset -c "$CPU" "$@"
