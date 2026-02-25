#!/usr/bin/env python3
"""
Generate speedup and efficiency charts from benchmark_results.csv.

Produces three plots:
  1. Original versions (V1/V2/V3) speedup vs thread count
  2. Optimized versions speedup vs thread count
  3. Novel approaches speedup vs thread count

Usage: python plot_benchmarks.py
Reads:  benchmark_results.csv
Writes: charts/speedup_original.png
        charts/speedup_optimized.png
        charts/speedup_novel.png
        charts/speedup_all_best.png
"""

import csv
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --- Read CSV ---
data = {}  # {(version, baseline): [(threads, time), ...]}
baselines = {}

with open("benchmark_results.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        version = row["version"]
        threads = int(row["threads"])
        time = float(row["time_seconds"])
        baseline = row["baseline"]

        if threads == 1:
            baselines[baseline] = time
        else:
            key = (version, baseline)
            if key not in data:
                data[key] = []
            data[key].append((threads, time))

# Sort each version's data by thread count
for key in data:
    data[key].sort()

os.makedirs("charts", exist_ok=True)

# --- Style ---
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2,
    "lines.markersize": 8,
})

COLORS = {
    "version1_parallel_for": "#2196F3",
    "version2_sections": "#F44336",
    "version3_combined": "#4CAF50",
    "version1_optimized": "#2196F3",
    "version2_optimized": "#F44336",
    "version3_optimized": "#4CAF50",
    "novel_simd_avx2": "#FF9800",
    "novel_omp_simd": "#9C27B0",
    "novel_tiled": "#009688",
    "novel_tasks": "#795548",
    "novel_branchless": "#607D8B",
    "novel_ultimate": "#E91E63",
}

LABELS = {
    "version1_parallel_for": "V1: parallel for",
    "version2_sections": "V2: sections",
    "version3_combined": "V3: combined",
    "version1_optimized": "V1 optimized",
    "version2_optimized": "V2 optimized",
    "version3_optimized": "V3 optimized",
    "novel_simd_avx2": "SIMD AVX2",
    "novel_omp_simd": "OMP SIMD (2-pass)",
    "novel_tiled": "Cache tiling",
    "novel_tasks": "Task-based D&C",
    "novel_branchless": "Branchless",
    "novel_ultimate": "Ultimate (SIMD+tiling)",
}

MARKERS = {
    "version1_parallel_for": "o",
    "version2_sections": "s",
    "version3_combined": "^",
    "version1_optimized": "o",
    "version2_optimized": "s",
    "version3_optimized": "^",
    "novel_simd_avx2": "D",
    "novel_omp_simd": "v",
    "novel_tiled": "P",
    "novel_tasks": "X",
    "novel_branchless": "h",
    "novel_ultimate": "*",
}


def plot_group(versions_baseline, title, filename, baseline_key, show_ideal=True):
    """Plot speedup chart for a group of versions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    base_time = baselines[baseline_key]
    threads_set = set()

    for version, bl in versions_baseline:
        key = (version, bl)
        if key not in data:
            continue
        threads = [t for t, _ in data[key]]
        speedups = [base_time / time for _, time in data[key]]
        efficiencies = [sp / t for sp, t in zip(speedups, threads)]
        threads_set.update(threads)

        label = LABELS.get(version, version)
        color = COLORS.get(version, "#000000")
        marker = MARKERS.get(version, "o")

        ax1.plot(threads, speedups, marker=marker, color=color, label=label)
        ax2.plot(threads, efficiencies, marker=marker, color=color, label=label)

    if show_ideal and threads_set:
        t_range = sorted(threads_set)
        ax1.plot(t_range, t_range, "--", color="gray", alpha=0.5, label="Ideal (linear)")
        ax2.axhline(y=1.0, linestyle="--", color="gray", alpha=0.5, label="Ideal (100%)")

    ax1.set_xlabel("Threads")
    ax1.set_ylabel("Speedup")
    ax1.set_title("Speedup vs Thread Count")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.xaxis.set_major_locator(ticker.FixedLocator(sorted(threads_set) if threads_set else [2, 4, 8, 16]))

    ax2.set_xlabel("Threads")
    ax2.set_ylabel("Efficiency (Speedup / Threads)")
    ax2.set_title("Efficiency vs Thread Count")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.xaxis.set_major_locator(ticker.FixedLocator(sorted(threads_set) if threads_set else [2, 4, 8, 16]))

    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close(fig)


# --- Plot 1: Original versions ---
plot_group(
    [("version1_parallel_for", "ptr"),
     ("version2_sections", "ptr"),
     ("version3_combined", "ptr")],
    "Original Versions (baseline: sequential int***)",
    "charts/speedup_original.png",
    "ptr",
)

# --- Plot 2: Optimized versions ---
plot_group(
    [("version1_optimized", "flat"),
     ("version2_optimized", "flat"),
     ("version3_optimized", "flat")],
    "Optimized Versions (baseline: sequential_flat contiguous)",
    "charts/speedup_optimized.png",
    "flat",
)

# --- Plot 3: Novel approaches ---
plot_group(
    [("novel_simd_avx2", "novel"),
     ("novel_omp_simd", "novel"),
     ("novel_tiled", "novel"),
     ("novel_tasks", "novel"),
     ("novel_branchless", "novel"),
     ("novel_ultimate", "novel")],
    "Novel Approaches (baseline: sequential_flat contiguous)",
    "charts/speedup_novel.png",
    "flat",
)

# --- Plot 4: Best of each category ---
fig, ax = plt.subplots(figsize=(10, 6))

# Gather best time per thread count for select versions
best_versions = [
    ("version1_parallel_for", "ptr", "V1 original (ptr)"),
    ("version1_optimized", "flat", "V1 optimized (flat)"),
    ("novel_simd_avx2", "novel", "SIMD AVX2"),
    ("novel_tiled", "novel", "Cache tiling"),
    ("novel_ultimate", "novel", "Ultimate (SIMD+tiling)"),
]

for version, bl, label in best_versions:
    key = (version, bl)
    if key not in data:
        continue

    # Use sequential ptr baseline for originals, flat for everything else
    if bl == "ptr":
        base = baselines["ptr"]
    else:
        base = baselines["flat"]

    threads = [t for t, _ in data[key]]
    times = [time for _, time in data[key]]
    color = COLORS.get(version, "#000000")
    marker = MARKERS.get(version, "o")

    ax.plot(threads, times, marker=marker, color=color, label=f"{label}", linewidth=2.5)

ax.set_xlabel("Threads", fontsize=12)
ax.set_ylabel("Time (seconds)", fontsize=12)
ax.set_title("Absolute Time: Best Versions Compared", fontsize=14, fontweight="bold")
ax.legend(loc="upper right", fontsize=10)
ax.xaxis.set_major_locator(ticker.FixedLocator([2, 4, 8, 16]))
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("charts/time_comparison.png", dpi=150, bbox_inches="tight")
print("Saved: charts/time_comparison.png")
plt.close(fig)

print("\nAll charts generated successfully!")
