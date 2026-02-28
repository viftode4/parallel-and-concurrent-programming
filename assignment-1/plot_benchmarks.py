#!/usr/bin/env python3
"""
Generate polished speedup and efficiency charts from benchmark_results.csv.

Usage: python plot_benchmarks.py
Reads:  benchmark_results.csv
Writes: charts/speedup_original.png
        charts/speedup_optimized.png
        charts/speedup_novel.png
        charts/time_comparison.png
"""

import csv
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
    "font.size": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#cccccc",
    "grid.linewidth": 0.6,
    "grid.linestyle": ":",
    "grid.alpha": 1.0,
    "axes.axisbelow": True,
    "lines.linewidth": 1.6,
    "lines.markersize": 6,
    "legend.framealpha": 1.0,
    "legend.edgecolor": "#cccccc",
    "legend.fancybox": False,
    "legend.fontsize": 8.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.labelcolor": "black",
    "text.color": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# ── Palette — muted, print-friendly ───────────────────────────────────────────
BG        = "white"
ACCENT    = "#1a4f72"
IDEAL_CLR = "#999999"

# Distinguishable even in greyscale print (mix of shade + marker shape)
PALETTE = {
    "version1_parallel_for": "#1a4f72",
    "version2_sections":     "#c0392b",
    "version3_combined":     "#2e7d32",
    "version1_optimized":    "#1a4f72",
    "version2_optimized":    "#c0392b",
    "version3_optimized":    "#2e7d32",
    "novel_simd_avx2":       "#b45309",
    "novel_omp_simd":        "#6a0dad",
    "novel_tiled":           "#00695c",
    "novel_tasks":           "#4e342e",
    "novel_branchless":      "#546e7a",
    "novel_ultimate":        "#b71c1c",
}

LABELS = {
    "version1_parallel_for": "V1: parallel for",
    "version2_sections":     "V2: sections",
    "version3_combined":     "V3: combined",
    "version1_optimized":    "V1 optimized",
    "version2_optimized":    "V2 optimized",
    "version3_optimized":    "V3 optimized",
    "novel_simd_avx2":       "SIMD AVX2",
    "novel_omp_simd":        "OMP SIMD (2-pass)",
    "novel_tiled":           "Cache tiling + prefetch",
    "novel_tasks":           "Task-based D&C",
    "novel_branchless":      "Branchless XOR",
    "novel_ultimate":        "Ultimate (SIMD + tiling)",
}

MARKERS = {
    "version1_parallel_for": "o",
    "version2_sections":     "s",
    "version3_combined":     "^",
    "version1_optimized":    "o",
    "version2_optimized":    "s",
    "version3_optimized":    "^",
    "novel_simd_avx2":       "D",
    "novel_omp_simd":        "v",
    "novel_tiled":           "P",
    "novel_tasks":           "X",
    "novel_branchless":      "h",
    "novel_ultimate":        "*",
}

MARKER_SIZES = {k: 9 for k in MARKERS}
MARKER_SIZES["novel_ultimate"] = 13   # star needs to be bigger

# ── Read CSV ───────────────────────────────────────────────────────────────────
data = {}       # {(version, baseline): [(threads, time), ...]}
baselines = {}  # {baseline_key: sequential_time}

with open("benchmark_results.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        version  = row["version"]
        threads  = int(row["threads"])
        time     = float(row["time_seconds"])
        baseline = row["baseline"]

        if threads == 1:
            baselines[baseline] = time
        else:
            key = (version, baseline)
            data.setdefault(key, []).append((threads, time))

for key in data:
    data[key].sort()

os.makedirs("charts", exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10, fontweight="bold", pad=7)
    ax.set_xlabel(xlabel, fontsize=9.5)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.tick_params(labelsize=8.5)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#888888")
        ax.spines[spine].set_linewidth(0.7)


def plot_group(versions_baseline, title, filename, baseline_key, show_ideal=True):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(title, fontsize=10, style="italic", y=1.01)

    base_time   = baselines[baseline_key]
    threads_set = set()

    for version, bl in versions_baseline:
        key = (version, bl)
        if key not in data:
            continue
        pts      = data[key]
        threads  = [t for t, _ in pts]
        speedups = [base_time / time for _, time in pts]
        effs     = [sp / t for sp, t in zip(speedups, threads)]
        threads_set.update(threads)

        label  = LABELS.get(version, version)
        color  = PALETTE.get(version, "#333333")
        marker = MARKERS.get(version, "o")
        ms     = MARKER_SIZES.get(version, 6)

        ax1.plot(threads, speedups, marker=marker, ms=ms, color=color, label=label, zorder=3)
        ax2.plot(threads, effs,     marker=marker, ms=ms, color=color, label=label, zorder=3)

    t_range = sorted(threads_set) if threads_set else [2, 4, 8, 16]

    if show_ideal:
        ax1.plot(t_range, t_range, "--", color=IDEAL_CLR, lw=1.1,
                 label="Ideal (linear)", zorder=1)
        ax2.axhline(1.0, linestyle="--", color=IDEAL_CLR, lw=1.1,
                    label="Ideal (100%)", zorder=1)

    style_ax(ax1, "Speedup vs. Thread Count",    "Threads", "Speedup")
    style_ax(ax2, "Efficiency vs. Thread Count", "Threads", "Efficiency")

    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(ticker.FixedLocator(t_range))
        ax.legend(loc="upper left" if ax is ax1 else "upper right", handlelength=2.0)

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close(fig)


# ── Plot 1: Original versions ─────────────────────────────────────────────────
plot_group(
    [("version1_parallel_for", "ptr"),
     ("version2_sections",     "ptr"),
     ("version3_combined",     "ptr")],
    "Original Versions  ·  baseline: sequential int***  (T = 0.494 s)",
    "charts/speedup_original.png",
    "ptr",
)

# ── Plot 2: Optimized versions ────────────────────────────────────────────────
plot_group(
    [("version1_optimized", "flat"),
     ("version2_optimized", "flat"),
     ("version3_optimized", "flat")],
    "Optimized Versions  ·  baseline: sequential_flat contiguous  (T = 0.678 s)",
    "charts/speedup_optimized.png",
    "flat",
)

# ── Plot 3: Novel approaches ──────────────────────────────────────────────────
plot_group(
    [("novel_simd_avx2",   "novel"),
     ("novel_omp_simd",    "novel"),
     ("novel_tiled",       "novel"),
     ("novel_tasks",       "novel"),
     ("novel_branchless",  "novel"),
     ("novel_ultimate",    "novel")],
    "Novel Approaches  ·  baseline: sequential_flat contiguous  (T = 0.678 s)",
    "charts/speedup_novel.png",
    "flat",
    show_ideal=False,
)

# ── Plot 4: Absolute time — best versions ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.5))
best_versions = [
    ("version1_parallel_for", "ptr",   "V1 original (int***)"),
    ("version1_optimized",    "flat",  "V1 optimized (flat)"),
    ("novel_simd_avx2",       "novel", "SIMD AVX2"),
    ("novel_tiled",           "novel", "Cache tiling + prefetch"),
    ("novel_ultimate",        "novel", "Ultimate (SIMD + tiling)"),
]

for version, bl, label in best_versions:
    key = (version, bl)
    if key not in data:
        continue
    pts     = data[key]
    threads = [t for t, _ in pts]
    times   = [time for _, time in pts]
    color   = PALETTE.get(version, "#555555")
    marker  = MARKERS.get(version, "o")
    ms      = MARKER_SIZES.get(version, 9)

    ax.plot(threads, times, marker=marker, ms=ms, color=color, label=label,
            linewidth=2.4, markeredgecolor="white", markeredgewidth=0.9, zorder=3)

    # Annotate minimum (best) time
    idx = int(np.argmin(times))
    ax.annotate(
        f"{times[idx]:.3f}s",
        xy=(threads[idx], times[idx]),
        xytext=(6, 5),
        textcoords="offset points",
        fontsize=7.5,
        color=color,
        fontweight="bold",
    )

# Sequential baseline reference line
seq_time = baselines["ptr"]
ax.axhline(seq_time, linestyle=":", color="#7f8c8d", lw=1.4, zorder=1)
ax.text(16.2, seq_time, f"sequential\n{seq_time:.3f}s",
        va="center", fontsize=8, color="#7f8c8d")

ax.set_xlabel("Threads", fontsize=9.5)
ax.set_ylabel("Time (seconds)", fontsize=9.5)
ax.set_title("Absolute Execution Time — Best Versions",
             fontsize=10, fontweight="bold", pad=7)
ax.xaxis.set_major_locator(ticker.FixedLocator([2, 4, 8, 16]))
ax.set_ylim(bottom=0)
ax.legend(loc="upper right", handlelength=2.0)
ax.tick_params(labelsize=8.5)

for spine in ["left", "bottom"]:
    ax.spines[spine].set_color("#888888")
    ax.spines[spine].set_linewidth(0.7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig("charts/time_comparison.png", dpi=300, bbox_inches="tight")
print("Saved: charts/time_comparison.png")
plt.close(fig)

print("\nAll charts generated successfully!")
