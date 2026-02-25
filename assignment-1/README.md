# Assignment 1: OpenMP Min/Max in 3D Matrix

Finding minimum and maximum values (with indices) in a 500x500x500 3D matrix using OpenMP parallel constructs. The project implements the 3 required parallel versions, then explores optimized variants and 6 novel approaches to push performance from 0.494s (sequential) down to 0.015s (ultimate).

## Building

```bash
make all        # builds all 14 versions into bin/
make clean      # removes bin/
```

Requires GCC with OpenMP support. The SIMD versions additionally require AVX2 (`-mavx2`), handled automatically by the Makefile.

## Running

```bash
# Single version
OMP_NUM_THREADS=8 ./bin/version1_parallel_for

# Full benchmark suite (all 14 versions, 2/4/8/16 threads, best of 3, correctness checks)
bash run_benchmarks.sh

# Generate speedup/efficiency charts (requires Python 3 + matplotlib)
python plot_benchmarks.py
```

All versions output identical results for correctness verification:
```
Min = -1 at (499, 499, 499)
Max = 100000 at (250, 250, 250)
Time: X.XXXXXX seconds
```

## Input Design (`read_input` in `common.h`)

- Allocates a 500x500x500 matrix (125 million elements, ~477 MB)
- Fills with pseudo-random values in [0, 99999] using a fixed seed (42) for reproducibility
- Uses `wide_rand()` to work around MinGW's RAND_MAX = 32767 limitation (combines two `rand()` calls)
- Plants a unique minimum (-1) at position (499, 499, 499) and a unique maximum (100000) at (250, 250, 250) so correctness can be verified deterministically
- Two memory layouts provided:
  - `read_input()` — `int***` pointer-of-pointer (250,501 separate mallocs, 3-level indirection)
  - `read_input_flat()` — contiguous `int*` (single malloc, arithmetic indexing via `IDX` macro)

## Versions Overview

### Required versions (Q1-Q3) — `int***` layout, matching pseudocode exactly

| File | Construct | Data Races | Solution |
|------|-----------|------------|----------|
| `sequential.c` | — | N/A | Direct translation of assignment pseudocode |
| `version1_parallel_for.c` | `parallel for` | All 6 index variables (`min_i/j/k`, `max_i/j/k`) — check-then-act, torn indices, lost updates | Thread-private local copies + `#pragma omp critical` merge |
| `version2_sections.c` | `sections` | None — each section writes to separate variables | Two sections: one for min, one for max. Max parallelism = 2 threads |
| `version3_combined.c` | `parallel for` + `sections` | None between sections; V1-style races within each section's inner parallel for | Named critical sections (`critical(min_merge)`, `critical(max_merge)`) for inner merges |

Why not `atomic`? The update involves 6 variables and a conditional check — a compound operation that `atomic` cannot protect. `atomic` only handles single memory operations.

### Optimized versions — contiguous `int*` layout

These demonstrate how memory layout and OpenMP features affect performance. Compared against `sequential_flat` for fairness.

| File | Key Improvements |
|------|-----------------|
| `sequential_flat.c` | Fair baseline using contiguous `int*` with `IDX(i,j,k,N,P)` macro |
| `version1_optimized.c` | `#pragma omp declare reduction(minloc/maxloc)` replaces critical section with O(log T) tree reduction + `collapse(2)` for 250K-iteration space + `MinMaxLoc` struct stores value (avoids re-indexing `a[min_i][min_j][min_k]` every comparison) |
| `version2_optimized.c` | Contiguous memory eliminates the cache contention that made original V2 slower than sequential |
| `version3_optimized.c` | Nested parallelism + declare reduction + collapse(2) within each section |

### Novel approaches — advanced techniques

| File | Technique | Best Time | Key Insight |
|------|-----------|-----------|-------------|
| `novel_simd_avx2.c` | AVX2 intrinsics (`_mm256_cmpgt_epi32`, `_mm256_blendv_epi8`) — processes 8 ints per instruction | 0.031s @4T | Fastest pure technique, but degrades beyond 4 threads (memory bandwidth saturated) |
| `novel_omp_simd.c` | `#pragma omp parallel for simd reduction(min/max)` — compiler auto-vectorises. Two-pass: values first (vectorised), then indices (scalar over cache-hot data) | 0.045s @8T | ~65% of hand-written AVX2 with zero intrinsics maintenance |
| `novel_tiled.c` | L2 cache tiling (8x8x500 = 125KB tiles) + `__builtin_prefetch` + single-pass min AND max | 0.031s @16T | Best scaling curve — halves memory bandwidth vs two-section approaches |
| `novel_tasks.c` | Recursive `#pragma omp task` with `final()` clause, work-stealing scheduler, 64K-element leaf tasks | 0.030s @16T | Demonstrates task paradigm; matches parallel for on uniform workloads |
| `novel_branchless.c` | XOR-based conditional select: `mask = -(cond); result = (new & mask) \| (old & ~mask)` — no branches | 0.041s @16T | Slowest novel approach — proves branchless is counterproductive on random data (branch predictor >99.99% accurate) |
| `novel_ultimate.c` | AVX2 SIMD + cache tiling + prefetch combined — addresses compute, bandwidth, and latency bottlenecks simultaneously | **0.015s @4T** | The champion: 45.2x speedup. ~2x faster than either SIMD or tiling alone |

## Performance Results

### Required versions (baseline: sequential `int***` = 0.494s)

| Version | 2T | 4T | 8T | 16T |
|---------|----|----|----|----|
| V1 parallel for | 2.32x | 5.31x | 8.37x | 12.67x |
| V2 sections | 0.66x | 0.64x | 0.64x | 0.63x |
| V3 combined | 1.58x | 3.01x | 4.75x | 7.26x |

- V1 shows good scaling but plateaus below theoretical due to memory bandwidth
- V2 is slower than sequential — pointer-chasing + double memory traffic from two full passes
- V3 is slower than V1 due to nested parallelism overhead

### Best of each category (absolute time)

| Version | Best Time | Speedup | Threads |
|---------|-----------|---------|---------|
| Sequential (`int***`) | 0.494s | 1.0x | 1 |
| V1 parallel for | 0.039s | 12.7x | 16 |
| V1 optimized | 0.037s | 18.3x | 8 |
| Ultimate (SIMD + tiling) | **0.015s** | **45.2x** | 4 |

### Theoretical vs Empirical

The report includes theoretical speedup analysis assuming all instructions take equal time (T_V1 = 2N/p, T_V2 = N, T_V3 = 2N/p) and explains why empirical results diverge: this is a **memory-bound** workload where data movement, not arithmetic, is the bottleneck.

## Key Findings

1. **Memory layout matters most**: Switching from `int***` (3-level pointer chasing) to contiguous `int*` is the single biggest improvement
2. **Memory bandwidth is the wall**: All approaches plateau around 0.030s at high thread counts regardless of compute optimisation
3. **SIMD + tiling breaks through**: Combining them (0.015s) is ~2x faster than either alone — SIMD reduces compute so the CPU keeps up with prefetched data, while tiling ensures L2 residency
4. **`declare reduction` > `critical`**: Tree-based O(log T) merge eliminates the serial bottleneck
5. **Branchless is counterproductive**: On random data, branch prediction is >99.99% accurate; branchless adds unconditional work
6. **Compiler auto-vectorisation is good**: OMP SIMD achieves ~65% of hand-written AVX2 with zero maintenance

## Benchmarking Infrastructure

- `run_benchmarks.sh` — runs all 14 versions at 2/4/8/16 threads, best of 3 runs for stability, automated correctness checking against expected min/max values and positions, outputs CSV + summary tables. Uses two separate baselines: `sequential` (ptr) for originals, `sequential_flat` for optimized/novel versions
- `plot_benchmarks.py` — reads `benchmark_results.csv`, generates 4 charts in `charts/`:
  - Original versions speedup + efficiency
  - Optimized versions speedup + efficiency
  - Novel approaches speedup + efficiency
  - Absolute time comparison of best versions
- `benchmark_results.csv` — raw timing data from the latest full run

## Project Structure

```
assignment-1/
  src/
    common.h                  # read_input (both layouts), MinMaxLoc, declare reduction, wide_rand
    sequential.c              # Baseline (int*** layout)
    sequential_flat.c         # Baseline (contiguous layout)
    version1_parallel_for.c   # Q1: parallel for + critical
    version2_sections.c       # Q2: parallel sections
    version3_combined.c       # Q3: nested sections + parallel for
    version1_optimized.c      # V1 + flat + declare reduction + collapse
    version2_optimized.c      # V2 + flat memory
    version3_optimized.c      # V3 + flat + declare reduction + collapse
    novel_simd_avx2.c         # AVX2 intrinsics
    novel_omp_simd.c          # OMP SIMD directive (two-pass)
    novel_tiled.c             # Cache tiling + prefetch
    novel_tasks.c             # Task-based divide & conquer
    novel_branchless.c        # Branchless bitwise min/max
    novel_ultimate.c          # SIMD + tiling + prefetch combined
  Makefile                    # Builds all 14 versions
  run_benchmarks.sh           # Full benchmark suite
  plot_benchmarks.py          # Chart generation
  benchmark_results.csv       # Raw results
  report.md                   # Full report (Q1-Q4 + further optimisations + novel approaches)
  charts/                     # Generated speedup/efficiency plots
  README.md                   # This file
```
