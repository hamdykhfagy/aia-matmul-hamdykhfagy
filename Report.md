# Lab Report – Matrix Multiplication on CPU
**Course:** AI Accelerators (AIA)
**Lab:** Praktikum 1
**Team members:** _(fill in)_
**Date:** _(fill in)_

---

## Task 1 – System Characterisation

> Fill in the details of your machine. Use tools such as `lscpu`, `lstopo`, `/proc/cpuinfo`.

| Property | Value |
|---|---|
| CPU model | CPU — Intel Core Ultra 7 265KF (Arrow Lake) |
| Number of cores / threads | 20 cores, 20 threads (8 P-cores + 12 E-cores, no HT) |
| Base / Boost clock speed (GHz) | P-core: 3.9 / 5.5 GHz · E-core: 3.3 / 4.6 GHz |
| SIMD ISA (SSE4.2 / AVX2 / AVX-512 …) | AVX2 (+ AVX-VNNI, VAES, GFNI — no AVX-512) |
| SIMD width (bits / floats per vector) | 256 bits / 8 × FP32 per vector |
| MAC units per core | 2 FMA units (256-bit each), both P- and E-cores |
| L1 cache size (per core) | P-core: 48 KB d + 64 KB i · E-core: 32 KB d + 64 KB i |
| L2 cache size (per core) | P-core: 3 MB (private) · E-core: 4 MB per cluster of 4 (1 MB/core) |
| L3 cache size (shared) | 30 MB (all cores) |
| Peak theoretical throughput (GFLOP/s) | ~3,174 GFLOP/s FP32 (3.17 TFLOP/s)|

**How did you calculate peak throughput?**

_(formula: cores × clock × SIMD_width × MACs_per_cycle)_
                         
  - Formula: cores × boost_GHz × 2 FMA_units × 8 FP32/vector × 2 ops/FMA                                                                                                             
  - P-cores: 8 × 5.5 × 32 = 1,408 GFLOP/s                                                                                                                                              
  - E-cores: 12 × 4.6 × 32 = 1,766 GFLOP/s                                                                                                                                           
  - This assumes sustained all-core max-turbo, which is the theoretical ceiling — real sustained throughput will be lower due to thermal/power limits.                                 


---

## Task 2 – Loop Reordering

> Measure each loop ordering for matrix sizes 64, 128, 256, 512, 1024, 2048, 4096.

| Loop order | N=256 (GFLOP/s) | N=1024 (GFLOP/s) | N=4096 (GFLOP/s) |
|---|---|---|---|
| i-j-k (naive) | | | |
| i-k-j | | | |
| j-k-i | | | |
| k-i-j | | | |
| _(add more)_ | | | |

**Best ordering found:** ___

**Why does this ordering perform best?**

_(Explain in terms of spatial locality and cache reuse of A, B, and C)_

---

## Task 3 – Vectorization

> List the compiler flags you tested and their effect.

| Flags added | N=1024 (GFLOP/s) | Speedup vs. naive |
|---|---|---|
| -O3 only (baseline) | | 1.0× |
| -O3 -march=native | | |
| -O3 -march=native -ffast-math | | |
| -O3 -march=native -ffast-math -funroll-loops | | |
| -O3 -march=native -ffast-math -fopenmp-simd | | |

**Did you add any `#pragma` hints to the source?** If yes, which ones?

**What speedup did you achieve? Why?**

---

## Task 4 – Loop Tiling

> Experiment with tile sizes to find the sweet spot for your cache hierarchy.

| Tile size | N=1024 (GFLOP/s) | N=4096 (GFLOP/s) |
|---|---|---|
| 32 | | |
| 64 | | |
| 128 | | |
| 256 | | |

**Best tile size:** ___

**Why does this tile size work best for your machine?**

---

## Task 5 – Multithreading

> Measure scaling as you increase the number of OpenMP threads.

| Threads | N=4096 (GFLOP/s) | Speedup |
|---|---|---|
| 1 | | 1.0× |
| 2 | | |
| 4 | | |
| 8 | | |
| _(max physical cores)_ | | |

**Does throughput scale linearly with threads?** Why / why not?

---

## Task 6 – Performance Analysis

**Is your implementation compute-bound or memory-bound?** Justify with arithmetic intensity (FLOPs / bytes).

**Comparison vs. PyTorch (N=4096):**

| Implementation | GFLOP/s | % of PyTorch |
|---|---|---|
| Naive C | | |
| Best optimised C | | |
| PyTorch (CPU) | | 100% |

**What is the gap and why does it exist?**

---

## Task 7 – Key Takeaways

_Write 3–5 sentences summarising the most important lessons learned from this lab._

---

## Figures

> Place your performance plots (GFLOP/s vs. matrix size) in the `figures/` folder and reference them here.

![Performance comparison](figures/performance.png)
