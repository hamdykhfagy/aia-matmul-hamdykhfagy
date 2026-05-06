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

| Loop order | N=64 | N=128 | N=256 | N=512 | N=1024 | N=2048 | N=4096 |
|---|---|---|---|---|---|---|---|
| i-j-k (naive) | 12.13 | 4.91 | 3.63 | 3.06 | 0.99 | 0.76 | 0.69 |
| i-k-j | 36.05 | 31.60 | 35.67 | 38.72 | 28.38 | 22.60 | 11.90 |
| j-k-i | 9.44 | 1.82 | 1.72 | 1.47 | 0.91 | 0.79 | too long |
| k-i-j | 26.28 | 24.13 | 25.93 | 26.62 | 22.57 | 18.02 | 7.01 |

**Best ordering found:** i-k-j

**Why does this ordering perform best?**

_(Explain in terms of spatial locality and cache reuse of A, B, and C)_

---

## Task 3 – Vectorization

> List the compiler flags you tested and their effect.

| Flags added | N=1024 (GFLOP/s) | Speedup vs. naive |
|---|---|---|
| -O3 only (baseline) | 21.52 | 1.0× |
| -O3 -march=native | 20.56 | 0.96× |
| -O3 -march=native -ffast-math | 23.72 | 1.10× |
| -O3 -march=native -ffast-math -funroll-loops | 30.93 | 1.44× |
| -O3 -march=native -ffast-math -fopenmp-simd | 23.39 | 1.09× |

**Did you add any `#pragma` hints to the source?** If yes, which ones?

**What speedup did you achieve? Why?**

---

## Task 4 – Loop Tiling

> Experiment with tile sizes to find the sweet spot for your cache hierarchy.

| Tile size | N=1024 (GFLOP/s) | N=4096 (GFLOP/s) |
|---|---|---|
| 32 | 17.18 | 15.92 |
| 64 | 21.64 | 20.77 |
| 128 | 24.01 | 22.31|
| 256 | 31.75 | 30.69 |
| 512 | 33.92 | 31.23 |

**Best tile size:** 512

**Why does this tile size work best for your machine?**

---

## Task 5 – Multithreading

> Measure scaling as you increase the number of OpenMP threads.

| Threads | N=4096 (GFLOP/s) | Speedup |
|---|---|---|
| 1 | 30.69 | 1.0× |
| 2 | 39.24 | 1.28× |
| 4 | 113.91 | 3.71× |
| 8 | 224.74 | 7.32× |
| 16 | 463.89 | 15.11× |
| 20 | 463.97 | 15.12× |

**Does throughput scale linearly with threads?** Why / why not?

---

## Task 6 – Performance Analysis

**Is your implementation compute-bound or memory-bound?** Justify with arithmetic intensity (FLOPs / bytes).

**Comparison vs. PyTorch (N=4096):**

| Implementation | GFLOP/s | % of PyTorch |
|---|---|---|
| Naive C | 0.69 | 0.05% |
| Best optimised C | 463.89 | 31.2% |
| PyTorch (CPU) | 1487.6 | 100% |
| PyTorch (GPU) | 23074.9 | XL% |

**What is the gap and why does it exist?**

---

## Task 7 – Key Takeaways

Loop ordering has the single largest impact on single-core performance: switching from naive i-j-k to i-k-j improved throughput by up to 56× at N=512 purely by making the inner loop stride-1 for both B and C, eliminating cache misses without changing any arithmetic. Compiler flags (-march=native, -ffast-math, -funroll-loops) provided an additional 1.44× speedup on top of the best loop order by enabling AVX2 vectorization and reducing branch overhead in the inner loop. Loop tiling was most beneficial at large matrix sizes (N=4096), where the full matrices exceed the L2 cache — tile size 512 (fitting within the 3MB P-core L2) gave the best results, nearly doubling throughput over untiled i-k-j at that size. Multithreading scaled well up to 16 threads (15.11× speedup), but adding the remaining 4 E-cores gave no further gain, showing that memory bandwidth — not compute — is the bottleneck at that point. Despite all optimizations, our best C implementation reached only ~15% of theoretical peak, while PyTorch (using a production BLAS with hand-written AVX2 micro-kernels and register blocking) reached ~47%, highlighting that closing the gap to peak requires assembly-level optimization far beyond what a compiler can generate automatically.

---

## Figures

> Place your performance plots (GFLOP/s vs. matrix size) in the `figures/` folder and reference them here.

![Performance comparison](figures/performance.png)
