### E2.6 L run provenance (10 finished runs; K = 5 realizations, N = 150 dev scenarios, 60 s, NumPy float64)

| Run seed | Folder | Host | GPU | Commit | Dirty | Stop reason | Gens | Robot-s | Selected (in-run) | Ckpts |
|---:|---|---|---|---|---|---|---:|---:|---|---:|
| 0 | `e2-seed0` | Pavilion | NVIDIA GeForce RTX 3080 | `4582fe7` | clean | budget | 651 | 1.500e+09 | `checkpoints/best_2.996e07.npz` | 32 |
| 1 | `e2-seed1` | Evans-G16 | NVIDIA GeForce RTX 5070 Ti Laptop GPU | `4582fe7` | clean | cma-es stop: {'tolupsigma': 1e+20} | 539 | 1.242e+09 | `checkpoints/mean_8.986e08.npz` | 30 |
| 2 | `e2-seed2` | Evans-G16 | NVIDIA GeForce RTX 5070 Ti Laptop GPU | `4582fe7` | clean | cma-es stop: {'tolupsigma': 1e+20} | 530 | 1.221e+09 | `checkpoints/mean_5.99e08.npz` | 30 |
| 3 | `e2-seed3` | Evans-G16 | NVIDIA GeForce RTX 5070 Ti Laptop GPU | `4582fe7` | clean | cma-es stop: {'tolupsigma': 1e+20} | 570 | 1.313e+09 | `checkpoints/best_7.488e08.npz` | 30 |
| 4 | `e2-seed4` | Evans-G16 | NVIDIA GeForce RTX 5070 Ti Laptop GPU | `4582fe7` | clean | cma-es stop: {'tolupsigma': 1e+20} | 545 | 1.256e+09 | `checkpoints/best_9.999e08.npz` | 30 |
| 5 | `e2-seed5` | Evans-G16 | NVIDIA GeForce RTX 5070 Ti Laptop GPU | `4582fe7` | clean | cma-es stop: {'tolupsigma': 1e+20} | 594 | 1.369e+09 | `checkpoints/mean_8.986e08.npz` | 32 |
| 6 | `e2-seed6` | Evans-G16 | NVIDIA GeForce RTX 5070 Ti Laptop GPU | `4582fe7` | clean | cma-es stop: {'tolupsigma': 1e+20} | 552 | 1.272e+09 | `checkpoints/best_7.488e08.npz` | 30 |
| 7 | `e2-seed7` | Evans-G16 | NVIDIA GeForce RTX 5070 Ti Laptop GPU | `4582fe7` | clean | cma-es stop: {'tolupsigma': 1e+20} | 549 | 1.265e+09 | `checkpoints/best_2.996e07.npz` | 30 |
| 8 | `e2-seed8` | Evans-G16 | NVIDIA GeForce RTX 5070 Ti Laptop GPU | `4582fe7` | clean | cma-es stop: {'tolupsigma': 1e+20} | 520 | 1.198e+09 | `checkpoints/mean_8.986e08.npz` | 28 |
| 9 | `e2-seed9` | Evans-G16 | NVIDIA GeForce RTX 5070 Ti Laptop GPU | `4582fe7` | clean | cma-es stop: {'tolupsigma': 1e+20} | 507 | 1.168e+09 | `checkpoints/mean_7.488e08.npz` | 28 |

### E2.6 per-run best candidate (plus frozen MB PID baseline)

| Seed | Host | Best candidate | Matches final | Mean-of-means | Median (KxN) | Both-up | Falls/(KxN) | Falls/realization |
|---:|---|---|---|---:|---:|---:|---:|---|
| — | — | `PID, model-based` | — | 98.957 | 28.093 | 28.872 | 27/750 | `[6, 4, 5, 6, 6]` |
| 0 | Pavilion | `L / s0 / final` | yes | 543.025 | 540.570 | 540.992 | 1/750 | `[1, 0, 0, 0, 0]` |
| 1 | Evans-G16 | `L / s1 / mean_7.488e08` | no | 19.794 | 19.701 | 19.724 | 0/750 | `[0, 0, 0, 0, 0]` |
| 2 | Evans-G16 | `L / s2 / mean_1.198e09` | no | 19.760 | 19.653 | 19.689 | 0/750 | `[0, 0, 0, 0, 0]` |
| 3 | Evans-G16 | `L / s3 / mean_8.986e08` | no | 19.749 | 19.687 | 19.676 | 0/750 | `[0, 0, 0, 0, 0]` |
| 4 | Evans-G16 | `L / s4 / best_8.986e08` | no | 19.767 | 19.705 | 19.696 | 0/750 | `[0, 0, 0, 0, 0]` |
| 5 | Evans-G16 | `L / s5 / best_5.99e08` | no | 19.786 | 19.754 | 19.705 | 0/750 | `[0, 0, 0, 0, 0]` |
| 6 | Evans-G16 | `L / s6 / mean_7.488e08` | no | 19.762 | 19.589 | 19.689 | 0/750 | `[0, 0, 0, 0, 0]` |
| 7 | Evans-G16 | `L / s7 / final` | yes | 620.035 | 600.718 | 601.295 | 10/750 | `[2, 2, 2, 2, 2]` |
| 8 | Evans-G16 | `L / s8 / mean_7.488e08` | no | 19.724 | 19.576 | 19.655 | 0/750 | `[0, 0, 0, 0, 0]` |
| 9 | Evans-G16 | `L / s9 / mean_8.986e08` | no | 19.726 | 19.578 | 19.655 | 0/750 | `[0, 0, 0, 0, 0]` |

### E2.6 top 25 candidates overall (plus `PID, model-based`)

| Rank | Candidate | Run seed | Mean-of-means | Median (KxN) | Both-up | Falls/(KxN) | Falls/realization |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `L / s8 / mean_7.488e08` | 8 | 19.724 | 19.576 | 19.655 | 0/750 | `[0, 0, 0, 0, 0]` |
| 2 | `L / s9 / mean_8.986e08` | 9 | 19.726 | 19.578 | 19.655 | 0/750 | `[0, 0, 0, 0, 0]` |
| 3 | `L / s9 / best_8.986e08` | 9 | 19.730 | 19.582 | 19.659 | 0/750 | `[0, 0, 0, 0, 0]` |
| 4 | `L / s9 / best_9.999e08` | 9 | 19.732 | 19.582 | 19.662 | 0/750 | `[0, 0, 0, 0, 0]` |
| 5 | `L / s9 / mean_9.999e08` | 9 | 19.732 | 19.582 | 19.662 | 0/750 | `[0, 0, 0, 0, 0]` |
| 6 | `L / s9 / best_1.168e09` | 9 | 19.733 | 19.609 | 19.662 | 0/750 | `[0, 0, 0, 0, 0]` |
| 7 | `L / s9 / mean_1.168e09` | 9 | 19.733 | 19.609 | 19.662 | 0/750 | `[0, 0, 0, 0, 0]` |
| 8 | `L / s9 / mean_5.99e08` | 9 | 19.745 | 19.661 | 19.672 | 0/750 | `[0, 0, 0, 0, 0]` |
| 9 | `L / s9 / best_7.488e08` | 9 | 19.745 | 19.606 | 19.682 | 0/750 | `[0, 0, 0, 0, 0]` |
| 10 | `L / s3 / mean_8.986e08` | 3 | 19.749 | 19.687 | 19.676 | 0/750 | `[0, 0, 0, 0, 0]` |
| 11 | `L / s3 / best_8.986e08` | 3 | 19.751 | 19.648 | 19.675 | 0/750 | `[0, 0, 0, 0, 0]` |
| 12 | `L / s9 / final` | 9 | 19.751 | 19.630 | 19.680 | 0/750 | `[0, 0, 0, 0, 0]` |
| 13 | `L / s9 / mean_7.488e08` | 9 | 19.751 | 19.630 | 19.680 | 0/750 | `[0, 0, 0, 0, 0]` |
| 14 | `L / s9 / best_2.995e08` | 9 | 19.753 | 19.640 | 19.686 | 0/750 | `[0, 0, 0, 0, 0]` |
| 15 | `L / s2 / mean_1.198e09` | 2 | 19.760 | 19.653 | 19.689 | 0/750 | `[0, 0, 0, 0, 0]` |
| 16 | `L / s2 / best_1.221e09` | 2 | 19.760 | 19.653 | 19.689 | 0/750 | `[0, 0, 0, 0, 0]` |
| 17 | `L / s2 / mean_1.221e09` | 2 | 19.760 | 19.653 | 19.690 | 0/750 | `[0, 0, 0, 0, 0]` |
| 18 | `L / s2 / best_1.198e09` | 2 | 19.760 | 19.653 | 19.690 | 0/750 | `[0, 0, 0, 0, 0]` |
| 19 | `L / s9 / best_5.99e08` | 9 | 19.762 | 19.707 | 19.696 | 0/750 | `[0, 0, 0, 0, 0]` |
| 20 | `L / s6 / mean_7.488e08` | 6 | 19.762 | 19.589 | 19.689 | 0/750 | `[0, 0, 0, 0, 0]` |
| 21 | `L / s2 / best_9.999e08` | 2 | 19.763 | 19.638 | 19.693 | 0/750 | `[0, 0, 0, 0, 0]` |
| 22 | `L / s9 / mean_2.995e08` | 9 | 19.764 | 19.637 | 19.697 | 0/750 | `[0, 0, 0, 0, 0]` |
| 23 | `L / s4 / best_8.986e08` | 4 | 19.767 | 19.705 | 19.696 | 0/750 | `[0, 0, 0, 0, 0]` |
| 24 | `L / s2 / mean_9.999e08` | 2 | 19.769 | 19.662 | 19.695 | 0/750 | `[0, 0, 0, 0, 0]` |
| 25 | `L / s3 / mean_7.488e08` | 3 | 19.769 | 19.699 | 19.691 | 0/750 | `[0, 0, 0, 0, 0]` |
| 137 | `PID, model-based` | — | 98.957 | 28.093 | 28.872 | 27/750 | `[6, 4, 5, 6, 6]` |

### Machine & seed-parity disclosure

| Group | Seeds | Runs | Best mean-of-means median | Range |
|---|---|---:|---:|---|
| host `Evans-G16` | [1, 2, 3, 4, 5, 6, 7, 8, 9] | 9 | 19.762 | 19.724–620.035 |
| host `Pavilion` | [0] | 1 | 543.025 | 543.025–543.025 |
| even seeds | [0, 2, 4, 6, 8] | 5 | 19.762 | 19.724–543.025 |
| odd seeds | [1, 3, 5, 7, 9] | 5 | 19.786 | 19.726–620.035 |

**Selected E2.6 L candidate:** `L / s8 / mean_7.488e08` (mean-of-means 19.724, median 19.576, both-up 19.655, falls 0/750 [0, 0, 0, 0, 0]).

*Pre-registered K-realization re-score (E2_NOTES.md, approved 2026-11-25 16:44, before any E2.6 re-score ran): K = 5 NumPy float64 realizations of the dev tier (seeds 5000-5049 x 3 start tilts = N 150, 60 s); realization k adds k x 1e-15 rad to every scenario's initial tilt (k = 0..4, k = 0 is the plain dev tier); same machine, CUDA_VISIBLE_DEVICES=-1. Candidates: every run's final.npz and its saved checkpoint candidates, plus the frozen MB PID baseline. Selection key: mean over realizations of the mean dev cost; ties equal at the reported 3-decimal precision go to fewer total falls. Per-run machine and stop_reason reported for disclosure.*
