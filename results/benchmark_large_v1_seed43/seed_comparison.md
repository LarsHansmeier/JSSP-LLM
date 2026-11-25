# Seed comparison: seed 42 (results/benchmark_large_v1) vs seed 43

Same configuration except the seed; 20-minute runs; rank = by final makespan among the 5 methods (1 = best, ties averaged).

## Ranking per instance

| instance | seed 42 (best → worst) | seed 43 (best → worst) | changed |
|---|---|---|---|
| ta41 | B v2 < C v3 < random < A v2 < gurobi | B v2 < C v3 < random < A v2 < gurobi | no |
| ta61 | C v3 < B v2 < random < gurobi < A v2 | B v2 < C v3 < random < A v2 < gurobi | gurobi 4.0→5.0, A v2 5.0→4.0, B v2 2.0→1.0, C v3 1.0→2.0 |
| swv12 | B v2 < C v3 < random < A v2 < gurobi | B v2 < C v3 < random < A v2 < gurobi | no |
| ta71 | B v2 < random < C v3 < A v2 < gurobi | C v3 < B v2 < random < A v2 < gurobi | random 2.0→3.0, B v2 1.0→2.0, C v3 3.0→1.0 |

Mean rank over the instances: random 2.75 → 3.0, gurobi 4.75 → 5.0, A v2 4.25 → 4.0, B v2 1.25 → 1.25, C v3 2.0 → 1.75

## Largest primal-gap differences (seed 43 − seed 42)

| instance | method | gap seed 42 | gap seed 43 | difference | makespan 42 → 43 |
|---|---|---|---|---|---|
| swv12 | B v2 | 15.42 % | 20.51 % | 5.09 % | 3466 → 3619 |
| ta41 | gurobi | 20.27 % | 16.85 % | -3.42 % | 2427 → 2358 |
| swv12 | A v2 | 29.5 % | 32.9 % | 3.4 % | 3889 → 3991 |
| swv12 | C v3 | 20.01 % | 23.04 % | 3.03 % | 3604 → 3695 |
| ta61 | A v2 | 13.42 % | 10.6 % | -2.82 % | 3253 → 3172 |
| ta71 | B v2 | 4.8 % | 6.72 % | 1.92 % | 5726 → 5831 |
| ta61 | gurobi | 13.04 % | 11.4 % | -1.64 % | 3242 → 3195 |
| ta41 | random | 12.19 % | 10.85 % | -1.34 % | 2264 → 2237 |

Mean absolute gap difference: 1.46 %; per method: random 0.69 %, gurobi 1.31 %, A v2 1.93 %, B v2 2.27 %, C v3 1.11 %
