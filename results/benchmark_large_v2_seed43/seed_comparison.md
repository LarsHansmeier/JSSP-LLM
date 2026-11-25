# Seed comparison: seed 42 (results/benchmark_large_v2) vs seed 43

Same configuration except the seed (fix-and-optimize RNG, Gurobi Seed, CP-SAT random_seed); 20-minute runs.
Rank = by final makespan among the 7 methods per instance (1 = best, ties averaged).

## Ranking per instance

| instance | seed 42 (best → worst) | seed 43 (best → worst) | changed |
|---|---|---|---|
| swv11 | CPSAT_8w < CPSAT_1w < B v2 < C v3 < random < A v2 < gurobi | CPSAT_8w < CPSAT_1w < B v2 < A v2 < random < C v3 < gurobi | A v2 6.0→4.0, C v3 4.0→6.0 |
| ta31 | CPSAT_8w < CPSAT_1w < C v3 < random < B v2 < A v2 < gurobi | CPSAT_8w < CPSAT_1w < C v3 < random < B v2 < A v2 < gurobi | no |
| ta51 | CPSAT_8w < CPSAT_1w < B v2 < random < C v3 < gurobi < A v2 | CPSAT_8w < CPSAT_1w < B v2 < random < C v3 < A v2 < gurobi | gurobi 6.0→7.0, A v2 7.0→6.0 |
| ta72 | CPSAT_8w < B v2 < C v3 < random < gurobi < A v2 < CPSAT_1w | CPSAT_8w < B v2 < random < C v3 < gurobi < A v2 < CPSAT_1w | random 4.0→3.0, C v3 3.0→4.0 |

Mean rank over the instances (seed 42 → seed 43): random 4.25 → 4.0, gurobi 6.25 → 6.5, A v2 6.25 → 5.5, B v2 3.25 → 3.25, C v3 3.75 → 4.5, CPSAT_1w 3.25 → 3.25, CPSAT_8w 1.0 → 1.0

## Largest primal-gap differences (seed 43 − seed 42)

| instance | method | gap seed 42 | gap seed 43 | difference | makespan 42 → 43 |
|---|---|---|---|---|---|
| swv11 | C v3 | 22.63 % | 32.26 % | 9.63 % | 3668 → 3956 |
| swv11 | B v2 | 21.46 % | 15.55 % | -5.92 % | 3633 → 3456 |
| swv11 | A v2 | 34.2 % | 29.52 % | -4.68 % | 4014 → 3874 |
| ta51 | random | 15.0 % | 10.98 % | -4.02 % | 3174 → 3063 |
| ta51 | gurobi | 19.35 % | 22.97 % | 3.62 % | 3294 → 3394 |
| ta51 | A v2 | 19.82 % | 16.7 % | -3.12 % | 3307 → 3221 |
| ta51 | CPSAT_1w | 4.02 % | 1.01 % | -3.01 % | 2871 → 2788 |
| ta31 | gurobi | 10.77 % | 13.55 % | 2.78 % | 1954 → 2003 |

Mean absolute gap difference: 1.83 %; per method: random 1.33 %, gurobi 2.07 %, A v2 2.28 %, B v2 2.26 %, C v3 3.0 %, CPSAT_1w 1.45 %, CPSAT_8w 0.44 %
