All 8 large instances × 2 seeds, built by scripts/benchmark_large_v2_seed.jl --combined-2seeds
from (read only): benchmark_large_v1 (seed 42), benchmark_large_cpsat_v1 (seed 42), benchmark_large_v2 (seed 42), benchmark_large_v1_seed43 (seed 43), benchmark_large_v2_seed43 (seed 43)
comparison_all.csv: one row per instance × seed × method (final makespan, primal gap, primal integral over 1200 s)
  rank_all  = rank by final makespan among all methods present for that instance and seed
              (n_methods_in_rank_all: 7, or 5 for ta41/ta61/swv12/ta71 with seed 43 – no CP-SAT seed-43 runs there)
  rank_our5 = rank among random, gurobi, A v2, B v2, C v3 (always 5 methods; comparable across all 16 pairs)
summary.csv: per method over all its instance-seed pairs (16 for our methods; 12 for CP-SAT: seed 42 on all 8
  instances, seed 43 only on swv11/ta31/ta51/ta72); n_best_of_our_five counts ties for every tied method
per_instance.csv: per instance × method mean over the seeds, min / max primal gap over the seeds
written: 2026-11-25T20:22:28.979
