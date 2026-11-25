# =============================================================================
# JSSPLab.jl — Main module: includes all parts
# =============================================================================
module JSSPLab

include("instance.jl")
include("jsplib.jl")
include("schedule.jl")
include("checker.jl")
include("initial.jl")
include("bounds.jl")
include("model.jl")
include("matheuristic.jl")     # Fix-and-Optimize loop (random_fao)
include("reference_neighborhoods.jl")   # window_free (tuning reference, not a policy)
include("score_based.jl")      # Policy A: score-based release (policyA_selector)
include("structure_based.jl")  # Policy B: structural neighbourhoods (policyB_selector)
include("dynamic.jl")          # Policy C: adaptive deterministic control (policyC_selector)
include("dynamic_v2.jl")       # Policy C v2: revised control (policyC_v2_selector)
include("dynamic_v3.jl")       # Policy C v3 (final): per-schedule duplicates (policyC_v3_selector)
include("hybrid.jl")           # Policy D v1: hybrid (policyD_selector)
# Later:
# include("policies.jl")       # Policy A, B, C (replace select_free)
# include("experiments.jl")    # Benchmark & logging

export JSSPInstance, read_instance, read_standard, read_taillard, print_instance,
       load_jsplib, list_jsplib, jsplib_info, JSPLIB_DIR,
       operations, ops_on_machine, total_processing_time, trivial_lower_bound,
       Schedule, makespan, machine_sequences, schedule_from_sequences, print_schedule,
       check_schedule, initial_schedule, lower_bound, one_machine_lower_bound,
       build_model, solve_model, machine_pairs, set_warm_start!, schedule_from_model,
       precedes, random_fao, random_free, window_free, policyA_selector, policyB_selector, policyB_v2_selector, policyB_v3_selector, policyB_log, policyC_selector, policyC_log, policyC_v2_selector, policyC_v2_log,
       policyC_v3_selector, policyC_v3_log, schedule_fingerprint,
       policyD_selector, policyD_log, SOLVE_EXACT,
       SKIP_ITERATION, STOP_SEARCH,
       compute_structural_features, fao_diagnostics, keep_awake, DEFAULT_SEED, FREE_CAP

end # module
