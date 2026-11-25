# =============================================================================
# initial.jl — Start solution via priority rule (Giffler-Thompson-like)
# =============================================================================
#
# All matheuristic variants should start from the SAME start solution.
# This function deterministically returns a feasible schedule (no randomness).
#
# Idea (list scheduling): while operations are open, look at the next operation
# of every job, compute its earliest start and schedule the operation that can
# start earliest. Ties are broken by the priority rule:
#   :spt  – shortest processing time first
#   :mwkr – most work remaining in the job first (often better, was used here)
# =============================================================================

"""
    initial_schedule(inst; rule = :mwkr) -> Schedule

Creates a deterministic, feasible start solution.
"""
function initial_schedule(inst::JSSPInstance; rule::Symbol = :mwkr)
    start = fill(-1, inst.n, inst.m)
    next_op = ones(Int, inst.n)        # next operation per job
    job_ready = zeros(Int, inst.n)     # job available from
    mach_ready = zeros(Int, inst.m)    # machine available from
    remaining = [sum(inst.ptime[j, :]) for j in 1:inst.n]

    for _ in 1:length(operations(inst))
        best = nothing
        best_key = nothing
        for j in 1:inst.n
            k = next_op[j]
            k <= n_ops(inst, j) || continue
            i = inst.machine[j, k]
            est = max(job_ready[j], mach_ready[i])
            prio = rule == :spt  ? inst.ptime[j, k] :
                   rule == :mwkr ? -remaining[j] :
                   error("unknown rule $rule")
            key = (est, prio, j)          # j as last tie-breaker -> deterministic
            if best_key === nothing || key < best_key
                best, best_key = (j, k), key
            end
        end
        j, k = best
        i = inst.machine[j, k]
        start[j, k] = best_key[1]
        job_ready[j] = mach_ready[i] = start[j, k] + inst.ptime[j, k]
        remaining[j] -= inst.ptime[j, k]
        next_op[j] += 1
    end
    return Schedule(start)
end
