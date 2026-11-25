# =============================================================================
# checker.jl — Independent feasibility checker
# =============================================================================
#
# "The solver is not the validator": this code uses NOTHING from the
# optimisation model. It checks a solution against the instance data only.
#
# Checks:
#   1. All start times are set and ≥ 0
#   2. The order within every job is respected
#   3. No two operations overlap on the same machine
#   4. (optional) The claimed makespan matches the recomputed one
#
# The condition "a job is never on two machines at once" follows from 2.
# =============================================================================

"""
    check_schedule(inst, s; claimed_makespan = nothing, verbose = true) -> (ok, errors)

Checks the feasibility of a schedule. Returns `ok::Bool` and a list of
error messages.
"""
function check_schedule(inst::JSSPInstance, s::Schedule;
                        claimed_makespan::Union{Nothing,Real} = nothing,
                        verbose::Bool = true)
    errors = String[]

    # 0. Dimensions
    if size(s.start) != (inst.n, inst.m)
        push!(errors, "start time matrix has size $(size(s.start)), expected $((inst.n, inst.m))")
        return false, errors
    end

    # 1. Start times set and non-negative
    for (j, k) in operations(inst)
        s.start[j, k] >= 0 || push!(errors, "operation ($j,$k): start time $(s.start[j,k]) < 0")
    end

    # 2. Job order
    for j in 1:inst.n, k in 2:n_ops(inst, j)
        prev_end = s.start[j, k-1] + inst.ptime[j, k-1]
        if s.start[j, k] < prev_end
            push!(errors, "job $j: operation $k starts at $(s.start[j,k]), " *
                          "but its predecessor only ends at $prev_end")
        end
    end

    # 3. Machine conflicts: sort by start time, compare neighbours
    for i in 1:inst.m
        ops = sort(ops_on_machine(inst, i); by = op -> s.start[op...])
        for t in 2:length(ops)
            (j1, k1), (j2, k2) = ops[t-1], ops[t]
            end1 = s.start[j1, k1] + inst.ptime[j1, k1]
            if s.start[j2, k2] < end1
                push!(errors, "machine $i: job $j1 (until $end1) overlaps with " *
                              "job $j2 (from $(s.start[j2,k2]))")
            end
        end
    end

    # 4. Recompute makespan
    if claimed_makespan !== nothing && isempty(errors)
        actual = makespan(inst, s)
        if abs(actual - claimed_makespan) > 1e-6
            push!(errors, "claimed makespan $claimed_makespan ≠ recomputed $actual")
        end
    end

    ok = isempty(errors)
    if verbose
        if ok
            println("✓ schedule feasible (makespan = $(makespan(inst, s)))")
        else
            println("✗ schedule INFEASIBLE – $(length(errors)) errors:")
            foreach(e -> println("   - ", e), first(errors, 20))
            length(errors) > 20 && println("   ... and $(length(errors) - 20) more")
        end
    end
    return ok, errors
end
