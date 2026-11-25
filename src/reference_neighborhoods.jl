# =============================================================================
# reference_neighborhoods.jl — Reference neighbourhoods for tuning (NOT policies)
# =============================================================================

"""
    window_free(inst, incumbent, history, rng, n_free_y) -> Vector of machine pairs

Contiguous reference neighbourhood: draws a random time t in [0, makespan), visits the
operations by temporal distance to t (0 if t ∈ [start, end), otherwise distance to the
nearer interval end; ties by operation index) and frees, for each new operation a, all
pairs with already selected operations on the same machine – until exactly `n_free_y`
pairs are free (all pairs if the instance has fewer).

Only a reference for tuning, not a policy.
"""
function window_free(inst::JSSPInstance, incumbent::Schedule, history,
                     rng::AbstractRNG, n_free_y::Int)
    t = rand(rng) * makespan(inst, incumbent)
    ops = operations(inst)
    function dist(op)
        s = incumbent.start[op...]
        e = s + inst.ptime[op...]
        return s <= t < e ? 0.0 : min(abs(t - s), abs(t - e))
    end
    order = sort(eachindex(ops); by = i -> (dist(ops[i]), i))

    pairs_of = Dict(op => Tuple{Tuple{Int,Int},Tuple{Int,Int}}[] for op in ops)
    for (a, b) in machine_pairs(inst)
        push!(pairs_of[a], (a, b))
        push!(pairs_of[b], (a, b))
    end

    selected = Set{Tuple{Int,Int}}()
    chosen = Tuple{Tuple{Int,Int},Tuple{Int,Int}}[]
    for i in order
        a = ops[i]
        for p in pairs_of[a]
            length(chosen) >= n_free_y && return chosen
            other = p[1] == a ? p[2] : p[1]
            other in selected && push!(chosen, p)
        end
        push!(selected, a)
    end
    return chosen
end
