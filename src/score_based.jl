# =============================================================================
# score_based.jl — Policy A: score-based release (deterministic)
# =============================================================================
#
# Plugs into the existing Fix-and-Optimize loop (random_fao) as `select_free`;
# start solution, model, fixing, warm start, time limits, acceptance and logging
# are unchanged. Only the choice of released operations differs.
#
# Release size: the loop passes the shared per-iteration y budget n (drawn from
# n_free_min:n_free_max, capped at FREE_CAP of all machine pairs). Two variants:
#   policyA_selector()                  "PolicyA_ScoreBased" (default): R = top k operations,
#                                       k = max(1, floor(n / (n_jobs - 1))) – every operation has
#                                       n_jobs - 1 pairs on its machine, so at most n y are freed
#   policyA_selector(fill_budget=true)  "PolicyA_ScoreBased_fill": operations are added in
#                                       score order as long as the distinct free y stay ≤ n; an
#                                       operation that would exceed n is skipped, selection stops
#                                       once exactly n y are free (same target as random_free)
#
# Released y (rule "any"): y[(a,b)] is free if a ∈ R or b ∈ R, otherwise fixed.
#
# Features of operation o = (i, j) (job i, position j), from the CURRENT incumbent,
# each in [0, 1] (denominator 0 -> feature 0):
#   f1 remaining work     = Σ_{l ≥ j} p[i,l] / max_i' Σ_l p[i',l]
#   f2 machine load       = L[machine[i,j]] / max_m L[m],  L[m] = Σ p of operations on m
#   f3 local waiting      = W[i,j] / max W,  W[i,j] = S[i,j] - (S[i,j-1] + p[i,j-1]), W[i,1] = S[i,1]
#   f4 release frequency  = 1 - count[o] / maxc  (maxc = max count; maxc == 0 -> f4 = 1),
#                           count[o] = number of releases of o since the last improvement
#   score = 0.30 f3 + 0.20 f1 + 0.20 f2 + 0.30 f4
# Selection: sort by score descending, ties by smaller job, then smaller position;
# R by the variant's rule (see above); count[o] += 1 for o ∈ R.
# Reset: if the incumbent makespan is strictly smaller than at the previous call,
# all counters are set to 0 before scoring.
# =============================================================================

"Operation count k for the default variant from the shared y budget n (see file header)."
policyA_k(inst::JSSPInstance, n_free_y::Int) =
    inst.n > 1 ? clamp(fld(n_free_y, inst.n - 1), 1, length(operations(inst))) :
                 length(operations(inst))

"""
    policyA_scores(inst, incumbent, count) -> Vector of NamedTuples

Features f1–f4 and score of every operation, sorted by score descending (ties: smaller
job, then smaller position). `count[o]` = releases of o since the last improvement.
"""
function policyA_scores(inst::JSSPInstance, incumbent::Schedule,
                        count::Dict{Tuple{Int,Int},Int})
    p, S = inst.ptime, incumbent.start
    ops = operations(inst)

    maxjob = maximum(sum(p[i, :]) for i in 1:inst.n)
    load = [sum(p[i, j] for (i, j) in ops_on_machine(inst, m); init = 0) for m in 1:inst.m]
    maxload = maximum(load)
    W(i, j) = S[i, j] - (j == 1 ? 0 : S[i, j-1] + p[i, j-1])
    maxW = maximum(W(i, j) for (i, j) in ops)
    maxc = maximum(values(count); init = 0)

    nz(x, d) = d > 0 ? x / d : 0.0
    scored = map(ops) do (i, j)
        f1 = nz(sum(p[i, j:end]), maxjob)
        f2 = nz(load[inst.machine[i, j]], maxload)
        f3 = nz(W(i, j), maxW)
        f4 = maxc == 0 ? 1.0 : 1.0 - count[(i, j)] / maxc
        (op = (i, j), count = count[(i, j)], f1 = f1, f2 = f2, f3 = f3, f4 = f4,
         # explicit `*`: in Julia `0.30f3` would be the Float32 literal 300.0
         score = 0.30 * f3 + 0.20 * f1 + 0.20 * f2 + 0.30 * f4)
    end
    return sort!(scored; by = s -> (-s.score, s.op[1], s.op[2]))
end

"""
    policyA_release(inst, incumbent, count, k) -> Vector{Tuple{Int,Int}}

Default variant: release set R = top k operations by score; increments `count[o]` for o ∈ R.
"""
function policyA_release(inst::JSSPInstance, incumbent::Schedule,
                         count::Dict{Tuple{Int,Int},Int}, k::Int)
    R = [s.op for s in policyA_scores(inst, incumbent, count)[1:k]]
    foreach(o -> count[o] += 1, R)
    return R
end

"""
    policyA_release_fill(inst, incumbent, count, n_free_y) -> (R, free_y)

Goes through the operations in score order and adds o to R if the distinct free y
(pairs from `machine_pairs(inst)` with at least one operation in R) stay ≤ `n_free_y`;
otherwise o is skipped. Stops once exactly `n_free_y` y are free. Increments
`count[o]` for o ∈ R and returns R and the set of free pairs.
"""
function policyA_release_fill(inst::JSSPInstance, incumbent::Schedule,
                              count::Dict{Tuple{Int,Int},Int}, n_free_y::Int;
                              y_floor::Union{Nothing,Int} = nothing)
    pairs_of = Dict(op => Tuple{Tuple{Int,Int},Tuple{Int,Int}}[] for op in operations(inst))
    for (a, b) in machine_pairs(inst)                    # same pair representation as the loop
        push!(pairs_of[a], (a, b))
        push!(pairs_of[b], (a, b))
    end

    R = Tuple{Int,Int}[]
    free_y = Set{Tuple{Tuple{Int,Int},Tuple{Int,Int}}}()
    for s in policyA_scores(inst, incumbent, count)
        length(free_y) >= n_free_y && break             # budget reached
        new_pairs = [p for p in pairs_of[s.op] if !(p in free_y)]
        # would exceed: skip – unless (y_floor, default nothing = old behaviour) the set is
        # still below the range minimum, then the overshooting operation is freed anyway
        length(free_y) + length(new_pairs) <= n_free_y ||
            (y_floor !== nothing && length(free_y) < y_floor) || continue
        push!(R, s.op)
        union!(free_y, new_pairs)
    end
    foreach(o -> count[o] += 1, R)
    return R, free_y
end

"""
    policyA_selector(; fill_budget = false) -> select_free function for random_fao

Creates Policy A with fresh state (call once per run, so runs are independent):
`count` (releases per operation since the last improvement, all 0) and
`last_makespan` (incumbent makespan at the previous call, `nothing` at first).
The returned function has the `select_free` signature; it ignores `history` and
`rng` (fully deterministic) and frees all machine pairs with at least one operation
in R. `fill_budget = false`: R = top k (`policyA_release`); `true`: R filled up to
`n_free_y` free y (`policyA_release_fill`).
"""
function policyA_selector(; fill_budget::Bool = false, y_floor::Union{Nothing,Int} = nothing)
    count = Dict{Tuple{Int,Int},Int}()
    last_makespan = Ref{Union{Nothing,Int}}(nothing)
    return function policyA_free(inst, incumbent, history, rng, n_free_y)
        isempty(count) && foreach(o -> count[o] = 0, operations(inst))
        ms = makespan(inst, incumbent)                    # integral -> exact comparison
        if last_makespan[] !== nothing && ms < last_makespan[]   # new incumbent: reset
            foreach(o -> count[o] = 0, keys(count))
        end
        last_makespan[] = ms
        if fill_budget
            _, free_y = policyA_release_fill(inst, incumbent, count, n_free_y; y_floor)
            return collect(free_y)
        end
        R = Set(policyA_release(inst, incumbent, count, policyA_k(inst, n_free_y)))
        return [(a, b) for (a, b) in machine_pairs(inst) if a in R || b in R]
    end
end
