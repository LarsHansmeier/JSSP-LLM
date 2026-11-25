# =============================================================================
# structure_based.jl — Policy B: structural neighbourhoods (deterministic)
# =============================================================================
#
# Plugs into the existing Fix-and-Optimize loop (random_fao) as `select_free`;
# start solution, model, fixing, warm start, time limits, acceptance and logging of
# the loop are unchanged. Only the choice of released operations differs.
#
# Release size: the loop passes the shared per-iteration y budget n (drawn from
# n_free_min:n_free_max, capped at FREE_CAP of all machine pairs). As in Policy A it
# is converted into an operation count k = policyA_k(inst, n) = max(1, floor(n / (n_jobs - 1))).
# Policy B releases exactly k distinct operations (all if the instance has fewer).
# Released y (rule "any", as Policy A): y[(a,b)] is free if a ∈ R or b ∈ R.
# Variant policyB_selector(fill_budget = true) ("PolicyB_StructureBased_fill"): k is
# chosen per candidate – starting from policyA_k (which always fits, since every
# operation has n_jobs - 1 pairs) it is increased as long as the candidate's release
# set (same rules, fill/trim) frees at most n y; stops at the first k that exceeds n.
#
# Features from the incumbent (every iteration): left-shifted start times r (heads),
# tails q (longest path from the end of o to the sink in the disjunctive graph of the
# incumbent), Cmax, slack = Cmax - (r + p + q); critical ⇔ slack == 0 (integers);
# critical blocks = maximal runs (≥ 2) of consecutive critical operations on a machine.
#
# Neighbourhood types in round-robin order CB → M → J → TW, type index =
# mod(iter, 4) + 1 with iter = 0, 1, 2, … (first iteration uses CB). Each type ranks
# its candidates by score (descending, ties: lowest index) and builds a release set,
# which is filled / trimmed to exactly k by slack and sorted (canonical form).
#
# Variant PolicyB_v2 (policyB_v2_selector, "PolicyB_StructureBased_v2"): as the fill
# variant, but CB is replaced by BB (block borders of all critical blocks, see
# bb_candidates) and the round-robin by the fixed cycle BB → M → BB → TW → BB → J.
# Variant PolicyB_v3 (policyB_v3_selector, "PolicyB_StructureBased_v3"): as v2, only the
# BB candidates differ – candidate 0 = all borders (v2), candidate i = borders of block i
# expanded along its machine (single-block focus), so BB has B + 1 distinct sets.
#
# Tabu: released sets that gave no improvement since the last improvement. The first
# non-tabu candidate of the current type is used; if all are tabu, the next types
# are tried; if everything is tabu, the k operations with the smallest slack.
# Tabu update (at the next call, from `history[end].improved`): improvement → clear,
# otherwise → add the previous release set.
# =============================================================================

const Op = Tuple{Int,Int}
const POLICY_B_TYPES = (:CB, :M, :J, :TW)
# PolicyB_v2: fixed weighted rotation, block borders (BB) instead of CB, every second step BB
const POLICY_B_V2_CYCLE = (:BB, :M, :BB, :TW, :BB, :J)

"Structural features of an incumbent (matrices indexed [job, position] like `Schedule`)."
struct StructuralFeatures
    r::Matrix{Int}                 # left-shifted start times (heads)
    q::Matrix{Int}                 # tails: longest path from the end of o to the sink
    slack::Matrix{Int}             # Cmax - (r + p + q)
    critical::Matrix{Bool}         # slack == 0
    blocks::Vector{Vector{Op}}     # critical blocks (≥ 2 consecutive critical ops on a machine)
    Cmax::Int
    seqs::Vector{Vector{Op}}       # machine sequences of the incumbent
    routes::Vector{Vector{Op}}     # job routes
end

"Job and machine successors of operation `o` in the disjunctive graph of `seqs`."
function _successors(inst::JSSPInstance, msucc::Dict{Op,Op}, o::Op)
    j, k = o
    s = Op[]
    k < inst.m && push!(s, (j, k + 1))
    haskey(msucc, o) && push!(s, msucc[o])
    return s
end

"""
    compute_structural_features(inst, incumbent) -> StructuralFeatures

Left-shifts the incumbent (`schedule_from_sequences` on its machine sequences), computes
tails in reverse topological order of the disjunctive graph, slack, critical operations
and critical blocks.
"""
function compute_structural_features(inst::JSSPInstance, incumbent::Schedule)
    p = inst.ptime
    ops = operations(inst)
    seqs = machine_sequences(inst, incumbent)
    r = schedule_from_sequences(inst, seqs).start

    msucc = Dict{Op,Op}()
    for seq in seqs, t in 1:length(seq)-1
        msucc[seq[t]] = seq[t+1]
    end
    # topological order (Kahn, lowest index first)
    indeg = Dict(o => 0 for o in ops)
    for o in ops, s in _successors(inst, msucc, o)
        indeg[s] += 1
    end
    queue = [o for o in ops if indeg[o] == 0]
    order = Op[]
    while !isempty(queue)
        o = popfirst!(queue)
        push!(order, o)
        for s in _successors(inst, msucc, o)
            indeg[s] -= 1
            indeg[s] == 0 && push!(queue, s)
        end
    end
    length(order) == length(ops) || error("disjunctive graph of the incumbent is cyclic")

    q = zeros(Int, inst.n, inst.m)
    for o in Iterators.reverse(order)
        q[o...] = maximum((p[s...] + q[s...] for s in _successors(inst, msucc, o)); init = 0)
    end
    Cmax = maximum(r[o...] + p[o...] for o in ops)
    slack = [Cmax - (r[j, k] + p[j, k] + q[j, k]) for j in 1:inst.n, k in 1:inst.m]
    critical = slack .== 0

    blocks = Vector{Op}[]
    for seq in seqs
        run = Op[]
        for o in seq
            if critical[o...]
                push!(run, o)
            else
                length(run) >= 2 && push!(blocks, run)
                run = Op[]
            end
        end
        length(run) >= 2 && push!(blocks, run)
    end
    routes = [[(j, k) for k in 1:inst.m] for j in 1:inst.n]
    return StructuralFeatures(r, q, slack, critical, blocks, Cmax, seqs, routes)
end

"""
Fills `R` to exactly `k` operations by ascending slack (ties: lowest index) or trims it
to the `k` with the smallest slack; returns the set sorted (canonical form).
"""
function _fill_trim(R::Vector{Op}, k::Int, F::StructuralFeatures, slack_order::Vector{Op})
    R = unique(R)
    if length(R) < k
        inR = Set(R)
        for o in slack_order
            length(R) >= k && break
            o in inR || push!(R, o)
        end
    elseif length(R) > k
        R = sort(R; by = o -> (F.slack[o...], o))[1:k]
    end
    return sort(R)
end

"Contiguous window of length `len` in `seq` with minimal total slack (ties: earliest)."
function _min_slack_window(seq::Vector{Op}, len::Int, F::StructuralFeatures)
    tot(w) = sum(F.slack[o...] for o in seq[w:w+len-1])
    w = argmin(w -> (tot(w), w), 1:length(seq)-len+1)
    return seq[w:w+len-1]
end

"""
Operations of `seq` alternately outward from the block at positions `s:e`: one before,
one after, two before, two after, … (CB expansion; also used by BB in PolicyB_v3).
"""
function _expansion_order(seq::Vector{Op}, s::Int, e::Int)
    order = Op[]
    d = 1
    while s - d >= 1 || e + d <= length(seq)
        s - d >= 1 && push!(order, seq[s-d])
        e + d <= length(seq) && push!(order, seq[e+d])
        d += 1
    end
    return order
end

"""
    cb_candidates(inst, F, k) -> ranked Vector of (id, build)

CB (critical block): score of a block = Σ p over its operations (long blocks first). The
set is the block, expanded alternately one before / one after, two before / two after, …
along its machine sequence until k operations or the sequence is exhausted.
"""
function cb_candidates(inst::JSSPInstance, F::StructuralFeatures, k0::Int)
    p = inst.ptime
    score = [sum(p[o...] for o in b) for b in F.blocks]
    ranking = sort(1:length(F.blocks); by = i -> (-score[i], i))
    return map(ranking) do i
        b = F.blocks[i]
        m = inst.machine[b[1]...]
        build = function (k)
            seq = F.seqs[m]
            s = findfirst(==(b[1]), seq)
            e = s + length(b) - 1
            R = copy(b)
            for o in _expansion_order(seq, s, e)
                length(R) >= k && break
                push!(R, o)
            end
            R
        end
        ("block $i (M$m, $(length(b)) ops)", build)
    end
end

"""
    m_candidates(inst, F, k) -> ranked Vector of (id, build)

M (machine): score of machine m = Σ p of its critical operations + 0.01 · Σ p of all its
operations (critical load first, total load as tie-breaker). The set is the window of k
consecutive operations on m with minimal total slack; if m has ≤ k operations, all of them
are taken and the next-ranked machines fill the remaining capacity with the same rule.
"""
function m_candidates(inst::JSSPInstance, F::StructuralFeatures, k0::Int)
    p = inst.ptime
    score(m) = sum((p[o...] for o in F.seqs[m] if F.critical[o...]); init = 0) +
               0.01 * sum(p[o...] for o in F.seqs[m])
    ranking = sort(1:inst.m; by = m -> (-score(m), m))
    return map(1:length(ranking)) do c
        build = function (k)
            R = Op[]
            for m in ranking[c:end]
                cap = k - length(R)
                cap == 0 && break
                seq = F.seqs[m]
                append!(R, length(seq) <= cap ? seq : _min_slack_window(seq, cap, F))
            end
            R
        end
        ("machine $(ranking[c])", build)
    end
end

"""
    j_candidates(inst, F, k) -> ranked Vector of (id, build)

J (job): score of job j = waiting time W_j = C_j − r_first − Σ p (time the job spends
waiting between its operations). The set is all operations of the top-ranked job, then the
next one, …; the last job is truncated to its latest operations so that exactly k are released.
"""
function j_candidates(inst::JSSPInstance, F::StructuralFeatures, k0::Int)
    p = inst.ptime
    W(j) = F.r[j, inst.m] + p[j, inst.m] - F.r[j, 1] - sum(p[j, :])
    ranking = sort(1:inst.n; by = j -> (-W(j), j))
    return map(1:length(ranking)) do c
        build = function (k)
            R = Op[]
            for j in ranking[c:end]
                cap = k - length(R)
                cap == 0 && break
                route = F.routes[j]
                append!(R, length(route) <= cap ? route : route[end-cap+1:end])
            end
            R
        end
        ("job $(ranking[c])", build)
    end
end

"""
    tw_candidates(inst, F, k) -> ranked Vector of (id, build)

TW (time window): operations sorted by left-shifted start r (ties: lowest index); a
candidate is each window of k consecutive operations in that order, scored by Σ p of its
critical operations (time windows containing much of the critical path first).
Candidates are `(id, build)`, `build(k)` returning the window from the same start with length k.
"""
function tw_candidates(inst::JSSPInstance, F::StructuralFeatures, k0::Int)
    p = inst.ptime
    by_r = sort(operations(inst); by = o -> (F.r[o...], o))
    nwin = max(1, length(by_r) - k0 + 1)
    win(w, k) = by_r[w:min(w + k - 1, length(by_r))]
    score(w) = sum((p[o...] for o in win(w, k0) if F.critical[o...]); init = 0)
    ranking = sort(1:nwin; by = w -> (-score(w), w))      # windows ranked at length k0
    return [("window $w (r = $(F.r[by_r[w]...]))", k -> win(w, k)) for w in ranking]
end

"Border operations of a critical block: all if length ≤ 3, else the first two and last two."
_border_ops(b::Vector{Op}) = length(b) <= 3 ? copy(b) : [b[1], b[2], b[end-1], b[end]]

"""
Border operations of the blocks in cyclic order starting at `ranked[i]`; a block whose
borders would push the free y above `n_free_y` is skipped (no check if `nothing`).
"""
function _bb_border_list(ranked::Vector{Vector{Op}}, i::Int, n_free_y::Union{Nothing,Int},
                         pairs_of)
    R = Op[]
    free = Set{Tuple{Op,Op}}()
    B = length(ranked)
    for t in 0:B-1
        borders = _border_ops(ranked[mod(i - 1 + t, B) + 1])
        if n_free_y !== nothing
            new_free = union(free, (pairs_of[o] for o in borders)...)
            length(new_free) <= n_free_y || continue      # would exceed the budget: skip block
            free = new_free
        end
        append!(R, borders)
    end
    return R
end

"""
    bb_candidates(inst, F, k; n_free_y = nothing, pairs_of = nothing) -> ranked Vector of (id, build, n_border)

BB (block borders): by the N5 result for the JSSP (Nowicki & Smutnicki, 1996), swapping two
adjacent operations inside a critical block can only reduce the makespan if the swap involves
the first two or the last two operations of the block – interior swaps cannot improve Cmax.
BB therefore releases the borders (all operations if length ≤ 3, else first two and last
two) of all critical blocks at once. Blocks are ranked by Σ p (as CB; ties: lowest machine,
then earliest position); candidate i walks the blocks cyclically from the i-th ranked block,
skipping blocks whose borders would exceed the y budget, and is then filled by slack.

`single_focus = true` (PolicyB_v3): B + 1 candidates `(id, build, n_border, index)` –
index 0 = all borders in rank order (as v2's first candidate); index i = only the borders
of the i-th ranked block, expanded alternately outward along its machine sequence
(`_expansion_order`) until the y budget is reached, the next operation would exceed it or
the sequence is exhausted; all are then filled by slack.
"""
function bb_candidates(inst::JSSPInstance, F::StructuralFeatures, k0::Int;
                       n_free_y::Union{Nothing,Int} = nothing, pairs_of = nothing,
                       single_focus::Bool = false)
    p = inst.ptime
    score = [sum(p[o...] for o in b) for b in F.blocks]
    ranked = F.blocks[sort(1:length(F.blocks); by = i -> (-score[i], i))]
    B = length(ranked)
    single_focus || return map(1:B) do i
        borders = _bb_border_list(ranked, i, n_free_y, pairs_of)
        ("borders from block $i (of $B)", k -> copy(borders), length(borders))
    end
    B == 0 && return []
    all_b = _bb_border_list(ranked, 1, n_free_y, pairs_of)
    cands = Any[("all borders ($B blocks)", k -> copy(all_b), length(all_b), 0)]
    for i in 1:B
        b = ranked[i]
        m = inst.machine[b[1]...]
        seq = F.seqs[m]
        s = findfirst(==(b[1]), seq)
        R = _border_ops(b)
        n_border = length(R)
        free = union(Set{Tuple{Op,Op}}(), (pairs_of[o] for o in R)...)
        for o in _expansion_order(seq, s, s + length(b) - 1)
            n_free_y !== nothing && length(free) >= n_free_y && break     # target reached
            new_free = union(free, pairs_of[o])
            n_free_y !== nothing && length(new_free) > n_free_y && break  # would exceed
            push!(R, o)
            free = new_free
        end
        push!(cands, ("block $i borders + machine nbrs (M$m, $(length(b)) ops)",
                      k -> copy(R), n_border, i))
    end
    return cands
end

const _CANDIDATES = Dict(:CB => cb_candidates, :M => m_candidates,
                         :J => j_candidates, :TW => tw_candidates)

"Machine pairs per operation (from `machine_pairs(inst)`), for counting free y."
function _pairs_of(inst::JSSPInstance)
    pairs_of = Dict(o => Tuple{Op,Op}[] for o in operations(inst))
    for (a, b) in machine_pairs(inst)
        push!(pairs_of[a], (a, b))
        push!(pairs_of[b], (a, b))
    end
    return pairs_of
end

"Number of distinct machine pairs with at least one operation in `R` (free y under rule \"any\")."
_n_free_y(R, pairs_of) = length(union(Set{Tuple{Op,Op}}(), (pairs_of[o] for o in R)...))

"""
Release set of a candidate: exactly `k` operations, or – if `n_free_y` is given – the
largest k ≥ `k` (increased one at a time, stopping at the first k that exceeds the budget)
whose release set frees at most `n_free_y` y.
"""
function _release_set(build, k::Int, F::StructuralFeatures, slack_order::Vector{Op},
                      n_free_y::Union{Nothing,Int}, pairs_of; y_floor::Union{Nothing,Int} = nothing)
    R = _fill_trim(build(k), k, F, slack_order)
    n_free_y === nothing && return R
    while k < length(slack_order)
        R2 = _fill_trim(build(k + 1), k + 1, F, slack_order)
        if _n_free_y(R2, pairs_of) > n_free_y
            # y_floor (default nothing = old behaviour): while the set is still below the
            # range minimum, the operation that would overshoot n is freed anyway
            (y_floor !== nothing && _n_free_y(R, pairs_of) < y_floor) || break
        end
        R, k = R2, k + 1
    end
    return R
end

"""
    policyB_select(inst, F, k, type_index, tabu; n_free_y = nothing, cycle = POLICY_B_TYPES,
                   bb_focus = false) -> (type, candidate_id, R, n_border_ops, bb_candidate_index)

Tries the types of `cycle` in order starting at `cycle[type_index]` (cyclically, at most
`length(cycle)` steps) and returns the first release set (exactly k, sorted) that is not in
`tabu`; if all are tabu, the k operations with the smallest slack (type `:fallback`). With
`n_free_y`, k is the lower bound and every set is enlarged up to the y budget (see
`_release_set`). `n_border_ops` = border operations before fill (BB only, else `missing`);
`bb_focus` selects the PolicyB_v3 BB candidates (`bb_candidates(...; single_focus = true)`),
whose index is returned as `bb_candidate_index` (else `missing`). Within a type, a set
equal to an earlier candidate's set is skipped (it has the same tabu status anyway).
"""
function policyB_select(inst::JSSPInstance, F::StructuralFeatures, k::Int,
                        type_index::Int, tabu::Set{Vector{Op}};
                        n_free_y::Union{Nothing,Int} = nothing, cycle = POLICY_B_TYPES,
                        bb_focus::Bool = false, y_floor::Union{Nothing,Int} = nothing)
    slack_order = sort(operations(inst); by = o -> (F.slack[o...], o))
    pairs_of = n_free_y === nothing ? Dict(o => Tuple{Op,Op}[] for o in slack_order) : _pairs_of(inst)
    L = length(cycle)
    for d in 0:L-1
        typ = cycle[mod(type_index - 1 + d, L) + 1]
        cands = typ == :BB ? bb_candidates(inst, F, k; n_free_y, pairs_of, single_focus = bb_focus) :
                             _CANDIDATES[typ](inst, F, k)
        seen = Set{Vector{Op}}()
        for c in cands
            R = _release_set(c[2], k, F, slack_order, n_free_y, pairs_of; y_floor)
            R in seen && continue                     # duplicate of an earlier candidate
            push!(seen, R)
            R in tabu || return (typ, c[1], R, length(c) >= 3 ? c[3] : missing,
                                 length(c) >= 4 ? c[4] : missing)
        end
    end
    R = _release_set(kk -> slack_order[1:kk], k, F, slack_order, n_free_y, pairs_of; y_floor)
    return (:fallback, "smallest slack", R, missing, missing)
end

"""
    StructuralPolicyState()

Per-run state of Policy B (create one per run): tabu list, previous release set and a log
with one row per call (iteration, type, candidate, |R|, critical ops in R, k, free y, tabu
size). Callable with the `select_free` signature of `random_fao`; ignores `rng`.
"""
mutable struct StructuralPolicyState
    tabu::Set{Vector{Op}}
    last_R::Union{Nothing,Vector{Op}}
    log::Vector{NamedTuple}
    fill_budget::Bool              # false: exactly k = policyA_k; true: enlarge up to n free y
    cycle::Tuple                   # type rotation (POLICY_B_TYPES or POLICY_B_V2_CYCLE)
    bb_focus::Bool                 # PolicyB_v3 BB candidates (see bb_candidates)
    y_floor::Union{Nothing,Int}    # range minimum for _release_set (nothing = old behaviour)
end
StructuralPolicyState(; fill_budget::Bool = false, cycle::Tuple = POLICY_B_TYPES,
                      bb_focus::Bool = false, y_floor::Union{Nothing,Int} = nothing) =
    StructuralPolicyState(Set{Vector{Op}}(), nothing, NamedTuple[], fill_budget, cycle, bb_focus, y_floor)

"""
    policyB_selector(; fill_budget = false)

Policy B for `random_fao(...; select_free = policyB_selector())` – a fresh state per run.
`fill_budget = true`: each release set is enlarged until it frees (almost) the full y
budget n instead of exactly k = policyA_k operations.
"""
policyB_selector(; fill_budget::Bool = false, cycle::Tuple = POLICY_B_TYPES,
                 bb_focus::Bool = false, y_floor::Union{Nothing,Int} = nothing) =
    StructuralPolicyState(; fill_budget, cycle, bb_focus, y_floor)

"""
    policyB_v2_selector()

PolicyB_v2: the fill variant with block borders (BB) instead of CB and the fixed weighted
rotation BB → M → BB → TW → BB → J (static; only the tabu list depends on the history).
"""
policyB_v2_selector(; y_floor::Union{Nothing,Int} = nothing) =
    policyB_selector(fill_budget = true, cycle = POLICY_B_V2_CYCLE, y_floor = y_floor)

"""
    policyB_v3_selector()

PolicyB_v3: PolicyB_v2 with diverse BB candidates – all borders, then each block's borders
expanded along its machine (B + 1 distinct sets instead of one).
"""
policyB_v3_selector() = policyB_selector(fill_budget = true, cycle = POLICY_B_V2_CYCLE,
                                         bb_focus = true)

function (st::StructuralPolicyState)(inst::JSSPInstance, incumbent::Schedule, history, rng,
                                     n_free_y::Int)
    iter = length(history)                         # 0, 1, 2, … (first call: cycle[1])
    if !isempty(history) && st.last_R !== nothing  # result of the previous release set
        history[end].improved ? empty!(st.tabu) : push!(st.tabu, st.last_R)
    end
    k = min(policyA_k(inst, n_free_y), length(operations(inst)))
    F = compute_structural_features(inst, incumbent)
    typ, id, R, n_border, bb_idx = policyB_select(inst, F, k, mod(iter, length(st.cycle)) + 1,
                                                  st.tabu; n_free_y = st.fill_budget ? n_free_y : nothing,
                                                  cycle = st.cycle, bb_focus = st.bb_focus,
                                                  y_floor = st.y_floor)
    st.last_R = R
    Rset = Set(R)
    free = [(a, b) for (a, b) in machine_pairs(inst) if a in Rset || b in Rset]
    push!(st.log, (iteration = iter + 1, type = string(typ), candidate = id, k = k,
                   budget = n_free_y,
                   n_released = length(R), n_critical = count(o -> F.critical[o...], R),
                   n_blocks = length(F.blocks), n_free_y = length(free),
                   n_tabu = length(st.tabu), n_border_ops = n_border,
                   bb_candidate_index = bb_idx))
    return free
end

"""
    policyB_log(st, iterations) -> Vector of NamedTuples

Policy B log joined with the loop's iteration log: adds `improved` (new best solution in
that iteration; `missing` if the loop stopped before solving it).
"""
function policyB_log(st::StructuralPolicyState, iterations)
    imp = Dict(it.iteration => it.improved for it in iterations)
    return [merge(row, (improved = get(imp, row.iteration, missing),)) for row in st.log]
end
