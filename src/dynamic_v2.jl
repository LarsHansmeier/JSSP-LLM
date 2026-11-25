# =============================================================================
# dynamic_v2.jl — Policy C v2: adaptive deterministic control (revised)
# =============================================================================
#
# Same interface as the other policies (`select_free`); reuses from Policy C v1
# (dynamic.jl) the constants α, F, the warm-up CB → TW → MA, the type priority and
# the critical blocks (`policyC_blocks`), and from Policy B the structural features
# and the free-y counting (`_pairs_of`). Released y: rule "any" as A/B/C v1.
#
# Changes to v1:
# 1. Size is a free-y target y_target ∈ [y_min, y_max] (the shared budget range
#    n_free_min:n_free_max, capped at FREE_CAP of all machine pairs as for A/B), start
#    y_min, step Δ_y = max(1, round(0.1 (y_max - y_min))). Each type yields an ordered
#    operation list from its cursor; operations are added in that order until the free
#    y reach y_target; an operation that would push them above y_max is skipped and
#    the next one is tried (overshoot rule of Policy A's fill).
#      CB: critical blocks from c_CB (cyclic), then machine predecessors/successors of
#          the listed operations (increasing start time), then all remaining operations
#          by start time, cyclically from the start of block c_CB.
#      TW: operations by start time from p_TW (cyclic); on failure p_TW += ceil(|R|/2).
#      MA: machines by load (descending) from c_MA (cyclic), operations by start time.
# 2. Duplicate subproblems are skipped: released sets already solved for the current
#    incumbent (cleared whenever the incumbent improves) are not solved again – the
#    policy returns SKIP_ITERATION, which counts as "not improved, solved" (cursor,
#    size, fail_streak, u, fails_at_max) without a score update. After 1000
#    consecutive skips the policy returns STOP_SEARCH (stop_reason "exhausted").
# 3. No fixed restart and no score reset: fails_at_max counts consecutive failures
#    with y_target == y_max; at ≥ F, y_target = y_min (scores, cursors, sets kept).
# 4. Type switch tie-break (|Δscore| < 1e-12): least recently used type, then CB > TW > MA.
# =============================================================================

const POLICY_C2_MAX_SKIPS = 1000          # consecutive skips before STOP_SEARCH

"""
    PolicyCv2State(; n_free_min = 200, n_free_max = 800)

Per-run state of Policy C v2 (create one per run, e.g. via `policyC_v2_selector()`); the
range must be the one passed to `random_fao`. Callable with the `select_free` signature;
ignores `rng` and the loop's per-iteration budget n.
"""
mutable struct PolicyCv2State
    n_free_min::Int
    n_free_max::Int
    y_min::Int
    y_max::Int
    delta_y::Int
    tau::Symbol
    y_target::Int
    fail_streak::Int
    u::Int                               # iterations since the last improvement (incl. skips)
    fails_at_max::Int
    scores::Dict{Symbol,Float64}
    cursor::Dict{Symbol,Int}             # c_CB, p_TW, c_MA (1-based, cyclic)
    last_used::Dict{Symbol,Int}          # last iteration in which a type was used
    solved_sets::Set{Vector{Op}}         # released sets solved for the current incumbent
    consec_skips::Int
    last_Z::Int
    last_n_released::Int
    log::Vector{NamedTuple}
end
PolicyCv2State(; n_free_min::Int = 200, n_free_max::Int = 800) =
    PolicyCv2State(n_free_min, n_free_max, 0, 0, 0, :CB, 0, 0, 0, 0,
                   Dict(:CB => 0.0, :TW => 0.0, :MA => 0.0), Dict(:CB => 1, :TW => 1, :MA => 1),
                   Dict(:CB => 0, :TW => 0, :MA => 0), Set{Vector{Op}}(), 0, 0, 0, NamedTuple[])

"Policy C v2 for `random_fao(...; select_free = policyC_v2_selector())` – fresh state per run."
policyC_v2_selector(; n_free_min::Int = 200, n_free_max::Int = 800) =
    PolicyCv2State(; n_free_min, n_free_max)

_rotate(v, i) = isempty(v) ? v : vcat(v[mod1(i, length(v)):end], v[1:mod1(i, length(v))-1])

"CB order: blocks from `cursor`, their machine neighbours, then the rest by start time
(cyclically from the start of block `cursor`). Returns (order, n_blocks)."
function policyC2_cb_order(inst::JSSPInstance, S::Matrix{Int}, F::StructuralFeatures, cursor::Int)
    by_S = sort(operations(inst); by = o -> (S[o...], o))
    blocks = policyC_blocks(F, S)
    nb = length(blocks)
    nb == 0 && return by_S, 0
    c = mod1(cursor, nb)
    order = reduce(vcat, (blocks[mod1(c + j, nb)] for j in 0:nb-1))
    listed = Set(order)
    pos = Dict(o => (m, i) for (m, seq) in enumerate(F.seqs) for (i, o) in enumerate(seq))
    for o in sort(copy(order); by = x -> (S[x...], x))
        m, i = pos[o]
        for j in (i - 1, i + 1)
            1 <= j <= length(F.seqs[m]) || continue
            nbr = F.seqs[m][j]
            nbr in listed || (push!(order, nbr); push!(listed, nbr))
        end
    end
    t0 = S[blocks[c][1]...]
    rest = [o for o in by_S if !(o in listed)]
    first_late = something(findfirst(o -> S[o...] >= t0, rest), 1)
    append!(order, _rotate(rest, first_late))
    return order, nb
end

"TW order: operations by start time (ties: index) from position `cursor`, cyclic."
policyC2_tw_order(inst::JSSPInstance, S::Matrix{Int}, cursor::Int) =
    _rotate(sort(operations(inst); by = o -> (S[o...], o)), cursor)

"MA order: machines by load (descending; ties: index) from `cursor`, cyclic; operations
of a machine by start time."
function policyC2_ma_order(inst::JSSPInstance, S::Matrix{Int}, cursor::Int)
    load(m) = sum(inst.ptime[o...] for o in ops_on_machine(inst, m); init = 0)
    machines = _rotate(sort(1:inst.m; by = m -> (-load(m), m)), cursor)
    return reduce(vcat, (sort(ops_on_machine(inst, m); by = o -> (S[o...], o)) for m in machines))
end

"""
    policyC2_fill(order, pairs_of, y_target, y_max) -> (R, n_free_y, exhausted)

Adds operations in `order` until the free y (pairs with ≥ 1 released operation) reach
`y_target`; an operation that would push them above `y_max` is skipped. `exhausted` =
the list ended before `y_target` was reached.
"""
function policyC2_fill(order::Vector{Op}, pairs_of, y_target::Int, y_max::Int)
    R = Op[]
    free = Set{Tuple{Op,Op}}()
    for o in order
        length(free) >= y_target && return R, length(free), false
        new = count(p -> !(p in free), pairs_of[o])
        length(free) + new > y_max && continue                 # would overshoot: skip
        push!(R, o)
        union!(free, pairs_of[o])
    end
    return R, length(free), length(free) < y_target
end

"Better-scoring of the other two types; ties (< 1e-12): least recently used, then CB > TW > MA."
function _policyC2_switch(tau::Symbol, scores, last_used)
    a, b = [x for x in POLICY_C_PRIORITY if x != tau]
    abs(scores[a] - scores[b]) >= 1e-12 && return scores[a] > scores[b] ? a : b
    last_used[a] != last_used[b] && return last_used[a] < last_used[b] ? a : b
    return a
end

function (st::PolicyCv2State)(inst::JSSPInstance, incumbent::Schedule, history, rng,
                              n_free_y::Int)
    t = length(history) + 1
    Z = makespan(inst, incumbent)
    pairs = machine_pairs(inst)
    if t == 1                                           # per-instance constants (as A/B: cap)
        cap = floor(Int, FREE_CAP * length(pairs))
        st.y_min = min(st.n_free_min, cap)
        st.y_max = max(st.y_min, min(st.n_free_max, cap))
        st.delta_y = max(1, round(Int, 0.1 * (st.y_max - st.y_min)))
        st.y_target = st.y_min
    end

    restart = false
    if t >= 2                                           # result of the previous iteration
        h = history[end]
        skipped = h.sub_status == "SKIPPED_DUPLICATE"
        improved = Z < st.last_Z
        if !skipped                                     # no score update for skipped iterations
            r = improved ? (st.last_Z - Z) / st.last_Z : 0.0
            st.scores[st.tau] = (1 - POLICY_C_ALPHA) * st.scores[st.tau] + POLICY_C_ALPHA * r
        end
        improved && empty!(st.solved_sets)              # improved incumbent: nothing solved for it yet
        st.u = improved ? 0 : st.u + 1
        if t - 1 >= length(POLICY_C_WARMUP)             # decision rules after iteration ≥ 3
            solved = skipped || h.sub_status == "OPTIMAL"
            st.fails_at_max = (improved || st.y_target < st.y_max) ? 0 : st.fails_at_max + 1
            # 1. size (in free y)
            improved || (st.y_target = solved ? min(st.y_max, st.y_target + st.delta_y) :
                                                max(st.y_min, st.y_target - st.delta_y))
            # 2. type
            old_tau = st.tau
            if improved
                st.fail_streak = 0
            else
                st.fail_streak += 1
                if st.fail_streak >= POLICY_C_F
                    st.tau = _policyC2_switch(st.tau, st.scores, st.last_used)
                    st.fail_streak = 0
                end
            end
            # 3. restart only after F failures at maximum size (scores, cursors, sets kept)
            if st.fails_at_max >= POLICY_C_F
                st.y_target = st.y_min
                st.fails_at_max = 0
                restart = true
            end
            # 4. cursor of an unchanged type after a failure
            if st.tau == old_tau && !improved
                st.cursor[st.tau] += st.tau == :TW ? cld(st.last_n_released, 2) : 1
            end
        end
    end
    if t <= length(POLICY_C_WARMUP)                     # warm-up: CB, TW, MA at y_min
        st.tau = POLICY_C_WARMUP[t]
        st.y_target = st.y_min
    end
    st.last_Z = Z
    st.last_used[st.tau] = t

    S = incumbent.start
    nb = missing
    if st.tau == :CB
        order, nb = policyC2_cb_order(inst, S, compute_structural_features(inst, incumbent), st.cursor[:CB])
    elseif st.tau == :TW
        order = policyC2_tw_order(inst, S, st.cursor[:TW])
    else
        order = policyC2_ma_order(inst, S, st.cursor[:MA])
    end
    R, n_free, exhausted = policyC2_fill(order, _pairs_of(inst), st.y_target, st.y_max)
    R = sort(R)
    st.last_n_released = length(R)

    skip = R in st.solved_sets
    stop = skip && st.consec_skips >= POLICY_C2_MAX_SKIPS
    st.consec_skips = skip ? st.consec_skips + 1 : 0
    push!(st.log, (t = t, type = string(st.tau), y_target = st.y_target, n_released = length(R),
                   n_free_y = n_free, Z_inc = Z, s_CB = st.scores[:CB], s_TW = st.scores[:TW],
                   s_MA = st.scores[:MA], restart = restart, fail_streak = st.fail_streak, u = st.u,
                   cursor = st.cursor[st.tau], n_blocks = nb, y_min = st.y_min, y_max = st.y_max,
                   skipped = skip && !stop, fails_at_max = st.fails_at_max,
                   last_used_CB = st.last_used[:CB], last_used_TW = st.last_used[:TW],
                   last_used_MA = st.last_used[:MA], stop_reason = stop ? "exhausted" : missing,
                   released = join(["$(o[1])-$(o[2])" for o in R], " ")))
    stop && return STOP_SEARCH
    skip && return SKIP_ITERATION

    @assert !(R in st.solved_sets) "Policy C v2: duplicate subproblem would be solved"
    @assert n_free <= st.y_max && (n_free >= st.y_min || exhausted) "Policy C v2: free y $n_free outside [$(st.y_min), $(st.y_max)]"
    push!(st.solved_sets, R)
    Rset = Set(R)
    return [(a, b) for (a, b) in pairs if a in Rset || b in Rset]
end

"""
    policyC_v2_log(st, iterations) -> Vector of NamedTuples

Policy C v2 log joined with the loop's iteration log: elapsed time, Gurobi status
(`SKIPPED_DUPLICATE` for skipped iterations), subproblem gap and makespan, `improved`.
"""
function policyC_v2_log(st::PolicyCv2State, iterations)
    it = Dict(r.iteration => r for r in iterations)
    return map(st.log) do row
        r = get(it, row.t, nothing)
        merge(row, (time = r === nothing ? missing : r.time,
                    sub_status = r === nothing ? missing : r.sub_status,
                    sub_gap = r === nothing ? missing : r.sub_gap,
                    sub_makespan = r === nothing ? missing : r.sub_makespan,
                    improved = r === nothing ? missing : r.improved))
    end
end
