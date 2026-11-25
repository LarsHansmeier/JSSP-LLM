# =============================================================================
# dynamic_v3.jl — Policy C v3 (final): adaptive deterministic control
# =============================================================================
#
# Policy C v2 (dynamic_v2.jl) with two changes; everything else is identical (types and
# their orders via policyC2_*_order, cursors, free-y target with Δ_y and fill via
# policyC2_fill, size update, warm-up, score update with α, F, restart at maximum size
# without score reset, skip semantics, 1000-skip guard).
#
# A. Type switch (after F failures) among the other two types a, b:
#    if max(s_a, s_b) ≥ ε = 1e-4 → higher score (|s_a - s_b| < 1e-12 counts as a tie);
#    otherwise / on a tie → least recently used type, then CB > TW > MA.
#    The score is a smoothed relative improvement, so 1e-4 corresponds to 0.01 %; below
#    that a type has effectively stopped improving and its score carries no information
#    (v2 compared decayed values like 1e-12 vs 0 and starved TW).
# B. Duplicates per schedule: a released set is a duplicate iff the pair (fingerprint of
#    the current incumbent schedule, sorted released set) was solved before. The
#    fingerprint is the exact machine sequences (operation IDs per machine). The set of
#    pairs is never cleared, so after a sideways move (equal makespan, different
#    sequences – accepted by the loop) sets can be solved again around the new schedule,
#    and returning to an earlier schedule does not repeat its subproblems.
# =============================================================================

const POLICY_C3_EPS = 1e-4                 # ε: minimum meaningful score (0.01 % improvement)

const Fingerprint = Vector{Vector{Int}}

"Schedule fingerprint: machine sequences as operation IDs ((j-1)·m + k), exact and comparable."
schedule_fingerprint(inst::JSSPInstance, s::Schedule) =
    [[(j - 1) * inst.m + k for (j, k) in seq] for seq in machine_sequences(inst, s)]

"""
    PolicyCv3State(; n_free_min = 200, n_free_max = 800)

Per-run state of Policy C v3 (create one per run, e.g. via `policyC_v3_selector()`). As
`PolicyCv2State`, but duplicates are keyed by (schedule fingerprint, released set) and
never cleared; schedules get ids in order of first visit.
"""
mutable struct PolicyCv3State
    n_free_min::Int
    n_free_max::Int
    y_min::Int
    y_max::Int
    delta_y::Int
    tau::Symbol
    y_target::Int
    fail_streak::Int
    u::Int
    fails_at_max::Int
    scores::Dict{Symbol,Float64}
    cursor::Dict{Symbol,Int}
    last_used::Dict{Symbol,Int}
    solved::Set{Tuple{Fingerprint,Vector{Op}}}   # (fingerprint, released set) solved so far
    schedule_ids::Dict{Fingerprint,Int}         # fingerprint => id (order of first visit)
    last_fp::Union{Nothing,Fingerprint}
    n_sideways::Int
    consec_skips::Int
    last_Z::Int
    last_n_released::Int
    log::Vector{NamedTuple}
end
PolicyCv3State(; n_free_min::Int = 200, n_free_max::Int = 800) =
    PolicyCv3State(n_free_min, n_free_max, 0, 0, 0, :CB, 0, 0, 0, 0,
                   Dict(:CB => 0.0, :TW => 0.0, :MA => 0.0), Dict(:CB => 1, :TW => 1, :MA => 1),
                   Dict(:CB => 0, :TW => 0, :MA => 0), Set{Tuple{Fingerprint,Vector{Op}}}(),
                   Dict{Fingerprint,Int}(), nothing, 0, 0, 0, 0, NamedTuple[])

"Policy C v3 for `random_fao(...; select_free = policyC_v3_selector())` – fresh state per run."
policyC_v3_selector(; n_free_min::Int = 200, n_free_max::Int = 800) =
    PolicyCv3State(; n_free_min, n_free_max)

"""
    _policyC3_switch(tau, scores, last_used) -> (type, mode)

Change A: among the two types other than `tau`, the higher score if the scores are
meaningful (max ≥ ε) and not tied (≥ 1e-12 apart) – mode "score"; otherwise the least
recently used – "lru"; if that is tied too, CB > TW > MA – "priority".
"""
function _policyC3_switch(tau::Symbol, scores, last_used)
    a, b = [x for x in POLICY_C_PRIORITY if x != tau]
    if max(scores[a], scores[b]) >= POLICY_C3_EPS && abs(scores[a] - scores[b]) >= 1e-12
        return (scores[a] > scores[b] ? a : b), "score"
    end
    last_used[a] != last_used[b] && return (last_used[a] < last_used[b] ? a : b), "lru"
    return a, "priority"
end

function (st::PolicyCv3State)(inst::JSSPInstance, incumbent::Schedule, history, rng,
                              n_free_y::Int)
    t = length(history) + 1
    Z = makespan(inst, incumbent)
    pairs = machine_pairs(inst)
    fp = schedule_fingerprint(inst, incumbent)
    sid = get!(st.schedule_ids, fp, length(st.schedule_ids) + 1)
    if t == 1                                           # per-instance constants (as v2)
        cap = floor(Int, FREE_CAP * length(pairs))
        st.y_min = min(st.n_free_min, cap)
        st.y_max = max(st.y_min, min(st.n_free_max, cap))
        st.delta_y = max(1, round(Int, 0.1 * (st.y_max - st.y_min)))
        st.y_target = st.y_min
    end
    sideways = st.last_fp !== nothing && Z == st.last_Z && fp != st.last_fp
    st.n_sideways += sideways

    restart = false
    switch_mode = missing
    if t >= 2                                           # result of the previous iteration
        h = history[end]
        skipped = h.sub_status == "SKIPPED_DUPLICATE"
        improved = Z < st.last_Z
        if !skipped                                     # no score update for skipped iterations
            r = improved ? (st.last_Z - Z) / st.last_Z : 0.0
            st.scores[st.tau] = (1 - POLICY_C_ALPHA) * st.scores[st.tau] + POLICY_C_ALPHA * r
        end
        st.u = improved ? 0 : st.u + 1
        if t - 1 >= length(POLICY_C_WARMUP)             # decision rules after iteration ≥ 3
            solved = skipped || h.sub_status == "OPTIMAL"
            st.fails_at_max = (improved || st.y_target < st.y_max) ? 0 : st.fails_at_max + 1
            # 1. size (in free y)
            improved || (st.y_target = solved ? min(st.y_max, st.y_target + st.delta_y) :
                                                max(st.y_min, st.y_target - st.delta_y))
            # 2. type (Change A)
            old_tau = st.tau
            if improved
                st.fail_streak = 0
            else
                st.fail_streak += 1
                if st.fail_streak >= POLICY_C_F
                    st.tau, switch_mode = _policyC3_switch(st.tau, st.scores, st.last_used)
                    st.fail_streak = 0
                end
            end
            # 3. restart only after F failures at maximum size (scores, cursors kept)
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
    st.last_fp = fp
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

    skip = (fp, R) in st.solved                         # Change B: duplicate per schedule
    stop = skip && st.consec_skips >= POLICY_C2_MAX_SKIPS
    st.consec_skips = skip ? st.consec_skips + 1 : 0
    push!(st.log, (t = t, type = string(st.tau), y_target = st.y_target, n_released = length(R),
                   n_free_y = n_free, Z_inc = Z, s_CB = st.scores[:CB], s_TW = st.scores[:TW],
                   s_MA = st.scores[:MA], restart = restart, fail_streak = st.fail_streak, u = st.u,
                   cursor = st.cursor[st.tau], n_blocks = nb, y_min = st.y_min, y_max = st.y_max,
                   skipped = skip && !stop, fails_at_max = st.fails_at_max,
                   last_used_CB = st.last_used[:CB], last_used_TW = st.last_used[:TW],
                   last_used_MA = st.last_used[:MA], stop_reason = stop ? "exhausted" : missing,
                   schedule_id = sid, sideways = sideways, switch_mode = switch_mode,
                   released = join(["$(o[1])-$(o[2])" for o in R], " ")))
    stop && return STOP_SEARCH
    skip && return SKIP_ITERATION

    @assert !((fp, R) in st.solved) "Policy C v3: (schedule, released set) would be solved twice"
    @assert n_free <= st.y_max && (n_free >= st.y_min || exhausted) "Policy C v3: free y $n_free outside [$(st.y_min), $(st.y_max)]"
    push!(st.solved, (fp, R))
    Rset = Set(R)
    return [(a, b) for (a, b) in pairs if a in Rset || b in Rset]
end

"""
    policyC_v3_log(st, iterations) -> Vector of NamedTuples

Policy C v3 log joined with the loop's iteration log (as `policyC_v2_log`). `sideways`
refers to the iteration of the row: true if its result moved the incumbent to another
schedule with equal makespan (detected at the next call, hence shifted by one row).
"""
function policyC_v3_log(st::PolicyCv3State, iterations)
    it = Dict(r.iteration => r for r in iterations)
    arrived_sideways = Dict(row.t => row.sideways for row in st.log)
    return map(st.log) do row
        r = get(it, row.t, nothing)
        merge(row, (sideways = get(arrived_sideways, row.t + 1, missing),
                    time = r === nothing ? missing : r.time,
                    sub_status = r === nothing ? missing : r.sub_status,
                    sub_gap = r === nothing ? missing : r.sub_gap,
                    sub_makespan = r === nothing ? missing : r.sub_makespan,
                    improved = r === nothing ? missing : r.improved))
    end
end
