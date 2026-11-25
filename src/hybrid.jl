# =============================================================================
# hybrid.jl — Policy D v1 (hybrid): structural neighbourhoods + adaptive size +
#             per-schedule duplicates + score-based fill + escape
# =============================================================================
#
# Plugs into the Fix-and-Optimize loop (random_fao) as `select_free`, like A/B/C.
# Reused as they are (no file of A/B/C changed): compute_structural_features,
# bb_candidates, m_candidates, j_candidates, tw_candidates, _pairs_of (Policy B,
# structure_based.jl), policyC2_fill (C v2), schedule_fingerprint / Fingerprint (C v3),
# policyA_k (A), random_free (random baseline). Released y: rule "any".
#
# Types: odd iterations BB (borders of all critical blocks ≥ 2, one candidate);
# even iterations τ_even ∈ {M, J, TW} (B's candidates; cursor per type, next rank after
# a failure, back to rank 1 when the incumbent schedule changes). Warm-up: iterations
# 1–6 (evens M, J, TW) at y_min. τ_even switches after F failed even slots by C v3's rule
# (score if ≥ ε, else least recently used, else M > J > TW).
#
# Released set: core operations of the candidate, then the other operations by descending
# fill score (revised Policy A score, ties: job, position), added with policyC2_fill (same
# free-y counting, skip an operation that would exceed y_max) until y_target is reached.
#   fill_score = 0.30 f_crit + 0.25 f_wait + 0.15 f_load + 0.30 f_freq
#   f_crit = 1 - slack/Cmax, f_wait = W/max W (waiting before o in its job), f_load =
#   L_machine/max L, f_freq = 1 - cnt/max cnt (1 if max cnt = 0); denominator 0 → 0.
# Size and duplicates as C v3: y_target ∈ [y_min, y_max], ±Δ_y, (fingerprint, R) pairs
# never solved twice (skip = "solved, not improved", no score update).
# Size cycle end (fails_at_max ≥ F): fingerprint unchanged since the cycle start →
# exhaustion → ESCAPE: blocks of E iterations released by random_free (policy-local RNG,
# y_max); leave on any schedule change. After 3 unsuccessful blocks and N ≤ 250
# operations → EXACT: full model for the remaining time (loop marker SOLVE_EXACT).
# Variants: :full ("PolicyD_Hybrid_v1"), :noExact (stay in ESCAPE), :noEscape (restart
# as C on exhaustion).
# =============================================================================

using Random: MersenneTwister

const POLICY_D_F = 2                       # failures before switching τ_even / ending a size cycle
const POLICY_D_ALPHA = 0.3                 # score smoothing
const POLICY_D_EPS = 1e-4                  # minimum meaningful score (0.01 % improvement)
const POLICY_D_W = (crit = 0.30, wait = 0.25, load = 0.15, freq = 0.30)   # fill-score weights
const POLICY_D_E = 20                      # iterations per escape block
const POLICY_D_EXACT_BLOCKS = 3            # escape blocks before EXACT
const POLICY_D_N_EXACT = 250               # max operations for EXACT
const POLICY_D_EVEN = (:M, :J, :TW)        # even-slot types, also the priority M > J > TW
const POLICY_D_WARMUP = 6                  # iterations 1–6 (BB, M, BB, J, BB, TW) at y_min
const POLICY_D_VARIANTS = (:full, :noExact, :noEscape)

"Δ_y: size step in free y."
policyD_delta(y_min::Int, y_max::Int) = max(1, round(Int, 0.1 * (y_max - y_min)))

"Level-2 gate: the full model is only used for instances with at most N_exact operations."
policyD_exact_allowed(inst::JSSPInstance) = length(operations(inst)) <= POLICY_D_N_EXACT

"""
    PolicyDState(; variant = :full, n_free_min = 200, n_free_max = 800, seed = 42)

Per-run state of Policy D (create one per run via `policyD_selector`). The range must be
the one passed to `random_fao`; `seed` seeds the policy-local escape RNG.
"""
mutable struct PolicyDState
    variant::Symbol
    n_free_min::Int
    n_free_max::Int
    y_min::Int
    y_max::Int
    delta_y::Int
    mode::Symbol                          # :STRUCT, :ESCAPE, :EXACT
    last_mode::Symbol                     # mode of the previous iteration
    tau::Symbol                           # type of the previous / current iteration
    tau_even::Symbol
    y_target::Int
    fail_streak::Int
    fails_at_max::Int
    scores::Dict{Symbol,Float64}          # s_BB (logged only), s_M, s_J, s_TW
    last_used::Dict{Symbol,Int}
    cursor::Dict{Symbol,Int}              # rank per type M, J, TW (1-based, cyclic)
    solved::Set{Tuple{Fingerprint,Vector{Op}}}                 # structural (fp, R)
    solved_escape::Set{Tuple{Fingerprint,Vector{Tuple{Op,Op}}}}  # escape (fp, free pairs)
    cnt::Dict{Op,Int}
    fp_at_cycle_start::Union{Nothing,Fingerprint}
    last_fp::Union{Nothing,Fingerprint}
    schedule_ids::Dict{Fingerprint,Int}
    n_sideways::Int
    escape_left::Int
    escape_blocks::Int
    n_exhaustions::Int
    n_escape_iters::Int
    rng::MersenneTwister                  # policy-local, only used in ESCAPE
    last_Z::Int
    log::Vector{NamedTuple}
end
function PolicyDState(; variant::Symbol = :full, n_free_min::Int = 200, n_free_max::Int = 800,
                      seed::Integer = DEFAULT_SEED)
    variant in POLICY_D_VARIANTS || error("unknown Policy D variant $variant")
    return PolicyDState(variant, n_free_min, n_free_max, 0, 0, 0, :STRUCT, :STRUCT, :BB, :M, 0, 0, 0,
                        Dict(:BB => 0.0, :M => 0.0, :J => 0.0, :TW => 0.0),
                        Dict(:BB => 0, :M => 0, :J => 0, :TW => 0), Dict(:M => 1, :J => 1, :TW => 1),
                        Set{Tuple{Fingerprint,Vector{Op}}}(), Set{Tuple{Fingerprint,Vector{Tuple{Op,Op}}}}(),
                        Dict{Op,Int}(), nothing, nothing, Dict{Fingerprint,Int}(), 0, 0, 0, 0, 0,
                        MersenneTwister(seed), 0, NamedTuple[])
end

"Policy D for `random_fao(...; select_free = policyD_selector())` – a fresh state per run."
policyD_selector(; variant::Symbol = :full, n_free_min::Int = 200, n_free_max::Int = 800,
                 seed::Integer = DEFAULT_SEED) = PolicyDState(; variant, n_free_min, n_free_max, seed)

"""
    policyD_fill_scores(inst, incumbent, F, cnt) -> Dict(op => (f_crit, f_wait, f_load, f_freq, score))

Revised Policy A score used as fill order: criticality (1 - slack/Cmax), waiting time
before the operation in its job (as in Policy A), machine load and release frequency.
"""
function policyD_fill_scores(inst::JSSPInstance, incumbent::Schedule, F::StructuralFeatures,
                             cnt::Dict{Op,Int})
    p, S = inst.ptime, incumbent.start
    ops = operations(inst)
    nz(x, d) = d > 0 ? x / d : 0.0
    W(i, j) = S[i, j] - (j == 1 ? 0 : S[i, j-1] + p[i, j-1])
    maxW = maximum(W(i, j) for (i, j) in ops)
    load = [sum(p[o...] for o in ops_on_machine(inst, m); init = 0) for m in 1:inst.m]
    maxL = maximum(load)
    maxc = maximum(values(cnt); init = 0)
    return Dict(map(ops) do (i, j)
        f_crit = F.Cmax > 0 ? 1 - F.slack[i, j] / F.Cmax : 0.0
        f_wait = nz(W(i, j), maxW)
        f_load = nz(load[inst.machine[i, j]], maxL)
        f_freq = maxc == 0 ? 1.0 : 1 - get(cnt, (i, j), 0) / maxc
        score = POLICY_D_W.crit * f_crit + POLICY_D_W.wait * f_wait +
                POLICY_D_W.load * f_load + POLICY_D_W.freq * f_freq
        (i, j) => (f_crit = f_crit, f_wait = f_wait, f_load = f_load, f_freq = f_freq, score = score)
    end)
end

"""
    policyD_switch(tau, scores, last_used) -> (type, mode)

Copy of `_policyC3_switch` (dynamic_v3.jl) with the priority M > J > TW instead of
CB > TW > MA: higher score if max ≥ ε and not tied, else least recently used, else priority.
"""
function policyD_switch(tau::Symbol, scores, last_used)
    a, b = [x for x in POLICY_D_EVEN if x != tau]
    if max(scores[a], scores[b]) >= POLICY_D_EPS && abs(scores[a] - scores[b]) >= 1e-12
        return (scores[a] > scores[b] ? a : b), "score"
    end
    last_used[a] != last_used[b] && return (last_used[a] < last_used[b] ? a : b), "lru"
    return a, "priority"
end

"Core operations and rank of the current candidate of type `tau` (B v2's candidate builders)."
function policyD_core(st::PolicyDState, inst::JSSPInstance, F::StructuralFeatures, tau::Symbol)
    tau == :BB && return (isempty(F.blocks) ? Op[] : bb_candidates(inst, F, 0)[1][2](0)), 1
    k = min(policyA_k(inst, st.y_target), length(operations(inst)))   # window length as in B
    cands = tau == :M ? m_candidates(inst, F, k) : tau == :J ? j_candidates(inst, F, k) :
                                                           tw_candidates(inst, F, k)
    c = mod1(st.cursor[tau], length(cands))
    len = tau == :M ? min(k, inst.n) : tau == :J ? inst.m : k
    return cands[c][2](len), c
end

function (st::PolicyDState)(inst::JSSPInstance, incumbent::Schedule, history, rng, n_free_y::Int)
    t = length(history) + 1
    Z = makespan(inst, incumbent)
    pairs = machine_pairs(inst)
    fp = schedule_fingerprint(inst, incumbent)
    sid = get!(st.schedule_ids, fp, length(st.schedule_ids) + 1)
    if t == 1                                           # per-instance constants (as C v3)
        cap = floor(Int, FREE_CAP * length(pairs))
        st.y_min = min(st.n_free_min, cap)
        st.y_max = max(st.y_min, min(st.n_free_max, cap))
        st.delta_y = policyD_delta(st.y_min, st.y_max)
        st.y_target = st.y_min
        foreach(o -> st.cnt[o] = 0, operations(inst))
        st.fp_at_cycle_start = fp
    end
    fp_changed = st.last_fp !== nothing && fp != st.last_fp
    sideways = fp_changed && Z == st.last_Z
    st.n_sideways += sideways

    switch_mode = missing
    exhausted_now = false
    if t >= 2                                           # result of the previous iteration
        h = history[end]
        skipped = h.sub_status == "SKIPPED_DUPLICATE"
        improved = Z < st.last_Z
        if st.last_mode == :STRUCT
            if !skipped                                 # score update (not for skips)
                r = improved ? (st.last_Z - Z) / st.last_Z : 0.0
                st.scores[st.tau] = (1 - POLICY_D_ALPHA) * st.scores[st.tau] + POLICY_D_ALPHA * r
            end
            st.tau in POLICY_D_EVEN && !improved && (st.cursor[st.tau] += 1)
            if iseven(t - 1) && t - 1 >= POLICY_D_WARMUP    # even slot after warm-up
                if improved
                    st.fail_streak = 0
                else
                    st.fail_streak += 1
                    if st.fail_streak >= POLICY_D_F
                        st.tau_even, switch_mode = policyD_switch(st.tau_even, st.scores, st.last_used)
                        st.fail_streak = 0
                    end
                end
            end
            if t - 1 >= POLICY_D_WARMUP                 # size control after warm-up
                solved = skipped || h.sub_status == "OPTIMAL"
                st.fails_at_max = (improved || st.y_target < st.y_max) ? 0 : st.fails_at_max + 1
                improved || (st.y_target = solved ? min(st.y_max, st.y_target + st.delta_y) :
                                                    max(st.y_min, st.y_target - st.delta_y))
                if st.fails_at_max >= POLICY_D_F        # end of a size cycle
                    st.fails_at_max = 0
                    if fp == st.fp_at_cycle_start       # no change during the cycle: exhausted
                        st.n_exhaustions += 1
                        exhausted_now = true
                    end
                    if exhausted_now && st.variant != :noEscape
                        st.mode = :ESCAPE
                        st.y_target = st.y_max          # escape releases at y_max
                        st.escape_left = POLICY_D_E
                        st.escape_blocks = 0
                    else                                # restart as in C (scores kept)
                        st.y_target = st.y_min
                        st.fp_at_cycle_start = fp
                    end
                end
            end
        elseif st.last_mode == :ESCAPE
            if fp_changed                               # improvement or sideways: back to STRUCT
                st.mode = :STRUCT
                st.y_target = st.y_min
                st.fails_at_max = 0
                st.fp_at_cycle_start = fp
                st.escape_blocks = 0
            else
                st.escape_left -= 1
                if st.escape_left == 0                  # block of E iterations without change
                    st.escape_blocks += 1
                    st.escape_left = POLICY_D_E
                    if st.escape_blocks >= POLICY_D_EXACT_BLOCKS && st.variant == :full &&
                       policyD_exact_allowed(inst)
                        st.mode = :EXACT
                    end
                end
            end
        end
    end
    fp_changed && foreach(x -> st.cursor[x] = 1, POLICY_D_EVEN)   # new schedule: rank 1
    st.last_Z = Z
    st.last_fp = fp
    st.last_mode = st.mode

    row(; type, rank = missing, n_free = missing, R = Op[], n_crit = missing, skipped = false) =
        push!(st.log, (t = t, mode = string(st.mode), type = string(type), rank = rank,
                       y_target = st.y_target, n_free_y = n_free, n_released = length(R),
                       n_critical = n_crit, Z_inc = Z, schedule_id = sid, sideways = sideways,
                       skipped = skipped, s_BB = st.scores[:BB], s_M = st.scores[:M],
                       s_J = st.scores[:J], s_TW = st.scores[:TW], switch_mode = switch_mode,
                       fails_at_max = st.fails_at_max, escape_blocks = st.escape_blocks,
                       exhausted = exhausted_now, released = join(["$(o[1])-$(o[2])" for o in R], " ")))

    if st.mode == :EXACT                                # Level 2: full model, remaining time
        st.tau = :EXACT
        row(type = :EXACT, n_free = length(pairs))
        return SOLVE_EXACT
    elseif st.mode == :ESCAPE                           # Level 1: random baseline release
        st.tau = :RAND
        st.n_escape_iters += 1
        free = sort(random_free(inst, incumbent, history, st.rng, st.y_max))
        skip = (fp, free) in st.solved_escape
        row(type = :RAND, n_free = length(free), skipped = skip)
        skip && return SKIP_ITERATION
        @assert st.y_min <= length(free) <= st.y_max "Policy D: free y outside [y_min, y_max]"
        push!(st.solved_escape, (fp, free))
        return free
    end

    # STRUCT: odd iterations BB, even iterations τ_even (warm-up M, J, TW)
    if isodd(t)
        st.tau = :BB
    else
        t <= POLICY_D_WARMUP && (st.tau_even = POLICY_D_EVEN[t ÷ 2])
        st.tau = st.tau_even
    end
    t <= POLICY_D_WARMUP && (st.y_target = st.y_min)
    st.last_used[st.tau] = t
    F = compute_structural_features(inst, incumbent)
    core, rank = policyD_core(st, inst, F, st.tau)
    fs = policyD_fill_scores(inst, incumbent, F, st.cnt)
    inR = Set(core)
    rest = sort([o for o in operations(inst) if !(o in inR)]; by = o -> (-fs[o].score, o))
    R, n_free, list_exhausted = policyC2_fill(vcat(core, rest), _pairs_of(inst), st.y_target, st.y_max)
    R = sort(R)
    foreach(o -> st.cnt[o] += 1, R)
    skip = (fp, R) in st.solved
    row(type = st.tau, rank = rank, n_free = n_free, R = R,
        n_crit = count(o -> F.critical[o...], R), skipped = skip)
    skip && return SKIP_ITERATION
    @assert !((fp, R) in st.solved) "Policy D: (schedule, released set) would be solved twice"
    @assert n_free <= st.y_max && (n_free >= st.y_min || list_exhausted) "Policy D: free y $n_free outside [$(st.y_min), $(st.y_max)]"
    push!(st.solved, (fp, R))
    Rset = Set(R)
    return [(a, b) for (a, b) in pairs if a in Rset || b in Rset]
end

"""
    policyD_log(st, iterations) -> Vector of NamedTuples

Policy D log joined with the loop's iteration log (elapsed time, Gurobi status,
subproblem gap and makespan, improved). `sideways` refers to the iteration of the row
(its result moved the incumbent to another schedule with equal makespan).
"""
function policyD_log(st::PolicyDState, iterations)
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
