# =============================================================================
# dynamic.jl — Policy C: adaptive deterministic control
# =============================================================================
#
# Plugs into the existing Fix-and-Optimize loop (random_fao) as `select_free`, like
# Policies A and B; start solution, model, fixing, warm start, per-subproblem time
# limit, Gurobi settings and acceptance are those of the loop. Released y (rule "any",
# as A/B): y[(a,b)] is free if a ∈ R or b ∈ R.
#
# Release size: Policy C controls the number k of released operations itself, within
# [k_min, k_max] derived from the shared y budget range n_free_min:n_free_max (capped at
# FREE_CAP of all machine pairs) with the same conversion as Policies A/B:
#   k_min = policyA_k(n_free_min),  k_max = policyA_k(min(n_free_max, cap)).
# The per-iteration n drawn by the loop is not used (C adapts k instead).
#
# Neighbourhood types (cursor per type, advanced after a failed iteration of that type):
#   CB  critical blocks (runs of consecutive critical operations on a machine, incl.
#       length 1) in start-time order; whole blocks cyclically from cursor c_CB until k;
#       if all critical operations are in and |R| < k, machine predecessors/successors
#       of released operations (increasing start time); trim to k (latest starts of the
#       last block dropped).
#   TW  operations sorted by incumbent start time; k consecutive from position p_TW
#       (cyclic); on failure p_TW += ceil(k/2).
#   MA  machines by load (Σ p) descending; whole machines from cursor c_MA while they
#       fit, remaining slots from the next machine, earliest start first.
#
# Control (deterministic): warm-up t = 1, 2, 3 → CB, TW, MA with k = k_min. After every
# iteration t ≥ 3: score s_τ = (1-α) s_τ + α r (r = relative makespan improvement, else 0;
# updated after every iteration incl. warm-up), then
#   1. size: improved → keep k; not improved & solved (OPTIMAL) → k + Δ; not improved &
#      hard (time limit) → k - Δ (clamped to [k_min, k_max]);
#   2. type: improved → keep, fail_streak = 0; else fail_streak += 1 and if ≥ F switch
#      to the better-scoring of the other two types (ties CB > TW > MA), fail_streak = 0;
#   3. restart: if u (iterations since the last improvement) ≥ U_R: k = k_min, scores 0, u = 0;
#   4. type unchanged and iteration failed → advance that type's cursor.
# =============================================================================

const POLICY_C_ALPHA = 0.3          # α: score smoothing
const POLICY_C_F = 2                # F: failures before switching type
const POLICY_C_UR = 10              # U_R: iterations without improvement before restart
const POLICY_C_WARMUP = (:CB, :TW, :MA)
const POLICY_C_PRIORITY = (:CB, :TW, :MA)   # tie-break for type choice
"Δ: step of the size adaptation."
policyC_delta(k_min::Int, k_max::Int) = max(1, round(Int, 0.1 * (k_max - k_min)))

"""
    PolicyCState(; n_free_min = 200, n_free_max = 800)

Per-run state of Policy C (create one per run, e.g. via `policyC_selector()`); the range
must be the one passed to `random_fao`. Callable with the `select_free` signature;
ignores `rng` and the loop's per-iteration budget n.
"""
mutable struct PolicyCState
    n_free_min::Int
    n_free_max::Int
    k_min::Int
    k_max::Int
    delta::Int
    tau::Symbol                         # type of the current / last iteration
    k::Int
    fail_streak::Int
    u::Int                              # iterations since the last improvement
    scores::Dict{Symbol,Float64}
    cursor::Dict{Symbol,Int}            # c_CB, p_TW, c_MA (1-based, used cyclically)
    last_Z::Int                         # incumbent makespan at the previous decision
    log::Vector{NamedTuple}
end
PolicyCState(; n_free_min::Int = 200, n_free_max::Int = 800) =
    PolicyCState(n_free_min, n_free_max, 0, 0, 0, :CB, 0, 0, 0,
                 Dict(:CB => 0.0, :TW => 0.0, :MA => 0.0), Dict(:CB => 1, :TW => 1, :MA => 1),
                 0, NamedTuple[])

"Policy C for `random_fao(...; select_free = policyC_selector())` – a fresh state per run."
policyC_selector(; n_free_min::Int = 200, n_free_max::Int = 800) =
    PolicyCState(; n_free_min, n_free_max)

"Critical blocks for CB: maximal runs (length ≥ 1) of consecutive critical operations on a
machine, ordered by the incumbent start time of their first operation (ties: index)."
function policyC_blocks(F::StructuralFeatures, S::Matrix{Int})
    blocks = Vector{Op}[]
    for seq in F.seqs
        run = Op[]
        for o in seq
            if F.critical[o...]
                push!(run, o)
            else
                isempty(run) || push!(blocks, run)
                run = Op[]
            end
        end
        isempty(run) || push!(blocks, run)
    end
    return sort(blocks; by = b -> (S[b[1]...], b[1]))
end

"""
    policyC_cb(inst, incumbent, F, k, cursor) -> (R, n_blocks)

CB: whole critical blocks cyclically from block `cursor` until k operations; if all are
in and |R| < k, immediate machine predecessors/successors of released operations in
increasing start time (repeated while it grows), then any remaining operations by start
time; trimmed to exactly k (the last block's latest-starting operations are dropped).
"""
function policyC_cb(inst::JSSPInstance, incumbent::Schedule, F::StructuralFeatures,
                    k::Int, cursor::Int)
    S = incumbent.start
    blocks = policyC_blocks(F, S)
    nb = length(blocks)
    R = Op[]
    for j in 0:nb-1
        length(R) >= k && break
        append!(R, blocks[mod1(cursor + j, nb)])
    end
    inR = Set(R)
    pos = Dict(o => (m, i) for (m, seq) in enumerate(F.seqs) for (i, o) in enumerate(seq))
    grown = true
    while length(R) < k && grown                       # machine neighbours of released ops
        grown = false
        for o in sort(collect(R); by = x -> (S[x...], x))
            m, i = pos[o]
            for nbr in (i > 1 ? F.seqs[m][i-1] : nothing, i < length(F.seqs[m]) ? F.seqs[m][i+1] : nothing)
                (nbr === nothing || nbr in inR || length(R) >= k) && continue
                push!(R, nbr); push!(inR, nbr); grown = true
            end
        end
    end
    if length(R) < k                                   # (only if the machines are exhausted)
        for o in sort(operations(inst); by = x -> (S[x...], x))
            length(R) >= k && break
            o in inR || (push!(R, o); push!(inR, o))
        end
    end
    return R[1:k], nb
end

"TW: k consecutive operations (cyclic) from position `cursor` of the start-time order."
function policyC_tw(inst::JSSPInstance, incumbent::Schedule, k::Int, cursor::Int)
    S = incumbent.start
    by_S = sort(operations(inst); by = o -> (S[o...], o))
    N = length(by_S)
    return [by_S[mod1(cursor + i, N)] for i in 0:k-1]
end

"MA: whole machines (by load, descending; ties: index) from `cursor` while they fit into
k, remaining slots from the next machine, earliest start first."
function policyC_ma(inst::JSSPInstance, incumbent::Schedule, k::Int, cursor::Int)
    S = incumbent.start
    load(m) = sum(inst.ptime[o...] for o in ops_on_machine(inst, m); init = 0)
    order = sort(1:inst.m; by = m -> (-load(m), m))
    R = Op[]
    for j in 0:inst.m-1
        cap = k - length(R)
        cap == 0 && break
        ops = sort(ops_on_machine(inst, order[mod1(cursor + j, inst.m)]); by = o -> (S[o...], o))
        append!(R, length(ops) <= cap ? ops : ops[1:cap])
    end
    return R
end

"Better-scoring of the two types other than `tau` (ties: CB > TW > MA)."
function _policyC_switch(tau::Symbol, scores::Dict{Symbol,Float64})
    others = [x for x in POLICY_C_PRIORITY if x != tau]
    return scores[others[2]] > scores[others[1]] ? others[2] : others[1]
end

function (st::PolicyCState)(inst::JSSPInstance, incumbent::Schedule, history, rng,
                            n_free_y::Int)
    t = length(history) + 1
    Z = makespan(inst, incumbent)
    N = length(operations(inst))
    pairs = machine_pairs(inst)
    if t == 1                                          # per-instance constants
        cap = floor(Int, FREE_CAP * length(pairs))
        st.k_min = min(policyA_k(inst, min(st.n_free_min, cap)), N)
        st.k_max = max(st.k_min, min(policyA_k(inst, min(st.n_free_max, cap)), N))
        st.delta = policyC_delta(st.k_min, st.k_max)
        st.k = st.k_min
    end

    restart = false
    if t >= 2                                          # result of the previous iteration
        h = history[end]
        improved = Z < st.last_Z
        r = improved ? (st.last_Z - Z) / st.last_Z : 0.0
        st.scores[st.tau] = (1 - POLICY_C_ALPHA) * st.scores[st.tau] + POLICY_C_ALPHA * r
        st.u = improved ? 0 : st.u + 1
        if t - 1 >= length(POLICY_C_WARMUP)            # decision rules after iteration ≥ 3
            solved = h.sub_status == "OPTIMAL"
            # 1. size
            improved || (st.k = solved ? min(st.k_max, st.k + st.delta) :
                                         max(st.k_min, st.k - st.delta))
            # 2. type
            old_tau = st.tau
            if improved
                st.fail_streak = 0
            else
                st.fail_streak += 1
                if st.fail_streak >= POLICY_C_F
                    st.tau = _policyC_switch(st.tau, st.scores)
                    st.fail_streak = 0
                end
            end
            # 3. restart
            if st.u >= POLICY_C_UR
                st.k = st.k_min
                foreach(x -> st.scores[x] = 0.0, keys(st.scores))
                st.u = 0
                restart = true
            end
            # 4. cursor of an unchanged type after a failure
            if st.tau == old_tau && !improved
                st.cursor[st.tau] += st.tau == :TW ? cld(st.k, 2) : 1
            end
        end
    end
    if t <= length(POLICY_C_WARMUP)                    # warm-up: CB, TW, MA with k_min
        st.tau = POLICY_C_WARMUP[t]
        st.k = st.k_min
    end
    st.last_Z = Z

    k = min(st.k, N)
    F = compute_structural_features(inst, incumbent)
    nb = missing                                       # number of critical blocks (CB only)
    if st.tau == :CB
        R, nb = policyC_cb(inst, incumbent, F, k, st.cursor[:CB])
    elseif st.tau == :TW
        R = policyC_tw(inst, incumbent, k, st.cursor[:TW])
    else
        R = policyC_ma(inst, incumbent, k, st.cursor[:MA])
    end
    @assert allunique(R) && length(R) == k && st.k_min <= length(R) <= st.k_max "Policy C: |R| = $(length(R)) outside [$(st.k_min), $(st.k_max)]"

    Rset = Set(R)
    free = [(a, b) for (a, b) in pairs if a in Rset || b in Rset]
    push!(st.log, (t = t, type = string(st.tau), k = k, n_released = length(R),
                   n_free_y = length(free), Z_inc = Z, s_CB = st.scores[:CB],
                   s_TW = st.scores[:TW], s_MA = st.scores[:MA], restart = restart,
                   fail_streak = st.fail_streak, u = st.u, cursor = st.cursor[st.tau],
                   n_blocks = nb, k_min = st.k_min, k_max = st.k_max,
                   released = join(["$(o[1])-$(o[2])" for o in sort(R)], " ")))
    return free
end

"""
    policyC_log(st, iterations) -> Vector of NamedTuples

Policy C log joined with the loop's iteration log: adds elapsed time, Gurobi status,
subproblem gap, subproblem makespan and `improved` (`missing` if not solved).
"""
function policyC_log(st::PolicyCState, iterations)
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
