# =============================================================================
# matheuristic.jl — Fix-and-Optimize (FaO) based on the MIP from model.jl
# =============================================================================
#
# Idea: starting from the current solution, each iteration frees n sequencing
# variables y[(a,b)], with n drawn uniformly from n_free_min:n_free_max and capped
# at FREE_CAP (75 %) of all machine pairs. A fixed n: n_free_min = n_free_max.
# All other y are fixed to their value in the current solution and Gurobi
# optimises the resulting subproblem (warm start = current solution).
#
# Which y are freed is decided by `select_free(inst, incumbent, history, rng, n_free_y)`,
# which returns a Vector of machine pairs (a, b) as produced by `machine_pairs(inst)`:
#   - random_free (here): random operations, nearest machine neighbours first
#     -> baseline "random_fao"
#   - Policy A/B/C (later): own function with the same signature
# `incumbent` is the CURRENT solution (the search is centred on it), `history`
# the iteration log so far (Vector of IterationRecord).
#
# Acceptance: a subproblem solution with makespan < current (or ≤ with
# accept_equal = true and changed machine sequences) becomes the new current
# solution after passing check_schedule. The best solution is tracked separately;
# only a strictly smaller makespan adds a point to the anytime trace.
#
# The MIP is built ONCE. Per iteration only changes are passed to Gurobi (delta):
# fix/unfix only for y whose status changes, fixed values and start values only
# if they changed since the previous iteration.
#
# Time limit: hard. The remaining time is checked before every iteration and
# before every solve (< 0.05 s -> stop); the subproblem gets
# max(0.01, min(sub_time_limit, remaining)). Fixing, warm start and reading the
# solution count towards the time. Note: if the machine goes to sleep, the wall
# clock keeps running -> start long runs with keep_awake() or `caffeinate -i`.
#
# Bound: subproblems provide no global bound. Therefore `lower_bound(inst)`
# (static lower bound, bounds.jl) is computed once at the start. If the best
# solution reaches this bound it is provably optimal -> stop with status "OPTIMAL".
#
# Reproducibility: all randomness comes from MersenneTwister(seed), default 42;
# Gurobi also runs with Seed = 42 (see build_model). Because subproblems are
# solved under wall-clock time limits, runs can still differ slightly between
# machines or under different load.
# =============================================================================

using Random
using Statistics: median, mean

const DEFAULT_SEED = 42
# Markers a `select_free` may return instead of a pair list (used by Policy C v2):
const SKIP_ITERATION = :skip_iteration   # no Gurobi call; logged as sub_status "SKIPPED_DUPLICATE"
const STOP_SEARCH = :stop_search         # end the search; incumbent kept, trace continued to time_limit
const SOLVE_EXACT = :solve_exact         # (Policy D) solve the full model for the remaining time
const FREE_CAP = 0.75          # at most this share of all machine pairs is freed

"One entry in the Fix-and-Optimize iteration log."
const IterationRecord = NamedTuple{
    (:iteration, :time, :n_free_y, :sub_status, :sub_makespan, :improved,
     :setup_time, :sub_time, :accepted, :sub_gap),
    Tuple{Int, Float64, Int, String, Union{Missing,Int}, Bool, Float64, Float64, Bool,
          Union{Missing,Float64}}}

"""
    random_free(inst, incumbent, history, rng, n_free_y) -> Vector of machine pairs

Baseline neighbourhood: visits the operations in random order and frees their y
variables until exactly `n_free_y` are free. For each operation the pairs with its
nearest neighbours in the current machine sequence come first, so an operation
whose pairs are only partially freed can still move locally.
"""
function random_free(inst::JSSPInstance, incumbent::Schedule, history,
                     rng::AbstractRNG, n_free_y::Int)
    pos = Dict{Tuple{Int,Int},Int}()                     # position in machine sequence
    for seq in machine_sequences(inst, incumbent), (p, op) in enumerate(seq)
        pos[op] = p
    end
    pairs_of = Dict(op => Tuple{Tuple{Int,Int},Tuple{Int,Int}}[] for op in operations(inst))
    for (a, b) in machine_pairs(inst)
        push!(pairs_of[a], (a, b))
        push!(pairs_of[b], (a, b))
    end

    chosen = Set{Tuple{Tuple{Int,Int},Tuple{Int,Int}}}()
    for op in shuffle(rng, operations(inst))
        length(chosen) >= n_free_y && break
        cand = sort!([p for p in pairs_of[op] if !(p in chosen)];
                     by = p -> abs(pos[p[1]] - pos[p[2]]))
        for p in cand
            length(chosen) >= n_free_y && break
            push!(chosen, p)
        end
    end
    return collect(chosen)
end

"y values (0/1) of all machine pairs for schedule `s`."
y_values(inst, s::Schedule, pairs) = Int8[precedes(inst, s, a, b) ? 1 : 0 for (a, b) in pairs]

"""
    random_fao(inst; time_limit = 60.0, sub_time_limit = 1.0, n_free_min = 200,
               n_free_max = 800, seed = 42, threads = 4, accept_equal = true,
               select_free = random_free, ub = nothing, wall_start = nothing, verbose = false)

Fix-and-Optimize with a random neighbourhood (baseline for policies A/B/C).

- Start solution: `initial_schedule(inst)` (same as the Gurobi variant).
- Per iteration: n = rand(n_free_min:n_free_max), capped at floor(FREE_CAP · #pairs);
  `select_free` picks n free y, all others are fixed; the subproblem gets the time
  limit max(0.01, min(sub_time_limit, remaining)).
- `ub` (optional) is passed to `build_model` (Big-M = ub). The model is built once;
  M is not updated during the search.
- All times are wall-clock times since `wall_start` (default: function start),
  including start solution and model build; `time_limit` is hard for everything.

Returns the same format as `solve_model`: `bound` = static lower bound
`lower_bound(inst)`, `gap` accordingly; status "OPTIMAL" if the best solution reaches
the bound, otherwise "TIME_LIMIT". Additionally `iterations` (iteration log),
`diagnostics` (see `fao_diagnostics`) and `params`.
"""
function random_fao(inst::JSSPInstance; time_limit::Real = 60.0, sub_time_limit::Real = 1.0,
                    n_free_min::Integer = 200, n_free_max::Integer = 800,
                    seed::Integer = DEFAULT_SEED, threads::Integer = 4,
                    accept_equal::Bool = true, select_free = random_free,
                    ub::Union{Nothing,Int} = nothing,
                    wall_start::Union{Nothing,Float64} = nothing, verbose::Bool = false)
    1 <= n_free_min <= n_free_max ||
        error("expected 1 ≤ n_free_min ≤ n_free_max, got $n_free_min, $n_free_max")
    t0 = something(wall_start, time())
    elapsed() = time() - t0
    rng = MersenneTwister(seed)

    # Start solution = current = best solution
    cur = initial_schedule(inst)
    check_schedule(inst, cur; verbose = false)[1] || error("$(inst.name): start solution infeasible")
    cur_ms = makespan(inst, cur)
    best, best_ms = cur, cur_ms
    lb = lower_bound(inst)
    trace = TracePoint[(time = elapsed(), incumbent = best_ms, bound = lb)]

    # Build the model once; remember Gurobi-side state for delta updates
    model = build_model(inst; time_limit = sub_time_limit, threads = threads, silent = true,
                        seed = seed, ub = ub)
    S, y, Cmax = model[:S], model[:y], model[:Cmax]
    pairs = machine_pairs(inst)
    yvars = [y[p] for p in pairs]
    ops = operations(inst)
    Svars = [S[op] for op in ops]
    cur_y = y_values(inst, cur, pairs)             # y values of the current solution
    y_state = fill(Int8(-1), length(pairs))        # -1 free, 0/1 fixed to this value
    y_start = fill(Int8(-1), length(pairs))        # last start value set (-1: none)
    S_start = fill(-1, length(ops))
    build_time = elapsed()

    n_cap = floor(Int, FREE_CAP * length(pairs))
    history = IterationRecord[]

    it = 0
    stopped = false                                # STOP_SEARCH returned by select_free
    exact = missing                                # SOLVE_EXACT: (start_time, improved, status, bound)
    while best_ms > lb && time_limit - elapsed() >= 0.05
        it += 1
        t_setup = time()
        n = rand(rng, n_free_min:n_free_max)
        n = min(n, n_cap)
        sel = select_free(inst, cur, history, rng, n)
        if sel === STOP_SEARCH
            stopped = true
            break
        elseif sel === SOLVE_EXACT                    # full model for the remaining time, then stop
            remaining = time_limit - elapsed()
            remaining < 0.05 && break
            t_ex = elapsed()
            # existing builder and Gurobi configuration, MIP start = current incumbent; anytime
            # trace recorded via the same callback as solve_model, shifted to the loop's clock
            full = build_model(inst; time_limit = remaining, threads = threads, silent = true,
                               seed = seed, ub = ub, warm_start = cur)
            ex_trace = TracePoint[]
            attach_trace_callback!(full, ex_trace)
            t_sub = time()
            optimize!(full)
            sub_time = time() - t_sub
            status = string(termination_status(full))
            sub_ms = missing
            improved = accepted = false
            if has_values(full)
                cand = schedule_from_model(full, inst)
                sub_ms = makespan(inst, cand)
                ok, errs = check_schedule(inst, cand; verbose = false)
                ok || @warn "$(inst.name): full-model solution rejected by the checker" errors = first(errs, 5)
                if ok && sub_ms < best_ms
                    for tp in ex_trace                 # intermediate incumbents of the full model
                        ismissing(tp.incumbent) && continue
                        sub_ms <= tp.incumbent < trace[end].incumbent &&
                            push!(trace, (time = t_ex + tp.time, incumbent = tp.incumbent, bound = lb))
                    end
                    trace[end].incumbent > sub_ms &&
                        push!(trace, (time = elapsed(), incumbent = sub_ms, bound = lb))
                    cur, cur_ms, best, best_ms = cand, sub_ms, cand, sub_ms
                    improved = accepted = true
                end
            end
            raw_bnd = objective_bound(full)
            exact = (start_time = t_ex, improved = improved, status = status,
                     bound = isfinite(raw_bnd) ? ceil(Int, raw_bnd - 1e-6) : missing)
            push!(history, (iteration = it, time = elapsed(), n_free_y = length(pairs),
                            sub_status = "EXACT_" * status, sub_makespan = sub_ms,
                            improved = improved, setup_time = time() - t_setup - sub_time,
                            sub_time = sub_time, accepted = accepted,
                            sub_gap = has_values(full) ? relative_gap(full) : missing))
            break
        elseif sel === SKIP_ITERATION                 # duplicate subproblem: no Gurobi call
            push!(history, (iteration = it, time = elapsed(), n_free_y = 0,
                            sub_status = "SKIPPED_DUPLICATE", sub_makespan = missing,
                            improved = false, setup_time = time() - t_setup, sub_time = 0.0,
                            accepted = false, sub_gap = missing))
            continue
        end
        free_set = Set(sel)

        # (1) fix/unfix y – pass only status changes to Gurobi
        n_free = 0
        for idx in eachindex(pairs)
            if pairs[idx] in free_set
                n_free += 1
                if y_state[idx] != -1
                    unfix(yvars[idx])
                    y_state[idx] = -1
                end
            elseif y_state[idx] != cur_y[idx]
                fix(yvars[idx], cur_y[idx]; force = true)
                y_state[idx] = cur_y[idx]
            end
        end

        # (2) warm start = current solution – set only changed values
        for idx in eachindex(pairs)
            if y_start[idx] != cur_y[idx]
                set_start_value(yvars[idx], cur_y[idx])
                y_start[idx] = cur_y[idx]
            end
        end
        for (n, (j, k)) in enumerate(ops)
            if S_start[n] != cur.start[j, k]
                set_start_value(Svars[n], cur.start[j, k])
                S_start[n] = cur.start[j, k]
            end
        end
        set_start_value(Cmax, cur_ms)
        setup_time = time() - t_setup

        # (3) check remaining time again (setup counts), then solve
        remaining = time_limit - elapsed()
        remaining < 0.05 && break
        sub_limit = max(0.01, min(sub_time_limit, remaining))
        set_attribute(model, "TimeLimit", sub_limit)
        t_sub = time()
        optimize!(model)
        sub_time = time() - t_sub
        sub_time > sub_limit + 5 &&
            @warn "$(inst.name), iteration $it: subproblem ran $(round(sub_time, digits = 1)) s " *
                  "with limit $(round(sub_limit, digits = 2)) s – machine asleep/paused?"

        # (4) read, check, accept
        status = string(termination_status(model))
        sub_ms = missing
        sub_gap = has_values(model) ? relative_gap(model) : missing   # Gurobi MIP gap of the subproblem
        improved = accepted = false
        if has_values(model)
            cand = schedule_from_model(model, inst)
            sub_ms = makespan(inst, cand)
            # An equal makespan only counts as a new solution if the machine sequences (y)
            # change – not if only start times are shifted.
            cand_y = sub_ms <= cur_ms ? y_values(inst, cand, pairs) : cur_y
            if sub_ms < cur_ms || (accept_equal && sub_ms == cur_ms && cand_y != cur_y)
                ok, errs = check_schedule(inst, cand; verbose = false)
                if ok
                    cur, cur_ms, accepted = cand, sub_ms, true
                    cur_y = cand_y
                    if cur_ms < best_ms
                        best, best_ms, improved = cur, cur_ms, true
                        push!(trace, (time = elapsed(), incumbent = best_ms, bound = lb))
                        verbose && println("  it $it: new incumbent $best_ms (t = $(round(elapsed(), digits = 2)) s)")
                    end
                else
                    @warn "$(inst.name), iteration $it: subproblem solution (makespan $sub_ms) " *
                          "rejected by the checker" errors = first(errs, 5)
                end
            end
        end
        push!(history, (iteration = it, time = elapsed(), n_free_y = n_free,
                        sub_status = status, sub_makespan = sub_ms, improved = improved,
                        setup_time = setup_time, sub_time = sub_time, accepted = accepted,
                        sub_gap = sub_gap))
    end

    runtime = elapsed()
    # bound: static lower bound, or the full model's bound if SOLVE_EXACT was used
    bound = (ismissing(exact) || ismissing(exact.bound)) ? lb : max(lb, exact.bound)
    push!(trace, (time = runtime, incumbent = best_ms, bound = bound))
    # stopped by the policy: the incumbent is held until the time limit (same trace end as others)
    stopped && runtime < time_limit && push!(trace, (time = Float64(time_limit), incumbent = best_ms, bound = lb))
    return (
        status      = best_ms <= bound ? "OPTIMAL" : "TIME_LIMIT",
        incumbent   = best_ms,
        bound       = bound,
        gap         = best_ms == 0 ? 0.0 : (best_ms - bound) / best_ms,
        runtime     = runtime,
        build_time  = build_time,
        trace       = trace,
        schedule    = best,
        model       = model,
        iterations  = history,
        stop_reason = stopped ? "exhausted" : missing,
        exact       = exact,
        diagnostics = fao_diagnostics(history),
        params      = (time_limit = time_limit, sub_time_limit = sub_time_limit,
                       n_free_min = n_free_min, n_free_max = n_free_max,
                       free_cap = FREE_CAP, n_pairs = length(pairs), seed = seed,
                       threads = threads, accept_equal = accept_equal, ub = ub,
                       select_free = string(select_free)),
    )
end

"""
    fao_diagnostics(iterations) -> NamedTuple

Key figures of a FaO run: number of iterations, share of subproblems ending with
TIME_LIMIT, medians of n_free_y / sub_time / setup_time, share of improving and
accepted iterations, and time of the last improvement (`missing` if none).
Only real Gurobi calls are counted; skipped duplicate iterations (Policy C v2) are
reported separately as `n_skipped`.
"""
function fao_diagnostics(all_iterations)
    iterations = [r for r in all_iterations if r.sub_status != "SKIPPED_DUPLICATE"]
    n = length(iterations)
    med(f) = n == 0 ? missing : median(f(r) for r in iterations)
    improvements = [r.time for r in iterations if r.improved]
    return (
        iterations         = n,
        share_time_limit   = n == 0 ? missing : count(r -> r.sub_status == "TIME_LIMIT", iterations) / n,
        median_n_free_y    = med(r -> r.n_free_y),
        median_sub_time    = med(r -> r.sub_time),
        median_setup_time  = med(r -> r.setup_time),
        share_improved     = n == 0 ? missing : length(improvements) / n,
        share_accepted     = n == 0 ? missing : count(r -> r.accepted, iterations) / n,
        time_last_improvement = isempty(improvements) ? missing : last(improvements),
        n_skipped          = length(all_iterations) - n,
    )
end

"""
    keep_awake()

Prevents sleep on macOS while this Julia process is running
(`caffeinate -i -w <pid>`). Otherwise the wall clock keeps running during sleep and
time limits / anytime curves are distorted. Does nothing on other systems.
"""
function keep_awake()
    Sys.isapple() && Sys.which("caffeinate") !== nothing &&
        run(pipeline(`caffeinate -i -w $(getpid())`; stdout = devnull, stderr = devnull); wait = false)
    return nothing
end
