# =============================================================================
# model.jl — Exact MIP model (Manne, disjunctive with big-M)
# =============================================================================
#
# Step 1 of the assignment.
#
# Variables
#   S[j,k] ≥ 0          start time of operation (j,k)
#   y[a,b] ∈ {0,1}      for every pair a=(j1,k1), b=(j2,k2) on the same machine:
#                       1 ⇔ a before b
#   Cmax ≥ 0            makespan
#
# Objective
#   min Cmax
#
# Constraints
#   (1) Job order:         S[j,k] ≥ S[j,k-1] + p[j,k-1]
#   (2) Disjunctive:       S[b] ≥ S[a] + p[a] - M*(1 - y[a,b])
#                          S[a] ≥ S[b] + p[b] - M*y[a,b]
#   (3) Makespan:          Cmax ≥ S[j,m] + p[j,m]   for every job j
#
# M = total_processing_time(inst), or the makespan of a known solution (keyword `ub`)
# For Fix-and-Optimize the model is built ONCE and y is controlled via
# fix(y[...], value; force = true) / unfix(y[...]).
# Gurobi runs with Seed = 42 by default (reproducibility).
# =============================================================================

using JuMP
using Gurobi

# One Gurobi environment for all models (avoids repeated licence messages)
const GRB_ENV = Ref{Gurobi.Env}()
function gurobi_env()
    isassigned(GRB_ENV) || (GRB_ENV[] = Gurobi.Env())
    return GRB_ENV[]
end

"All unordered pairs of operations that run on the same machine."
function machine_pairs(inst::JSSPInstance)
    pairs = Tuple{Tuple{Int,Int},Tuple{Int,Int}}[]
    for i in 1:inst.m
        ops = ops_on_machine(inst, i)
        for x in 1:length(ops), z in x+1:length(ops)
            push!(pairs, (ops[x], ops[z]))
        end
    end
    return pairs
end

"""
    build_model(inst; time_limit = 60.0, threads = 4, silent = false, warm_start = nothing,
                seed = 42, ub = nothing)

Builds the JSSP MIP and returns the JuMP model. Access to the variables:
`model[:S][(j,k)]`, `model[:y][(a,b)]` (for `(a,b)` from `machine_pairs(inst)`),
`model[:Cmax]`.

`warm_start` (optional, a `Schedule`) is passed to Gurobi as MIP start.
`seed` sets Gurobi's `Seed` parameter (reproducible solver behaviour).
`ub` (optional): makespan of a known solution. Then M = ub and S[(j,k)] ≤ ub - p[j,k],
Cmax ≤ ub; otherwise M = total_processing_time(inst).
"""
function build_model(inst::JSSPInstance; time_limit = 60.0, threads = 4, silent = false,
                     warm_start::Union{Nothing,Schedule} = nothing, seed::Integer = 42,
                     ub::Union{Nothing,Int} = nothing)
    model = Model(() -> Gurobi.Optimizer(gurobi_env()))
    silent && set_silent(model)              # first, otherwise Gurobi prints "Set parameter ..."
    set_attribute(model, "TimeLimit", time_limit)
    set_attribute(model, "Threads", threads)
    set_attribute(model, "Seed", seed)

    ops = operations(inst)
    pairs = machine_pairs(inst)
    p = inst.ptime

    # Big-M. Without ub: M = sum of all processing times (always valid).
    # With ub (makespan of a known solution): we only search for solutions with
    # makespan ≤ ub. For these, 0 ≤ S ≤ ub - p holds, so S_b - S_a - p_a ≥ -ub, and
    # M = ub suffices to "switch off" every disjunction. The optimum is preserved
    # because it is ≤ ub.
    M = ub === nothing ? total_processing_time(inst) : ub

    # Variables
    @variable(model, 0 <= S[ops] <= M)       # S[(j,k)]; upper bound M is always valid
    @variable(model, y[pairs], Bin)          # y[(a,b)] = 1  ⇔  a before b
    @variable(model, 0 <= Cmax <= M)
    if ub !== nothing                        # tighter bounds: every operation ends by ub
        for (j, k) in ops
            set_upper_bound(S[(j, k)], ub - p[j, k])
        end
    end

    # Objective
    @objective(model, Min, Cmax)

    # (1) Job order
    @constraint(model, job_prec[j in 1:inst.n, k in 2:inst.m],
                S[(j, k)] >= S[(j, k-1)] + p[j, k-1])

    # (2) Disjunctive machine constraints
    @constraint(model, a_before_b[(a, b) in pairs],
                S[b] >= S[a] + p[a...] - M * (1 - y[(a, b)]))
    @constraint(model, b_before_a[(a, b) in pairs],
                S[a] >= S[b] + p[b...] - M * y[(a, b)])

    # (3) Makespan: last operation of every job
    @constraint(model, makespan_def[j in 1:inst.n],
                Cmax >= S[(j, inst.m)] + p[j, inst.m])

    warm_start === nothing || set_warm_start!(model, inst, warm_start)
    return model
end

"""
    precedes(inst, s, a, b) -> Bool

Does operation `a` come before `b` (on the same machine) in schedule `s`? This is the
value of `y[(a,b)]` belonging to `s`. Ties in start time (only possible with duration 0):
`a` first if `a` then also ends before `b` starts.
"""
precedes(inst::JSSPInstance, s::Schedule, a, b) =
    (s.start[a...], s.start[a...] + inst.ptime[a...]) <=
    (s.start[b...], s.start[b...] + inst.ptime[b...])

"""
    set_warm_start!(model, inst, s::Schedule)

Sets S, y and Cmax as MIP start (e.g. the solution from `initial_schedule`).
"""
function set_warm_start!(model::Model, inst::JSSPInstance, s::Schedule)
    S, y, Cmax = model[:S], model[:y], model[:Cmax]
    for (j, k) in operations(inst)
        set_start_value(S[(j, k)], s.start[j, k])
    end
    for (a, b) in machine_pairs(inst)
        set_start_value(y[(a, b)], precedes(inst, s, a, b) ? 1.0 : 0.0)
    end
    set_start_value(Cmax, makespan(inst, s))
    return model
end

"""
    schedule_from_model(model, inst) -> Schedule

Reads the start times from the current solution. The data are integral, so small
numerical deviations (e.g. 4.9999999) are rounded. Whether the result is feasible is
checked independently afterwards by `check_schedule`.
"""
function schedule_from_model(model::Model, inst::JSSPInstance)
    S = model[:S]
    start = [round(Int, value(S[(j, k)])) for j in 1:inst.n, k in 1:inst.m]
    return Schedule(start)
end

# --- Anytime-Performance -------------------------------------------------------
#
# During solving, a Gurobi callback records when the incumbent (best solution
# found) or the bound (lower bound) changes. The result is a vector of points
# (time, incumbent, bound) – the basis for anytime plots. Values as in solve_model:
# incumbent rounded, bound rounded up (makespan is integral); `missing` as long as
# no value exists yet.

const TracePoint = NamedTuple{(:time, :incumbent, :bound),
                              Tuple{Float64, Union{Missing,Int}, Union{Missing,Int}}}

_clean_inc(x) = (isfinite(x) && abs(x) < 1e90) ? round(Int, x) : missing
_clean_bnd(x) = (isfinite(x) && abs(x) < 1e90) ? ceil(Int, x - 1e-6) : missing

function _push_trace!(trace::Vector{TracePoint}, t, inc, bnd)
    if isempty(trace) || !isequal(trace[end].incumbent, inc) || !isequal(trace[end].bound, bnd)
        push!(trace, (time = Float64(t), incumbent = inc, bound = bnd))
    end
end

"Registers the callback that writes incumbent/bound over time into `trace`."
function attach_trace_callback!(model::Model, trace::Vector{TracePoint})
    function cb(cb_data, cb_where::Cint)
        cb_where == GRB_CB_MIP || cb_where == GRB_CB_MIPSOL || return
        t, bst, bnd = Ref{Cdouble}(), Ref{Cdouble}(), Ref{Cdouble}()
        GRBcbget(cb_data, cb_where, GRB_CB_RUNTIME, t)
        if cb_where == GRB_CB_MIP
            GRBcbget(cb_data, cb_where, GRB_CB_MIP_OBJBST, bst)
            GRBcbget(cb_data, cb_where, GRB_CB_MIP_OBJBND, bnd)
        else    # new solution found: its value may be better than OBJBST
            sol = Ref{Cdouble}()
            GRBcbget(cb_data, cb_where, GRB_CB_MIPSOL_OBJ, sol)
            GRBcbget(cb_data, cb_where, GRB_CB_MIPSOL_OBJBST, bst)
            GRBcbget(cb_data, cb_where, GRB_CB_MIPSOL_OBJBND, bnd)
            bst[] = min(bst[], sol[])
        end
        _push_trace!(trace, t[], _clean_inc(bst[]), _clean_bnd(bnd[]))
        return
    end
    MOI.set(model, Gurobi.CallbackFunction(), cb)
    return model
end

"""
    solve_model(inst; kwargs...) -> NamedTuple

Solves the model and returns the required key figures:
`status, incumbent, bound, gap, runtime, schedule, model`.
`incumbent` and `bound` are integral (the makespan is integral);
`gap = (incumbent - bound) / incumbent`. `bound` is `missing` if no finite bound exists;
`trace` is the anytime history: vector of `(time, incumbent, bound)`,
one entry per change, plus an end point at `runtime`.
`incumbent`/`gap` are `missing` and `schedule` is `nothing` if no solution was found.
Keywords are passed on to `build_model` (e.g. `time_limit`, `silent`, `warm_start`, `seed`).

`wall_start` (optional, a `time()` value): common zero point for comparisons with
other methods. Then `trace` times and `runtime` are wall-clock times since
`wall_start` (incl. model build), and the Gurobi time limit is reduced by the time
already used, so that `time_limit` applies in total. `build_time` is the time from
`wall_start` (or function start) until Gurobi starts.
"""
function solve_model(inst::JSSPInstance; time_limit = 60.0,
                     wall_start::Union{Nothing,Float64} = nothing, kwargs...)
    t0 = something(wall_start, time())
    model = build_model(inst; time_limit = time_limit, kwargs...)
    build_time = time() - t0
    wall_start === nothing ||
        set_attribute(model, "TimeLimit", max(0.0, time_limit - build_time))
    trace = TracePoint[]
    attach_trace_callback!(model, trace)
    optimize!(model)

    # All data are integral -> makespan is integral. So round the incumbent
    # (removes noise like 54.99999999) and round the bound up (stays valid).
    found = has_values(model)
    incumbent = found ? round(Int, objective_value(model)) : missing
    raw_bound = objective_bound(model)
    bound = isfinite(raw_bound) ? ceil(Int, raw_bound - 1e-6) : missing
    gap = (found && bound !== missing) ?
          (incumbent == 0 ? 0.0 : (incumbent - bound) / incumbent) : missing

    # End point: final values at the end time (the callback may miss the last state)
    runtime = solve_time(model)
    push!(trace, (time = runtime, incumbent = incumbent, bound = bound))
    if wall_start !== nothing          # shift to the common zero point
        trace = [(time = tp.time + build_time, incumbent = tp.incumbent, bound = tp.bound)
                 for tp in trace]
        runtime += build_time
    end

    return (
        status    = termination_status(model),
        incumbent = incumbent,
        bound     = bound,
        gap       = gap,
        runtime   = runtime,
        build_time = build_time,
        trace     = trace,
        schedule  = found ? schedule_from_model(model, inst) : nothing,
        model     = model,
    )
end
