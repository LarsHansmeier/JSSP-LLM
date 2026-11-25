# =============================================================================
# calibrate_subproblems.jl — subproblem size vs. difficulty / improvement chance
# =============================================================================
#
# Measurement only (no policy is run, nothing in src/ is changed). Per instance:
#   1. reference incumbent: window baseline (random_fao + window_free, current settings)
#      for 45 s from the standard initial solution;
#   2. free-y levels 250 … 8000 (geometric, ≤ 0.8·Y_tot); per level 6 released sets,
#      2 each from random_free (local RNG, seed 42), a window release at evenly spaced
#      time points and the BB core (Policy B) + slack fill, filled with policyC2_fill;
#   3. each subproblem solved once with T_cap = 15 s on the FaO model of the window run
#      (same builder and Gurobi configuration as the benchmark), y fixed / freed as in
#      the random_fao loop, warm start = reference; a callback logs every incumbent
#      (seconds, work units, makespan), so one run answers every T ≤ T_cap;
#   4. early stop: ≥ 4 of 6 samples at a level hit T_cap → at most one larger level
#      with 3 samples; per-instance and global time budgets.
# Analysis and the recommendation (decision rules of the task) are derived from the
# logged runs only.
#
# Run:   julia --project=. scripts/calibrate_subproblems.jl
# Output in results/calibration_v1/:
#   samples.csv, incumbents.csv, summary.csv (instance × level × generator × T),
#   instances.csv, skipped.csv, timing.csv, recommendation.txt,
#   time_to_opt.png, p_imp.png, rate.png
# =============================================================================

import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")); io = devnull)
ENV["GKSwstype"] = "100"     # GR without window (file output)

module JSSPLabCalibration

using CSV, DataFrames, Plots, Statistics, Random, JuMP, Gurobi
include(joinpath(@__DIR__, "..", "src", "JSSPLab.jl"))
using .JSSPLab
import .JSSPLab: Op, compute_structural_features, bb_candidates, _pairs_of, policyC2_fill,
                 y_values, schedule_from_model, keep_awake

const T_START = time()
# CALIB_SMOKE=<dir>: quick functional test (ft10, short limits) written to <dir> instead
const SMOKE = get(ENV, "CALIB_SMOKE", "")
const OUT = isempty(SMOKE) ? normpath(joinpath(@__DIR__, "..", "results", "calibration_v1")) : SMOKE
const INSTANCES = isempty(SMOKE) ? ["ta21", "ta41", "ta51", "ta61", "swv12"] : ["ft10"]
const EXTRA = "ta71"                      # only if time remains
const WINDOW_T = isempty(SMOKE) ? 45.0 : 5.0   # reference run of the window baseline
const LEVELS = [250, 500, 1000, 2000, 4000, 8000]
const T_CAP = isempty(SMOKE) ? 15.0 : 2.0
const T_GRID = isempty(SMOKE) ? [0.5, 1.0, 2.0, 5.0, 10.0, 15.0] : [0.5, 1.0, 2.0]
const GENERATORS = ["random", "window", "bb"]
const INSTANCE_BUDGET = isempty(SMOKE) ? 240.0 : 40.0   # s per instance (incl. the window run)
const GLOBAL_BUDGET = isempty(SMOKE) ? 22.5 * 60 : 60.0 # s: no new sample after this (hard cap 25 min)
const RUN_MINUTES = 20.0                  # planned long runs (rule 1)
const MIN_SOLVES = 300                    # rule 1: ≥ 300 solves in 20 min → ≤ 4 s per solve

elapsed_total() = time() - T_START

# --- released sets -----------------------------------------------------------------
"""
Copy of `window_free` (src/reference_neighborhoods.jl) with the time point `t` given
instead of drawn from an RNG (for evenly spaced windows). Frees exactly `n_free_y` pairs.
"""
function calib_window_free(inst::JSSPInstance, incumbent::Schedule, n_free_y::Int, t::Float64)
    ops = operations(inst)
    function dist(op)
        s = incumbent.start[op...]
        e = s + inst.ptime[op...]
        return s <= t < e ? 0.0 : min(abs(t - s), abs(t - e))
    end
    order = sort(eachindex(ops); by = i -> (dist(ops[i]), i))
    pairs_of = Dict(op => Tuple{Op,Op}[] for op in ops)
    for (a, b) in machine_pairs(inst)
        push!(pairs_of[a], (a, b))
        push!(pairs_of[b], (a, b))
    end
    selected = Set{Op}()
    chosen = Tuple{Op,Op}[]
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

"BB core (borders of all critical blocks, Policy B) + slack-ordered fill (rotated by
`offset`) with policyC2_fill up to `L` free y (never above L). Returns (free pairs, |R|)."
function bb_release(inst, ref, F, pairs_of, L, offset)
    core = isempty(F.blocks) ? Op[] : bb_candidates(inst, F, 0)[1][2](0)
    inR = Set(core)
    rest = sort([o for o in operations(inst) if !(o in inR)]; by = o -> (F.slack[o...], o))
    rest = isempty(rest) ? rest : vcat(rest[mod1(offset, length(rest)):end], rest[1:mod1(offset, length(rest))-1])
    R, _, _ = policyC2_fill(vcat(core, rest), pairs_of, L, L)
    Rs = Set(R)
    return [(a, b) for (a, b) in machine_pairs(inst) if a in Rs || b in Rs], length(R)
end

"Number of distinct operations touched by a set of pairs (|R| for random/window releases)."
ops_touched(pairs) = length(unique(vcat(first.(pairs), last.(pairs))))

# --- one capped subproblem solve -----------------------------------------------------
"""
Fixes all y not in `free` to the reference (as in the random_fao loop), warm-starts from
the reference, solves with TimeLimit `tlim` and logs every incumbent via callback.
"""
function solve_sample(model, inst, pairs, yvars, ref, ref_y, free::Set, tlim)
    for idx in eachindex(pairs)
        if pairs[idx] in free
            is_fixed(yvars[idx]) && unfix(yvars[idx])
        else
            fix(yvars[idx], ref_y[idx]; force = true)
        end
    end
    set_warm_start!(model, inst, ref)
    set_attribute(model, "TimeLimit", tlim)
    inc_log = Tuple{Float64,Float64,Int}[]
    function cb(cb_data, cb_where::Cint)
        cb_where == Gurobi.GRB_CB_MIPSOL || return
        t, w, obj = Ref{Cdouble}(), Ref{Cdouble}(), Ref{Cdouble}()
        Gurobi.GRBcbget(cb_data, cb_where, Gurobi.GRB_CB_RUNTIME, t)
        Gurobi.GRBcbget(cb_data, cb_where, Gurobi.GRB_CB_WORK, w)
        Gurobi.GRBcbget(cb_data, cb_where, Gurobi.GRB_CB_MIPSOL_OBJ, obj)
        push!(inc_log, (t[], w[], round(Int, obj[])))
        return
    end
    MOI.set(model, Gurobi.CallbackFunction(), cb)
    optimize!(model)
    status = string(termination_status(model))
    work = MOI.get(model, Gurobi.ModelAttribute("Work"))
    best = has_values(model) ? schedule_from_model(model, inst) : nothing
    return (status = status, time = solve_time(model), work = work,
            bound = objective_bound(model), gap = has_values(model) ? relative_gap(model) : missing,
            best = best, inc_log = inc_log)
end

# --- per instance -----------------------------------------------------------------
function calibrate_instance(name, samples, incs, skipped, timing, instances)
    t_inst = time()
    inst_left() = INSTANCE_BUDGET - (time() - t_inst)
    inst = load_jsplib(name)
    s0 = initial_schedule(inst)
    ms0 = makespan(inst, s0)
    pairs = machine_pairs(inst)
    Y_tot = length(pairs)
    println("\n== $name ($(inst.n)×$(inst.m), N = $(inst.n * inst.m), Y_tot = $Y_tot): window run $(WINDOW_T) s …")
    res = random_fao(inst; time_limit = WINDOW_T, select_free = window_free, ub = ms0)
    ref = res.schedule
    ref_ms = makespan(inst, ref)
    check_schedule(inst, ref; verbose = false)[1] || error("$name: reference infeasible")
    push!(instances, (instance = name, jobs = inst.n, machines = inst.m, N = inst.n * inst.m,
                      Y_tot = Y_tot, initial_makespan = ms0, reference_makespan = ref_ms,
                      window_iterations = res.diagnostics.iterations))
    println("   initial $ms0 → reference $ref_ms ($(res.diagnostics.iterations) window iterations)")

    model = res.model                      # FaO model: same builder and Gurobi configuration
    yvars = [model[:y][p] for p in pairs]
    ref_y = y_values(inst, ref, pairs)
    pairs_of = _pairs_of(inst)
    F = compute_structural_features(inst, ref)
    levels = [L for L in LEVELS if L <= 0.8 * Y_tot]
    foreach(L -> push!(skipped, (instance = name, level = L, generator = "all", reason = "> 0.8·Y_tot")),
            [L for L in LEVELS if L > 0.8 * Y_tot])

    stop_after = nothing                   # index of the last level to test (early stop)
    reason_stop = ""
    for (li, L) in enumerate(levels)
        if stop_after !== nothing && li > stop_after
            push!(skipped, (instance = name, level = L, generator = "all", reason = reason_stop))
            continue
        end
        reps = stop_after !== nothing ? 1 : 2          # extra level after early stop: 3 samples
        rng = MersenneTwister(42)
        n_cap = 0
        n_done = 0
        for gen in GENERATORS, rep in 1:reps
            if inst_left() < T_CAP + 1 || GLOBAL_BUDGET - elapsed_total() < T_CAP + 1
                push!(skipped, (instance = name, level = L, generator = "$gen #$rep",
                                reason = inst_left() < T_CAP + 1 ? "instance budget" : "global budget"))
                continue
            end
            free, nR = if gen == "random"
                p = random_free(inst, ref, [], rng, L); (p, ops_touched(p))
            elseif gen == "window"
                p = calib_window_free(inst, ref, L, (rep - 0.5) / 2 * ref_ms); (p, ops_touched(p))
            else
                bb_release(inst, ref, F, pairs_of, L, rep == 1 ? 1 : length(operations(inst)) ÷ 2)
            end
            r = solve_sample(model, inst, pairs, yvars, ref, ref_y, Set(free), T_CAP)
            best_ms = r.best === nothing ? missing : makespan(inst, r.best)
            feasible = r.best === nothing ? missing : check_schedule(inst, r.best; verbose = false)[1]
            imp = [x for x in r.inc_log if x[3] < ref_ms]
            bestv = isempty(r.inc_log) ? ref_ms : minimum(x[3] for x in r.inc_log)
            first_best = isempty(r.inc_log) ? nothing : r.inc_log[findfirst(x -> x[3] == bestv, r.inc_log)]
            sid = nrow(samples) + 1
            push!(samples, (sample = sid, instance = name, generator = gen, rep = rep, level = L,
                            level_frac = L / Y_tot, free_y = length(free), n_released = nR,
                            status = r.status, time = r.time, work = r.work,
                            time_opt = r.status == "OPTIMAL" ? r.time : missing,
                            work_opt = r.status == "OPTIMAL" ? r.work : missing,
                            bound = r.bound, gap = r.gap, best_makespan = best_ms,
                            reference_makespan = ref_ms, delta = ismissing(best_ms) ? missing : ref_ms - best_ms,
                            t_first_imp = isempty(imp) ? missing : imp[1][1],
                            w_first_imp = isempty(imp) ? missing : imp[1][2],
                            t_best = (first_best === nothing || bestv >= ref_ms) ? missing : first_best[1],
                            w_best = (first_best === nothing || bestv >= ref_ms) ? missing : first_best[2],
                            feasible = feasible); promote = true)
            foreach(x -> push!(incs, (sample = sid, time = x[1], work = x[2], makespan = x[3])), r.inc_log)
            n_done += 1
            n_cap += r.status != "OPTIMAL"
            println("   L=$(rpad(L, 5)) $(rpad(gen, 6)) #$rep free=$(rpad(length(free), 5)) $(rpad(r.status, 11)) ",
                    "t=$(round(r.time, digits = 2)) s  Δ=$(ismissing(best_ms) ? "–" : ref_ms - best_ms)",
                    feasible === false ? "  INFEASIBLE" : "")
        end
        CSV.write(joinpath(OUT, "samples.csv"), samples)
        if stop_after === nothing && n_done == 6 && n_cap >= 4
            stop_after = li + 1                        # later levels are logged as skipped above
            reason_stop = "early stop (≥ 4/6 hit T_cap at $L)"
        end
    end
    push!(timing, (instance = name, seconds = round(time() - t_inst, digits = 1)))
end

# --- analysis ----------------------------------------------------------------------
"Metrics of a group of samples for limit T (from the logged incumbents only)."
function metrics(sub::DataFrame, incs::DataFrame, T)
    n = nrow(sub)
    n == 0 && return nothing
    topt = [ismissing(x) ? Inf : x for x in sub.time_opt]
    used = min.(T, topt)
    dT = map(eachrow(sub)) do s
        rows = incs[(incs.sample .== s.sample) .& (incs.time .<= T), :]
        isempty(rows) ? 0 : max(0, s.reference_makespan - minimum(rows.makespan))
    end
    first_imp = [ismissing(x) ? Inf : x for x in sub.t_first_imp]
    return (n = n, p_opt = mean(topt .<= T), p_imp = mean(first_imp .<= T),
            E_delta = mean(dT), E_used = mean(used), rate = mean(dT) / mean(used),
            median_t_opt = median(topt), work_per_s = median(sub.work ./ max.(sub.time, 1e-9)))
end

function analyse(samples::DataFrame, incs::DataFrame, inst_df::DataFrame)
    rows = DataFrame()
    for g in groupby(samples, [:instance, :level]), T in T_GRID, gen in ["all"; GENERATORS]
        sub = gen == "all" ? DataFrame(g) : DataFrame(g[g.generator .== gen, :])
        m = metrics(sub, incs, T)
        m === nothing && continue
        push!(rows, (instance = g.instance[1], level = g.level[1], level_frac = g.level_frac[1],
                     generator = gen, T = T, m...))
    end
    return rows
end

fmt(x) = x isa AbstractFloat ? (isinf(x) ? ">$(Int(T_CAP))" : string(round(x, digits = 3))) : string(x)

"Decision rules 1–5 of the task, applied exactly; returns the report text."
function recommend(summary::DataFrame, inst_df::DataFrame, skipped::DataFrame)
    io = IOBuffer()
    S = summary[summary.generator .== "all", :]
    names = inst_df.instance
    lines(args...) = println(io, args...)
    # Rule 1: T_sub
    lines("Rule 1 – T_sub (≥ $MIN_SOLVES solves in $RUN_MINUTES min ⇒ E[time used] ≤ $(RUN_MINUTES * 60 / MIN_SOLVES) s at the typical size)")
    best_T = nothing; best_val = -Inf; best_sizes = Dict{String,Int}()
    for T in T_GRID
        per = map(names) do nm
            d = S[(S.instance .== nm) .& (S.T .== T), :]
            isempty(d) && return nothing
            i = argmax(collect(zip(d.rate, -d.level)))          # max rate, ties: smaller size
            (nm, d.level[i], d.rate[i], d.E_used[i])
        end
        per = filter(!isnothing, per)
        isempty(per) && continue
        med_rate = median(x[3] for x in per)
        med_used = median(x[4] for x in per)
        ok = med_used <= RUN_MINUTES * 60 / MIN_SOLVES
        lines("  T = $T s: median rate at best size = $(fmt(med_rate)) /s, median E[time used] = $(fmt(med_used)) s",
              ok ? "" : "  → excluded (> 4 s)", "   best sizes: ", join(["$(x[1])=$(x[2])" for x in per], ", "))
        if ok && med_rate > best_val
            best_T, best_val = T, med_rate
            best_sizes = Dict(x[1] => x[2] for x in per)
        end
    end
    best_T === nothing && (lines("  CONFLICT: no T satisfies the solve-count condition"); return String(take!(io)), nothing)
    wps = median(skipmissing(S[S.T .== best_T, :work_per_s]))
    lines("  ⇒ T_sub = $best_T s  (≈ WorkLimit $(round(wps * best_T, digits = 2)) work units; median $(round(wps, digits = 3)) work units/s)")

    # Rule 2: y_max
    lines("\nRule 2 – y_max: largest size with p_opt(T_sub) ≥ 0.5 on every instance (size > 0.8·Y_tot: not applicable; not measured: counts as fail)")
    y_max = nothing
    for L in LEVELS
        verdict = map(eachrow(inst_df)) do r
            L > 0.8 * r.Y_tot && return (r.instance, "n/a", true)
            d = S[(S.instance .== r.instance) .& (S.level .== L) .& (S.T .== best_T), :]
            isempty(d) && return (r.instance, "not measured", false)
            (r.instance, fmt(d.p_opt[1]), d.p_opt[1] >= 0.5)
        end
        ok = all(v -> v[3], verdict) && any(v -> v[2] != "n/a", verdict)
        lines("  $L: ", join(["$(v[1])=$(v[2])" for v in verdict], ", "), ok ? "  ✓" : "  ✗")
        ok && (y_max = L)
    end
    y_max === nothing && (lines("  CONFLICT: no size satisfies rule 2"); return String(take!(io)), nothing)

    # Rule 3: y_min
    lines("\nRule 3 – y_min = max(a, b, c)")
    a = nothing
    for L in LEVELS
        sub = S[(S.level .== L) .& (S.T .== best_T), :]
        isempty(sub) && continue
        # median solve time over all samples at this size: per-instance medians of time used
        med = median(min.(best_T, sub.median_t_opt))
        if med >= 0.05
            a = L; lines("  a) smallest size with median solve time ≥ 0.05 s: $L (median over instances $(fmt(med)) s)"); break
        end
    end
    a === nothing && lines("  a) no size with median solve time ≥ 0.05 s")
    b = nothing
    for L in LEVELS
        cnt = count(nm -> any((S.instance .== nm) .& (S.level .== L) .& (S.T .== best_T) .& (S.p_imp .> 0)), names)
        if cnt >= 3
            b = L; lines("  b) smallest size with p_imp(T_sub) > 0 on ≥ 3 instances: $L ($cnt instances)"); break
        end
    end
    b === nothing && lines("  b) no size with p_imp(T_sub) > 0 on ≥ 3 instances")
    c = y_max ÷ 4
    lines("  c) y_max / 4 = $c")
    parts = filter(!isnothing, [a, b, c])
    y_min = maximum(parts)
    y_min > y_max && lines("  CONFLICT: y_min = $y_min > y_max = $y_max (closest feasible choice: y_min = y_max = $y_max)")
    y_min = min(y_min, y_max)

    # Rule 4: absolute vs relative
    lines("\nRule 4 – absolute vs. relative (best sizes at T_sub)")
    bs = [best_sizes[nm] for nm in names if haskey(best_sizes, nm)]
    fr = [best_sizes[nm] / inst_df.Y_tot[findfirst(==(nm), names)] for nm in names if haskey(best_sizes, nm)]
    cv(v) = std(v) / mean(v)
    lines("  best sizes: ", join(["$nm=$(best_sizes[nm]) ($(round(100 * best_sizes[nm] / inst_df.Y_tot[findfirst(==(nm), names)], digits = 2)) % of Y_tot)" for nm in names if haskey(best_sizes, nm)], ", "))
    lines("  coefficient of variation: absolute $(fmt(cv(bs))), relative $(fmt(cv(fr)))")
    relative = cv(fr) < cv(bs)
    lines("  ⇒ ", relative ? "relative range (fraction of Y_tot) aligns better" : "absolute range (free y) aligns better")
    lines("\nRecommendation")
    lines("  T_sub = $best_T s, WorkLimit ≈ $(round(wps * best_T, digits = 2)) (recommended: WorkLimit, reproducible)")
    if relative
        f_min = median(y_min ./ inst_df.Y_tot); f_max = median(y_max ./ inst_df.Y_tot)
        lines("  range relative to Y_tot: [$(round(100 * f_min, digits = 2)) %, $(round(100 * f_max, digits = 2)) %]")
    else
        lines("  range: [y_min, y_max] = [$y_min, $y_max] free y")
    end
    lines("  per instance (capped at FREE_CAP = 75 % of Y_tot as in the loop) and expected solves in $RUN_MINUTES min:")
    for r in eachrow(inst_df)
        lo, hi = relative ? (round(Int, median(y_min ./ inst_df.Y_tot) * r.Y_tot), round(Int, median(y_max ./ inst_df.Y_tot) * r.Y_tot)) : (y_min, y_max)
        cap = floor(Int, 0.75 * r.Y_tot)
        d = S[(S.instance .== r.instance) .& (S.T .== best_T) .& (S.level .<= min(hi, cap)) .& (S.level .>= min(lo, cap)), :]
        eu = isempty(d) ? missing : mean(d.E_used)
        lines("    $(rpad(r.instance, 6)) [$(min(lo, cap)), $(min(hi, cap))]  E[time used] ≈ $(ismissing(eu) ? "–" : fmt(eu)) s  ⇒ ≤ ",
              ismissing(eu) ? "–" : round(Int, RUN_MINUTES * 60 / eu), " solves (upper bound, loop overhead not included)")
    end
    return String(take!(io)), (T_sub = best_T, y_min = y_min, y_max = y_max, relative = relative)
end

function plots(summary::DataFrame, samples::DataFrame, inst_df::DataFrame)
    S = summary[summary.generator .== "all", :]
    p = plot(; xscale = :log10, yscale = :log10, xlabel = "free y", ylabel = "median time to optimality [s]",
             title = "Time to optimality vs. size (censored at T_cap = $(Int(T_CAP)) s)", legend = :topleft,
             size = (800, 500), dpi = 120, titlefontsize = 11)
    for nm in inst_df.instance
        d = sort(S[(S.instance .== nm) .& (S.T .== T_CAP), :], :level)
        plot!(p, d.level, clamp.(d.median_t_opt, 1e-3, T_CAP); marker = :circle, lw = 2, label = nm)
    end
    hline!(p, [T_CAP]; ls = :dash, color = :black, label = "T_cap")
    savefig(p, joinpath(OUT, "time_to_opt.png"))
    for (col, file, lbl) in ((:p_imp, "p_imp.png", "p_imp(T)"), (:rate, "rate.png", "rate(T) [makespan units / s]"))
        panels = map(T_GRID) do T
            q = plot(; xscale = :log10, title = "T = $T s", xlabel = "free y", ylabel = lbl, titlefontsize = 10,
                     legend = T == T_GRID[1] ? :topright : false)
            for nm in inst_df.instance
                d = sort(S[(S.instance .== nm) .& (S.T .== T), :], :level)
                plot!(q, d.level, d[!, col]; marker = :circle, lw = 2, label = nm)
            end
            q
        end
        savefig(plot(panels...; layout = (2, 3), size = (1200, 700), dpi = 110), joinpath(OUT, file))
    end
end

function main()
    mkpath(OUT)
    keep_awake()
    samples, incs, skipped, timing, instances = DataFrame(), DataFrame(), DataFrame(), DataFrame(), DataFrame()
    println("Calibration: $(join(INSTANCES, ", ")) (+ $EXTRA if time remains), T_cap = $T_CAP s, ",
            "budget $(INSTANCE_BUDGET) s per instance, global $(GLOBAL_BUDGET / 60) min")
    random_fao(load_jsplib("ft06"); time_limit = 1.0, select_free = window_free)   # JIT warm-up
    for nm in INSTANCES
        if GLOBAL_BUDGET - elapsed_total() < WINDOW_T + 2 * T_CAP
            push!(skipped, (instance = nm, level = 0, generator = "all", reason = "global budget (instance not started)"))
            continue
        end
        calibrate_instance(nm, samples, incs, skipped, timing, instances)
    end
    if GLOBAL_BUDGET - elapsed_total() >= INSTANCE_BUDGET
        calibrate_instance(EXTRA, samples, incs, skipped, timing, instances)
    else
        push!(skipped, (instance = EXTRA, level = 0, generator = "all",
                        reason = "global budget (only $(round(Int, GLOBAL_BUDGET - elapsed_total())) s left)"))
    end
    push!(timing, (instance = "TOTAL (incl. startup)", seconds = round(elapsed_total(), digits = 1)))
    CSV.write(joinpath(OUT, "samples.csv"), samples)
    CSV.write(joinpath(OUT, "incumbents.csv"), incs)
    CSV.write(joinpath(OUT, "instances.csv"), instances)
    CSV.write(joinpath(OUT, "skipped.csv"), isempty(skipped) ? DataFrame(instance = String[], level = Int[], generator = String[], reason = String[]) : skipped)
    summary = analyse(samples, incs, instances)
    CSV.write(joinpath(OUT, "summary.csv"), summary)
    text, rec = recommend(summary, instances, isempty(skipped) ? DataFrame() : skipped)
    plots(summary, samples, instances)
    push!(timing, (instance = "TOTAL (incl. analysis)", seconds = round(elapsed_total(), digits = 1)))
    CSV.write(joinpath(OUT, "timing.csv"), timing)
    write(joinpath(OUT, "recommendation.txt"), text)
    println("\n", text, "\nResults: ", OUT, "  (total $(round(elapsed_total() / 60, digits = 1)) min)")
end

end # module JSSPLabCalibration

JSSPLabCalibration.main()
