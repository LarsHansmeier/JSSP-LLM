# =============================================================================
# run_anytime.jl — Document, plot & compare anytime performance
# =============================================================================
#
# Methods (selected via --method=..., default: both):
#   gurobi      full MIP from src/model.jl (warm start = initial_schedule, Gurobi Seed = 42)
#   Both methods use Big-M = makespan of initial_schedule (build_model keyword `ub`).
#   random_fao  Random Fix-and-Optimize from src/matheuristic.jl (baseline for policies),
#               seed 42 by default (several seeds possible via --seeds)
#   both        both methods per instance, one after the other + comparison plot
#
# Run:
#   julia --project=. scripts/run_anytime.jl                          # both, DEFAULT_INSTANCES, 60 s
#   julia --project=. scripts/run_anytime.jl 30 ft10 la16             # 30 s, own instances
#   julia --project=. scripts/run_anytime.jl 60 prefix:la             # all la instances
#   julia --project=. scripts/run_anytime.jl --method=gurobi 60       # Gurobi only
#   julia --project=. scripts/run_anytime.jl --method=random_fao 60   # Random FaO only
# Further options:
#   --threads=4                         both methods
#   --no-warmstart                      gurobi only
#   --seeds=42  (or 42:46, 42,43)       random_fao only
#   --nfree=200:800 --sub-time=1 --no-accept-equal   (number of free y variables per
#                       subproblem: range 200:800, drawn per iteration, or fixed, e.g. --nfree=400;
#                       capped at 75 % of all machine pairs; defaults shown)
# From the REPL: JSSPLabAnytime.run_anytime(["ft06"]; time_limit = 10, n_free_min = 100, n_free_max = 300)
#
# Gurobi output in results/gurobi/:
#   <name>.csv                    history: time, incumbent, bound, gap
#   plots/<name>.png              incumbent & bound over time
# Random FaO output in results/random_fao/ (per seed k):
#   <name>_seed<k>.csv            history
#   <name>_seed<k>_iterations.csv iteration log
#   <name>_diagnostics.txt        diagnostics per seed + mean
#   <name>_median.csv             median (and min/max) over seeds – basis of the plots
#   plots/<name>.png              median incumbent, min/max, static lower bound
#   summary_seeds.csv             one row per (instance, seed)
# For both methods:
#   summary.csv                   final values (FaO: mean/min/max over seeds), diagnostics
#   plots/overview.png            primal gap to the best known solution over time
#   params.json                   all parameters, Gurobi and Julia version
# With --method=both additionally in results/comparison/:
#   <name>.png, summary.csv, overview.png
#
# Metrics: gap_own_bound = (incumbent − own bound)/incumbent
#          (Gurobi: MIP bound; FaO: static lower bound lower_bound)
#          primal_gap_bks = (incumbent − best known)/best known (JSPLIB)
#
# Time axis: wall-clock time since method start, incl. start solution and model
# build – the same zero point for both methods; the time limit applies per run
# to everything. The machine is kept awake during the run (keep_awake / caffeinate).
# =============================================================================

import Pkg
const _ROOT = normpath(joinpath(@__DIR__, ".."))
Base.active_project() == joinpath(_ROOT, "Project.toml") || Pkg.activate(_ROOT; io = devnull)
ENV["GKSwstype"] = "100"     # GR without window (file output)

module JSSPLabAnytime

using CSV, DataFrames, Plots, Dates, Statistics
include(joinpath(@__DIR__, "..", "src", "JSSPLab.jl"))
using .JSSPLab

const ROOT = normpath(joinpath(@__DIR__, ".."))
const METHODS = (:gurobi, :random_fao)
method_dir(method::Symbol) = joinpath(ROOT, "results", string(method))
const COMPARISON_DIR = joinpath(ROOT, "results", "comparison")
const METHOD_STYLE = Dict(:gurobi => (color = :royalblue, label = "Gurobi (volles MIP)"),
                          :random_fao => (color = :seagreen, label = "Random FaO"))

# Small but diverse default selection (size 6×6 … 50×10, easy to hard)
const DEFAULT_INSTANCES = ["ft06", "la01", "la16", "ft10", "abz5", "orb01", "la21", "ta01", "swv12"]

# --- Arguments -----------------------------------------------------------------
"Seeds from \"42:46\" or \"42,43\"."
parse_seeds(s) = occursin(':', s) ? collect(parse(Int, split(s, ':')[1]):parse(Int, split(s, ':')[2])) :
                                    parse.(Int, split(s, ','))

"Splits the command line into instance names and keyword arguments for run_anytime."
function parse_args(args)
    names, kw = String[], Dict{Symbol,Any}()
    for a in args
        key, val = occursin('=', a) ? split(a, '='; limit = 2) : (a, "")
        if a == "--no-warmstart"
            kw[:warm_start] = false
        elseif a == "--no-accept-equal"
            kw[:accept_equal] = false
        elseif key == "--method"
            kw[:method] = Symbol(val)
        elseif key == "--seeds" || key == "--seed"
            kw[:seeds] = parse_seeds(val)
        elseif key == "--threads"
            kw[:threads] = parse(Int, val)
        elseif key == "--sub-time"
            kw[:sub_time_limit] = parse(Float64, val)
        elseif key == "--nfree"
            lo, hi = occursin(':', val) ? split(val, ':') : (val, val)   # "200:800" or "400"
            kw[:n_free_min], kw[:n_free_max] = parse(Int, lo), parse(Int, hi)
        elseif startswith(a, "--")
            error("Unbekannte Option $a")
        elseif tryparse(Float64, a) !== nothing
            kw[:time_limit] = parse(Float64, a)
        elseif a == "all"
            append!(names, list_jsplib())
        elseif startswith(a, "prefix:")
            append!(names, list_jsplib(prefix = a[8:end]))
        else
            push!(names, a)
        end
    end
    isempty(names) && (names = copy(DEFAULT_INSTANCES))
    return unique(names), kw
end

# --- Documentation (CSV, JSON, TXT) ----------------------------------------------
_relgap(i, b) = ismissing(i) || ismissing(b) || i == 0 ? missing : (i - b) / i

function trace_dataframe(trace)
    df = DataFrame(trace)
    df.gap = _relgap.(df.incumbent, df.bound)
    return df
end

"""
Median (and min/max) of several step curves on the union of all time points.
Each curve is valid from its first to its last point (after that: final value).
"""
function aggregate_traces(dfs::Vector{DataFrame})
    times = sort!(unique(vcat([df.time for df in dfs]...)))
    value_at(df, t) = df.incumbent[something(findlast(<=(t), df.time), 1)]
    vals = [[value_at(df, t) for df in dfs] for t in times]
    bound = dfs[1].bound[1]                          # static, identical for all seeds
    out = DataFrame(time = times, incumbent = median.(vals), lo = minimum.(vals),
                    hi = maximum.(vals), bound = fill(bound, length(times)))
    out.gap = _relgap.(out.incumbent, out.bound)
    return out
end

_json(x::AbstractString) = "\"" * escape_string(x) * "\""
_json(x::Symbol) = _json(string(x))
_json(x::Union{Real,Bool}) = string(x)
_json(x::Nothing) = "null"
_json(x::Union{AbstractVector,Tuple,AbstractRange}) = "[" * join(_json.(collect(x)), ", ") * "]"

"Writes the parameters (flat Dict) as a JSON file."
function write_params(path, params::AbstractDict)
    open(path, "w") do io
        println(io, "{")
        ks = sort!(collect(keys(params)); by = string)
        for (i, k) in enumerate(ks)
            println(io, "  ", _json(string(k)), ": ", _json(params[k]), i < length(ks) ? "," : "")
        end
        println(io, "}")
    end
end

function gurobi_version()
    a, b, c = Ref{Cint}(), Ref{Cint}(), Ref{Cint}()
    JSSPLab.Gurobi.GRBversion(a, b, c)
    return "$(a[]).$(b[]).$(c[])"
end

_fmt(x) = ismissing(x) ? "–" : x isa AbstractFloat ? string(round(x, digits = 4)) : string(x)

"Diagnostics file: one line per seed, plus means."
function write_diagnostics(path, name, seed_diags)
    keys_ = collect(keys(first(values(seed_diags))))
    open(path, "w") do io
        println(io, "# Random FaO – Diagnose für $name")
        println(io, "# Spalten: ", join(keys_, ", "))
        for seed in sort!(collect(keys(seed_diags)))
            d = seed_diags[seed]
            println(io, "seed $seed: ", join(["$k=$(_fmt(d[k]))" for k in keys_], "  "))
        end
        avg(k) = (v = collect(skipmissing(d[k] for d in values(seed_diags))); isempty(v) ? missing : mean(v))
        println(io, "mittel:  ", join(["$k=$(_fmt(avg(k)))" for k in keys_], "  "))
    end
end

# --- Visualisation ------------------------------------------------------------
"Step curve without missing values; time 0 is raised for the log axis."
function step_series(df, col)
    rows = .!ismissing.(df[!, col])
    return max.(df.time[rows], 1e-3), collect(skipmissing(df[!, col]))
end

bound_label(method) = method == :random_fao ? "statische untere Schranke" : "Bound (untere Schranke)"

"Reference lines: optimum or best known UB, and start solution."
function add_reference_lines!(p, info, initial_ms)
    ref = something(info.optimum, info.upper, missing)
    if !ismissing(ref)
        lbl = info.optimum === nothing ? "Best known UB ($ref)" : "Optimum ($ref)"
        hline!(p, [ref]; ls = :dash, color = :black, lw = 1.5, label = lbl)
    end
    initial_ms === nothing ||
        hline!(p, [initial_ms]; ls = :dot, color = :gray, lw = 1.5, label = "Startlösung ($initial_ms)")
    return p
end

"Draws the incumbent of a method (with several seeds: median + min/max)."
function add_method_series!(p, df, method; color, prefix = "", n_seeds = 1)
    x, y = step_series(df, :incumbent)
    inc_label = n_seeds > 1 ? "Incumbent (Median, $n_seeds Seeds)" : "Incumbent (beste Lösung)"
    isempty(x) || plot!(p, x, y; seriestype = :steppost, lw = 2.5, color = color,
                        label = prefix * inc_label)
    if hasproperty(df, :lo) && n_seeds > 1
        for (col, lbl) in ((:lo, prefix * "Min/Max über Seeds"), (:hi, ""))
            x, y = step_series(df, col)
            plot!(p, x, y; seriestype = :steppost, lw = 1, alpha = 0.6, color = color,
                  label = lbl)
        end
    end
    return p
end

function add_bound_series!(p, df, method; color, prefix = "", ls = :solid)
    x, y = step_series(df, :bound)
    isempty(x) || plot!(p, x, y; seriestype = :steppost, lw = 2, ls = ls, color = color,
                        label = prefix * bound_label(method))
    return p
end

function plot_trace(name, df, info, title_suffix, initial_ms, xmax, method; n_seeds = 1)
    p = plot(; title = "$name ($(info.jobs)×$(info.machines)) – $method – $title_suffix",
             xlabel = "Zeit [s] (log)", ylabel = "Makespan", xscale = :log10,
             xlims = (1e-3, xmax * 1.05),
             legend = :outerbottom, legend_columns = 2, size = (800, 560),
             titlefontsize = 11, dpi = 120)
    add_method_series!(p, df, method; color = :royalblue, n_seeds = n_seeds)
    add_bound_series!(p, df, method; color = :darkorange)
    return add_reference_lines!(p, info, initial_ms)
end

"Overview: primal gap (incumbent − best known)/best known over time, all instances."
function plot_overview(traces, time_limit, method)
    what = method == :random_fao ? " (Median über Seeds)" : ""
    p = plot(; title = "Anytime-Performance $method$what – Primal Gap zur besten bekannten Lösung",
             xlabel = "Zeit [s] (log)", ylabel = "Primal Gap [%]", xscale = :log10,
             xlims = (1e-3, time_limit * 1.05), legend = :outerright,
             size = (950, 520), titlefontsize = 11, dpi = 120,
             palette = :tab20)                           # easily distinguishable colours
    for (name, df, ref) in traces
        ismissing(ref) && continue                       # ta71–80: no reference value
        x, y = step_series(df, :incumbent)
        isempty(x) && continue
        plot!(p, x, 100 .* (y .- ref) ./ ref; seriestype = :steppost, lw = 2, label = name)
    end
    return p
end

"Comparison of both methods on one instance (same time zero point)."
function plot_comparison(name, info, dfs::AbstractDict, finals::AbstractDict, initial_ms,
                         xmax; n_seeds = 1)
    title = join(["$m $(finals[m])" for m in METHODS if haskey(finals, m)], " vs. ")
    p = plot(; title = "$name ($(info.jobs)×$(info.machines)) – $title",
             xlabel = "Zeit [s] (log)", ylabel = "Makespan", xscale = :log10,
             xlims = (1e-3, xmax * 1.05),
             legend = :outerbottom, legend_columns = 2, size = (820, 620),
             titlefontsize = 11, dpi = 120)
    for m in METHODS
        haskey(dfs, m) || continue
        st = METHOD_STYLE[m]
        add_method_series!(p, dfs[m], m; color = st.color, prefix = "$(st.label): ",
                           n_seeds = m == :random_fao ? n_seeds : 1)
        add_bound_series!(p, dfs[m], m; color = st.color, prefix = "$(st.label): ", ls = :dash)
    end
    return add_reference_lines!(p, info, initial_ms)
end

"Overview: final primal gap per instance (FaO: mean, error bars min–max)."
function plot_comparison_overview(cmp::DataFrame)
    df = filter(r -> !ismissing(r.gurobi_primal_gap_bks) &&
                     !ismissing(r.random_fao_primal_gap_bks_mean), cmp)
    n = nrow(df)
    p = plot(; title = "Primal Gap zur besten bekannten Lösung am Ende",
             ylabel = "Primal Gap [%]", xticks = (1:n, df.instance), xrotation = 45,
             legend = :topleft, size = (max(600, 70n + 200), 480), titlefontsize = 11,
             dpi = 120, bottom_margin = 5Plots.mm)
    n == 0 && return p
    bar!(p, (1:n) .- 0.2, 100 .* df.gurobi_primal_gap_bks; bar_width = 0.4,
         color = METHOD_STYLE[:gurobi].color, label = METHOD_STYLE[:gurobi].label,
         linecolor = :white)
    m = 100 .* df.random_fao_primal_gap_bks_mean
    hi = 100 .* df.random_fao_primal_gap_bks_max
    err = (m .- 100 .* df.random_fao_primal_gap_bks_min, hi .- m)
    bar!(p, (1:n) .+ 0.2, m; bar_width = 0.4, yerror = err, color = METHOD_STYLE[:random_fao].color,
         label = METHOD_STYLE[:random_fao].label * " (Mittel, Fehlerbalken Min–Max)",
         linecolor = :black, linewidth = 0.5, markerstrokecolor = :black, markerstrokewidth = 1.5)
    ymax = maximum(vcat(hi, 100 .* df.gurobi_primal_gap_bks, [1.0]))
    ylims!(p, (0, 1.2 * ymax))
    return p
end

# --- Solving -------------------------------------------------------------------
"""
Solves an instance with the chosen method (FaO: with one seed). Both return
the same result format with wall-clock times since method start `t0`.
"""
function run_method(method::Symbol, inst; time_limit, threads, warm_start, seed = DEFAULT_SEED,
                    sub_time_limit, n_free_min, n_free_max, accept_equal, kw...)
    t0 = time()
    ub = makespan(inst, initial_schedule(inst))       # Big-M for both methods
    if method == :gurobi
        ws = warm_start ? initial_schedule(inst) : nothing
        res = solve_model(inst; time_limit = time_limit, threads = threads, silent = true,
                          warm_start = ws, wall_start = t0, ub = ub)
        return res, ws === nothing ? nothing : makespan(inst, ws)
    elseif method == :random_fao
        res = random_fao(inst; time_limit = time_limit, sub_time_limit = sub_time_limit,
                         n_free_min = n_free_min, n_free_max = n_free_max, seed = seed,
                         threads = threads, accept_equal = accept_equal, ub = ub,
                         wall_start = t0)
        return res, res.trace[1].incumbent
    end
    error("Unbekannte Methode $method, erlaubt: $(join(METHODS, ", "))")
end

"Solve once briefly so JIT compilation does not distort the first instance."
function warmup(method; kw...)
    inst = read_instance(joinpath(ROOT, "data", "instances", "example3x3.txt"))
    run_method(method, inst; kw..., time_limit = 0.5)
    return nothing
end

"Check validity (independent checker)."
function verify(inst, res, label)
    ok = res.schedule !== nothing &&
         check_schedule(inst, res.schedule; claimed_makespan = res.incumbent, verbose = false)[1]
    ok || res.schedule === nothing || @warn "$label: Lösung vom Checker als UNZULÄSSIG erkannt!"
    return ok
end

function time_to_best(df, inc)
    ismissing(inc) && return missing
    i = findfirst(r -> !ismissing(r.incumbent) && r.incumbent == inc, eachrow(df))
    return i === nothing ? missing : df.time[i]
end

function print_line(name, label, res, ref; extra = "", ok = true)
    println(rpad(name, 7), rpad(label, 18), rpad(string(res.status), 11),
            " Inc=", lpad(string(res.incumbent), 5),
            "  Bound=", lpad(ismissing(res.bound) ? "–" : string(res.bound), 5),
            "  Gap=", ismissing(res.gap) ? "    –   " : lpad(string(round(100res.gap, digits = 2)), 6) * " %",
            "  Ref=", ismissing(ref) ? "–" : ref,
            "  Zeit=", round(res.runtime, digits = 1), " s", extra, "  ", ok ? "✓" : "✗")
end

"Gurobi on one instance: history, plot, summary row."
function process_gurobi(name; kw...)
    inst, info = load_jsplib(name), jsplib_info(name)
    outdir = method_dir(:gurobi)
    ref = something(info.optimum, info.upper, missing)
    res, initial_ms = run_method(:gurobi, inst; kw...)
    ok = verify(inst, res, "$name [gurobi]")

    df = trace_dataframe(res.trace)
    CSV.write(joinpath(outdir, "$name.csv"), df)
    suffix = ismissing(res.gap) ? "Makespan $(res.incumbent)" : "Gap $(round(100res.gap, digits = 2)) %"
    savefig(plot_trace(name, df, info, suffix, initial_ms, max(kw[:time_limit], res.runtime), :gurobi),
            joinpath(outdir, "plots", "$name.png"))
    print_line(name, "gurobi", res, ref; ok = ok)

    row = (instance = name, method = "gurobi", jobs = inst.n, machines = inst.m,
           time_limit = kw[:time_limit], warm_start = kw[:warm_start], status = string(res.status),
           initial = something(initial_ms, missing), incumbent = res.incumbent, bound = res.bound,
           gap_own_bound = res.gap,
           primal_gap_bks = ismissing(ref) || ismissing(res.incumbent) ? missing : (res.incumbent - ref) / ref,
           runtime = res.runtime, build_time = res.build_time, time_to_best = time_to_best(df, res.incumbent),
           optimum = something(info.optimum, missing), best_known_lb = something(info.lower, missing),
           best_known_ub = something(info.upper, missing), feasible = ok, timestamp = string(now()))
    return row, df, initial_ms, res.runtime
end

_stats(v) = (v = collect(skipmissing(v)); isempty(v) ? (missing, missing, missing) :
                                                       (mean(v), minimum(v), maximum(v)))

"Random FaO on one instance with all seeds: CSVs per seed, median history, plot, rows."
function process_fao(name; seeds, kw...)
    inst, info = load_jsplib(name), jsplib_info(name)
    outdir = method_dir(:random_fao)
    ref = something(info.optimum, info.upper, missing)
    seed_rows, dfs, diags = NamedTuple[], DataFrame[], Dict{Int,Any}()
    initial_ms, tmax = nothing, 0.0
    for seed in seeds
        res, ini = run_method(:random_fao, inst; kw..., seed = seed)
        initial_ms = ini
        ok = verify(inst, res, "$name [random_fao, seed $seed]")
        df = trace_dataframe(res.trace)
        CSV.write(joinpath(outdir, "$(name)_seed$(seed).csv"), df)
        CSV.write(joinpath(outdir, "$(name)_seed$(seed)_iterations.csv"), DataFrame(res.iterations))
        push!(dfs, df)
        diags[seed] = res.diagnostics
        tmax = max(tmax, res.runtime)
        d = res.diagnostics
        print_line(name, "random_fao s$seed", res, ref; ok = ok,
                   extra = "  It=$(d.iterations)  frei(med)=$(_fmt(d.median_n_free_y))")
        push!(seed_rows, (instance = name, seed = seed, status = res.status, initial = ini,
                          incumbent = res.incumbent, bound = res.bound, gap_own_bound = res.gap,
                          primal_gap_bks = ismissing(ref) ? missing : (res.incumbent - ref) / ref,
                          runtime = res.runtime, build_time = res.build_time,
                          time_to_best = time_to_best(df, res.incumbent), feasible = ok, d...))
    end
    write_diagnostics(joinpath(outdir, "$(name)_diagnostics.txt"), name, diags)

    med = aggregate_traces(dfs)
    CSV.write(joinpath(outdir, "$(name)_median.csv"), med)
    inc = [r.incumbent for r in seed_rows]
    suffix = "Median $(median(inc)) (Min $(minimum(inc)), Max $(maximum(inc)), $(length(seeds)) Seeds)"
    savefig(plot_trace(name, med, info, suffix, initial_ms, max(kw[:time_limit], tmax), :random_fao;
                       n_seeds = length(seeds)),
            joinpath(outdir, "plots", "$name.png"))

    col(k) = [r[k] for r in seed_rows]
    inc_mean, inc_min, inc_max = _stats(col(:incumbent))
    gap_mean, gap_min, gap_max = _stats(col(:gap_own_bound))
    pg_mean, pg_min, pg_max = _stats(col(:primal_gap_bks))
    dmean(k) = _stats(col(k))[1]
    row = (instance = name, method = "random_fao", jobs = inst.n, machines = inst.m,
           time_limit = kw[:time_limit], seeds = join(seeds, " "), n_seeds = length(seeds),
           n_optimal = count(==("OPTIMAL"), col(:status)), initial = initial_ms,
           incumbent_mean = inc_mean, incumbent_min = inc_min, incumbent_max = inc_max,
           incumbent_median = median(inc), bound = seed_rows[1].bound,
           gap_own_bound_mean = gap_mean, gap_own_bound_min = gap_min, gap_own_bound_max = gap_max,
           primal_gap_bks_mean = pg_mean, primal_gap_bks_min = pg_min, primal_gap_bks_max = pg_max,
           runtime_mean = dmean(:runtime), time_to_best_mean = dmean(:time_to_best),
           iterations_mean = dmean(:iterations), share_time_limit_mean = dmean(:share_time_limit),
           median_n_free_y_mean = dmean(:median_n_free_y), median_sub_time_mean = dmean(:median_sub_time),
           median_setup_time_mean = dmean(:median_setup_time), share_improved_mean = dmean(:share_improved),
           share_accepted_mean = dmean(:share_accepted),
           time_last_improvement_mean = dmean(:time_last_improvement),
           optimum = something(info.optimum, missing), best_known_lb = something(info.lower, missing),
           best_known_ub = something(info.upper, missing), feasible = all(col(:feasible)),
           timestamp = string(now()))
    return row, med, initial_ms, tmax, seed_rows
end

# --- Main program ----------------------------------------------------------------
"Writes the parameter file for a method (all parameters, versions)."
function write_method_params(method, names; kw...)
    params = Dict{Symbol,Any}(:method => method, :instances => names,
                              :timestamp => string(now()), :julia => string(VERSION),
                              :gurobi => gurobi_version(), :time_base => "wall clock incl. model build",
                              :big_m => "initial_makespan")
    for (k, v) in kw
        params[k] = v isa AbstractRange ? collect(v) : v
    end
    if method == :random_fao
        merge!(params, Dict(:rng => "MersenneTwister", :select_free => "random_free",
                            :start_solution => "initial_schedule (mwkr)", :free_cap => FREE_CAP,
                            :bound => "lower_bound (statisch: trivial & Ein-Maschinen-Relaxierung)"))
    else
        params[:start_solution] = kw[:warm_start] ? "initial_schedule (mwkr) als MIP-Start" : "keine"
        params[:gurobi_seed] = DEFAULT_SEED
    end
    write_params(joinpath(method_dir(method), "params.json"), params)
end

"Final values of both methods side by side (one row per instance)."
function comparison_table(g::DataFrame, f::DataFrame)
    better(a, b) = ismissing(a) ? "random_fao" : ismissing(b) ? "gurobi" :
                   a < b ? "gurobi" : b < a ? "random_fao" : "gleich"
    return DataFrame(
        instance = g.instance, jobs = g.jobs, machines = g.machines,
        reference = coalesce.(g.optimum, g.best_known_ub), initial = g.initial,
        gurobi_incumbent = g.incumbent, gurobi_bound = g.bound,
        gurobi_gap_own_bound = g.gap_own_bound, gurobi_primal_gap_bks = g.primal_gap_bks,
        gurobi_status = g.status, gurobi_time_to_best = g.time_to_best,
        random_fao_incumbent_mean = f.incumbent_mean, random_fao_incumbent_min = f.incumbent_min,
        random_fao_incumbent_max = f.incumbent_max, random_fao_bound = f.bound,
        random_fao_gap_own_bound_mean = f.gap_own_bound_mean,
        random_fao_primal_gap_bks_mean = f.primal_gap_bks_mean,
        random_fao_primal_gap_bks_min = f.primal_gap_bks_min,
        random_fao_primal_gap_bks_max = f.primal_gap_bks_max,
        random_fao_n_optimal = f.n_optimal, random_fao_n_seeds = f.n_seeds,
        random_fao_time_to_best_mean = f.time_to_best_mean,
        random_fao_iterations_mean = f.iterations_mean,
        better = better.(g.incumbent, f.incumbent_mean),   # FaO: mean over seeds
    )
end

"""
    run_anytime(names; method = :both, time_limit = 60.0, threads = 4, warm_start = true,
                seeds = [42], sub_time_limit = 1.0, n_free_min = 200, n_free_max = 800,
                accept_equal = true)

Solves all instances with `method` (`:gurobi`, `:random_fao` or `:both`). With `:both`
both methods are solved directly one after the other per instance (FaO with all seeds)
and the comparison plot is written immediately.
"""
function run_anytime(names; method::Symbol = :both, time_limit::Real = 60.0,
                     threads::Integer = 4, warm_start::Bool = true, seeds = [DEFAULT_SEED],
                     sub_time_limit::Real = 1.0, n_free_min::Integer = 200,
                     n_free_max::Integer = 800,
                     accept_equal::Bool = true)
    methods = method == :both ? collect(METHODS) :
              method in METHODS ? [method] :
              error("Unbekannte Methode $method, erlaubt: $(join(METHODS, ", ")), both")
    compare = length(methods) > 1
    seeds = collect(seeds)
    kw = (time_limit = Float64(time_limit), threads = threads, warm_start = warm_start,
          sub_time_limit = Float64(sub_time_limit), n_free_min = Int(n_free_min),
          n_free_max = Int(n_free_max),
          accept_equal = accept_equal)
    keep_awake()                                   # no sleep during the run

    for m in methods
        mkpath(joinpath(method_dir(m), "plots"))
        write_method_params(m, names; kw..., (m == :random_fao ? (seeds = seeds,) : (;))...)
    end
    compare && mkpath(COMPARISON_DIR)

    n_runs = (:gurobi in methods ? 1 : 0) + (:random_fao in methods ? length(seeds) : 0)
    println("Anytime-Lauf [$(join(methods, " + "))]: $(length(names)) Instanzen, ",
            "Zeitlimit $(time_limit) s je Lauf, FaO-Seeds $(seeds) ",
            "(≈ $(round(length(names) * n_runs * time_limit / 60, digits = 1)) min)")
    foreach(m -> warmup(m; kw...), methods)

    summaries = Dict(m => DataFrame() for m in methods)
    seed_summary = DataFrame()
    traces = Dict(m => Tuple{String,DataFrame,Union{Missing,Int}}[] for m in methods)
    for name in names
        info = jsplib_info(name)
        ref = something(info.optimum, info.upper, missing)
        dfs, finals = Dict{Symbol,DataFrame}(), Dict{Symbol,String}()
        initial_ms, xmax = nothing, Float64(time_limit)
        if :gurobi in methods
            row, df, ini, rt = process_gurobi(name; kw...)
            push!(summaries[:gurobi], row; promote = true, cols = :union)
            dfs[:gurobi], finals[:gurobi] = df, string(row.incumbent)
            initial_ms, xmax = something(initial_ms, ini, Some(nothing)), max(xmax, rt)
        end
        if :random_fao in methods
            row, med, ini, rt, srows = process_fao(name; seeds = seeds, kw...)
            push!(summaries[:random_fao], row; promote = true, cols = :union)
            foreach(r -> push!(seed_summary, r; promote = true, cols = :union), srows)
            dfs[:random_fao], finals[:random_fao] = med, "Ø $(round(row.incumbent_mean, digits = 1))"
            initial_ms, xmax = something(initial_ms, ini), max(xmax, rt)
        end
        for (m, df) in dfs
            push!(traces[m], (name, df, ref))
        end
        compare && savefig(plot_comparison(name, info, dfs, finals, initial_ms, xmax;
                                           n_seeds = length(seeds)),
                           joinpath(COMPARISON_DIR, "$name.png"))
    end

    for m in methods
        CSV.write(joinpath(method_dir(m), "summary.csv"), summaries[m])
        savefig(plot_overview(traces[m], time_limit, m),
                joinpath(method_dir(m), "plots", "overview.png"))
        println("Ergebnisse $m: ", method_dir(m))
    end
    :random_fao in methods &&
        CSV.write(joinpath(method_dir(:random_fao), "summary_seeds.csv"), seed_summary)
    if compare
        cmp = comparison_table(summaries[:gurobi], summaries[:random_fao])
        CSV.write(joinpath(COMPARISON_DIR, "summary.csv"), cmp)
        savefig(plot_comparison_overview(cmp), joinpath(COMPARISON_DIR, "overview.png"))
        println("Vergleich:      ", COMPARISON_DIR)
    end
    return summaries
end

function main(args = ARGS)
    names, kw = parse_args(args)
    return run_anytime(names; kw...)
end

end # module JSSPLabAnytime

JSSPLabAnytime.main()
