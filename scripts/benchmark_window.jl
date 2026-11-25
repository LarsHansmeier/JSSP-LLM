# =============================================================================
# benchmark_window.jl — Gurobi vs. Random FaO (random_free) vs. Random FaO (window_free)
# =============================================================================
#
# Same settings as run_anytime.jl: wall-clock time incl. model build, seed 42,
# Big-M = makespan of initial_schedule for all methods, n drawn per iteration from
# n_free_min:n_free_max (default 200:800, capped at 75 % of all machine pairs).
#
# Run:   julia --project=. scripts/benchmark_window.jl            # 60 s per run
#        julia --project=. scripts/benchmark_window.jl 30 ft10    # 30 s, own instances
# Options:
#   --methods=PolicyA_ScoreBased      run only these methods (comma-separated)
#   --out=benchmark_policyA_v2        output folder results/<out>/ (default: benchmark)
#   --reuse=benchmark                 take the other methods' runs/traces from
#                                     results/<reuse>/ instead of re-running them; several
#                                     folders: --reuse=f1,f2:methodA+methodB (first folder wins)
#
# Output in results/<out>/ (default results/benchmark/):
#   runs.csv            one row per (instance, method); column `source` = where it comes from
#   comparison.csv      one row per instance: primal gap, share TIME_LIMIT, median sub_time
#   traces.csv          anytime history of all runs (time, incumbent, instance, method)
#   anytime.png         primal gap over time, one panel per instance
#   median_sub_time.png median subproblem time per instance (random vs. window)
#   policyB_log.csv     Policy B only: one row per iteration (type, candidate, |R|, critical ops, improved)
#   policyC_log.csv     Policy C only: one row per iteration (type, k / y_target, |R|, status, gap,
#                       Z_inc, scores, restart; v2 also skipped, fails_at_max, last_used_*, stop_reason)
# =============================================================================

import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")); io = devnull)
ENV["GKSwstype"] = "100"     # GR without window (file output)

module JSSPLabBenchmark

using CSV, DataFrames, Plots, Statistics
include(joinpath(@__DIR__, "..", "src", "JSSPLab.jl"))
using .JSSPLab

const RESULTS = normpath(joinpath(@__DIR__, "..", "results"))
const METHODS = ["gurobi", "random", "window", "PolicyA_ScoreBased", "PolicyA_ScoreBased_fill",
                 "PolicyB_StructureBased", "PolicyB_StructureBased_fill", "PolicyB_StructureBased_v2",
                 "PolicyB_StructureBased_v3", "PolicyC_Dynamic", "PolicyC_Dynamic_v2",
                 "PolicyC_Dynamic_v3", "PolicyD_Hybrid_v1", "PolicyD_noExact", "PolicyD_noEscape"]
const FAO_METHODS = ["random", "window", "PolicyA_ScoreBased", "PolicyA_ScoreBased_fill",
                     "PolicyB_StructureBased", "PolicyB_StructureBased_fill",
                     "PolicyB_StructureBased_v2", "PolicyB_StructureBased_v3",
                     "PolicyC_Dynamic", "PolicyC_Dynamic_v2", "PolicyC_Dynamic_v3",
                     "PolicyD_Hybrid_v1", "PolicyD_noExact", "PolicyD_noEscape"]
const STYLE = Dict("gurobi" => (:royalblue, :solid), "random" => (:seagreen, :dash),
                   "window" => (:darkorange, :solid),
                   "PolicyA_ScoreBased" => (:crimson, :solid),
                   "PolicyA_ScoreBased_fill" => (:purple, :dash),
                   "PolicyB_StructureBased" => (:black, :solid),
                   "PolicyB_StructureBased_fill" => (:gray, :dash),
                   "PolicyB_StructureBased_v2" => (:teal, :solid),
                   "PolicyB_StructureBased_v3" => (:olive, :solid),
                   "PolicyC_Dynamic" => (:brown, :solid),
                   "PolicyC_Dynamic_v2" => (:magenta, :solid),
                   "PolicyC_Dynamic_v3" => (:darkgreen, :solid),
                   "PolicyD_Hybrid_v1" => (:red, :solid), "PolicyD_noExact" => (:red, :dash),
                   "PolicyD_noEscape" => (:red, :dot))
const POLICY_D_METHODS = Dict("PolicyD_Hybrid_v1" => :full, "PolicyD_noExact" => :noExact,
                              "PolicyD_noEscape" => :noEscape)

function run_one(method, inst; time_limit, n_free_min, n_free_max)
    t0 = time()
    s0 = initial_schedule(inst)
    ub = makespan(inst, s0)
    method == "gurobi" &&
        return solve_model(inst; time_limit = time_limit, silent = true, warm_start = s0,
                           wall_start = t0, ub = ub)
    if haskey(POLICY_D_METHODS, method)                         # fresh Policy D state per run
        st = policyD_selector(; variant = POLICY_D_METHODS[method], n_free_min, n_free_max,
                              seed = DEFAULT_SEED)
        res = random_fao(inst; time_limit = time_limit, n_free_min = n_free_min,
                         n_free_max = n_free_max, ub = ub, wall_start = t0, select_free = st)
        log = policyD_log(st, res.iterations)
        esc = [r for r in log if r.type == "RAND"]
        return merge(res, (policy_log = log, log_name = "policyD_log.csv",
                           n_schedules = length(st.schedule_ids), n_sideways = st.n_sideways,
                           n_exhaustions = st.n_exhaustions, n_escape_iters = length(esc),
                           n_escape_improvements = count(r -> r.improved === true, esc),
                           exact_start_time = ismissing(res.exact) ? missing : res.exact.start_time,
                           exact_improved = ismissing(res.exact) ? missing : res.exact.improved,
                           exact_status = ismissing(res.exact) ? missing : res.exact.status))
    end
    if method == "PolicyC_Dynamic_v3"                           # fresh Policy C v3 state per run
        st = policyC_v3_selector(; n_free_min, n_free_max)
        res = random_fao(inst; time_limit = time_limit, n_free_min = n_free_min,
                         n_free_max = n_free_max, ub = ub, wall_start = t0, select_free = st)
        return merge(res, (policy_log = policyC_v3_log(st, res.iterations), log_name = "policyC_log.csv",
                           n_schedules = length(st.schedule_ids), n_sideways = st.n_sideways))
    end
    if method == "PolicyC_Dynamic_v2"                           # fresh Policy C v2 state per run
        st = policyC_v2_selector(; n_free_min, n_free_max)
        res = random_fao(inst; time_limit = time_limit, n_free_min = n_free_min,
                         n_free_max = n_free_max, ub = ub, wall_start = t0, select_free = st)
        return merge(res, (policy_log = policyC_v2_log(st, res.iterations), log_name = "policyC_log.csv"))
    end
    if method == "PolicyC_Dynamic"                              # fresh Policy C state per run
        st = policyC_selector(; n_free_min, n_free_max)
        res = random_fao(inst; time_limit = time_limit, n_free_min = n_free_min,
                         n_free_max = n_free_max, ub = ub, wall_start = t0, select_free = st)
        return merge(res, (policy_log = policyC_log(st, res.iterations), log_name = "policyC_log.csv"))
    end
    if startswith(method, "PolicyB")                            # fresh Policy B state per run
        st = method == "PolicyB_StructureBased_v2" ? policyB_v2_selector() :
             method == "PolicyB_StructureBased_v3" ? policyB_v3_selector() :
             policyB_selector(fill_budget = method == "PolicyB_StructureBased_fill")
        res = random_fao(inst; time_limit = time_limit, n_free_min = n_free_min,
                         n_free_max = n_free_max, ub = ub, wall_start = t0, select_free = st)
        return merge(res, (policy_log = policyB_log(st, res.iterations), log_name = "policyB_log.csv"))
    end
    return random_fao(inst; time_limit = time_limit, n_free_min = n_free_min,
                      n_free_max = n_free_max, ub = ub, wall_start = t0,
                      select_free = method == "random" ? random_free :
                                    method == "window" ? window_free :
                                    policyA_selector(fill_budget = method == "PolicyA_ScoreBased_fill"))
                                    # fresh Policy A state per run
end

function main(args = ARGS)
    opt(key, default) = (i = findfirst(startswith("--$key="), args);
                         i === nothing ? default : split(args[i], '='; limit = 2)[2])
    methods = split(opt("methods", join(METHODS, ",")), ',')
    all(in(METHODS), methods) || error("--methods: erlaubt sind $(join(METHODS, ", "))")
    OUT = joinpath(RESULTS, opt("out", "benchmark"))
    reuse = opt("reuse", nothing)
    args = filter(a -> !startswith(a, "--"), args)
    time_limit = something(tryparse.(Float64, args)..., 60.0)   # first number, if any
    names = filter(a -> tryparse(Float64, a) === nothing, args)
    isempty(names) && (names = ["ft10", "la21", "ta01", "swv12"])
    n_free_min, n_free_max = 200, 800
    mkpath(OUT)
    keep_awake()
    println("Benchmark: $(join(names, ", ")) × $(join(methods, ", ")), $time_limit s je Lauf ",
            "(≈ $(round(length(names) * length(methods) * time_limit / 60, digits = 1)) min)",
            reuse === nothing ? "" : ", übrige Methoden aus results/$reuse/")

    # warm-up so JIT compilation does not distort the first run
    ft06 = load_jsplib("ft06")
    foreach(m -> run_one(m, ft06; time_limit = 1.0, n_free_min, n_free_max), methods)

    runs, traces = DataFrame(), DataFrame()
    policy_logs = Dict{String,DataFrame}()        # file name => per-iteration policy log
    for name in names, method in methods
        inst, info = load_jsplib(name), jsplib_info(name)
        ref = something(info.optimum, info.upper)
        res = run_one(method, inst; time_limit, n_free_min, n_free_max)
        ok = check_schedule(inst, res.schedule; claimed_makespan = res.incumbent, verbose = false)[1]
        d = method == "gurobi" ? nothing : res.diagnostics
        push!(runs, (instance = name, method = method, source = "this run", reference = ref,
                     initial = res.trace[1].incumbent, incumbent = res.incumbent,
                     primal_gap = (res.incumbent - ref) / ref, bound = res.bound,
                     runtime = res.runtime, feasible = ok,
                     iterations = d === nothing ? missing : d.iterations,
                     share_time_limit = d === nothing ? missing : d.share_time_limit,
                     median_sub_time = d === nothing ? missing : d.median_sub_time,
                     median_n_free_y = d === nothing ? missing : d.median_n_free_y,
                     n_skipped = d === nothing ? missing : d.n_skipped,
                     stop_reason = d === nothing ? missing : res.stop_reason,
                     n_schedules = get(res, :n_schedules, missing),
                     n_sideways = get(res, :n_sideways, missing),
                     n_exhaustions = get(res, :n_exhaustions, missing),
                     n_escape_iters = get(res, :n_escape_iters, missing),
                     n_escape_improvements = get(res, :n_escape_improvements, missing),
                     exact_start_time = get(res, :exact_start_time, missing),
                     exact_improved = get(res, :exact_improved, missing),
                     exact_status = get(res, :exact_status, missing));
              promote = true)
        if haskey(res, :policy_log)               # Policy B / C: per-iteration log
            policy_log = get!(policy_logs, res.log_name, DataFrame())
            foreach(row -> push!(policy_log, merge((instance = name, method = method), row);
                                 promote = true, cols = :union), res.policy_log)
            CSV.write(joinpath(OUT, res.log_name), policy_log)
        end
        for tp in res.trace
            ismissing(tp.incumbent) || push!(traces, (instance = name, method = method,
                                                      time = tp.time, incumbent = tp.incumbent))
        end
        r = runs[end, :]
        println(rpad(name, 7), rpad(method, 8), "Inc=", lpad(r.incumbent, 5), "  PG=",
                round(100r.primal_gap, digits = 2), " %",
                d === nothing ? "" : "  It=$(r.iterations)  TL-Anteil=$(round(r.share_time_limit, digits = 3))" *
                                     "  sub(med)=$(round(r.median_sub_time, digits = 3)) s",
                ok ? "  ✓" : "  ✗")
        CSV.write(joinpath(OUT, "runs.csv"), runs)              # save continuously
    end
    if reuse !== nothing                     # other methods from earlier benchmarks
        # "folder" or "folder:m1+m2,folder2:m3": each method from the first listed folder with it
        taken = Set{String}(methods)
        for item in split(reuse, ',')
            folder, only = occursin(':', item) ? split(item, ':'; limit = 2) : (item, nothing)
            allowed = only === nothing ? nothing : Set(split(only, '+'))
            old = CSV.read(joinpath(RESULTS, folder, "runs.csv"), DataFrame)
            use = Set(m for m in unique(old.method) if m in METHODS && !(m in taken) &&
                                                       (allowed === nothing || m in allowed))
            keep(df) = filter(r -> r.method in use && r.instance in names, df)
            old_runs = keep(old)
            old_runs.source .= "results/$folder"
            runs = vcat(old_runs, runs; cols = :union)
            traces = vcat(keep(CSV.read(joinpath(RESULTS, folder, "traces.csv"), DataFrame)), traces)
            union!(taken, use)
        end
    end
    CSV.write(joinpath(OUT, "runs.csv"), runs)
    CSV.write(joinpath(OUT, "traces.csv"), traces)

    # Comparison: one row per instance (only methods present in `runs`)
    present = [m for m in METHODS if m in runs.method]
    fao = [m for m in FAO_METHODS if m in present]
    wide(col, ms) = unstack(filter(:method => in(ms), runs), :instance, :method, col;
                            renamecols = m -> "$(col)_$m")
    cmp = outerjoin(wide(:primal_gap, present), wide(:share_time_limit, fao),
                    wide(:median_sub_time, fao), wide(:iterations, fao); on = :instance)
    cmp.best = [present[argmin([r["primal_gap_$m"] for m in present])] for r in eachrow(cmp)]
    CSV.write(joinpath(OUT, "comparison.csv"), cmp)

    # Anytime plot: primal gap over time, one panel per instance
    panels = map(names) do name
        p = plot(; title = name, xscale = :log10, xlabel = "Zeit [s] (log)",
                 ylabel = "Primal Gap [%]", xlims = (1e-2, time_limit * 1.05), titlefontsize = 10)
        ref = runs.reference[findfirst(==(name), runs.instance)]
        for m in present
            t = filter(r -> r.instance == name && r.method == m, traces)
            color, ls = STYLE[m]
            plot!(p, max.(t.time, 1e-2), 100 .* (t.incumbent .- ref) ./ ref;
                  seriestype = :steppost, lw = 2, color = color, ls = ls, label = m)
        end
        p
    end
    savefig(plot(panels...; layout = length(panels), size = (1000, 750), dpi = 120,
                 plot_title = "$(join(present, " vs. ")) (n ∈ $n_free_min:$n_free_max)",
                 plot_titlefontsize = 11),
            joinpath(OUT, "anytime.png"))

    # Median subproblem time (log scale) per instance: random vs. window
    x = 1:length(names)
    p = plot(; xticks = (x, names), ylabel = "median_sub_time [s] (log)", yscale = :log10,
             title = "Teilproblem-Lösezeit: random (gestrichelt) vs. window (durchgezogen)",
             titlefontsize = 11, size = (800, 450), dpi = 120, legend = :topleft)
    for m in fao
        color, ls = STYLE[m]
        plot!(p, x, cmp[indexin(names, cmp.instance), "median_sub_time_$m"];
              color = color, ls = ls, lw = 2, marker = :circle, label = m)
    end
    savefig(p, joinpath(OUT, "median_sub_time.png"))

    println("\n", cmp, "\n\nErgebnisse: ", OUT)
    return cmp
end

end # module JSSPLabBenchmark

JSSPLabBenchmark.main()
