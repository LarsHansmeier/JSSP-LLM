# =============================================================================
# benchmark_large.jl — large-instance benchmark (random, Gurobi, A v2, B v2, C v3)
# =============================================================================
#
# Settings (fixed for this benchmark, passed only through existing parameters of
# random_fao / solve_model / the policy constructors – no policy is changed):
#   instances ta41, ta61, swv12, ta71; 1200 s wall clock per run (incl. model build);
#   free-y range [250, 4000] (capped per instance by FREE_CAP as in the loop);
#   per-subproblem TimeLimit 7.5 s; Gurobi Threads = 1, Seed = 42; start solution
#   initial_schedule, Big-M ub = its makespan (as in all earlier benchmarks).
#   Methods (existing names): random, gurobi, PolicyA_ScoreBased_fill ("A v2"),
#   PolicyB_StructureBased_v2, PolicyC_Dynamic_v3.
#
# Execution: the driver runs every pending run as a separate Julia process (sequential,
# one worker), interleaving methods and instances. Each run is written to
# results/<out>/runs/<instance>__<method>/ with a `done` marker written last; on restart,
# finished runs are skipped and unfinished ones are re-run from scratch. Afterwards all
# runs are aggregated (runs.csv, traces.csv, iterations.csv, policy logs, comparison.csv,
# plots, reference_values.csv, environment.txt). progress.csv is updated after every run.
#
#   julia --project=. scripts/benchmark_large.jl                         # full benchmark
#   julia --project=. scripts/benchmark_large.jl --time=90 --instances=swv12,ta71 \
#         --out=benchmark_large_v1_smoke                                 # smoke test
#   julia --project=. scripts/benchmark_large.jl --aggregate --out=...   # only aggregate
# =============================================================================

import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")); io = devnull)
ENV["GKSwstype"] = "100"     # GR without window (file output)

module JSSPLabLarge

using CSV, DataFrames, Dates, Plots, Statistics, JuMP
include(joinpath(@__DIR__, "..", "src", "JSSPLab.jl"))
using .JSSPLab
import .JSSPLab: keep_awake

const ROOT = normpath(joinpath(@__DIR__, ".."))
const METHODS = ["random", "gurobi", "PolicyA_ScoreBased_fill", "PolicyB_StructureBased_v2",
                 "PolicyC_Dynamic_v3"]
const FAO = filter(!=("gurobi"), METHODS)
const INSTANCES = ["ta41", "ta61", "swv12", "ta71"]
const Y_MIN, Y_MAX = 250, 4000
const SUB_T = 7.5
const THREADS = 1
const SEED = DEFAULT_SEED                   # 42
# reference values: existing JSPLIB metadata; ta71 has none there → published BKS
const EXTRA_REF = Dict("ta71" => (5464, "optimizizer.com/TA.php (LB = UB = 5464, [Ta1994]; accessed 2026-11-23); not in JSPLIB metadata"))
const REF_NOTES = Dict("ta41" => "optimizizer.com lists a newer UB 2006 (Shylo2008), LB 1912 – not used, to stay comparable with earlier benchmarks")
const STYLE = Dict("random" => (:seagreen, :dash), "gurobi" => (:royalblue, :solid),
                   "PolicyA_ScoreBased_fill" => (:purple, :solid),
                   "PolicyB_StructureBased_v2" => (:teal, :solid),
                   "PolicyC_Dynamic_v3" => (:darkgreen, :dot))

function reference(name)
    haskey(EXTRA_REF, name) && return EXTRA_REF[name]
    info = jsplib_info(name)
    info.optimum !== nothing && return (info.optimum, "JSPLIB metadata (optimum)")
    info.upper !== nothing && return (info.upper, "JSPLIB metadata (best known upper bound)")
    return (missing, "none")
end

run_dir(out, inst, m) = joinpath(ROOT, "results", out, "runs", "$(inst)__$(m)")

# --- one run (worker process) ------------------------------------------------------
function selector(method, inst)
    method == "random" && return random_free
    # A v2 and B v2 with y_floor = effective range minimum: free the overshooting operation
    # instead of ending below the range (default of both policies unchanged)
    y_lo = min(Y_MIN, floor(Int, FREE_CAP * length(machine_pairs(inst))))
    method == "PolicyA_ScoreBased_fill" && return policyA_selector(fill_budget = true, y_floor = y_lo)
    method == "PolicyB_StructureBased_v2" && return policyB_v2_selector(y_floor = y_lo)
    method == "PolicyC_Dynamic_v3" && return policyC_v3_selector(n_free_min = Y_MIN, n_free_max = Y_MAX)
    error("unknown method $method")
end

"Runs `method` on `inst` for `T` s; returns (result, policy state or nothing, incumbent check)."
function run_method(inst, method, T)
    s0 = initial_schedule(inst)
    ub = makespan(inst, s0)
    t0 = time()
    if method == "gurobi"
        res = solve_model(inst; time_limit = T, silent = true, warm_start = s0, wall_start = t0,
                          ub = ub, threads = THREADS, seed = SEED)
        return res, nothing, (seen = 0, infeasible = 0)
    end
    st = selector(method, inst)
    seen, bad = Ref(0), Ref(0)
    f = function (i, c, h, r, n)                      # incumbent check only, no decision change
        seen[] += 1
        check_schedule(i, c; verbose = false)[1] || (bad[] += 1)
        st(i, c, h, r, n)
    end
    res = random_fao(inst; time_limit = T, sub_time_limit = SUB_T, n_free_min = Y_MIN,
                     n_free_max = Y_MAX, threads = THREADS, seed = SEED, ub = ub,
                     wall_start = t0, select_free = f)
    return res, st, (seen = seen[], infeasible = bad[])
end

"""
`run_dir_override` (CLI `--run-dir=<path>`, used by scripts/benchmark_large_v2.jl): write into
that folder (created fresh by the calling driver, not deleted here) and additionally save the
final schedule as schedule.csv. Without it the behaviour is unchanged.
"""
function worker(inst_name, method, out, T; run_dir_override = "")
    dir = isempty(run_dir_override) ? run_dir(out, inst_name, method) : run_dir_override
    isempty(run_dir_override) && rm(dir; force = true, recursive = true)
    mkpath(dir)
    run_method(load_jsplib("ft06"), method, 2.0)        # JIT warm-up, not part of the run
    inst = load_jsplib(inst_name)
    started = now()
    res, st, inc = run_method(inst, method, T)
    ref, _ = reference(inst_name)
    ok = res.schedule !== nothing &&
         check_schedule(inst, res.schedule; claimed_makespan = res.incumbent, verbose = false)[1]
    pairs = length(machine_pairs(inst))
    cap = floor(Int, FREE_CAP * pairs)
    y_lo, y_hi = min(Y_MIN, cap), min(Y_MAX, cap)
    row = Dict{Symbol,Any}(:instance => inst_name, :method => method, :reference => ref,
        :initial => res.trace[1].incumbent, :incumbent => res.incumbent,
        :primal_gap => ismissing(ref) || ismissing(res.incumbent) ? missing : (res.incumbent - ref) / ref,
        :bound => res.bound, :status => string(res.status), :runtime => res.runtime,
        :feasible => ok, :incumbents_checked => inc.seen, :incumbents_infeasible => inc.infeasible,
        :peak_rss_mb => round(Sys.maxrss() / 2^20, digits = 1), :started => string(started),
        :y_range_effective => "$y_lo-$y_hi", :Y_tot => pairs, :sub_time_limit => method == "gurobi" ? missing : SUB_T,
        :threads => get_attribute(res.model, "Threads"),
        :gurobi_TimeLimit_last => get_attribute(res.model, "TimeLimit"))
    tr = DataFrame(res.trace)
    imp = [tr.time[i] for i in 2:nrow(tr) if !ismissing(tr.incumbent[i]) &&
           (ismissing(tr.incumbent[i-1]) || tr.incumbent[i] < tr.incumbent[i-1])]
    row[:time_last_improvement] = isempty(imp) ? missing : last(imp)
    if method != "gurobi"
        its = DataFrame(res.iterations)
        solved = its[.!in.(its.sub_status, Ref(["SKIPPED_DUPLICATE"])), :]
        d = res.diagnostics
        viol = count(n -> !(y_lo <= n <= y_hi), solved.n_free_y)
        merge!(row, Dict(:iterations => d.iterations, :share_time_limit => d.share_time_limit,
            :median_sub_time => d.median_sub_time, :median_n_free_y => d.median_n_free_y,
            :min_n_free_y => isempty(solved) ? missing : minimum(solved.n_free_y),
            :max_n_free_y => isempty(solved) ? missing : maximum(solved.n_free_y),
            :free_y_violations => viol, :n_skipped => d.n_skipped,
            :max_sub_time_limit_hit => maximum([r.sub_time for r in eachrow(its) if r.sub_status == "TIME_LIMIT"]; init = 0.0),
            :stop_reason => res.stop_reason))
        CSV.write(joinpath(dir, "iterations.csv"), its)
        if method == "PolicyB_StructureBased_v2"
            CSV.write(joinpath(dir, "policy_log.csv"), DataFrame(policyB_log(st, res.iterations)))
        elseif method == "PolicyC_Dynamic_v3"
            lg = DataFrame(policyC_v3_log(st, res.iterations))
            CSV.write(joinpath(dir, "policy_log.csv"), lg)
            merge!(row, Dict(:n_schedules => length(st.schedule_ids), :n_sideways => st.n_sideways,
                             :restarts => count(lg.restart)))
        end
    end
    CSV.write(joinpath(dir, "trace.csv"), select(tr, :time, :incumbent, :bound))
    CSV.write(joinpath(dir, "run.csv"), DataFrame([row]))
    if !isempty(run_dir_override) && res.schedule !== nothing   # final schedule for the driver's own check
        CSV.write(joinpath(dir, "schedule.csv"),
                  DataFrame(res.schedule.start, ["op$k" for k in 1:size(res.schedule.start, 2)]))
    end
    write(joinpath(dir, "done"), string(now()))          # written last: run complete
    println("$inst_name $method: incumbent $(res.incumbent), runtime $(round(res.runtime, digits = 1)) s, ",
            "feasible $ok, peak RSS $(row[:peak_rss_mb]) MB")
end

# --- driver ---------------------------------------------------------------------------
"Interleaved order: round r runs instance i with method (i + r) mod 5 – every round mixes methods."
run_order(insts, methods) = [(insts[i], methods[mod1(i + r - 1, length(methods))])
                             for r in 1:length(methods) for i in eachindex(insts)]

function write_progress(out, progress)
    CSV.write(joinpath(ROOT, "results", out, "progress.csv"), progress)
end

function environment(out, n_runs)
    gv = let a = Ref{Cint}(), b = Ref{Cint}(), c = Ref{Cint}()
        JSSPLab.Gurobi.GRBversion(a, b, c); "$(a[]).$(b[]).$(c[])"
    end
    cpu = strip(read(`sysctl -n machdep.cpu.brand_string`, String))
    pcores = strip(read(`sysctl -n hw.perflevel0.physicalcpu`, String))
    ecores = strip(read(`sysctl -n hw.perflevel1.physicalcpu`, String))
    mem = parse(Int, strip(read(`sysctl -n hw.memsize`, String))) ÷ 2^30
    os = strip(read(`sw_vers -productVersion`, String))
    open(joinpath(ROOT, "results", out, "environment.txt"), "w") do io
        println(io, "machine: $cpu, $(Sys.CPU_THREADS) cores ($pcores performance + $ecores efficiency), $mem GB RAM, macOS $os")
        println(io, "julia: $(VERSION), gurobi: $gv")
        println(io, "parallelism: 1 worker (sequential, one Julia process per run), Gurobi Threads = $THREADS, Seed = $SEED")
        println(io, "runs: $n_runs, free-y range [$Y_MIN, $Y_MAX] (capped by FREE_CAP = $FREE_CAP), sub-problem TimeLimit $SUB_T s")
        println(io, "written: $(now())")
    end
end

function drive(out, T, insts, methods)
    base = joinpath(ROOT, "results", out)
    mkpath(joinpath(base, "runs"))
    keep_awake()
    order = run_order(insts, methods)
    environment(out, length(order))
    progress = DataFrame(instance = String[], method = String[], start = String[], finish = String[], status = String[])
    for (inst, m) in order
        if isfile(joinpath(run_dir(out, inst, m), "done"))
            push!(progress, (inst, m, "", "", "done (earlier)"))
            continue
        end
        push!(progress, (inst, m, string(now()), "", "running"))
        write_progress(out, progress)
        log = joinpath(base, "runs", "$(inst)__$(m).log")
        cmd = `$(Base.julia_cmd()) --project=$ROOT $(@__FILE__) --worker --instance=$inst --method=$m --out=$out --time=$T`
        ok = try
            run(pipeline(cmd; stdout = log, stderr = log)); true
        catch
            false
        end
        ok &= isfile(joinpath(run_dir(out, inst, m), "done"))
        progress[end, :finish] = string(now())
        progress[end, :status] = ok ? "done" : "FAILED (see $(basename(log)); re-run from scratch on restart)"
        write_progress(out, progress)
        println(now(), "  $inst $m: ", progress[end, :status])
    end
    aggregate(out, T, insts, methods)
end

# --- aggregation -------------------------------------------------------------------------
function primal_integral(times, incs, ref, T)
    area, t_prev, g_prev = 0.0, 0.0, 100 * (incs[1] - ref) / ref
    for (t, z) in zip(times, incs)
        t = min(t, T)
        area += g_prev * (t - t_prev)
        t_prev, g_prev = t, 100 * (z - ref) / ref
    end
    return (area + g_prev * (T - t_prev)) / T
end

function aggregate(out, T, insts, methods)
    base = joinpath(ROOT, "results", out)
    runs, traces, its, logB, logC = DataFrame(), DataFrame(), DataFrame(), DataFrame(), DataFrame()
    for inst in insts, m in methods
        dir = run_dir(out, inst, m)
        isfile(joinpath(dir, "done")) || continue
        append!(runs, CSV.read(joinpath(dir, "run.csv"), DataFrame); cols = :union, promote = true)
        tr = CSV.read(joinpath(dir, "trace.csv"), DataFrame)
        tr.instance .= inst; tr.method .= m
        append!(traces, tr; cols = :union, promote = true)
        if isfile(joinpath(dir, "iterations.csv"))
            it = CSV.read(joinpath(dir, "iterations.csv"), DataFrame)
            it.instance .= inst; it.method .= m
            append!(its, it; cols = :union, promote = true)
        end
        if isfile(joinpath(dir, "policy_log.csv"))
            lg = CSV.read(joinpath(dir, "policy_log.csv"), DataFrame)
            insertcols!(lg, 1, :instance => inst, :method => m)
            append!(m == "PolicyC_Dynamic_v3" ? logC : logB, lg; cols = :union, promote = true)
        end
    end
    isempty(runs) && return println("no finished runs")
    # final check: full T (±5 s), or an early stop that is a valid outcome – Gurobi proved
    # optimality, or C v3 stopped "exhausted" (trace carried forward to T by the loop)
    runs.run_check = map(eachrow(runs)) do r
        t_end = maximum(traces[(traces.instance .== r.instance) .& (traces.method .== r.method), :time])
        abs(r.runtime - T) <= 5 ? "full" :
        r.method == "gurobi" && r.status == "OPTIMAL" ? "gurobi optimal" :
        hasproperty(r, :stop_reason) && isequal(r.stop_reason, "exhausted") && t_end >= T - 5 ?
            "C v3 exhausted (trace carried to T)" : "INCOMPLETE"
    end
    CSV.write(joinpath(base, "runs.csv"), runs)
    CSV.write(joinpath(base, "traces.csv"), select(traces, :instance, :method, :time, :incumbent, :bound))
    isempty(its) || CSV.write(joinpath(base, "iterations.csv"), its)
    isempty(logB) || CSV.write(joinpath(base, "policyB_log.csv"), logB)
    isempty(logC) || CSV.write(joinpath(base, "policyC_log.csv"), logC)
    refs = DataFrame(instance = insts, reference = [reference(i)[1] for i in insts],
                     source = [reference(i)[2] for i in insts],
                     note = [get(REF_NOTES, i, "") for i in insts])
    CSV.write(joinpath(base, "reference_values.csv"), refs)

    # comparison: final primal gap and primal integral over T
    cmp = DataFrame()
    for r in eachrow(runs)
        t = sort(traces[(traces.instance .== r.instance) .& (traces.method .== r.method) .& .!ismissing.(traces.incumbent), :], :time)
        pi = ismissing(r.reference) ? missing : primal_integral(t.time, t.incumbent, r.reference, T)
        push!(cmp, (instance = r.instance, method = r.method, incumbent = r.incumbent,
                    primal_gap = r.primal_gap, primal_integral = pi); promote = true)
    end
    CSV.write(joinpath(base, "comparison.csv"), cmp)

    # plots
    present = [m for m in methods if m in runs.method]
    ref_of = Dict(r.instance => r.reference for r in eachrow(refs))
    gap_panels = []
    for inst in insts
        inst in runs.instance || continue
        p = plot(; xscale = :log10, xlabel = "time [s] (log)", ylabel = "makespan", title = inst,
                 legend = :topright, size = (900, 550), dpi = 120, titlefontsize = 11, xlims = (1, T * 1.05))
        q = plot(; xscale = :log10, xlabel = "time [s] (log)", ylabel = "primal gap [%]", title = inst,
                 titlefontsize = 10, xlims = (1, T * 1.05), legend = inst == insts[1] ? :topright : false)
        for m in present
            t = sort(traces[(traces.instance .== inst) .& (traces.method .== m) .& .!ismissing.(traces.incumbent), :], :time)
            isempty(t) && continue
            c, ls = STYLE[m]
            plot!(p, max.(t.time, 1.0), t.incumbent; seriestype = :steppost, lw = 2, color = c, ls = ls, label = m)
            plot!(q, max.(t.time, 1.0), 100 .* (t.incumbent .- ref_of[inst]) ./ ref_of[inst];
                  seriestype = :steppost, lw = 2, color = c, ls = ls, label = m)
        end
        hline!(p, [ref_of[inst]]; color = :black, ls = :dash, label = "reference $(ref_of[inst])")
        savefig(p, joinpath(base, "anytime_$inst.png"))
        push!(gap_panels, q)
    end
    savefig(plot(gap_panels...; layout = (2, 2), size = (1200, 850), dpi = 110), joinpath(base, "anytime_gap_all.png"))
    fao_present = [m for m in present if m != "gurobi"]
    insts_present = [i for i in insts if i in runs.instance]
    xs = 1:length(insts_present)
    for (col, file, lbl, ysc) in ((:median_sub_time, "median_sub_time.png", "median sub-problem time [s] (log)", :log10),
                                  (:share_time_limit, "share_time_limit.png", "share of sub-problems hitting 7.5 s", :identity))
        p = plot(; xticks = (xs, insts_present), ylabel = lbl, yscale = ysc, size = (850, 480), dpi = 120,
                 legend = :outertopright, title = string(col), titlefontsize = 11)
        for m in fao_present
            v = [let r = runs[(runs.instance .== i) .& (runs.method .== m), :]
                     isempty(r) || ismissing(r[1, col]) ? NaN : max(Float64(r[1, col]), ysc == :log10 ? 1e-4 : 0.0) end
                 for i in insts_present]
            c, ls = STYLE[m]
            plot!(p, xs, v; marker = :circle, lw = 2, color = c, ls = ls, label = m)
        end
        savefig(p, joinpath(base, file))
    end
    if !isempty(its)
        solved = its[its.sub_status .!= "SKIPPED_DUPLICATE", :]
        panels = map(insts_present) do i
            d = solved[solved.instance .== i, :]
            q = plot(; title = i, ylabel = "free y", titlefontsize = 10, legend = false,
                     xticks = (1:length(fao_present), ["random", "A v2", "B v2", "C v3"][1:length(fao_present)]))
            for (k, m) in enumerate(fao_present)          # box = quartiles, whiskers = min/max
                v = d[d.method .== m, :n_free_y]
                isempty(v) && continue
                q1, med, q3 = quantile(v, [0.25, 0.5, 0.75])
                c = STYLE[m][1]
                plot!(q, Shape([k - 0.3, k + 0.3, k + 0.3, k - 0.3], [q1, q1, q3, q3]); color = c, alpha = 0.5)
                plot!(q, [k - 0.3, k + 0.3], [med, med]; color = :black, lw = 2)
                plot!(q, [k, k], [minimum(v), q1]; color = c); plot!(q, [k, k], [q3, maximum(v)]; color = c)
            end
            q
        end
        savefig(plot(panels...; layout = (2, 2), size = (1100, 800), dpi = 110), joinpath(base, "free_y_distribution.png"))
    end
    println("aggregated $(nrow(runs)) runs → $base")
end

function main(args = ARGS)
    opt(k, d) = (i = findfirst(startswith("--$k="), args); i === nothing ? d : split(args[i], '='; limit = 2)[2])
    out = opt("out", "benchmark_large_v1")
    T = parse(Float64, opt("time", "1200"))
    insts = split(opt("instances", join(INSTANCES, ",")), ',')
    methods = split(opt("methods", join(METHODS, ",")), ',')
    if "--worker" in args
        worker(opt("instance", ""), opt("method", ""), out, T; run_dir_override = opt("run-dir", ""))
    elseif "--aggregate" in args
        aggregate(out, T, String.(insts), String.(methods))
    else
        drive(out, T, String.(insts), String.(methods))
    end
end

end # module JSSPLabLarge

JSSPLabLarge.main()
