# =============================================================================
# benchmark_cpsat.jl — OR-Tools CP-SAT as external reference on the large instances
# =============================================================================
#
# 1. export the instances + initial solution (scripts/export_instances.jl → input/*.json)
# 2. run scripts/cpsat_jssp.py sequentially (CPSAT_1w: 1 worker, CPSAT_8w: 8 workers;
#    1200 s, seed 42), all 1-worker runs first; finished runs are skipped on restart;
#    progress.csv is updated after every run
# 3. validate every final schedule in Julia (check_schedule, makespan == CP-SAT objective),
#    compute primal gap / primal integral against results/benchmark_large_v1/
#    reference_values.csv (read only) and aggregate runs.csv, traces.csv, comparison.csv
#    (incl. the 20 rows of results/benchmark_large_v1/comparison.csv, read only), plots and
#    environment.txt in results/<out>/
#
#   julia --project=. scripts/benchmark_cpsat.jl                                  # full run
#   julia --project=. scripts/benchmark_cpsat.jl --time=60 --instances=swv12 \
#         --out=benchmark_large_cpsat_v1_smoke                                    # smoke test
#   julia --project=. scripts/benchmark_cpsat.jl --aggregate                      # only aggregate
#
# Trace times: CP-SAT's callback reports seconds since solve start; they are shifted by the
# model build time so that, as for the other methods, time 0 is the start of the method
# (build included). The initial solution (hint) is the first trace point at time 0.
# =============================================================================

import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")); io = devnull)
ENV["GKSwstype"] = "100"     # GR without window (file output)

module JSSPLabCPSAT

using CSV, DataFrames, Dates, Plots, Statistics
include(joinpath(@__DIR__, "..", "src", "JSSPLab.jl"))
using .JSSPLab

const ROOT = normpath(joinpath(@__DIR__, ".."))
const PY = joinpath(ROOT, ".venv-cpsat", "bin", "python")
const LARGE = joinpath(ROOT, "results", "benchmark_large_v1")      # read only
const INSTANCES = ["ta41", "ta61", "swv12", "ta71"]
const VARIANTS = ["CPSAT_1w" => 1, "CPSAT_8w" => 8]
const SEED = 42
const OUR = ["random", "gurobi", "PolicyA_ScoreBased_fill", "PolicyB_StructureBased_v2", "PolicyC_Dynamic_v3"]
const STYLE = Dict("random" => (:seagreen, :dash), "gurobi" => (:royalblue, :solid),
                   "PolicyA_ScoreBased_fill" => (:purple, :solid),
                   "PolicyB_StructureBased_v2" => (:teal, :solid),
                   "PolicyC_Dynamic_v3" => (:darkgreen, :dot),
                   "CPSAT_1w" => (:darkorange, :solid), "CPSAT_8w" => (:red, :dash))

base(out) = joinpath(ROOT, "results", out)
raw(out, inst, m) = joinpath(base(out), "raw", "$(inst)_$(m)")

# --- driver ---------------------------------------------------------------------------
function write_environment(out)
    cpu = strip(read(`sysctl -n machdep.cpu.brand_string`, String))
    p = strip(read(`sysctl -n hw.perflevel0.physicalcpu`, String))
    e = strip(read(`sysctl -n hw.perflevel1.physicalcpu`, String))
    mem = parse(Int, strip(read(`sysctl -n hw.memsize`, String))) ÷ 2^30
    os = strip(read(`sw_vers -productVersion`, String))
    py = strip(read(`$PY -c "import sys; print(sys.version.split()[0])"`, String))
    ort = strip(read(`$PY -c "import ortools; print(ortools.__version__)"`, String))
    open(joinpath(base(out), "environment.txt"), "w") do io
        println(io, "machine: $cpu, $(Sys.CPU_THREADS) cores ($p performance + $e efficiency), $mem GB RAM, macOS $os")
        println(io, "julia: $(VERSION), python: $py (project venv .venv-cpsat), ortools: $ort")
        println(io, "variants: ", join(["$m = num_workers $w" for (m, w) in VARIANTS], ", "),
                "; random_seed $SEED, max_time_in_seconds per run (see runs.csv), other CP-SAT parameters default")
        println(io, "note: the machine has only $p performance cores, so the 8-worker variant also runs on efficiency cores")
        println(io, "runs: sequential, one at a time, not in parallel with other benchmarks")
        println(io, "written: $(now())")
    end
end

function drive(out, T, insts, variants)
    mkpath(joinpath(base(out), "raw"))
    keep_awake()
    input = joinpath(base(out), "input")
    run(`$(Base.julia_cmd()) --project=$ROOT $(joinpath(@__DIR__, "export_instances.jl")) --out=$input --instances=$(join(insts, ","))`)
    write_environment(out)
    progress = DataFrame(instance = String[], method = String[], start = String[], finish = String[], status = String[])
    for (m, w) in variants, inst in insts                 # all 1-worker runs first
        r = raw(out, inst, m)
        if isfile(r * ".done")
            push!(progress, (inst, m, "", "", "done (earlier)"))
            continue
        end
        foreach(f -> rm(f; force = true), [r * s for s in (".json", "_starts.csv", "_trace.csv", "_meta.csv", ".log")])
        push!(progress, (inst, m, string(now()), "", "running"))
        CSV.write(joinpath(base(out), "progress.csv"), progress)
        cmd = `$PY $(joinpath(@__DIR__, "cpsat_jssp.py")) --input $(joinpath(input, "$inst.json")) --output $(r * ".json") --workers $w --time $T --seed $SEED`
        ok = try
            run(pipeline(cmd; stdout = r * ".log", stderr = r * ".log")); isfile(r * "_meta.csv")
        catch
            false
        end
        ok && write(r * ".done", string(now()))
        progress[end, :finish] = string(now())
        progress[end, :status] = ok ? "done" : "FAILED (see raw/$(inst)_$(m).log; re-run on restart)"
        CSV.write(joinpath(base(out), "progress.csv"), progress)
        println(now(), "  $inst $m: ", progress[end, :status])
    end
    aggregate(out, T, insts, variants)
end

# --- validation and aggregation ------------------------------------------------------------
"Same formula as aggregate() in scripts/benchmark_large.jl: step-function mean gap (%) over [0, T]."
function primal_integral(times, incs, ref, T)
    area, t_prev, g_prev = 0.0, 0.0, 100 * (incs[1] - ref) / ref
    for (t, z) in zip(times, incs)
        t = min(t, T)
        area += g_prev * (t - t_prev)
        t_prev, g_prev = t, 100 * (z - ref) / ref
    end
    return (area + g_prev * (T - t_prev)) / T
end

function meta(path)
    d = CSV.read(path, DataFrame; types = String)
    return Dict(r.key => (ismissing(r.value) ? "" : r.value) for r in eachrow(d))
end

function aggregate(out, T, insts, variants)
    refs = CSV.read(joinpath(LARGE, "reference_values.csv"), DataFrame)          # read only
    ref_of = Dict(r.instance => r.reference for r in eachrow(refs))
    old_traces = CSV.read(joinpath(LARGE, "traces.csv"), DataFrame)              # read only
    old_cmp = CSV.read(joinpath(LARGE, "comparison.csv"), DataFrame)            # read only
    runs, traces, beats = DataFrame(), DataFrame(), DataFrame()
    for (m, w) in variants, inst in insts
        r = raw(out, inst, m)
        isfile(r * ".done") || continue
        md = meta(r * "_meta.csv")
        inst_obj = load_jsplib(inst)
        status = md["status"]
        obj = isempty(md["objective"]) ? missing : round(Int, parse(Float64, md["objective"]))
        bnd = isempty(md["bound"]) ? missing : ceil(Int, parse(Float64, md["bound"]) - 1e-6)
        build = parse(Float64, md["build_time"])
        wall = parse(Float64, md["wall_time"])
        init_ms = parse(Int, md["initial_makespan"])
        # 1./2. rebuild the schedule from the exported start times, independent check
        feasible, ms_julia, errs = false, missing, String[]
        if !ismissing(obj)
            st = Matrix{Int}(CSV.read(r * "_starts.csv", DataFrame))
            size(st) == (inst_obj.n, inst_obj.m) || error("$inst $m: start-time matrix $(size(st))")
            s = Schedule(st)
            ms_julia = makespan(inst_obj, s)
            feasible, errs = check_schedule(inst_obj, s; claimed_makespan = obj, verbose = false)
        end
        makespan_equal = !ismissing(obj) && ms_julia == obj
        makespan_equal || @warn "$inst $m: Julia makespan $ms_julia ≠ CP-SAT objective $obj"
        feasible || @warn "$inst $m: final schedule INFEASIBLE" errors = first(errs, 5)
        # trace: initial solution at 0, CP-SAT solutions shifted by the build time,
        # carried forward to T (also after a proven optimum)
        tr = CSV.read(r * "_trace.csv", DataFrame)
        times = vcat(0.0, build .+ tr.time)
        incs = vcat(init_ms, round.(Int, tr.objective))
        bnds = vcat(missing, ceil.(Int, tr.bound .- 1e-6))
        keep = [1; [i for i in 2:length(incs) if incs[i] < minimum(incs[1:i-1])]]   # improvements only
        times, incs, bnds = times[keep], incs[keep], bnds[keep]
        runtime = build + wall
        push!(times, max(T, runtime)); push!(incs, incs[end]); push!(bnds, bnd)
        ref = ref_of[inst]
        for (t, z, b) in zip(times, incs, bnds)
            push!(traces, (instance = inst, method = m, time = t, incumbent = z, bound = b); promote = true)
        end
        imp = times[2:end-1]
        check = !feasible ? "INFEASIBLE" : !makespan_equal ? "MAKESPAN MISMATCH" :
                status == "OPTIMAL" ? (runtime < T - 5 ? "cpsat optimal" : "full") :
                abs(runtime - T) <= 5 ? "full" : "INCOMPLETE"
        push!(runs, (instance = inst, method = m, reference = ref, initial = init_ms,
                     incumbent = obj, primal_gap = ismissing(obj) ? missing : (obj - ref) / ref,
                     bound = bnd, status = status, runtime = runtime, feasible = feasible,
                     incumbents_checked = 1, incumbents_infeasible = feasible ? 0 : 1,
                     peak_rss_mb = parse(Float64, md["peak_rss_mb"]), threads = w,
                     time_last_improvement = isempty(imp) ? missing : last(imp),
                     started = Dates.format(unix2datetime(mtime(r * ".json") - runtime), "yyyy-mm-ddTHH:MM:SS"),
                     num_workers = w, cpsat_status = status, final_bound = bnd, build_time = build,
                     makespan_julia = ms_julia, n_solutions = nrow(tr), run_check = check); promote = true)
        # first time CP-SAT beat the best final result of our methods (20-min runs)
        best_ours = minimum(old_cmp[old_cmp.instance .== inst, :incumbent])
        who = old_cmp.method[findfirst((old_cmp.instance .== inst) .& (old_cmp.incumbent .== best_ours))]
        i = findfirst(<(best_ours), incs)
        push!(beats, (instance = inst, method = m, best_of_our_methods = best_ours, by = who,
                      first_time_better = i === nothing ? missing : times[i]); promote = true)
    end
    isempty(runs) && return println("no finished runs")
    # same columns as results/benchmark_large_v1/runs.csv (not applicable → empty), plus CP-SAT columns
    old_cols = names(CSV.read(joinpath(LARGE, "runs.csv"), DataFrame; limit = 1))
    for c in old_cols
        hasproperty(runs, c) || (runs[!, c] .= missing)
    end
    runs = select(runs, vcat(old_cols, [c for c in names(runs) if !(c in old_cols)]))
    CSV.write(joinpath(base(out), "runs.csv"), runs)
    CSV.write(joinpath(base(out), "traces.csv"), traces)
    CSV.write(joinpath(base(out), "cpsat_vs_ours.csv"), beats)
    cmp = copy(old_cmp)
    for r in eachrow(runs)
        t = traces[(traces.instance .== r.instance) .& (traces.method .== r.method), :]
        push!(cmp, (instance = r.instance, method = r.method, incumbent = r.incumbent,
                    primal_gap = r.primal_gap,
                    primal_integral = primal_integral(t.time, t.incumbent, ref_of[r.instance], T)); promote = true)
    end
    CSV.write(joinpath(base(out), "comparison.csv"), cmp)

    # plots: our methods (results/benchmark_large_v1/traces.csv, read only) + CP-SAT
    all_tr = vcat(select(old_traces, :instance, :method, :time, :incumbent),
                  select(traces, :instance, :method, :time, :incumbent))
    methods = vcat(OUR, first.(variants))
    panels = []
    for inst in insts
        inst in runs.instance || continue
        p = plot(; xscale = :log10, xlabel = "time [s] (log)", ylabel = "makespan", title = inst,
                 legend = :topright, size = (900, 550), dpi = 120, titlefontsize = 11, xlims = (1, T * 1.05))
        q = plot(; xscale = :log10, xlabel = "time [s] (log)", ylabel = "primal gap [%]", title = inst,
                 titlefontsize = 10, xlims = (1, T * 1.05), legend = inst == insts[1] ? :topright : false)
        for m in methods
            t = sort(all_tr[(all_tr.instance .== inst) .& (all_tr.method .== m) .& .!ismissing.(all_tr.incumbent), :], :time)
            isempty(t) && continue
            c, ls = STYLE[m]
            plot!(p, max.(t.time, 1.0), t.incumbent; seriestype = :steppost, lw = 2, color = c, ls = ls, label = m)
            plot!(q, max.(t.time, 1.0), 100 .* (t.incumbent .- ref_of[inst]) ./ ref_of[inst];
                  seriestype = :steppost, lw = 2, color = c, ls = ls, label = m)
        end
        hline!(p, [ref_of[inst]]; color = :black, ls = :dash, label = "reference $(ref_of[inst])")
        savefig(p, joinpath(base(out), "anytime_$inst.png"))
        push!(panels, q)
    end
    savefig(plot(panels...; layout = length(panels) == 4 ? (2, 2) : length(panels), size = (1200, 850), dpi = 110),
            joinpath(base(out), "anytime_gap_all.png"))
    println("aggregated $(nrow(runs)) CP-SAT runs → $(base(out))")
    show(stdout, select(runs, :instance, :method, :incumbent, :makespan_julia, :primal_gap, :final_bound,
                        :cpsat_status, :runtime, :feasible, :run_check); allrows = true)
    println()
end

function main(args = ARGS)
    opt(k, d) = (i = findfirst(startswith("--$k="), args); i === nothing ? d : split(args[i], '='; limit = 2)[2])
    out = opt("out", "benchmark_large_cpsat_v1")
    T = parse(Float64, opt("time", "1200"))
    insts = String.(split(opt("instances", join(INSTANCES, ",")), ','))
    variants = [v for v in VARIANTS if first(v) in split(opt("methods", join(first.(VARIANTS), ",")), ',')]
    "--aggregate" in args ? aggregate(out, T, insts, variants) : drive(out, T, insts, variants)
end

end # module JSSPLabCPSAT

JSSPLabCPSAT.main()
