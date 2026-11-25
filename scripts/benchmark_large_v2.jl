# =============================================================================
# benchmark_large_v2.jl — large benchmark on swv11, ta31, ta51, ta72 (crash-safe, resumable)
# =============================================================================
#
# Same benchmark as results/benchmark_large_v1/ (+ the CP-SAT reference of
# results/benchmark_large_cpsat_v1/), identical settings, on four more instances.
# This file is only a driver / aggregator; the runs themselves are done by the existing
#   scripts/benchmark_large.jl --worker … --run-dir=<tmp>   (random, gurobi, A v2, B v2, C v3)
#   scripts/cpsat_jssp.py                                   (CPSAT_1w, CPSAT_8w)
# so all method settings are exactly those of the earlier benchmarks.
#
# Crash safety:
#   * every run (instance × method) writes only into runs/<inst>__<method>.tmp/
#   * after it finished AND its final schedule passed check_schedule (checked here again,
#     independently of the worker), the folder is renamed atomically to runs/<inst>__<method>/
#     and an empty DONE file is written (fsync'd). Complete ⇔ DONE exists.
#   * on every start: runs with DONE are skipped; any other run folder (*.tmp, *.FAILED,
#     final name without DONE) is deleted – a still-running orphaned child process of a killed
#     driver is killed first – and the run is redone from scratch. Never resumed mid-way.
#   * progress.csv (instance, method, start, finish, status) is rewritten atomically and fsync'd
#     after every change; rows still "running" without DONE become "interrupted" on restart.
#   * --aggregate reads only DONE folders and never writes into runs/.
#
#   julia --project=. scripts/benchmark_large_v2.jl                       # full benchmark
#   julia --project=. scripts/benchmark_large_v2.jl --time=60 --instances=swv11,ta72 \
#         --out=benchmark_large_v2_smoke                                  # smoke test
#   julia --project=. scripts/benchmark_large_v2.jl --aggregate [--out=…] # only aggregate
# With the default --out, aggregation also writes results/benchmark_large_combined/
# (reading the v1 / CP-SAT v1 folders read-only).
# =============================================================================

import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")); io = devnull)
ENV["GKSwstype"] = "100"     # GR without window (file output)

module JSSPLabLargeV2

using CSV, DataFrames, Dates, Plots, Statistics
include(joinpath(@__DIR__, "..", "src", "JSSPLab.jl"))
using .JSSPLab

const ROOT = normpath(joinpath(@__DIR__, ".."))
const PY = joinpath(ROOT, ".venv-cpsat", "bin", "python")
const DEFAULT_OUT = "benchmark_large_v2"
const INSTANCES = ["swv11", "ta31", "ta51", "ta72"]
const OUR = ["random", "gurobi", "PolicyA_ScoreBased_fill", "PolicyB_StructureBased_v2", "PolicyC_Dynamic_v3"]
const FAO = filter(!=("gurobi"), OUR)
const CPSAT = Dict("CPSAT_1w" => 1, "CPSAT_8w" => 8)
const ALL = vcat(OUR, ["CPSAT_1w", "CPSAT_8w"])
const SEED = 42
const Y_MIN, Y_MAX, SUB_T, THREADS = 250, 4000, 7.5, 1      # as in scripts/benchmark_large.jl
const SHORT = Dict("random" => "random", "gurobi" => "gurobi", "PolicyA_ScoreBased_fill" => "A v2",
                   "PolicyB_StructureBased_v2" => "B v2", "PolicyC_Dynamic_v3" => "C v3",
                   "CPSAT_1w" => "CPSAT_1w", "CPSAT_8w" => "CPSAT_8w")
const STYLE = Dict("random" => (:seagreen, :dash), "gurobi" => (:royalblue, :solid),
                   "PolicyA_ScoreBased_fill" => (:purple, :solid),
                   "PolicyB_StructureBased_v2" => (:teal, :solid),
                   "PolicyC_Dynamic_v3" => (:darkgreen, :dot),
                   "CPSAT_1w" => (:darkorange, :solid), "CPSAT_8w" => (:red, :dash))
# reference values: JSPLIB metadata; optimizizer.com only where JSPLIB has none (as for ta71)
const REFS = Dict(
    "swv11" => (2991, "JSPLIB metadata (best known upper bound; LB 2983)", false),
    "ta31"  => (1764, "JSPLIB metadata (optimum)", true),
    "ta51"  => (2760, "JSPLIB metadata (optimum)", true),
    "ta72"  => (5181, "optimizizer.com/TA.php (LB = UB = 5181, [Ta1994]; accessed 2026-11-24); not in JSPLIB metadata", true))

base(out) = joinpath(ROOT, "results", out)
final_dir(out, i, m) = joinpath(base(out), "runs", "$(i)__$(m)")
tmp_dir(out, i, m) = final_dir(out, i, m) * ".tmp"
is_done(out, i, m) = isfile(joinpath(final_dir(out, i, m), "DONE"))

"Log line with timestamp, flushed immediately (stdout is a file under nohup)."
say(args...) = (println(now(), "  ", args...); flush(stdout))

# --- durable file writes -----------------------------------------------------------------
fsync_io(io) = (flush(io); ccall(:fsync, Cint, (Cint,), fd(io)))
"Write `df` to `path` atomically: temp file, fsync, rename."
function write_durable(path, df::DataFrame)
    tmp = path * ".part"
    open(tmp, "w") do io
        CSV.write(io, df)
        fsync_io(io)
    end
    mv(tmp, path; force = true)
end
function touch_durable(path)
    open(io -> fsync_io(io), path, "w")
end

# --- order --------------------------------------------------------------------------------
"Our five methods interleaved exactly as in benchmark_large_v1, then all CPSAT_1w, then all CPSAT_8w."
function run_order(insts, methods)
    ours = [m for m in OUR if m in methods]
    seq = [(insts[i], ours[mod1(i + r - 1, length(ours))]) for r in 1:length(ours) for i in eachindex(insts)]
    for v in ("CPSAT_1w", "CPSAT_8w")
        v in methods && append!(seq, [(i, v) for i in insts])
    end
    return seq
end

# --- reference values / environment ----------------------------------------------------------
function write_references(out, insts)
    df = DataFrame(instance = insts, reference = [REFS[i][1] for i in insts],
                   source = [REFS[i][2] for i in insts], proven_optimal = [REFS[i][3] for i in insts])
    write_durable(joinpath(base(out), "reference_values.csv"), df)
end

function write_environment(out, n_runs, T)
    gv = let a = Ref{Cint}(), b = Ref{Cint}(), c = Ref{Cint}()
        JSSPLab.Gurobi.GRBversion(a, b, c); "$(a[]).$(b[]).$(c[])"
    end
    sh(c) = strip(read(c, String))
    cpu, p, e = sh(`sysctl -n machdep.cpu.brand_string`), sh(`sysctl -n hw.perflevel0.physicalcpu`), sh(`sysctl -n hw.perflevel1.physicalcpu`)
    mem = parse(Int, sh(`sysctl -n hw.memsize`)) ÷ 2^30
    py = sh(`$PY -c "import sys; print(sys.version.split()[0])"`)
    ort = sh(`$PY -c "import ortools; print(ortools.__version__)"`)
    open(joinpath(base(out), "environment.txt"), "w") do io
        println(io, "machine: $cpu, $(Sys.CPU_THREADS) cores ($p performance + $e efficiency), $mem GB RAM, macOS $(sh(`sw_vers -productVersion`))")
        println(io, "julia: $(VERSION), gurobi: $gv, python: $py (project venv .venv-cpsat), ortools: $ort")
        println(io, "runs: $n_runs, $(T) s each, strictly sequential (one process at a time), not in parallel with other benchmarks")
        println(io, "our methods: $(join(OUR, ", ")) via scripts/benchmark_large.jl --worker (one Julia process per run, ft06 warm-up)")
        println(io, "  free-y range [$Y_MIN, $Y_MAX] (capped by FREE_CAP = $FREE_CAP), y_floor = effective range minimum for A v2 and B v2,")
        println(io, "  sub-problem TimeLimit $SUB_T s, Gurobi Threads = $THREADS, Seed = $SEED, start = initial_schedule, Big-M ub = its makespan")
        println(io, "CP-SAT: CPSAT_1w = num_workers 1, CPSAT_8w = num_workers 8; random_seed $SEED, hint = initial_schedule (+ its makespan), other parameters default")
        println(io, "note: the machine has only $p performance cores, so the 8-worker variant also runs on efficiency cores")
        println(io, "crash safety: runs/<inst>__<method>.tmp → atomic rename → DONE; see header of scripts/benchmark_large_v2.jl")
        println(io, "written: $(now())")
    end
end

# --- start-up cleanup ----------------------------------------------------------------------
"Kill a still-running child of a killed driver (pid recorded in child.pid), if it is one of ours."
function kill_orphan(dir)
    f = joinpath(dir, "child.pid")
    isfile(f) || return
    pid = tryparse(Int, strip(read(f, String)))
    pid === nothing && return
    cmdline = try strip(read(`ps -p $pid -o command=`, String)) catch; "" end
    if occursin("benchmark_large.jl", cmdline) || occursin("cpsat_jssp.py", cmdline)
        say("killing orphaned child $pid ($(first(cmdline, 80)))")
        run(ignorestatus(`kill -9 $pid`))
        sleep(1)
    end
end

"Delete every run folder that is not complete (no DONE); returns the deleted names."
function cleanup(out)
    runs = joinpath(base(out), "runs")
    removed = String[]
    for d in readdir(runs)
        p = joinpath(runs, d)
        isdir(p) || continue
        # complete only at the final name with DONE (never a *.tmp / *.FAILED folder)
        !endswith(d, ".tmp") && !endswith(d, ".FAILED") && isfile(joinpath(p, "DONE")) && continue
        kill_orphan(p)
        rm(p; recursive = true, force = true)
        push!(removed, d)
    end
    return removed
end

function load_progress(out)
    f = joinpath(base(out), "progress.csv")
    if isfile(f)
        df = CSV.read(f, DataFrame; types = String)
        for c in names(df)
            df[!, c] = coalesce.(df[!, c], "")
        end
    else
        df = DataFrame(instance = String[], method = String[], start = String[], finish = String[], status = String[])
    end
    for r in eachrow(df)
        r.status == "running" || continue
        if is_done(out, r.instance, r.method)
            r.status = "done"
        else
            r.status = "interrupted"
            r.finish = "restart $(now())"
        end
    end
    return df
end

# --- one run -------------------------------------------------------------------------------
"Run a child process (stdout+stderr → log), recording its pid in `dir`/child.pid."
function run_child(cmd, dir, log)
    p = run(pipeline(cmd; stdout = log, stderr = log); wait = false)
    open(io -> (print(io, getpid(p)); fsync_io(io)), joinpath(dir, "child.pid"), "w")
    wait(p)
    return success(p)
end

read_matrix(path) = Matrix{Int}(CSV.read(path, DataFrame))

"Our methods: worker of scripts/benchmark_large.jl, then an independent check of schedule.csv."
function run_ours(out, inst, m, T, dir)
    cmd = `$(Base.julia_cmd()) --project=$ROOT $(joinpath(@__DIR__, "benchmark_large.jl")) --worker --instance=$inst --method=$m --out=$out --time=$T --run-dir=$dir`
    ok = run_child(cmd, dir, joinpath(dir, "worker.log"))
    ok && isfile(joinpath(dir, "done")) && isfile(joinpath(dir, "run.csv")) && isfile(joinpath(dir, "schedule.csv")) ||
        return false, "worker failed (see worker.log)"
    row = CSV.read(joinpath(dir, "run.csv"), DataFrame)[1, :]
    I = load_jsplib(inst)
    feas, errs = check_schedule(I, Schedule(read_matrix(joinpath(dir, "schedule.csv"))); claimed_makespan = row.incumbent, verbose = false)
    feas && row.feasible === true || return false, "final schedule INFEASIBLE: $(first(errs, 3))"
    # the worker's own marker `done` would be the same file as DONE on case-insensitive macOS
    # file systems → rename it, so that DONE is created only by the driver after validation
    mv(joinpath(dir, "done"), joinpath(dir, "worker_finished.txt"))
    return true, "done"
end

"CP-SAT: scripts/cpsat_jssp.py, then validation in Julia; writes run.csv and trace.csv into `dir`."
function run_cpsat(out, inst, m, T, dir)
    w = CPSAT[m]
    input = joinpath(base(out), "input", "$inst.json")
    cmd = `$PY $(joinpath(@__DIR__, "cpsat_jssp.py")) --input $input --output $(joinpath(dir, "cpsat.json")) --workers $w --time $T --seed $SEED`
    ok = run_child(cmd, dir, joinpath(dir, "cpsat.log"))
    ok && isfile(joinpath(dir, "cpsat_meta.csv")) || return false, "cpsat_jssp.py failed (see cpsat.log)"
    md = let d = CSV.read(joinpath(dir, "cpsat_meta.csv"), DataFrame; types = String)
        Dict(r.key => coalesce(r.value, "") for r in eachrow(d))
    end
    status = md["status"]
    obj = isempty(md["objective"]) ? missing : round(Int, parse(Float64, md["objective"]))
    ismissing(obj) && return false, "CP-SAT found no solution (status $status)"
    bnd = isempty(md["bound"]) ? missing : ceil(Int, parse(Float64, md["bound"]) - 1e-6)
    build, wall = parse(Float64, md["build_time"]), parse(Float64, md["wall_time"])
    init_ms = parse(Int, md["initial_makespan"])
    I = load_jsplib(inst)
    st = read_matrix(joinpath(dir, "cpsat_starts.csv"))
    size(st) == (I.n, I.m) || return false, "start-time matrix $(size(st))"
    s = Schedule(st)
    ms_julia = makespan(I, s)
    feas, errs = check_schedule(I, s; claimed_makespan = obj, verbose = false)
    feas || return false, "final schedule INFEASIBLE: $(first(errs, 3))"
    ms_julia == obj || return false, "MAKESPAN MISMATCH: Julia $ms_julia ≠ CP-SAT $obj"
    # trace exactly as in scripts/benchmark_cpsat.jl: initial at 0, solutions shifted by the
    # build time, improvements only, carried forward to max(T, runtime)
    tr = CSV.read(joinpath(dir, "cpsat_trace.csv"), DataFrame)
    times = vcat(0.0, build .+ tr.time)
    incs = vcat(init_ms, round.(Int, tr.objective))
    bnds = vcat(missing, ceil.(Int, tr.bound .- 1e-6))
    keep = [1; [i for i in 2:length(incs) if incs[i] < minimum(incs[1:i-1])]]
    times, incs, bnds = times[keep], incs[keep], bnds[keep]
    runtime = build + wall
    imp = times[2:end]
    push!(times, max(T, runtime)); push!(incs, incs[end]); push!(bnds, bnd)
    CSV.write(joinpath(dir, "trace.csv"), DataFrame(time = times, incumbent = incs, bound = bnds))
    ref = REFS[inst][1]
    CSV.write(joinpath(dir, "run.csv"), DataFrame([(instance = inst, method = m, reference = ref, initial = init_ms,
        incumbent = obj, primal_gap = (obj - ref) / ref, bound = bnd, status = status, runtime = runtime,
        feasible = feas, incumbents_checked = 1, incumbents_infeasible = 0,
        peak_rss_mb = round(parse(Float64, md["peak_rss_mb"]), digits = 1), threads = w,
        time_last_improvement = isempty(imp) ? missing : last(imp), num_workers = w, cpsat_status = status,
        final_bound = bnd, build_time = build, makespan_julia = ms_julia, n_solutions = nrow(tr))]))
    return true, "done"
end

function drive(out, T, insts, methods)
    mkpath(joinpath(base(out), "runs"))
    keep_awake()
    progress = load_progress(out)
    removed = cleanup(out)
    isempty(removed) || say("removed incomplete run folders (re-run from scratch): ", join(removed, ", "))
    write_durable(joinpath(base(out), "progress.csv"), progress)
    todo = run_order(insts, methods)
    write_references(out, insts)
    write_environment(out, length(todo), T)
    if any(m -> haskey(CPSAT, m), methods)
        run(`$(Base.julia_cmd()) --project=$ROOT $(joinpath(@__DIR__, "export_instances.jl")) --out=$(joinpath(base(out), "input")) --instances=$(join(insts, ","))`)
    end
    for (inst, m) in todo
        if is_done(out, inst, m)
            say("$inst $m: DONE earlier, skipped")
            continue
        end
        dir, fin = tmp_dir(out, inst, m), final_dir(out, inst, m)
        rm(dir; recursive = true, force = true); rm(fin; recursive = true, force = true)
        mkpath(dir)
        push!(progress, (inst, m, string(now()), "", "running"))
        write_durable(joinpath(base(out), "progress.csv"), progress)
        say("$inst $m: started")
        ok, msg = try
            haskey(CPSAT, m) ? run_cpsat(out, inst, m, T, dir) : run_ours(out, inst, m, T, dir)
        catch err
            false, "driver error: $(sprint(showerror, err))"
        end
        if ok
            isfile(joinpath(dir, "DONE")) && error("unexpected DONE in $dir before validation")
            rm(joinpath(dir, "child.pid"); force = true)
            mv(dir, fin)                                    # atomic rename (same file system)
            touch_durable(joinpath(fin, "DONE"))
            run(`sync`)
        else
            mv(dir, fin * ".FAILED"; force = true)          # kept for inspection, redone on restart
        end
        progress[end, :finish] = string(now())
        progress[end, :status] = ok ? "done" : "FAILED: $msg"
        write_durable(joinpath(base(out), "progress.csv"), progress)
        say("$inst $m: ", progress[end, :status])
    end
    aggregate(out, T, insts, methods)
end

# --- aggregation (reads only DONE folders, writes only top-level files) --------------------
"Same formula as scripts/benchmark_large.jl: step-function mean gap (%) over [0, T]."
function primal_integral(times, incs, ref, T)
    area, t_prev, g_prev = 0.0, 0.0, 100 * (incs[1] - ref) / ref
    for (t, z) in zip(times, incs)
        t = min(t, T)
        area += g_prev * (t - t_prev)
        t_prev, g_prev = t, 100 * (z - ref) / ref
    end
    return (area + g_prev * (T - t_prev)) / T
end

function gap_panel(traces, inst, methods, ref, T; legend = false)
    q = plot(; xscale = :log10, xlabel = "time [s] (log)", ylabel = "primal gap [%]", title = inst,
             titlefontsize = 10, xlims = (1, T * 1.05), legend = legend)
    for m in methods
        t = sort(traces[(traces.instance .== inst) .& (traces.method .== m) .& .!ismissing.(traces.incumbent), :], :time)
        isempty(t) && continue
        c, ls = STYLE[m]
        plot!(q, max.(t.time, 1.0), 100 .* (t.incumbent .- ref) ./ ref; seriestype = :steppost, lw = 2, color = c, ls = ls, label = m)
    end
    return q
end

function aggregate(out, T, insts, methods)
    B = base(out)
    write_references(out, insts)
    runs, traces, its, logs = DataFrame(), DataFrame(), DataFrame(), DataFrame()
    for inst in insts, m in methods
        is_done(out, inst, m) || continue
        dir = final_dir(out, inst, m)
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
            hasproperty(lg, :skipped) && (lg = lg[.!coalesce.(lg.skipped, false), :])   # C v3: skipped = no solve
            append!(logs, DataFrame(instance = inst, method = m, type = string.(lg.type),
                                    improved = coalesce.(lg.improved, false)); cols = :union, promote = true)
        end
    end
    isempty(runs) && return println("no finished runs")
    # reference and primal gap from this benchmark's reference table (the worker only knows JSPLIB)
    runs.reference = [REFS[i][1] for i in runs.instance]
    runs.primal_gap = (runs.incumbent .- runs.reference) ./ runs.reference
    runs.run_check = map(eachrow(runs)) do r
        t_end = maximum(traces[(traces.instance .== r.instance) .& (traces.method .== r.method), :time])
        if haskey(CPSAT, r.method)
            r.status == "OPTIMAL" && r.runtime < T - 5 ? "cpsat optimal" : abs(r.runtime - T) <= 5 ? "full" : "INCOMPLETE"
        else
            abs(r.runtime - T) <= 5 ? "full" :
            r.method == "gurobi" && r.status == "OPTIMAL" ? "gurobi optimal" :
            hasproperty(r, :stop_reason) && isequal(r.stop_reason, "exhausted") && t_end >= T - 5 ?
                "C v3 exhausted (trace carried to T)" : "INCOMPLETE"
        end
    end
    sort!(runs, [order(:instance, by = i -> findfirst(==(i), insts)), order(:method, by = m -> findfirst(==(m), ALL))])
    CSV.write(joinpath(B, "runs.csv"), runs)
    CSV.write(joinpath(B, "traces.csv"), select(traces, :instance, :method, :time, :incumbent, :bound))
    isempty(its) || CSV.write(joinpath(B, "iterations.csv"), its)

    cmp = DataFrame()
    for r in eachrow(runs)
        t = sort(traces[(traces.instance .== r.instance) .& (traces.method .== r.method) .& .!ismissing.(traces.incumbent), :], :time)
        push!(cmp, (instance = r.instance, method = r.method, incumbent = r.incumbent, primal_gap = r.primal_gap,
                    primal_integral = primal_integral(t.time, t.incumbent, r.reference, T)); promote = true)
    end
    CSV.write(joinpath(B, "comparison.csv"), cmp)

    # sub-problem statistics per instance × fix-and-optimize method
    sub = DataFrame()
    if !isempty(its)
        for inst in insts, m in FAO
            d = its[(its.instance .== inst) .& (its.method .== m) .& (its.sub_status .!= "SKIPPED_DUPLICATE"), :]
            isempty(d) && continue
            tl = d[d.sub_status .== "TIME_LIMIT", :]
            push!(sub, (instance = inst, method = m, gurobi_calls = nrow(d), median_free_y = median(d.n_free_y),
                        median_sub_time = median(d.sub_time), mean_sub_time = mean(d.sub_time),
                        share_optimal = count(==("OPTIMAL"), d.sub_status) / nrow(d),
                        share_time_limit = nrow(tl) / nrow(d),
                        share_solve_time_in_time_limited = sum(tl.sub_time; init = 0.0) / sum(d.sub_time),
                        # Gurobi reports gap 1e100 when it has none (sub-problem cut after ~0.1 s
                        # by the end of the run) → excluded from the median, counted separately
                        median_mip_gap_time_limited = let g = [x for x in skipmissing(tl.sub_gap) if x < 1e10]
                            isempty(g) ? missing : median(g) end,
                        time_limited_without_gap = count(x -> ismissing(x) || x >= 1e10, tl.sub_gap),
                        improving_iterations = count(coalesce.(d.improved, false))); promote = true)
        end
        CSV.write(joinpath(B, "subproblems.csv"), sub)
    end
    if !isempty(logs)
        ty = combine(groupby(logs, [:instance, :method, :type]), nrow => :iterations, :improved => (x -> count(x)) => :improving_iterations)
        CSV.write(joinpath(B, "types.csv"), ty)
    end

    # plots
    present = [m for m in ALL if m in runs.method]
    insts_p = [i for i in insts if i in runs.instance]
    lay = length(insts_p) == 4 ? (2, 2) : (1, length(insts_p))
    for inst in insts_p
        ref = REFS[inst][1]
        p = plot(; xscale = :log10, xlabel = "time [s] (log)", ylabel = "makespan", title = inst,
                 legend = :topright, size = (900, 550), dpi = 120, titlefontsize = 11, xlims = (1, T * 1.05))
        for m in present
            t = sort(traces[(traces.instance .== inst) .& (traces.method .== m) .& .!ismissing.(traces.incumbent), :], :time)
            isempty(t) && continue
            c, ls = STYLE[m]
            plot!(p, max.(t.time, 1.0), t.incumbent; seriestype = :steppost, lw = 2, color = c, ls = ls, label = m)
        end
        hline!(p, [ref]; color = :black, ls = :dash, label = "reference $ref")
        savefig(p, joinpath(B, "anytime_$inst.png"))
    end
    panels = [gap_panel(traces, i, present, REFS[i][1], T; legend = i == insts_p[1] ? :topright : false) for i in insts_p]
    savefig(plot(panels...; layout = lay, size = (1200, 850), dpi = 110), joinpath(B, "anytime_gap_all.png"))
    fao_p = [m for m in FAO if m in runs.method]
    xs = 1:length(insts_p)
    for (col, file, lbl, ysc) in ((:median_sub_time, "median_sub_time.png", "median sub-problem time [s] (log)", :log10),
                                  (:share_time_limit, "share_time_limit.png", "share of sub-problems hitting 7.5 s", :identity))
        p = plot(; xticks = (xs, insts_p), ylabel = lbl, yscale = ysc, size = (850, 480), dpi = 120,
                 legend = :outertopright, title = string(col), titlefontsize = 11)
        for m in fao_p
            v = [let r = runs[(runs.instance .== i) .& (runs.method .== m), :]
                     isempty(r) || ismissing(r[1, col]) ? NaN : max(Float64(r[1, col]), ysc == :log10 ? 1e-4 : 0.0) end
                 for i in insts_p]
            c, ls = STYLE[m]
            plot!(p, xs, v; marker = :circle, lw = 2, color = c, ls = ls, label = m)
        end
        savefig(p, joinpath(B, file))
    end
    if !isempty(its)
        solved = its[its.sub_status .!= "SKIPPED_DUPLICATE", :]
        bp = map(insts_p) do i
            d = solved[solved.instance .== i, :]
            q = plot(; title = i, ylabel = "free y", titlefontsize = 10, legend = false,
                     xticks = (1:length(fao_p), [SHORT[m] for m in fao_p]))
            for (k, m) in enumerate(fao_p)          # box = quartiles, whiskers = min/max
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
        savefig(plot(bp...; layout = lay, size = (1100, 800), dpi = 110), joinpath(B, "free_y_distribution.png"))
    end
    println("aggregated $(nrow(runs)) runs → $B")
    show(stdout, select(runs, :instance, :method, :incumbent, :primal_gap, :status, :runtime, :feasible,
                        :peak_rss_mb, :run_check); allrows = true)
    println()
    out == DEFAULT_OUT && combined(T)
end

# --- combined 8-instance view (old folders read only) -------------------------------------
function combined(T)
    V1, C1 = joinpath(ROOT, "results", "benchmark_large_v1"), joinpath(ROOT, "results", "benchmark_large_cpsat_v1")
    C = joinpath(ROOT, "results", "benchmark_large_combined")
    mkpath(C)
    rd(p) = CSV.read(p, DataFrame)
    c1 = rd(joinpath(C1, "comparison.csv"))
    cmp = vcat(select(rd(joinpath(V1, "comparison.csv")), :instance, :method, :incumbent, :primal_gap, :primal_integral),
               select(c1[startswith.(c1.method, "CPSAT"), :], :instance, :method, :incumbent, :primal_gap, :primal_integral),
               select(rd(joinpath(base(DEFAULT_OUT), "comparison.csv")), :instance, :method, :incumbent, :primal_gap, :primal_integral))
    refs = vcat(select(rd(joinpath(V1, "reference_values.csv")), :instance, :reference, :source),
                select(rd(joinpath(base(DEFAULT_OUT), "reference_values.csv")), :instance, :reference, :source))
    cmp = leftjoin(cmp, refs, on = :instance)
    insts8 = unique(cmp.instance)
    # rank per instance by final makespan among all methods present (ties: average rank)
    cmp.rank = zeros(nrow(cmp))
    cmp.best_of_our_five = falses(nrow(cmp))
    for i in insts8
        idx = findall(==(i), cmp.instance)
        z = cmp.incumbent[idx]
        cmp.rank[idx] = [count(<(v), z) + (count(==(v), z) + 1) / 2 for v in z]
        ours = [k for k in idx if cmp.method[k] in OUR]
        best = minimum(cmp.incumbent[ours])
        foreach(k -> cmp.best_of_our_five[k] = cmp.incumbent[k] == best, ours)
    end
    sort!(cmp, [order(:instance, by = i -> findfirst(==(i), insts8)), order(:method, by = m -> findfirst(==(m), ALL))])
    CSV.write(joinpath(C, "comparison_all.csv"), cmp)
    summ = combine(groupby(cmp, :method), nrow => :instances, :primal_gap => mean => :mean_primal_gap,
                   :primal_integral => mean => :mean_primal_integral, :rank => mean => :mean_rank,
                   :best_of_our_five => (x -> count(x)) => :n_best_of_our_five)
    summ.n_best_of_our_five = [m in OUR ? v : missing for (m, v) in zip(summ.method, summ.n_best_of_our_five)]
    sort!(summ, order(:method, by = m -> findfirst(==(m), ALL)))
    CSV.write(joinpath(C, "summary.csv"), summ)
    tr = vcat(select(rd(joinpath(V1, "traces.csv")), :instance, :method, :time, :incumbent),
              select(rd(joinpath(C1, "traces.csv")), :instance, :method, :time, :incumbent),
              select(rd(joinpath(base(DEFAULT_OUT), "traces.csv")), :instance, :method, :time, :incumbent))
    ref_of = Dict(r.instance => r.reference for r in eachrow(refs))
    panels = [gap_panel(tr, i, ALL, ref_of[i], T; legend = i == insts8[1] ? :topright : false) for i in insts8]
    savefig(plot(panels...; layout = (2, cld(length(panels), 2)), size = (2000, 900), dpi = 100,
                 left_margin = 10Plots.mm, bottom_margin = 8Plots.mm),
            joinpath(C, "anytime_gap_all8.png"))
    open(joinpath(C, "README.txt"), "w") do io
        println(io, "Combined view, built by scripts/benchmark_large_v2.jl from (read only):")
        println(io, "  results/benchmark_large_v1 (our 5 methods; ta41, ta61, swv12, ta71)")
        println(io, "  results/benchmark_large_cpsat_v1 (CPSAT_1w / CPSAT_8w on the same 4 instances)")
        println(io, "  results/benchmark_large_v2 (all 7 methods; swv11, ta31, ta51, ta72)")
        println(io, "rank: per instance by final makespan among the 7 methods (1 = best, ties averaged)")
        println(io, "n_best_of_our_five: instances where the method has the best final makespan of random,")
        println(io, "  gurobi, A v2, B v2, C v3 (ties counted for every tied method)")
        println(io, "written: $(now())")
    end
    println("combined: $(length(insts8)) instances → $C")
    show(stdout, summ; allrows = true); println()
end

function main(args = ARGS)
    opt(k, d) = (i = findfirst(startswith("--$k="), args); i === nothing ? d : split(args[i], '='; limit = 2)[2])
    out = String(opt("out", DEFAULT_OUT))
    T = parse(Float64, opt("time", "1200"))
    insts = String.(split(opt("instances", join(INSTANCES, ",")), ','))
    methods = String.(split(opt("methods", join(ALL, ",")), ','))
    all(in(ALL), methods) || error("unknown method in $methods")
    "--aggregate" in args ? aggregate(out, T, insts, methods) : drive(out, T, insts, methods)
end

end # module JSSPLabLargeV2

JSSPLabLargeV2.main()
