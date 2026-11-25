# =============================================================================
# benchmark_large_v3.jl — large benchmark on 5 more instances (ta21, yn1, ta42, ta62, swv13), 7 methods
# =============================================================================
#
# Generated as a mechanical copy of scripts/benchmark_large_v2_seed.jl (itself a copy of
# scripts/benchmark_large_v2.jl): driver, crash safety (.tmp → atomic rename → DONE), stall check
# after every run (1 rerun), CP-SAT validation, aggregation and plots are unchanged.
# Differences: the instance list, the reference values (with lower bounds), the output folders
# (results/benchmark_large_v3 for seed 42, results/benchmark_large_v3_seed<seed> otherwise), the
# seed comparison (seed 42 = results/benchmark_large_v3) and the final combined evaluation
# (--combined-13x2 → results/benchmark_large_combined_13x2). All method settings as in v2.
#
#   julia --project=. scripts/benchmark_large_v3.jl [--seed=42|43]            # full benchmark
#   julia --project=. scripts/benchmark_large_v3.jl --seed=42 --dry-run       # config only
#   julia --project=. scripts/benchmark_large_v3.jl --seed=42 --time=60 --instances=ta42 \
#         --out=benchmark_large_v3/smoke                                      # smoke test
#   julia --project=. scripts/benchmark_large_v3.jl --seed=… --check | --aggregate | --seed-compare
#   julia --project=. scripts/benchmark_large_v3.jl --combined-13x2
# =============================================================================

import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")); io = devnull)
ENV["GKSwstype"] = "100"     # GR without window (file output)

module JSSPLabLargeV3

using CSV, DataFrames, Dates, Plots, Statistics
include(joinpath(@__DIR__, "..", "src", "JSSPLab.jl"))
using .JSSPLab

const ROOT = normpath(joinpath(@__DIR__, ".."))
const PY = joinpath(ROOT, ".venv-cpsat", "bin", "python")
const V2_OUT = "benchmark_large_v2"            # settings reference for the dry run (read only)
const BASE42 = "benchmark_large_v3"            # seed-42 results of this benchmark
const INSTANCES = ["ta21", "yn1", "ta42", "ta62", "swv13"]
const OUR = ["random", "gurobi", "PolicyA_ScoreBased_fill", "PolicyB_StructureBased_v2", "PolicyC_Dynamic_v3"]
const FAO = filter(!=("gurobi"), OUR)
const CPSAT = Dict("CPSAT_1w" => 1, "CPSAT_8w" => 8)
const ALL = vcat(OUR, ["CPSAT_1w", "CPSAT_8w"])
const SEED = Ref(42)                           # set from --seed (the only change vs v2)
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
const REFS = Dict(      # (reference = best known UB, source, proven optimal, LB); public source where JSPLIB is outdated
    "ta21"  => (1642, "optimizizer.com/TA.php (LB = UB = 1642: LB [CPO2015], UB [Shylo2008]; accessed 2026-11-25); JSPLIB metadata outdated (LB 1539, UB 1644)", true, 1642),
    "yn1"   => (884, "optimizizer.com/Extra.php (Yn01: LB 869 [YMRL2026], UB 884 [GTA2013]; accessed 2026-11-25); JSPLIB metadata outdated (LB 826, UB 885)", false, 869),
    "ta42"  => (1939, "optimizizer.com/TA.php (LB 1887 [RL2024], UB 1939 [PMAEDC2019]; accessed 2026-11-25); JSPLIB metadata outdated (LB 1867, UB 1956)", false, 1887),
    "ta62"  => (2869, "JSPLIB metadata (optimum); optimizizer.com/TA.php agrees (LB = UB = 2869, accessed 2026-11-25)", true, 2869),
    "swv13" => (3104, "JSPLIB metadata (optimum); optimizizer.com/Extra.php agrees (LB = UB = 3104, accessed 2026-11-25)", true, 3104))

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
    df = DataFrame(instance = insts, reference = [REFS[i][1] for i in insts], lower_bound = [REFS[i][4] for i in insts],
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
        println(io, "our methods: $(join(OUR, ", ")) via scripts/benchmark_seed.jl --worker --seed=$(SEED[]) (same worker code as v2, one Julia process per run, ft06 warm-up)")
        println(io, "  free-y range [$Y_MIN, $Y_MAX] (capped by FREE_CAP = $FREE_CAP), y_floor = effective range minimum for A v2 and B v2,")
        println(io, "  sub-problem TimeLimit $SUB_T s, Gurobi Threads = $THREADS, Seed = $(SEED[]), start = initial_schedule, Big-M ub = its makespan")
        println(io, "CP-SAT: CPSAT_1w = num_workers 1, CPSAT_8w = num_workers 8; random_seed $(SEED[]), hint = initial_schedule (+ its makespan), other parameters default")
        println(io, "note: the machine has only $p performance cores, so the 8-worker variant also runs on efficiency cores")
        println(io, "crash safety: runs/<inst>__<method>.tmp → atomic rename → DONE; see header of scripts/benchmark_large_v2.jl; stall check after every run (scripts/stall_check.py)")
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
    if occursin("benchmark_seed.jl", cmdline) || occursin("cpsat_jssp.py", cmdline)
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

"Our methods: worker of scripts/benchmark_seed.jl (seed parameter), then an independent check of schedule.csv."
function run_ours(out, inst, m, T, dir)
    cmd = `$(Base.julia_cmd()) --project=$ROOT $(joinpath(@__DIR__, "benchmark_seed.jl")) --worker --instance=$inst --method=$m --seed=$(SEED[]) --time=$T --run-dir=$dir`
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
    cmd = `$PY $(joinpath(@__DIR__, "cpsat_jssp.py")) --input $input --output $(joinpath(dir, "cpsat.json")) --workers $w --time $T --seed $(SEED[])`
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

"""
Stall check of one finished run (scripts/stall_check.py). Attempt 1: a stalled run is moved to
runs_invalidated/ (→ rerun); attempt 2: a stalled run is kept and marked in repair_log.txt.
Returns true if the run is stalled.
"""
function stall_check(out, inst, m, attempt)
    opt = attempt == 1 ? `--invalidate --reason=stalled-after-run` : `--retry-note=stalled-again-after-rerun`
    cmd = `$PY $(joinpath(@__DIR__, "stall_check.py")) $(base(out)) --runs=$(inst)__$(m) --scan-dir=$(joinpath(base(out), "stall_checks")) $opt`
    p = run(pipeline(ignorestatus(cmd); stdout = stdout, stderr = stderr))
    flush(stdout)
    p.exitcode == 1 && say("WARNING: stall check of $inst $m failed (exit 1); run kept as not stalled")
    return p.exitcode == 2
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
        for attempt in 1:2                                  # second attempt only after a stall
            dir, fin = tmp_dir(out, inst, m), final_dir(out, inst, m)
            rm(dir; recursive = true, force = true); rm(fin; recursive = true, force = true)
            mkpath(dir)
            push!(progress, (inst, m, string(now()), "", "running"))
            write_durable(joinpath(base(out), "progress.csv"), progress)
            say("$inst $m: started", attempt > 1 ? " (rerun after stall)" : "")
            ok, msg = try
                haskey(CPSAT, m) ? run_cpsat(out, inst, m, T, dir) : run_ours(out, inst, m, T, dir)
            catch err
                false, "driver error: $(sprint(showerror, err))"
            end
            if ok
                isfile(joinpath(dir, "DONE")) && error("unexpected DONE in $dir before validation")
                rm(joinpath(dir, "child.pid"); force = true)
                mv(dir, fin)                                # atomic rename (same file system)
                touch_durable(joinpath(fin, "DONE"))
                run(`sync`)
            else
                mv(dir, fin * ".FAILED"; force = true)      # kept for inspection, redone on restart
            end
            stalled = ok && stall_check(out, inst, m, attempt)
            progress[end, :finish] = string(now())
            progress[end, :status] = !ok ? "FAILED: $msg" : !stalled ? "done" :
                attempt == 1 ? "STALLED → invalidated, rerun" : "done (STALLED again, kept; see repair_log.txt)"
            write_durable(joinpath(base(out), "progress.csv"), progress)
            say("$inst $m: ", progress[end, :status])
            (ok && stalled && attempt == 1) || break
        end
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
end

# --- v3 modes ----------------------------------------------------------------------------------
rank_of(z) = [count(<(v), z) + (count(==(v), z) + 1) / 2 for v in z]    # 1 = best, ties averaged

"Print the configuration and compare the settings with what v2 actually ran (only instances and seed differ)."
function dry_run(out, T, insts, methods)
    println("output:        results/$out")
    println("seed:          $(SEED[])  → worker (random_fao: MersenneTwister for budget draws / random release + sub-problem Gurobi Seed; solve_model: Gurobi Seed) and CP-SAT random_seed")
    println("instances:     ", join(insts, ", "))
    println("methods:       ", join(methods, ", "))
    println("order:         ", join(["$i/$(SHORT[m])" for (i, m) in run_order(insts, methods)], ", "))
    v2 = CSV.read(joinpath(base(V2_OUT), "runs.csv"), DataFrame)
    ours = v2[.!startswith.(v2.method, "CPSAT"), :]
    cps = v2[startswith.(v2.method, "CPSAT"), :]
    env2 = read(joinpath(base(V2_OUT), "environment.txt"), String)
    T2 = parse(Float64, match(r"runs: \d+, ([\d.]+) s each", env2)[1])
    println("settings (this | results/benchmark_large_v2):")
    chk(name, mine, theirs) = println("  ", rpad(name, 26), "$mine | $theirs → ", isequal(mine, theirs) ? "identical" : "DIFFERENT")
    chk("methods", sort(methods), sort(unique(String.(v2.method))))
    chk("time per run [s]", T, T2)
    chk("free-y range (before cap)", "$Y_MIN-$Y_MAX", only(unique(string.(ours.y_range_effective))))
    chk("FREE_CAP", FREE_CAP, 0.75)
    chk("sub-problem TimeLimit [s]", SUB_T, only(unique(skipmissing(ours.sub_time_limit))))
    chk("Gurobi Threads", THREADS, only(unique(ours.threads)))
    chk("CP-SAT num_workers", sort(collect(values(CPSAT))), sort(unique(cps.num_workers)))
    chk("y_floor rule", "min(250, cap)", "min(250, cap)")
    println("  ", rpad("seed", 26), "$(SEED[]) | 42")
    println("per instance (pairs, effective free-y range = [min(250,cap), min(4000,cap)], cap = floor(0.75·pairs)):")
    for i in insts
        I = load_jsplib(i)
        pairs = length(machine_pairs(I)); cap = floor(Int, FREE_CAP * pairs)
        init = makespan(I, initial_schedule(I))
        js = joinpath(base(out), "input", "$i.json")
        jinit = isfile(js) ? parse(Int, match(r"\"initial_makespan\":(\d+)", read(js, String))[1]) : missing
        println("  $i: $(I.n)×$(I.m), pairs $pairs, free-y range $(min(Y_MIN, cap))-$(min(Y_MAX, cap)), y_floor $(min(Y_MIN, cap)), ",
                "initial makespan $init", ismissing(jinit) ? "" : " (exported json $jinit → $(jinit == init ? "same" : "DIFFERENT"))",
                ", reference $(REFS[i][1]) (LB $(REFS[i][4]), proven optimal $(REFS[i][3]))")
    end
end

"Validate an aggregated folder (seed everywhere, feasibility, run_check, free y, CP-SAT makespan)."
function check(out, T, insts, methods)
    f = joinpath(base(out), "runs.csv")
    isfile(f) || (println("CHECK FAILED: no runs.csv in $(base(out))"); return false)
    runs = CSV.read(f, DataFrame)
    col(r, c) = hasproperty(r, c) ? r[c] : missing
    bad = String[]
    for i in insts, m in methods
        is_done(out, i, m) || push!(bad, "$i $m: no DONE")
    end
    for r in eachrow(runs)
        tag = "$(r.instance) $(r.method)"
        r.feasible === true || push!(bad, "$tag: final schedule infeasible")
        r.run_check in ("full", "gurobi optimal", "cpsat optimal", "C v3 exhausted (trace carried to T)") ||
            push!(bad, "$tag: run_check = $(r.run_check)")
        if haskey(CPSAT, r.method)
            isequal(col(r, :makespan_julia), r.incumbent) || push!(bad, "$tag: CP-SAT makespan ≠ Julia makespan")
            md = CSV.read(joinpath(final_dir(out, r.instance, r.method), "cpsat_meta.csv"), DataFrame; types = String)
            sd = md.value[findfirst(==("seed"), md.key)]
            sd == string(SEED[]) || push!(bad, "$tag: CP-SAT random_seed $sd")
        else
            isequal(col(r, :seed), SEED[]) || push!(bad, "$tag: seed $(col(r, :seed))")
            isequal(col(r, :gurobi_Seed), SEED[]) || push!(bad, "$tag: Gurobi Seed $(col(r, :gurobi_Seed))")
            coalesce(col(r, :incumbents_infeasible), 0) == 0 || push!(bad, "$tag: infeasible incumbents")
            r.method == "gurobi" && continue
            coalesce(col(r, :free_y_violations), 1) == 0 || push!(bad, "$tag: $(col(r, :free_y_violations)) free-y violations")
            coalesce(col(r, :max_sub_time_limit_hit), 0.0) <= SUB_T + 0.5 ||
                push!(bad, "$tag: sub-problem took $(col(r, :max_sub_time_limit_hit)) s")
        end
    end
    if isempty(bad)
        println("CHECK OK: $(nrow(runs)) runs, all DONE, feasible, run_check passed, free y in range, seed $(SEED[]) everywhere")
        return true
    end
    println("CHECK FAILED: ", join(bad, "; "))
    return false
end

"Seed 42 (results/benchmark_large_v3) vs this seed: seed_comparison.csv + .md (7 methods)."
function seed_compare(out)
    new = CSV.read(joinpath(base(out), "comparison.csv"), DataFrame)
    old = CSV.read(joinpath(base(BASE42), "comparison.csv"), DataFrame)
    s = SEED[]
    rows = DataFrame()
    for i in INSTANCES, m in ALL
        a = old[(old.instance .== i) .& (old.method .== m), :]
        b = new[(new.instance .== i) .& (new.method .== m), :]
        (isempty(a) || isempty(b)) && continue
        push!(rows, (instance = i, method = m,
            makespan_seed42 = a.incumbent[1], makespan_new = b.incumbent[1],
            gap_seed42 = a.primal_gap[1], gap_new = b.primal_gap[1],
            primal_integral_seed42 = a.primal_integral[1], primal_integral_new = b.primal_integral[1],
            mean_makespan = (a.incumbent[1] + b.incumbent[1]) / 2, mean_gap = (a.primal_gap[1] + b.primal_gap[1]) / 2,
            mean_primal_integral = (a.primal_integral[1] + b.primal_integral[1]) / 2,
            diff_makespan = b.incumbent[1] - a.incumbent[1], diff_gap = b.primal_gap[1] - a.primal_gap[1],
            diff_primal_integral = b.primal_integral[1] - a.primal_integral[1]); promote = true)
    end
    rows.rank_seed42 = zeros(nrow(rows)); rows.rank_new = zeros(nrow(rows))
    for i in unique(rows.instance)
        idx = findall(==(i), rows.instance)
        rows.rank_seed42[idx] = rank_of(rows.makespan_seed42[idx])
        rows.rank_new[idx] = rank_of(rows.makespan_new[idx])
    end
    CSV.write(joinpath(base(out), "seed_comparison.csv"),
              rename(rows, [c => replace(c, "_new" => "_seed$s") for c in names(rows) if endswith(c, "_new")]))
    pct(x) = string(round(100x, digits = 2), " %")
    open(joinpath(base(out), "seed_comparison.md"), "w") do io
        println(io, "# Seed comparison: seed 42 (results/benchmark_large_v3) vs seed $s\n")
        println(io, "Same configuration except the seed (fix-and-optimize RNG, Gurobi Seed, CP-SAT random_seed); 20-minute runs.")
        println(io, "Rank = by final makespan among the 7 methods per instance (1 = best, ties averaged).\n")
        println(io, "## Ranking per instance\n")
        println(io, "| instance | seed 42 (best → worst) | seed $s (best → worst) | changed |")
        println(io, "|---|---|---|---|")
        for i in unique(rows.instance)
            d = rows[rows.instance .== i, :]
            o42 = join([SHORT[m] for m in d.method[sortperm(d.rank_seed42)]], " < ")
            onew = join([SHORT[m] for m in d.method[sortperm(d.rank_new)]], " < ")
            ch = [SHORT[r.method] * " $(r.rank_seed42)→$(r.rank_new)" for r in eachrow(d) if r.rank_seed42 != r.rank_new]
            println(io, "| $i | $o42 | $onew | ", isempty(ch) ? "no" : join(ch, ", "), " |")
        end
        println(io, "\nMean rank over the instances (seed 42 → seed $s): ",
                join(["$(SHORT[m]) $(round(mean(rows.rank_seed42[rows.method .== m]), digits = 2)) → $(round(mean(rows.rank_new[rows.method .== m]), digits = 2))"
                      for m in ALL if m in rows.method], ", "), "\n")
        println(io, "## Largest primal-gap differences (seed $s − seed 42)\n")
        println(io, "| instance | method | gap seed 42 | gap seed $s | difference | makespan 42 → $s |")
        println(io, "|---|---|---|---|---|---|")
        for r in eachrow(first(sort(rows, :diff_gap; by = abs, rev = true), 8))
            println(io, "| $(r.instance) | $(SHORT[r.method]) | $(pct(r.gap_seed42)) | $(pct(r.gap_new)) | $(pct(r.diff_gap)) | $(r.makespan_seed42) → $(r.makespan_new) |")
        end
        println(io, "\nMean absolute gap difference: ", pct(mean(abs.(rows.diff_gap))), "; per method: ",
                join(["$(SHORT[m]) $(pct(mean(abs.(rows.diff_gap[rows.method .== m]))))" for m in ALL if m in rows.method], ", "))
    end
    println("seed comparison → $(joinpath(base(out), "seed_comparison.csv")) / .md")
end

"All 13 instances × 2 seeds → results/benchmark_large_combined_13x2 (all inputs read only)."
function combined_13x2(T)
    R = joinpath(ROOT, "results")
    C = joinpath(R, "benchmark_large_combined_13x2")
    mkpath(C)
    V1I = ["ta41", "ta61", "swv12", "ta71"]; V2I = ["swv11", "ta31", "ta51", "ta72"]
    CP = ["CPSAT_1w", "CPSAT_8w"]
    # (seed, folder, instances, methods expected, has runs/ folders)
    inputs = [(42, "benchmark_large_v1", V1I, OUR, true), (42, "benchmark_large_cpsat_v1", V1I, CP, false),
              (43, "benchmark_large_v1_seed43", V1I, OUR, true),
              (42, "benchmark_large_v2", V2I, ALL, true), (43, "benchmark_large_v2_seed43", V2I, ALL, true),
              (42, "benchmark_large_v3", INSTANCES, ALL, true), (43, "benchmark_large_v3_seed43", INSTANCES, ALL, true)]
    rows, traces, refs, notes = DataFrame(), DataFrame(), Dict{String,Float64}(), String[]
    for (s, f, insts, meths, hasruns) in inputs
        F = joinpath(R, f)
        cmpf = joinpath(F, "comparison.csv")
        cmp = isfile(cmpf) ? CSV.read(cmpf, DataFrame) : DataFrame(instance = String[], method = String[])
        isfile(cmpf) || push!(notes, "$f: no comparison.csv (folder incomplete)")
        rf = joinpath(F, "reference_values.csv")
        isfile(rf) && foreach(r -> refs[String(r.instance)] = r.reference, eachrow(CSV.read(rf, DataFrame)))
        tf = joinpath(F, "traces.csv")
        if isfile(tf)
            t = select(CSV.read(tf, DataFrame), :instance, :method, :time, :incumbent)
            t.seed .= s
            append!(traces, t[in.(t.method, Ref(meths)), :]; cols = :union, promote = true)
        end
        stalled = Set{Tuple{String,String}}()
        if hasruns && isdir(joinpath(F, "runs"))      # unrepaired stalls (scan only, nothing is moved)
            sd = joinpath(C, "stall_scans", f)
            run(pipeline(ignorestatus(`$PY $(joinpath(@__DIR__, "stall_check.py")) $F --scan-dir=$sd --quiet`); stdout = devnull))
            sr = joinpath(sd, "stall_runs.csv")
            isfile(sr) && foreach(r -> string(r.stalled) == "true" && push!(stalled, (String(r.instance), String(r.method))),
                                  eachrow(CSV.read(sr, DataFrame)))
        end
        for i in insts, m in meths
            r = cmp[(string.(cmp.instance) .== i) .& (string.(cmp.method) .== m), :]
            flag = isempty(r) ? "MISSING" : (i, m) in stalled ? "STALLED (unrepaired)" : ""
            push!(rows, (instance = i, seed = s, method = m, source = f,
                         incumbent = isempty(r) ? missing : r.incumbent[1],
                         primal_gap = isempty(r) ? missing : r.primal_gap[1],
                         primal_integral = isempty(r) ? missing : r.primal_integral[1], flag = flag); promote = true)
        end
    end
    ok = rows.flag .== ""                                   # flagged cells are listed but not used
    rows.rank_all = Vector{Union{Missing,Float64}}(missing, nrow(rows))
    rows.rank_our5 = Vector{Union{Missing,Float64}}(missing, nrow(rows))
    rows.best_of_our_five = Vector{Union{Missing,Bool}}(missing, nrow(rows))
    insts13 = unique(rows.instance)
    for i in insts13, s in (42, 43)
        idx = findall((rows.instance .== i) .& (rows.seed .== s) .& ok)
        isempty(idx) && continue
        rows.rank_all[idx] = rank_of(rows.incumbent[idx])
        o = [k for k in idx if rows.method[k] in OUR]
        isempty(o) && continue
        rows.rank_our5[o] = rank_of(rows.incumbent[o])
        best = minimum(rows.incumbent[o])
        foreach(k -> rows.best_of_our_five[k] = rows.incumbent[k] == best, o)
    end
    rows.n_methods_in_rank_all = [count((rows.instance .== r.instance) .& (rows.seed .== r.seed) .& ok) for r in eachrow(rows)]
    CSV.write(joinpath(C, "comparison_all.csv"), rows)
    # wins / losses per instance-seed pair against random and CPSAT_1w (valid cells only)
    z(i, s, m) = (k = findfirst((rows.instance .== i) .& (rows.seed .== s) .& (rows.method .== m) .& ok); k === nothing ? missing : rows.incumbent[k])
    meanskip(x) = (v = collect(skipmissing(x)); isempty(v) ? missing : mean(v))
    summ = DataFrame()
    for m in ALL
        d = rows[(rows.method .== m), :]
        v = d[d.flag .== "", :]
        wl(other) = begin
            w = l = t = 0
            for r in eachrow(v)
                o = z(r.instance, r.seed, other)
                (ismissing(o) || m == other) && continue
                r.incumbent < o ? (w += 1) : r.incumbent > o ? (l += 1) : (t += 1)
            end
            m == other ? (missing, missing, missing) : (w, l, t)
        end
        wr, lr, tr = wl("random"); wc, lc, tc = wl("CPSAT_1w")
        push!(summ, (method = m, instance_seed_pairs = nrow(v), missing_pairs = count(==("MISSING"), d.flag),
                     stalled_unrepaired_pairs = count(==("STALLED (unrepaired)"), d.flag),
                     mean_primal_gap = meanskip(v.primal_gap), mean_primal_integral = meanskip(v.primal_integral),
                     mean_rank_our5 = meanskip(v.rank_our5), mean_rank_all = meanskip(v.rank_all),
                     n_best_of_our_five = m in OUR ? count(isequal(true), v.best_of_our_five) : missing,
                     wins_vs_random = wr, losses_vs_random = lr, ties_vs_random = tr,
                     wins_vs_CPSAT_1w = wc, losses_vs_CPSAT_1w = lc, ties_vs_CPSAT_1w = tc); promote = true)
    end
    CSV.write(joinpath(C, "summary.csv"), summ)
    per = DataFrame()
    for i in insts13, m in ALL
        d = rows[(rows.instance .== i) .& (rows.method .== m), :]
        isempty(d) && continue
        v = d[d.flag .== "", :]
        zs(s) = (k = findfirst(==(s), d.seed); k === nothing ? missing : (d.flag[k] == "" ? d.incumbent[k] : d.flag[k]))
        push!(per, (instance = i, method = m, n_valid_seeds = nrow(v), makespan_seed42 = string(zs(42)), makespan_seed43 = string(zs(43)),
                    mean_makespan = meanskip(v.incumbent), mean_primal_gap = meanskip(v.primal_gap),
                    min_primal_gap = isempty(v) ? missing : minimum(v.primal_gap), max_primal_gap = isempty(v) ? missing : maximum(v.primal_gap),
                    mean_primal_integral = meanskip(v.primal_integral), mean_rank_our5 = meanskip(v.rank_our5),
                    flags = join(unique(filter(!isempty, d.flag)), "; ")); promote = true)
    end
    CSV.write(joinpath(C, "per_instance.csv"), per)
    # overview plots: primal gap over time, one panel per instance, one figure per seed
    for s in (42, 43)
        tr = traces[traces.seed .== s, :]
        panels = [gap_panel(tr, i, ALL, refs[i], T; legend = i == insts13[1] ? :topright : false) for i in insts13 if haskey(refs, i)]
        isempty(panels) && continue
        savefig(plot(panels...; layout = length(panels), size = (2400, 1500), dpi = 90, plot_title = "seed $s",
                     left_margin = 8Plots.mm, bottom_margin = 6Plots.mm), joinpath(C, "anytime_gap_all13_seed$s.png"))
    end
    open(joinpath(C, "README.txt"), "w") do io
        println(io, "All 13 large instances × 2 seeds, built by scripts/benchmark_large_v3.jl --combined-13x2 (inputs read only):")
        foreach(x -> println(io, "  seed $(x[1]): results/$(x[2]) ($(join(x[3], ", ")); $(x[4] == CP ? "CP-SAT" : x[4] == OUR ? "our 5 methods" : "all 7 methods"))"), inputs)
        println(io, "No CP-SAT seed-43 runs exist for ta41/ta61/swv12/ta71 (CP-SAT has 22 instance-seed pairs, our methods 26).")
        println(io, "comparison_all.csv: one row per instance × seed × method; flag = MISSING (no result) or")
        println(io, "  STALLED (unrepaired) (scripts/stall_check.py finds the run stalled). Flagged cells are listed but excluded")
        println(io, "  from ranks, means, wins/losses. rank_all = among all valid methods of that instance and seed")
        println(io, "  (n_methods_in_rank_all), rank_our5 = among random, gurobi, A v2, B v2, C v3 (1 = best, ties averaged).")
        println(io, "summary.csv: per method; wins/losses/ties = instance-seed pairs with a strictly smaller / larger / equal final makespan")
        println(io, "  than random resp. CPSAT_1w (pairs where both are valid).")
        println(io, "per_instance.csv: mean over the seeds, min / max primal gap over the seeds; makespan cells show the flag if flagged.")
        println(io, "reference values: from each input folder's reference_values.csv")
        isempty(notes) || println(io, "notes: ", join(notes, "; "))
        println(io, "written: $(now())")
    end
    println("combined 13×2: $(length(insts13)) instances → $C")
    show(stdout, summ; allrows = true); println()
end

function main(args = ARGS)
    opt(k, d) = (i = findfirst(startswith("--$k="), args); i === nothing ? d : split(args[i], '='; limit = 2)[2])
    SEED[] = parse(Int, opt("seed", "42"))
    out = String(opt("out", SEED[] == 42 ? BASE42 : "benchmark_large_v3_seed$(SEED[])"))
    startswith(out, "benchmark_large_v3") || error("--out must be a benchmark_large_v3* folder")
    T = parse(Float64, opt("time", "1200"))
    insts = String.(split(opt("instances", join(INSTANCES, ",")), ','))
    methods = String.(split(opt("methods", join(ALL, ",")), ','))
    all(in(ALL), methods) || error("unknown method in $methods")
    if "--dry-run" in args
        dry_run(out, T, insts, methods)
    elseif "--check" in args
        check(out, T, insts, methods) || exit(1)
    elseif "--aggregate" in args
        aggregate(out, T, insts, methods)
    elseif "--seed-compare" in args
        seed_compare(out)
    elseif "--combined-13x2" in args
        combined_13x2(T)
    else
        drive(out, T, insts, methods)
    end
end

end # module JSSPLabLargeV3

JSSPLabLargeV3.main()
