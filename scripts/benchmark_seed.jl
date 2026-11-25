# =============================================================================
# benchmark_seed.jl — repeat of the large benchmark (results/benchmark_large_v1) with another seed
# =============================================================================
#
# Identical to scripts/benchmark_large.jl as it ran for results/benchmark_large_v1 (worker,
# settings, reference values, aggregation and plots copied from that version), except:
#   * the seed is a CLI parameter (--seed=, default DEFAULT_SEED = 42). It is the single value
#     that v1 passed as `seed` to random_fao (MersenneTwister for the budget draws and the
#     random release selection; Gurobi Seed of every sub-problem) and to solve_model (Gurobi
#     Seed of the full model). No policy file is touched; no policy has its own RNG.
#   * crash-safe driver as in scripts/benchmark_large_v2.jl: a run writes into
#     runs/<inst>__<method>.tmp/, is checked (worker finished + final schedule passes
#     check_schedule, checked again here), renamed atomically and only then gets DONE.
#     On start, runs with DONE are skipped, everything else is deleted and redone from
#     scratch. A failing run is retried once, then the driver continues with the next run.
#   * run order: by instance (ta41, ta61, swv12, ta71), within an instance in the v1 method order.
#
#   julia --project=. scripts/benchmark_seed.jl --seed=43                     # full benchmark
#   julia --project=. scripts/benchmark_seed.jl --seed=43 --dry-run           # print config only
#   julia --project=. scripts/benchmark_seed.jl --seed=43 --time=60 --instances=ta41 \
#         --out=benchmark_large_v1_seed43/smoke                               # smoke test
#   julia --project=. scripts/benchmark_seed.jl --check --time=60 --out=…     # validate (exit 1 on failure)
#   julia --project=. scripts/benchmark_seed.jl --aggregate --seed=43         # only aggregate
#   julia --project=. scripts/benchmark_seed.jl --seed-compare --seed=43      # vs v1 (seed 42)
# Default --out = benchmark_large_v1_seed<seed>.
# =============================================================================

import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")); io = devnull)
ENV["GKSwstype"] = "100"     # GR without window (file output)

module JSSPLabSeed

using CSV, DataFrames, Dates, Plots, Statistics, JuMP
include(joinpath(@__DIR__, "..", "src", "JSSPLab.jl"))
using .JSSPLab

const ROOT = normpath(joinpath(@__DIR__, ".."))
const V1 = joinpath(ROOT, "results", "benchmark_large_v1")          # read only
const METHODS = ["random", "gurobi", "PolicyA_ScoreBased_fill", "PolicyB_StructureBased_v2",
                 "PolicyC_Dynamic_v3"]
const INSTANCES = ["ta41", "ta61", "swv12", "ta71"]
const Y_MIN, Y_MAX = 250, 4000
const SUB_T = 7.5
const THREADS = 1
const SEED = Ref(DEFAULT_SEED)               # set from --seed (the only change vs v1)
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

base(out) = joinpath(ROOT, "results", out)
final_dir(out, inst, m) = joinpath(base(out), "runs", "$(inst)__$(m)")
tmp_dir(out, inst, m) = final_dir(out, inst, m) * ".tmp"
is_done(out, inst, m) = isfile(joinpath(final_dir(out, inst, m), "DONE"))
say(args...) = (println(now(), "  ", args...); flush(stdout))

# --- one run (worker process) – as in v1 ------------------------------------------------
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
                          ub = ub, threads = THREADS, seed = SEED[])
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
                     n_free_max = Y_MAX, threads = THREADS, seed = SEED[], ub = ub,
                     wall_start = t0, select_free = f)
    return res, st, (seen = seen[], infeasible = bad[])
end

"Worker: writes into `dir` (created fresh by the driver); `done` is written last."
function worker(inst_name, method, dir, T)
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
        :gurobi_TimeLimit_last => get_attribute(res.model, "TimeLimit"),
        :seed => SEED[], :gurobi_Seed => try get_attribute(res.model, "Seed") catch; missing end)
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
    res.schedule === nothing || CSV.write(joinpath(dir, "schedule.csv"),
        DataFrame(res.schedule.start, ["op$k" for k in 1:size(res.schedule.start, 2)]))
    write(joinpath(dir, "done"), string(now()))          # written last: worker finished
    println("$inst_name $method (seed $(SEED[])): incumbent $(res.incumbent), runtime $(round(res.runtime, digits = 1)) s, ",
            "feasible $ok, peak RSS $(row[:peak_rss_mb]) MB")
end

# --- crash-safe driver (as scripts/benchmark_large_v2.jl) -----------------------------------
fsync_io(io) = (flush(io); ccall(:fsync, Cint, (Cint,), fd(io)))
function write_durable(path, df::DataFrame)
    open(io -> (CSV.write(io, df); fsync_io(io)), path * ".part", "w")
    mv(path * ".part", path; force = true)
end
touch_durable(path) = open(io -> fsync_io(io), path, "w")

run_order(insts, methods) = [(i, m) for i in insts for m in methods]      # by instance, v1 method order

function kill_orphan(dir)
    f = joinpath(dir, "child.pid")
    isfile(f) || return
    pid = tryparse(Int, strip(read(f, String)))
    pid === nothing && return
    cmdline = try strip(read(`ps -p $pid -o command=`, String)) catch; "" end
    if occursin("benchmark_seed.jl", cmdline)
        say("killing orphaned child $pid")
        run(ignorestatus(`kill -9 $pid`)); sleep(1)
    end
end

function cleanup(out)
    runs = joinpath(base(out), "runs")
    removed = String[]
    for d in readdir(runs)
        p = joinpath(runs, d)
        isdir(p) || continue
        !endswith(d, ".tmp") && !endswith(d, ".FAILED") && isfile(joinpath(p, "DONE")) && continue
        kill_orphan(p)
        rm(p; recursive = true, force = true)
        push!(removed, d)
    end
    return removed
end

function load_progress(out)
    f = joinpath(base(out), "progress.csv")
    df = if isfile(f)
        d = CSV.read(f, DataFrame; types = String)
        foreach(c -> d[!, c] = coalesce.(d[!, c], ""), names(d)); d
    else
        DataFrame(instance = String[], method = String[], seed = String[], start = String[], finish = String[], status = String[])
    end
    for r in eachrow(df)
        r.status == "running" || continue
        r.status = is_done(out, r.instance, r.method) ? "done" : "interrupted"
        r.finish = r.status == "done" ? r.finish : "restart $(now())"
    end
    return df
end

function environment(out, n_runs, T)
    gv = let a = Ref{Cint}(), b = Ref{Cint}(), c = Ref{Cint}()
        JSSPLab.Gurobi.GRBversion(a, b, c); "$(a[]).$(b[]).$(c[])"
    end
    sh(c) = strip(read(c, String))
    cpu, p, e = sh(`sysctl -n machdep.cpu.brand_string`), sh(`sysctl -n hw.perflevel0.physicalcpu`), sh(`sysctl -n hw.perflevel1.physicalcpu`)
    mem = parse(Int, sh(`sysctl -n hw.memsize`)) ÷ 2^30
    open(joinpath(base(out), "environment.txt"), "w") do io
        println(io, "machine: $cpu, $(Sys.CPU_THREADS) cores ($p performance + $e efficiency), $mem GB RAM, macOS $(sh(`sw_vers -productVersion`))")
        println(io, "julia: $(VERSION), gurobi: $gv")
        println(io, "parallelism: 1 worker (sequential, one Julia process per run), Gurobi Threads = $THREADS, Seed = $(SEED[])")
        println(io, "runs: $n_runs, free-y range [$Y_MIN, $Y_MAX] (capped by FREE_CAP = $FREE_CAP), sub-problem TimeLimit $SUB_T s")
        println(io, "time per run: $T s; run order: by instance, v1 method order; crash-safe driver (runs/<inst>__<method>.tmp → rename → DONE)")
        println(io, "script: scripts/benchmark_seed.jl (copy of the v1 benchmark_large.jl worker/aggregation, seed as parameter)")
        println(io, "written: $(now())")
    end
end

"Worker as a child process (pid in child.pid), then the driver's own check; (ok, message)."
function run_one(out, inst, m, T, dir)
    log = joinpath(dir, "worker.log")
    cmd = `$(Base.julia_cmd()) --project=$ROOT $(@__FILE__) --worker --instance=$inst --method=$m --seed=$(SEED[]) --time=$T --run-dir=$dir`
    p = run(pipeline(cmd; stdout = log, stderr = log); wait = false)
    open(io -> (print(io, getpid(p)); fsync_io(io)), joinpath(dir, "child.pid"), "w")
    wait(p)
    success(p) && all(f -> isfile(joinpath(dir, f)), ("done", "run.csv", "schedule.csv")) ||
        return false, "worker failed (exit $(p.exitcode); see worker.log)"
    row = CSV.read(joinpath(dir, "run.csv"), DataFrame)[1, :]
    I = load_jsplib(inst)
    feas, errs = check_schedule(I, Schedule(Matrix{Int}(CSV.read(joinpath(dir, "schedule.csv"), DataFrame)));
                                claimed_makespan = row.incumbent, verbose = false)
    feas && row.feasible === true || return false, "final schedule INFEASIBLE: $(first(errs, 3))"
    mv(joinpath(dir, "done"), joinpath(dir, "worker_finished.txt"))   # `done` ≡ `DONE` on macOS
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
    environment(out, length(todo), T)
    for (inst, m) in todo
        if is_done(out, inst, m)
            say("$inst $m: DONE earlier, skipped")
            continue
        end
        for attempt in 1:2                                  # at most one retry per run
            dir, fin = tmp_dir(out, inst, m), final_dir(out, inst, m)
            rm(dir; recursive = true, force = true); rm(fin; recursive = true, force = true)
            mkpath(dir)
            push!(progress, (inst, m, string(SEED[]), string(now()), "", "running"))
            write_durable(joinpath(base(out), "progress.csv"), progress)
            say("$inst $m: started (attempt $attempt)")
            ok, msg = try
                run_one(out, inst, m, T, dir)
            catch err
                false, "driver error: $(sprint(showerror, err))"
            end
            if ok
                rm(joinpath(dir, "child.pid"); force = true)
                mv(dir, fin)                                # atomic rename (same file system)
                touch_durable(joinpath(fin, "DONE"))
                run(`sync`)
            else
                mv(dir, fin * ".FAILED"; force = true)      # kept for inspection, redone on restart
            end
            progress[end, :finish] = string(now())
            progress[end, :status] = ok ? "done" : "FAILED (attempt $attempt): " * replace(msg, '\n' => ' ')
            write_durable(joinpath(base(out), "progress.csv"), progress)
            say("$inst $m: ", progress[end, :status])
            ok && break
        end
    end
    aggregate(out, T, insts, methods)
end

# --- aggregation (as v1; reads only DONE folders) ------------------------------------------
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
    B = base(out)
    runs, traces, its, logB, logC = DataFrame(), DataFrame(), DataFrame(), DataFrame(), DataFrame()
    for inst in insts, m in methods
        dir = final_dir(out, inst, m)
        is_done(out, inst, m) || continue
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
    runs.run_check = map(eachrow(runs)) do r
        t_end = maximum(traces[(traces.instance .== r.instance) .& (traces.method .== r.method), :time])
        abs(r.runtime - T) <= 5 ? "full" :
        r.method == "gurobi" && r.status == "OPTIMAL" ? "gurobi optimal" :
        hasproperty(r, :stop_reason) && isequal(r.stop_reason, "exhausted") && t_end >= T - 5 ?
            "C v3 exhausted (trace carried to T)" : "INCOMPLETE"
    end
    CSV.write(joinpath(B, "runs.csv"), runs)
    CSV.write(joinpath(B, "traces.csv"), select(traces, :instance, :method, :time, :incumbent, :bound))
    isempty(its) || CSV.write(joinpath(B, "iterations.csv"), its)
    isempty(logB) || CSV.write(joinpath(B, "policyB_log.csv"), logB)
    isempty(logC) || CSV.write(joinpath(B, "policyC_log.csv"), logC)
    refs = DataFrame(instance = insts, reference = [reference(i)[1] for i in insts],
                     source = [reference(i)[2] for i in insts],
                     note = [get(REF_NOTES, i, "") for i in insts])
    CSV.write(joinpath(B, "reference_values.csv"), refs)

    cmp = DataFrame()
    for r in eachrow(runs)
        t = sort(traces[(traces.instance .== r.instance) .& (traces.method .== r.method) .& .!ismissing.(traces.incumbent), :], :time)
        pi = ismissing(r.reference) ? missing : primal_integral(t.time, t.incumbent, r.reference, T)
        push!(cmp, (instance = r.instance, method = r.method, incumbent = r.incumbent,
                    primal_gap = r.primal_gap, primal_integral = pi); promote = true)
    end
    CSV.write(joinpath(B, "comparison.csv"), cmp)

    present = [m for m in methods if m in runs.method]
    ref_of = Dict(r.instance => r.reference for r in eachrow(refs))
    gap_panels = []
    insts_present = [i for i in insts if i in runs.instance]
    for inst in insts_present
        p = plot(; xscale = :log10, xlabel = "time [s] (log)", ylabel = "makespan", title = "$inst (seed $(SEED[]))",
                 legend = :topright, size = (900, 550), dpi = 120, titlefontsize = 11, xlims = (1, T * 1.05))
        q = plot(; xscale = :log10, xlabel = "time [s] (log)", ylabel = "primal gap [%]", title = inst,
                 titlefontsize = 10, xlims = (1, T * 1.05), legend = inst == insts_present[1] ? :topright : false)
        for m in present
            t = sort(traces[(traces.instance .== inst) .& (traces.method .== m) .& .!ismissing.(traces.incumbent), :], :time)
            isempty(t) && continue
            c, ls = STYLE[m]
            plot!(p, max.(t.time, 1.0), t.incumbent; seriestype = :steppost, lw = 2, color = c, ls = ls, label = m)
            plot!(q, max.(t.time, 1.0), 100 .* (t.incumbent .- ref_of[inst]) ./ ref_of[inst];
                  seriestype = :steppost, lw = 2, color = c, ls = ls, label = m)
        end
        hline!(p, [ref_of[inst]]; color = :black, ls = :dash, label = "reference $(ref_of[inst])")
        savefig(p, joinpath(B, "anytime_$inst.png"))
        push!(gap_panels, q)
    end
    lay = length(gap_panels) == 4 ? (2, 2) : (1, length(gap_panels))
    savefig(plot(gap_panels...; layout = lay, size = (1200, 850), dpi = 110), joinpath(B, "anytime_gap_all.png"))
    fao_present = [m for m in present if m != "gurobi"]
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
        savefig(p, joinpath(B, file))
    end
    if !isempty(its)
        solved = its[its.sub_status .!= "SKIPPED_DUPLICATE", :]
        short = Dict("random" => "random", "PolicyA_ScoreBased_fill" => "A v2",
                     "PolicyB_StructureBased_v2" => "B v2", "PolicyC_Dynamic_v3" => "C v3")
        panels = map(insts_present) do i
            d = solved[solved.instance .== i, :]
            q = plot(; title = i, ylabel = "free y", titlefontsize = 10, legend = false,
                     xticks = (1:length(fao_present), [short[m] for m in fao_present]))
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
        savefig(plot(panels...; layout = lay, size = (1100, 800), dpi = 110), joinpath(B, "free_y_distribution.png"))
    end
    println("aggregated $(nrow(runs)) runs → $B")
    show(stdout, select(runs, :instance, :method, :incumbent, :primal_gap, :runtime, :feasible, :run_check); allrows = true)
    println()
end

# --- checks / config / seed comparison --------------------------------------------------------
"Validate an aggregated folder; prints the reasons and returns false on any failure."
function check(out, T, insts, methods)
    f = joinpath(base(out), "runs.csv")
    isfile(f) || (println("CHECK FAILED: no runs.csv in $(base(out))"); return false)
    runs = CSV.read(f, DataFrame)
    bad = String[]
    for i in insts, m in methods
        is_done(out, i, m) || push!(bad, "$i $m: no DONE")
    end
    for r in eachrow(runs)
        r.feasible === true || push!(bad, "$(r.instance) $(r.method): final schedule infeasible")
        r.run_check in ("full", "gurobi optimal", "C v3 exhausted (trace carried to T)") ||
            push!(bad, "$(r.instance) $(r.method): run_check = $(r.run_check)")
        coalesce(r.incumbents_infeasible, 0) == 0 || push!(bad, "$(r.instance) $(r.method): infeasible incumbents")
        coalesce(r.seed, -1) == SEED[] || push!(bad, "$(r.instance) $(r.method): seed $(r.seed) ≠ $(SEED[])")
        coalesce(r.gurobi_Seed, -1) == SEED[] || push!(bad, "$(r.instance) $(r.method): Gurobi Seed $(r.gurobi_Seed)")
        r.method == "gurobi" && continue
        coalesce(r.free_y_violations, 1) == 0 || push!(bad, "$(r.instance) $(r.method): $(r.free_y_violations) free-y violations")
        coalesce(r.max_sub_time_limit_hit, 0.0) <= SUB_T + 0.5 ||
            push!(bad, "$(r.instance) $(r.method): sub-problem took $(r.max_sub_time_limit_hit) s")
    end
    if isempty(bad)
        println("CHECK OK: $(nrow(runs)) runs, all DONE, feasible, run_check passed, free y in range, seed $(SEED[])")
        return true
    end
    println("CHECK FAILED: ", join(bad, "; "))
    return false
end

function dry_run(out, T, insts, methods)
    println("output:        results/$out")
    println("seed:          $(SEED[])   (random_fao seed → MersenneTwister + sub-problem Gurobi Seed; solve_model Gurobi Seed)")
    println("instances:     $(join(insts, ", "))")
    println("methods:       $(join(methods, ", "))")
    println("time per run:  $T s")
    println("free-y range:  [$Y_MIN, $Y_MAX], capped by FREE_CAP = $FREE_CAP")
    println("sub TimeLimit: $SUB_T s;  Gurobi Threads = $THREADS;  start = initial_schedule (MWKR), Big-M ub = its makespan")
    println("order:         ", join(["$i/$m" for (i, m) in run_order(insts, methods)], ", "))
    v1 = CSV.read(joinpath(V1, "runs.csv"), DataFrame)
    println("per instance (this config | v1 runs.csv):")
    for i in insts
        I = load_jsplib(i)
        pairs = length(machine_pairs(I)); cap = floor(Int, FREE_CAP * pairs)
        y = "$(min(Y_MIN, cap))-$(min(Y_MAX, cap))"
        init = makespan(I, initial_schedule(I)); ref = reference(i)[1]
        r = v1[v1.instance .== i, :]
        same = all(r.Y_tot .== pairs) && all(r.y_range_effective .== y) && all(r.initial .== init) && all(r.reference .== ref)
        println("  $i: Y_tot $pairs, y range $y, y_floor(A v2, B v2) $(min(Y_MIN, cap)), initial makespan $init, reference $ref  | ",
                "v1: Y_tot $(unique(r.Y_tot)), y range $(unique(r.y_range_effective)), initial $(unique(r.initial)), ",
                "reference $(unique(r.reference)), sub_time_limit $(unique(skipmissing(r.sub_time_limit))), threads $(unique(r.threads))  → ",
                same ? "identical" : "DIFFERENT")
    end
end

function seed_compare(out, T)
    new = CSV.read(joinpath(base(out), "comparison.csv"), DataFrame)
    old = CSV.read(joinpath(V1, "comparison.csv"), DataFrame)
    s = SEED[]
    rank(z) = [count(<(v), z) + (count(==(v), z) + 1) / 2 for v in z]
    rows = DataFrame()
    for i in INSTANCES, m in METHODS
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
        rows.rank_seed42[idx] = rank(rows.makespan_seed42[idx])
        rows.rank_new[idx] = rank(rows.makespan_new[idx])
    end
    CSV.write(joinpath(base(out), "seed_comparison.csv"),
              rename(rows, [c => replace(c, "_new" => "_seed$s") for c in names(rows) if endswith(c, "_new")]))
    sh(m) = Dict("random" => "random", "gurobi" => "gurobi", "PolicyA_ScoreBased_fill" => "A v2",
                 "PolicyB_StructureBased_v2" => "B v2", "PolicyC_Dynamic_v3" => "C v3")[m]
    pct(x) = string(round(100x, digits = 2), " %")
    open(joinpath(base(out), "seed_comparison.md"), "w") do io
        println(io, "# Seed comparison: seed 42 (results/benchmark_large_v1) vs seed $s\n")
        println(io, "Same configuration except the seed; 20-minute runs; rank = by final makespan among the 5 methods (1 = best, ties averaged).\n")
        println(io, "## Ranking per instance\n")
        println(io, "| instance | seed 42 (best → worst) | seed $s (best → worst) | changed |")
        println(io, "|---|---|---|---|")
        for i in unique(rows.instance)
            d = rows[rows.instance .== i, :]
            o42 = join([sh(m) for m in d.method[sortperm(d.rank_seed42)]], " < ")
            onew = join([sh(m) for m in d.method[sortperm(d.rank_new)]], " < ")
            ch = [sh(r.method) * " $(r.rank_seed42)→$(r.rank_new)" for r in eachrow(d) if r.rank_seed42 != r.rank_new]
            println(io, "| $i | $o42 | $onew | ", isempty(ch) ? "no" : join(ch, ", "), " |")
        end
        println(io, "\nMean rank over the instances: ", join(["$(sh(m)) $(round(mean(rows.rank_seed42[rows.method .== m]), digits = 2)) → $(round(mean(rows.rank_new[rows.method .== m]), digits = 2))" for m in METHODS], ", "), "\n")
        println(io, "## Largest primal-gap differences (seed $s − seed 42)\n")
        println(io, "| instance | method | gap seed 42 | gap seed $s | difference | makespan 42 → $s |")
        println(io, "|---|---|---|---|---|---|")
        for r in eachrow(first(sort(rows, :diff_gap; by = abs, rev = true), 8))
            println(io, "| $(r.instance) | $(sh(r.method)) | $(pct(r.gap_seed42)) | $(pct(r.gap_new)) | $(pct(r.diff_gap)) | $(r.makespan_seed42) → $(r.makespan_new) |")
        end
        println(io, "\nMean absolute gap difference: ", pct(mean(abs.(rows.diff_gap))), "; per method: ",
                join(["$(sh(m)) $(pct(mean(abs.(rows.diff_gap[rows.method .== m]))))" for m in METHODS], ", "))
    end
    println("seed comparison → $(joinpath(base(out), "seed_comparison.csv")) / .md")
end

function main(args = ARGS)
    opt(k, d) = (i = findfirst(startswith("--$k="), args); i === nothing ? d : split(args[i], '='; limit = 2)[2])
    SEED[] = parse(Int, opt("seed", string(DEFAULT_SEED)))
    out = String(opt("out", "benchmark_large_v1_seed$(SEED[])"))
    T = parse(Float64, opt("time", "1200"))
    insts = String.(split(opt("instances", join(INSTANCES, ",")), ','))
    methods = String.(split(opt("methods", join(METHODS, ",")), ','))
    if "--worker" in args
        worker(String(opt("instance", "")), String(opt("method", "")), String(opt("run-dir", "")), T)
    elseif "--dry-run" in args
        dry_run(out, T, insts, methods)
    elseif "--check" in args
        check(out, T, insts, methods) || exit(1)
    elseif "--aggregate" in args
        aggregate(out, T, insts, methods)
    elseif "--seed-compare" in args
        seed_compare(out, T)
    else
        drive(out, T, insts, methods)
    end
end

end # module JSSPLabSeed

JSSPLabSeed.main()
