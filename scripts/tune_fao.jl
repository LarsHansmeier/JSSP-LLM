# =============================================================================
# tune_fao.jl — Parameter study for Random Fix-and-Optimize
# =============================================================================
#
# Grid = n_free_ys × sub_times, every combination on all instances and seeds.
# Everything is configurable from the command line. Defaults:
#   n_free_ys  = 100, 200, 400, 800   (number of free y variables per subproblem)
#   sub_times  = 1
#   instances  = ft10, la21, ta01, swv12 (swv12: large, sub_time_limit matters there)
#   seeds      = 42, time_limit = 30 s
#   -> 4 configurations × 4 instances × 1 seed = 16 runs ≈ 8 min
#
# Run:
#   julia --project=. scripts/tune_fao.jl                                   # defaults
#   julia --project=. scripts/tune_fao.jl 60                                # 60 s per run
#   julia --project=. scripts/tune_fao.jl ft10 ta41 --seeds=42,43           # own instances/seeds
#   julia --project=. scripts/tune_fao.jl --nfree=50,100,200 --subtimes=1,5
#   julia --project=. scripts/tune_fao.jl --out=tuning_round3               # other output folder
#   julia --project=. scripts/tune_fao.jl --dry-run                         # only show grid & duration
#   julia --project=. scripts/tune_fao.jl --selector=window                 # contiguous reference
#                                         neighbourhood (window_free), writes to results/tuning_window/
#
# Output in results/<out>/ (default results/tuning/):
#   runs.csv      one row per run (configuration, instance, seed) – written continuously
#   summary.csv   per configuration: mean primal gap (overall and per instance), mean rank,
#                 share of TIME_LIMIT subproblems, median sub_time, … – sorted
#   params.json   grid, instances, seeds, time limit, versions
# Note: a new run into the same folder overwrites the old files.
# =============================================================================

import Pkg
const _ROOT = normpath(joinpath(@__DIR__, ".."))
Base.active_project() == joinpath(_ROOT, "Project.toml") || Pkg.activate(_ROOT; io = devnull)

module JSSPLabTuning

using CSV, DataFrames, Dates, Statistics
include(joinpath(@__DIR__, "..", "src", "JSSPLab.jl"))
using .JSSPLab

const ROOT = normpath(joinpath(@__DIR__, ".."))

const DEFAULTS = (
    n_free_ys  = [100, 200, 400, 800],
    sub_times  = [1.0],
    instances  = ["ft10", "la21", "ta01", "swv12"],
    seeds      = [DEFAULT_SEED],
    time_limit = 30.0,
    threads    = 4,
    out        = "tuning",
    selector   = "random",
)

const SELECTORS = Dict("random" => random_free, "window" => window_free)

config_name(c) = "free$(c.n_free_y)_sub$(c.sub_time_limit)"

# --- Arguments -----------------------------------------------------------------
parse_list(s) = filter(!isempty, strip.(split(s, ',')))
parse_seeds(s) = occursin(':', s) ? collect(parse(Int, split(s, ':')[1]):parse(Int, split(s, ':')[2])) :
                                    parse.(Int, parse_list(s))

function parse_args(args)
    o = Dict{Symbol,Any}(pairs(DEFAULTS))
    names, dry, out_given = String[], false, false
    for a in args
        key, val = occursin('=', a) ? split(a, '='; limit = 2) : (a, "")
        if a == "--dry-run"
            dry = true
        elseif key == "--seeds"
            o[:seeds] = parse_seeds(val)
        elseif key == "--nfree"
            o[:n_free_ys] = parse.(Int, parse_list(val))
        elseif key == "--subtimes"
            o[:sub_times] = parse.(Float64, parse_list(val))
        elseif key == "--threads"
            o[:threads] = parse(Int, val)
        elseif key == "--out"
            o[:out] = String(val)
            out_given = true
        elseif key == "--selector"
            haskey(SELECTORS, val) || error("--selector muss random oder window sein")
            o[:selector] = String(val)
        elseif startswith(a, "--")
            error("Unbekannte Option $a")
        elseif tryparse(Float64, a) !== nothing
            o[:time_limit] = parse(Float64, a)
        else
            push!(names, a)
        end
    end
    isempty(names) || (o[:instances] = names)
    out_given || o[:selector] == "random" || (o[:out] = "tuning_$(o[:selector])")   # own folder
    foreach(jsplib_info, o[:instances])               # catch typos early
    return (; o...), dry
end

# --- Documentation ---------------------------------------------------------------
function gurobi_version()
    a, b, c = Ref{Cint}(), Ref{Cint}(), Ref{Cint}()
    JSSPLab.Gurobi.GRBversion(a, b, c)
    return "$(a[]).$(b[]).$(c[])"
end

function write_params(path, o)
    q(x) = "\"" * string(x) * "\""
    open(path, "w") do io
        println(io, "{")
        println(io, "  \"n_free_ys\": [", join(o.n_free_ys, ", "), "],")
        println(io, "  \"sub_times\": [", join(o.sub_times, ", "), "],")
        println(io, "  \"selector\": ", q(o.selector), ",")
        println(io, "  \"accept_equal\": true,")
        println(io, "  \"instances\": [", join(q.(o.instances), ", "), "],")
        println(io, "  \"seeds\": [", join(o.seeds, ", "), "],")
        println(io, "  \"time_limit\": $(o.time_limit),")
        println(io, "  \"threads\": $(o.threads),")
        println(io, "  \"gurobi\": ", q(gurobi_version()), ",")
        println(io, "  \"julia\": ", q(VERSION), ",")
        println(io, "  \"timestamp\": ", q(now()))
        println(io, "}")
    end
end

# --- Study -----------------------------------------------------------------------
function tune(o; dry_run = false)
    configs = [(n_free_y = nf, sub_time_limit = st) for nf in o.n_free_ys for st in o.sub_times]
    n_total = length(configs) * length(o.instances) * length(o.seeds)
    println("Parameter-Studie [$(o.selector)]: $(length(configs)) Konfigurationen × $(length(o.instances)) Instanzen ",
            "($(join(o.instances, ", "))) × $(length(o.seeds)) Seeds = $n_total Läufe à ",
            "$(o.time_limit) s (≈ $(round(n_total * o.time_limit / 60, digits = 1)) min)")
    foreach(c -> println("  ", config_name(c)), configs)
    dry_run && return nothing

    out_dir = joinpath(ROOT, "results", o.out)
    mkpath(out_dir)
    keep_awake()
    write_params(joinpath(out_dir, "params.json"), o)
    # warm-up run so JIT compilation does not distort the first instance
    random_fao(read_instance(joinpath(ROOT, "data", "instances", "example3x3.txt")); time_limit = 0.5)

    runs = DataFrame()
    sub_times = Dict{String,Vector{Float64}}()     # all sub_time values per configuration (for the median)
    insts = Dict(n => load_jsplib(n) for n in o.instances)
    run_no = 0
    for c in configs, name in o.instances, seed in o.seeds
        run_no += 1
        inst, info = insts[name], jsplib_info(name)
        ref = something(info.optimum, info.upper, missing)
        res = random_fao(inst; time_limit = o.time_limit, seed = seed, threads = o.threads,
                         n_free_min = c.n_free_y, n_free_max = c.n_free_y,   # fixed value
                         sub_time_limit = c.sub_time_limit,
                         select_free = SELECTORS[o.selector])
        ok = check_schedule(inst, res.schedule; claimed_makespan = res.incumbent, verbose = false)[1]
        ok || @warn "$name, $(config_name(c)), Seed $seed: Lösung UNZULÄSSIG!"
        append!(get!(sub_times, config_name(c), Float64[]), [r.sub_time for r in res.iterations])
        d = res.diagnostics
        pg = ismissing(ref) ? missing : (res.incumbent - ref) / ref
        push!(runs, (config = config_name(c), selector = o.selector, c..., n_pairs = res.params.n_pairs,
                     instance = name, seed = seed,
                     status = res.status, initial = res.trace[1].incumbent,
                     incumbent = res.incumbent, reference = ref, primal_gap_bks = pg,
                     gap_own_bound = res.gap, runtime = res.runtime, feasible = ok, d...);
              promote = true)
        CSV.write(joinpath(out_dir, "runs.csv"), runs)            # save continuously
        tl = ismissing(d.share_time_limit) ? "–" : round(d.share_time_limit, digits = 3)
        sub = ismissing(d.median_sub_time) ? "–" : round(d.median_sub_time, digits = 4)
        println(lpad(run_no, 3), "/$n_total  ", rpad(config_name(c), 18), rpad(name, 6), " s$seed",
                "  Inc=", lpad(res.incumbent, 5), "  PG=", ismissing(pg) ? "–" : "$(round(100pg, digits = 2)) %",
                "  It=", d.iterations, "  TL-Anteil=", tl, "  sub(med)=", sub, " s", ok ? "  ✓" : "  ✗")
    end

    mskip(f) = x -> (v = collect(skipmissing(x)); isempty(v) ? missing : f(v))
    summary = combine(groupby(runs, [:config, :selector, :n_free_y, :sub_time_limit]),
        :primal_gap_bks => mskip(mean) => :mean_primal_gap,
        :share_time_limit => mskip(mean) => :share_time_limit,
        :config => (c -> (v = sub_times[first(c)]; isempty(v) ? missing : median(v))) => :median_sub_time,
        :iterations => mean => :mean_iterations,
        :share_improved => mskip(mean) => :mean_share_improved,
        :time_last_improvement => mskip(mean) => :mean_time_last_improvement,
        :feasible => all => :all_feasible,
        nrow => :runs)
    for name in o.instances                           # primal gap and TL share per instance
        per = combine(groupby(filter(:instance => ==(name), runs), :config),
                      :primal_gap_bks => mskip(mean) => Symbol("primal_gap_$name"),
                      :share_time_limit => mskip(mean) => Symbol("share_time_limit_$name"))
        summary = leftjoin(summary, per; on = :config)
    end
    # Mean rank: per instance, rank configurations by primal gap (1 = best), then
    # average – robust against one instance with a large gap (e.g. swv12)
    # dominating the mean.
    pg_cols = [Symbol("primal_gap_$n") for n in o.instances]
    ranks = map(pg_cols) do col
        v = summary[!, col]
        [ismissing(x) ? missing : count(y -> !ismissing(y) && y < x, v) + 1 for x in v]
    end
    summary.mean_rank = [mskip(mean)([r[i] for r in ranks]) for i in 1:nrow(summary)]
    sort!(summary, :mean_primal_gap)
    CSV.write(joinpath(out_dir, "summary.csv"), summary)
    println("\nBeste Konfigurationen (mittlerer Primal Gap):")
    show(stdout, first(summary[:, [:config, :mean_primal_gap, :mean_rank, :share_time_limit, :median_sub_time]], 5);
         allcols = true)
    println("\n\nErgebnisse: ", out_dir)
    return summary
end

function main(args = ARGS)
    o, dry = parse_args(args)
    return tune(o; dry_run = dry)
end

end # module JSSPLabTuning

JSSPLabTuning.main()
