# =============================================================================
# export_instances.jl — export instances and the standard initial solution to JSON
# =============================================================================
#
# For the external CP-SAT reference (scripts/cpsat_jssp.py): Python never parses the
# instance files itself; it reads exactly what `load_jsplib` produced in Julia.
# Indexing is the Julia one: jobs and operations in order (1-based positions),
# machines 1-based as in `inst.machine` (the file's 0-based machines + 1).
#
#   julia --project=. scripts/export_instances.jl --out=<dir> --instances=ta41,ta61,swv12,ta71
# writes <dir>/<instance>.json with
#   name, n_jobs, n_machines, indexing, jobs = [[[machine, processing_time], …], …],
#   initial_start_times = [[start of op 1, …], …] (initial_schedule), initial_makespan
# =============================================================================

import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")); io = devnull)

module JSSPLabExport

include(joinpath(@__DIR__, "..", "src", "JSSPLab.jl"))
using .JSSPLab

"Minimal JSON writer for nested vectors of integers / strings (no extra package needed)."
json(x::Integer) = string(x)
json(x::AbstractString) = "\"" * escape_string(x) * "\""
json(x::AbstractVector) = "[" * join(json.(x), ",") * "]"
json(d::Vector{<:Pair}) = "{" * join([json(String(k)) * ":" * json(v) for (k, v) in d], ",") * "}"

function export_instance(name, dir)
    inst = load_jsplib(name)
    s0 = initial_schedule(inst)
    check_schedule(inst, s0; verbose = false)[1] || error("$name: initial schedule infeasible")
    jobs = [[[inst.machine[j, k], inst.ptime[j, k]] for k in 1:inst.m] for j in 1:inst.n]
    starts = [[s0.start[j, k] for k in 1:inst.m] for j in 1:inst.n]
    doc = ["name" => name, "n_jobs" => inst.n, "n_machines" => inst.m,
           "indexing" => "1-based machines (Julia inst.machine); jobs and operations in route order",
           "jobs" => jobs, "initial_start_times" => starts, "initial_makespan" => makespan(inst, s0)]
    path = joinpath(dir, "$name.json")
    write(path, json(doc))
    println("exported $name ($(inst.n)×$(inst.m), initial makespan $(makespan(inst, s0))) → $path")
end

function main(args = ARGS)
    opt(k, d) = (i = findfirst(startswith("--$k="), args); i === nothing ? d : split(args[i], '='; limit = 2)[2])
    dir = opt("out", "")
    isempty(dir) && error("--out=<dir> required")
    mkpath(dir)
    foreach(n -> export_instance(String(n), dir), split(opt("instances", "ta41,ta61,swv12,ta71"), ','))
end

end # module JSSPLabExport

JSSPLabExport.main()
