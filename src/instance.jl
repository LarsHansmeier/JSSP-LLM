# =============================================================================
# instance.jl — Data structure and reading of JSSP instances
# =============================================================================
#
# Internal convention (throughout the project):
#   - Jobs, machines and operations are 1-based (Julia standard).
#   - Operation (j, k) = k-th operation of job j.
#   - machine[j, k]  = machine on which operation (j, k) runs
#   - ptime[j, k]    = processing time of operation (j, k)
#
# Supported file formats:
#
# 1) "standard" (OR-Library / Fisher-Thompson, e.g. ft06, la01):
#        n m
#        m_1 p_1  m_2 p_2  ...  m_m p_m      <- one line per job
#    Machines are 0-based in the file and are incremented by 1 when reading.
#    Comment lines with '#' are allowed.
#
# 2) "taillard" (Taillard instances ta01 ...):
#        n m
#        n lines with processing times
#        n lines with machines (1-based)
#    Lines that do not consist only of numbers (e.g. "Times", "Machines")
#    are ignored.
#
# If a data set fits none of the formats: simply write another
# read_xxx function that returns a JSSPInstance.
# =============================================================================

"""
    JSSPInstance

A JSSP instance with `n` jobs and `m` machines. Every job has exactly `m`
operations (classic JSSP: every job visits every machine exactly once).
"""
struct JSSPInstance
    name::String
    n::Int                 # number of jobs
    m::Int                 # number of machines
    machine::Matrix{Int}   # n × m, machine of the k-th operation of job j
    ptime::Matrix{Int}     # n × m, processing time of the k-th operation of job j
end

"Number of operations of job j (always m here)."
n_ops(inst::JSSPInstance, j::Int) = inst.m

"All operations as a vector of (job, position) tuples."
operations(inst::JSSPInstance) = [(j, k) for j in 1:inst.n for k in 1:n_ops(inst, j)]

"All operations that run on machine `i`."
ops_on_machine(inst::JSSPInstance, i::Int) =
    [(j, k) for j in 1:inst.n for k in 1:n_ops(inst, j) if inst.machine[j, k] == i]

"Sum of all processing times — trivial upper bound, useful as big-M."
total_processing_time(inst::JSSPInstance) = sum(inst.ptime)

"""
Simple lower bound for the makespan:
max(longest job, most loaded machine).
"""
function trivial_lower_bound(inst::JSSPInstance)
    job_lb = maximum(sum(inst.ptime[j, :]) for j in 1:inst.n)
    mach_lb = maximum(sum(inst.ptime[j, k] for (j, k) in ops_on_machine(inst, i); init = 0)
                      for i in 1:inst.m)
    return max(job_lb, mach_lb)
end

# --- Reading -------------------------------------------------------------------

# Reads the file and returns only lines that consist exclusively of numbers.
function _numeric_lines(path::AbstractString)
    lines = Vector{Vector{Int}}()
    for raw in eachline(path)
        line = strip(split(raw, '#')[1])        # strip comments
        isempty(line) && continue
        tokens = split(line)
        all(t -> tryparse(Int, t) !== nothing, tokens) || continue
        push!(lines, parse.(Int, tokens))
    end
    return lines
end

"Reads an instance in standard format (pairs 'machine time', machines 0-based)."
function read_standard(path::AbstractString; name = splitext(basename(path))[1])
    lines = _numeric_lines(path)
    n, m = lines[1][1], lines[1][2]
    length(lines) >= n + 1 || error("$path: expected $n job lines, found $(length(lines) - 1)")
    machine = zeros(Int, n, m)
    ptime = zeros(Int, n, m)
    for j in 1:n
        row = lines[1 + j]
        length(row) == 2m || error("$path: job $j has $(length(row)) entries, expected $(2m)")
        for k in 1:m
            machine[j, k] = row[2k - 1] + 1     # 0-based -> 1-based
            ptime[j, k] = row[2k]
        end
    end
    inst = JSSPInstance(name, n, m, machine, ptime)
    validate_instance(inst)
    return inst
end

"Reads an instance in Taillard format (times matrix first, then machine matrix, 1-based)."
function read_taillard(path::AbstractString; name = splitext(basename(path))[1])
    lines = _numeric_lines(path)
    n, m = lines[1][1], lines[1][2]      # further header numbers (seeds, bounds) are ignored
    length(lines) >= 1 + 2n || error("$path: too few lines for Taillard format")
    ptime = permutedims(hcat(lines[2:1+n]...))
    machine = permutedims(hcat(lines[2+n:1+2n]...))
    size(ptime) == (n, m) && size(machine) == (n, m) ||
        error("$path: matrix sizes do not match n=$n, m=$m")
    inst = JSSPInstance(name, n, m, Matrix{Int}(machine), Matrix{Int}(ptime))
    validate_instance(inst)
    return inst
end

"""
    read_instance(path; format = :standard)

Reads a JSSP instance. `format` is `:standard` or `:taillard`.
"""
function read_instance(path::AbstractString; format::Symbol = :standard)
    format == :standard && return read_standard(path)
    format == :taillard && return read_taillard(path)
    error("unknown format: $format")
end

"Plausibility check of the data read."
function validate_instance(inst::JSSPInstance)
    all(inst.ptime .>= 0) || error("$(inst.name): processing times must be ≥ 0")   # orb07 contains a 0
    all(1 .<= inst.machine .<= inst.m) || error("$(inst.name): machine index outside 1:$(inst.m)")
    for j in 1:inst.n
        sort(inst.machine[j, :]) == collect(1:inst.m) ||
            @warn "$(inst.name): job $j does not visit every machine exactly once"
    end
    return true
end

function Base.show(io::IO, inst::JSSPInstance)
    print(io, "JSSPInstance($(inst.name): $(inst.n) jobs × $(inst.m) machines)")
end

"Prints the instance in readable form (machines 1-based)."
function print_instance(inst::JSSPInstance)
    println(inst)
    for j in 1:inst.n
        ops = join(["M$(inst.machine[j, k])($(inst.ptime[j, k]))" for k in 1:inst.m], " → ")
        println("  Job $j: ", ops)
    end
end
