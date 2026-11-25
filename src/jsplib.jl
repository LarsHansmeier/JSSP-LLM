# =============================================================================
# jsplib.jl — Access to the JSPLIB benchmark instances
# =============================================================================
#
# Source: https://github.com/tamy0612/JSPLIB (162 instances: abz, ft, la, orb,
# swv, ta, yn). The files are in data/instances/jsplib/ and all use the
# "standard" format (machines 0-based, '#' comments) -> read_standard.
#
# metadata.csv (generated from instances.json) contains per instance:
#   name, jobs, machines, optimum, lower, upper
# `optimum` is empty if the optimum is unknown; then the bounds lower/upper
# apply (best known solution = upper). With a known optimum,
# lower == upper == optimum. For ta71–ta80 JSPLIB provides no values
# (all three fields `nothing`).
#
# Example:
#   inst = load_jsplib("ft10")
#   info = jsplib_info("ft10")          # (name, jobs, machines, optimum, lower, upper)
#   names = list_jsplib(; max_jobs = 10, max_machines = 10)
# =============================================================================

const JSPLIB_DIR = normpath(joinpath(@__DIR__, "..", "data", "instances", "jsplib"))

"Metadata of a JSPLIB instance. `optimum === nothing` if unknown."
const JSPLIBInfo = NamedTuple{(:name, :jobs, :machines, :optimum, :lower, :upper),
                              Tuple{String, Int, Int, Union{Nothing,Int},
                                    Union{Nothing,Int}, Union{Nothing,Int}}}

const _JSPLIB_META = Ref{Union{Nothing, Vector{JSPLIBInfo}}}(nothing)

_parse_opt(s) = isempty(s) ? nothing : parse(Int, s)

function _jsplib_meta()
    if _JSPLIB_META[] === nothing
        rows = JSPLIBInfo[]
        for (lineno, line) in enumerate(eachline(joinpath(JSPLIB_DIR, "metadata.csv")))
            lineno == 1 && continue                     # Header
            isempty(strip(line)) && continue
            f = split(line, ',')
            push!(rows, (name = String(f[1]), jobs = parse(Int, f[2]),
                         machines = parse(Int, f[3]), optimum = _parse_opt(f[4]),
                         lower = _parse_opt(f[5]), upper = _parse_opt(f[6])))
        end
        _JSPLIB_META[] = rows
    end
    return _JSPLIB_META[]
end

"""
    list_jsplib(; prefix = "", max_jobs = typemax(Int), max_machines = typemax(Int),
                  only_known_optimum = false) -> Vector{String}

Names of the JSPLIB instances, optionally filtered, e.g.
`list_jsplib(prefix = "la", max_jobs = 15)`.
"""
function list_jsplib(; prefix::AbstractString = "", max_jobs::Int = typemax(Int),
                       max_machines::Int = typemax(Int), only_known_optimum::Bool = false)
    [r.name for r in _jsplib_meta()
     if startswith(r.name, prefix) && r.jobs <= max_jobs && r.machines <= max_machines &&
        (!only_known_optimum || r.optimum !== nothing)]
end

"Metadata (size, optimum or bounds) of a JSPLIB instance."
function jsplib_info(name::AbstractString)
    idx = findfirst(r -> r.name == name, _jsplib_meta())
    idx === nothing && error("unknown JSPLIB instance \"$name\"")
    return _jsplib_meta()[idx]
end

"""
    load_jsplib(name) -> JSSPInstance

Loads a JSPLIB instance by name (e.g. "ft06", "la01", "ta01").
"""
function load_jsplib(name::AbstractString)
    path = joinpath(JSPLIB_DIR, name)
    isfile(path) || error("JSPLIB file not found: $path")
    inst = read_standard(path; name = name)
    info = jsplib_info(name)
    (inst.n, inst.m) == (info.jobs, info.machines) ||
        error("$name: size $(inst.n)×$(inst.m) does not match metadata $(info.jobs)×$(info.machines)")
    return inst
end
