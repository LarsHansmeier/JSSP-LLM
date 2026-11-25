# Run:   julia --project=. scripts/run_example.jl
# (or in VS Code: open the file, "Julia: Execute File" — also repeatedly)

# Activate the project environment if not done yet (otherwise e.g. JuMP is missing).
import Pkg
const _ROOT = normpath(joinpath(@__DIR__, ".."))
Base.active_project() == joinpath(_ROOT, "Project.toml") || Pkg.activate(_ROOT; io = devnull)

# Own module: when re-run in the same REPL it is replaced completely
# (otherwise the JSSPLab exports in Main would be ambiguous -> UndefVarError).
# Variables are then reachable in the REPL, e.g. via JSSPLabExample.inst.
module JSSPLabExample

include(joinpath(@__DIR__, "..", "src", "JSSPLab.jl"))
using .JSSPLab

datadir = joinpath(@__DIR__, "..", "data", "instances")

for file in ["example3x3.txt", "ft06.txt"]
    println("\n" * "="^60)
    inst = read_instance(joinpath(datadir, file); format = :standard)
    print_instance(inst)
    println("Triviale untere Schranke: ", trivial_lower_bound(inst))

    s = initial_schedule(inst)
    print_schedule(inst, s)
    check_schedule(inst, s)

    # Cross-check: extract sequences and rebuild the schedule from them
    s2 = schedule_from_sequences(inst, machine_sequences(inst, s))
    println("Neu aufgebaut aus Reihenfolgen: Makespan ", makespan(inst, s2))
end

# Exact MIP (Gurobi) for ft06, with the start solution as warm start
println("\n" * "="^60)
inst = read_instance(joinpath(datadir, "ft06.txt"))
res = solve_model(inst; time_limit = 30, silent = true, warm_start = initial_schedule(inst))
println(res.status, "  Incumbent=", res.incumbent, "  Bound=", res.bound,
        "  Gap=", res.gap, "  Zeit=", round(res.runtime, digits = 2), "s")
check_schedule(inst, res.schedule; claimed_makespan = res.incumbent)

end # module JSSPLabExample
