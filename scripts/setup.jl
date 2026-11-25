# Run once (in the project folder):   julia scripts/setup.jl
# Creates Project.toml/Manifest.toml and installs all packages.

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.add(["JuMP", "Gurobi", "CSV", "DataFrames", "Plots"])
Pkg.instantiate()

println("\nTeste Gurobi-Lizenz ...")
using Gurobi
env = Gurobi.Env()
println("✓ Gurobi läuft. Setup abgeschlossen.")
