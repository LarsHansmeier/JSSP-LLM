# =============================================================================
# compare_tuning.jl — random_free vs. window_free at equal n_free_y
# =============================================================================
#
# Reads results/tuning/runs.csv (random, read only) and results/tuning_window/runs.csv
# and writes to results/tuning_window/:
#   comparison.csv        one row per (instance, n_free_y), means over seeds
#   median_sub_time.png   x = n_free_y (log), y = median_sub_time; dashed = random, solid = window
#
# Run:   julia --project=. scripts/compare_tuning.jl
# =============================================================================

import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")); io = devnull)
ENV["GKSwstype"] = "100"     # GR without window (file output)

using CSV, DataFrames, Plots, Statistics

const RES = normpath(joinpath(@__DIR__, "..", "results"))

function load(dir, selector)
    df = CSV.read(joinpath(RES, dir, "runs.csv"), DataFrame)
    df.selector .= selector              # older random runs have no selector column
    return combine(groupby(df, [:instance, :n_free_y, :selector]),
                   [:primal_gap_bks, :share_time_limit, :median_sub_time] .=> mean;
                   renamecols = false)
end

long = vcat(load("tuning", "random"), load("tuning_window", "window"))
wide(col) = unstack(long, [:instance, :n_free_y], :selector, col;
                    renamecols = s -> "$(col)_$s")
cmp = outerjoin(wide(:primal_gap_bks), wide(:share_time_limit), wide(:median_sub_time);
                on = [:instance, :n_free_y])
rename!(n -> replace(n, "primal_gap_bks" => "primal_gap"), cmp)
sort!(cmp, [:instance, :n_free_y])
CSV.write(joinpath(RES, "tuning_window", "comparison.csv"), cmp)

nf = sort(unique(long.n_free_y))
p = plot(; xscale = :log10, xticks = (nf, string.(nf)), xlabel = "n_free_y (log)", ylabel = "median_sub_time [s]",
         title = "Teilproblem-Lösezeit: random (gestrichelt) vs. window (durchgezogen)",
         legend = :outerright, size = (900, 500), titlefontsize = 11, dpi = 120)
for (i, inst) in enumerate(unique(long.instance)), (sel, ls) in (("random", :dash), ("window", :solid))
    d = sort(filter(r -> r.instance == inst && r.selector == sel, long), :n_free_y)
    nrow(d) > 0 && plot!(p, d.n_free_y, d.median_sub_time; ls = ls, marker = :circle,
                         color = i, lw = 2, label = "$inst ($sel)")
end
savefig(p, joinpath(RES, "tuning_window", "median_sub_time.png"))
println(cmp)
