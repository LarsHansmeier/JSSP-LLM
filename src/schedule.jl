# =============================================================================
# schedule.jl — Representation of a solution (schedule)
# =============================================================================
#
# A solution is described by the start times S[j, k] of every operation.
# From these one can derive:
#   - the makespan
#   - the order of the operations on every machine
#     (exactly the decisions that Fix-and-Optimize fixes/frees)
# =============================================================================

"""
    Schedule

Start times of all operations: `start[j, k]` = start time of the k-th operation of job j.
"""
struct Schedule
    start::Matrix{Int}
end

completion(inst::JSSPInstance, s::Schedule, j::Int, k::Int) = s.start[j, k] + inst.ptime[j, k]

"Makespan = latest completion time of all operations."
makespan(inst::JSSPInstance, s::Schedule) =
    maximum(completion(inst, s, j, k) for (j, k) in operations(inst))

"""
    machine_sequences(inst, s) -> Vector{Vector{Tuple{Int,Int}}}

For every machine the operations sorted by start time.
`seq[i]` is the processing order on machine i.
"""
function machine_sequences(inst::JSSPInstance, s::Schedule)
    return [sort(ops_on_machine(inst, i); by = op -> s.start[op...]) for i in 1:inst.m]
end

"""
    schedule_from_sequences(inst, seqs) -> Schedule

Builds the earliest possible (semi-active) schedule from fixed machine sequences:
every operation starts as soon as its job predecessor AND its machine predecessor are done.
Throws an error if the sequences are cyclic (i.e. infeasible).

Useful to compute start times after fixing sequences.
"""
function schedule_from_sequences(inst::JSSPInstance, seqs::Vector{Vector{Tuple{Int,Int}}})
    start = fill(-1, inst.n, inst.m)
    pos = zeros(Int, inst.m)          # next index per machine sequence
    next_op = ones(Int, inst.n)       # next operation to schedule per job
    scheduled = 0
    total = length(operations(inst))
    while scheduled < total
        progress = false
        for i in 1:inst.m
            pos[i] < length(seqs[i]) || continue
            j, k = seqs[i][pos[i] + 1]
            next_op[j] == k || continue               # job predecessor not yet scheduled
            job_ready = k == 1 ? 0 : start[j, k-1] + inst.ptime[j, k-1]
            mach_ready = pos[i] == 0 ? 0 : begin
                (pj, pk) = seqs[i][pos[i]]
                start[pj, pk] + inst.ptime[pj, pk]
            end
            start[j, k] = max(job_ready, mach_ready)
            pos[i] += 1
            next_op[j] += 1
            scheduled += 1
            progress = true
        end
        progress || error("machine sequences are cyclic – no feasible schedule possible")
    end
    return Schedule(start)
end

"Compact text output of the machine assignment."
function print_schedule(inst::JSSPInstance, s::Schedule)
    println("Makespan: ", makespan(inst, s))
    for (i, seq) in enumerate(machine_sequences(inst, s))
        items = join(["J$(j)[$(s.start[j,k])–$(completion(inst, s, j, k))]" for (j, k) in seq], "  ")
        println("  M$i: ", items)
    end
end
