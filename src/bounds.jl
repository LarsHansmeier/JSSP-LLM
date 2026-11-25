# =============================================================================
# bounds.jl — Lower bounds for the makespan (without solver)
# =============================================================================
#
# One-machine relaxation: for every machine i only its capacity is respected.
# Every operation (j,k) on i gets
#   head r = sum of processing times of its job predecessors (earliest start)
#   tail q = sum of processing times of its job successors   (remaining work)
# The problem 1 | r_j, pmtn | max(C_j + q_j) is solved exactly by the preemptive
# Jackson schedule (always process the available operation with the largest tail).
# Its value is a valid lower bound for the JSSP makespan; the maximum over all
# machines is at least as strong as trivial_lower_bound.
# =============================================================================

"Preemptive Jackson schedule: optimal value of 1 | r_j, pmtn | max(C_j + q_j)."
function jackson_preemptive(r::Vector{Int}, p::Vector{Int}, q::Vector{Int})
    n = length(p)
    rem = copy(p)
    t, obj = 0, 0
    while any(>(0), rem)
        avail = [j for j in 1:n if r[j] <= t && rem[j] > 0]
        if isempty(avail)                          # idle until the next release
            t = minimum(r[j] for j in 1:n if rem[j] > 0)
            continue
        end
        j = avail[argmax(q[avail])]
        future = [r[l] for l in 1:n if r[l] > t && rem[l] > 0]
        run = isempty(future) ? rem[j] : min(rem[j], minimum(future) - t)
        t += run
        rem[j] -= run
        rem[j] == 0 && (obj = max(obj, t + q[j]))
    end
    return obj
end

"One-machine bound: maximum of the preemptive Jackson schedule over all machines."
function one_machine_lower_bound(inst::JSSPInstance)
    best = 0
    for i in 1:inst.m
        ops = ops_on_machine(inst, i)
        isempty(ops) && continue
        r = [sum(inst.ptime[j, 1:k-1]; init = 0) for (j, k) in ops]
        q = [sum(inst.ptime[j, k+1:end]; init = 0) for (j, k) in ops]
        p = [inst.ptime[j, k] for (j, k) in ops]
        best = max(best, jackson_preemptive(r, p, q))
    end
    return best
end

"Best available combinatorial lower bound (trivial & one-machine relaxation)."
lower_bound(inst::JSSPInstance) = max(trivial_lower_bound(inst), one_machine_lower_bound(inst))
