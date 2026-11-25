"""cpsat_jssp.py — OR-Tools CP-SAT on the full JSSP (external reference).

Standard CP-SAT job shop formulation (as in the official OR-Tools jobshop example):
one interval per operation (start, duration = processing time, end), job precedence
end[j][k] <= start[j][k+1], AddNoOverlap per machine, makespan = max over jobs of the end
of the last operation, minimise makespan, horizon = sum of all processing times.
Warm start: the initial solution's start times (and its makespan) as hints.

Input is the JSON written by scripts/export_instances.jl (Julia indexing, never the raw
instance file). Output JSON: final start times per job and operation (same indexing),
solver status, objective, bound, build time, wall time, and the anytime trace
(seconds since solve start, objective, best bound) recorded by a solution callback.

    .venv-cpsat/bin/python scripts/cpsat_jssp.py --input in.json --output out.json \
        --workers 1 --time 1200 --seed 42
"""

import argparse
import json
import resource
import sys
import time

import ortools
from ortools.sat.python import cp_model


class TraceCallback(cp_model.CpSolverSolutionCallback):
    """Records (seconds since solve start, objective, best bound) for every new solution."""

    def __init__(self):
        super().__init__()
        self.trace = []

    def on_solution_callback(self):
        self.trace.append([self.wall_time, self.objective_value, self.best_objective_bound])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--time", type=float, default=1200.0)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    with open(a.input) as f:
        data = json.load(f)
    jobs = data["jobs"]

    t_build = time.time()
    model = cp_model.CpModel()
    horizon = sum(p for job in jobs for _, p in job)
    starts, ends, per_machine = [], [], {}
    for j, job in enumerate(jobs):
        sj, ej = [], []
        for k, (machine, p) in enumerate(job):
            s = model.new_int_var(0, horizon, f"s_{j}_{k}")
            e = model.new_int_var(0, horizon, f"e_{j}_{k}")
            iv = model.new_interval_var(s, p, e, f"i_{j}_{k}")
            per_machine.setdefault(machine, []).append(iv)
            sj.append(s)
            ej.append(e)
        for k in range(len(job) - 1):
            model.add(ej[k] <= sj[k + 1])
        starts.append(sj)
        ends.append(ej)
    for ivs in per_machine.values():
        model.add_no_overlap(ivs)
    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, [ej[-1] for ej in ends])
    model.minimize(makespan)
    # warm start: same starting point as all other methods (initial_schedule)
    for j, row in enumerate(data["initial_start_times"]):
        for k, s0 in enumerate(row):
            model.add_hint(starts[j][k], s0)
    model.add_hint(makespan, data["initial_makespan"])
    build_time = time.time() - t_build

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = a.time
    solver.parameters.random_seed = a.seed
    solver.parameters.num_workers = a.workers
    solver.parameters.log_search_progress = False
    cb = TraceCallback()
    status = solver.solve(model, cb)

    found = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    out = {
        "instance": data["name"],
        "num_workers": a.workers,
        "time_limit": a.time,
        "seed": a.seed,
        "status": solver.status_name(status),
        "objective": solver.objective_value if found else None,
        "bound": solver.best_objective_bound,
        "build_time": build_time,
        "wall_time": solver.wall_time,
        "start_times": [[solver.value(s) for s in sj] for sj in starts] if found else None,
        "trace": cb.trace,
        "initial_makespan": data["initial_makespan"],
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,  # bytes on macOS
        "python": sys.version.split()[0],
        "ortools": ortools.__version__,
    }
    with open(a.output, "w") as f:
        json.dump(out, f)
    # CSV companions for Julia (JSON.jl is not a direct project dependency): same content
    base = a.output[:-5] if a.output.endswith(".json") else a.output
    with open(base + "_starts.csv", "w") as f:
        m = len(jobs[0])
        f.write(",".join(f"op{k + 1}" for k in range(m)) + "\n")
        for row in (out["start_times"] or []):
            f.write(",".join(str(v) for v in row) + "\n")
    with open(base + "_trace.csv", "w") as f:
        f.write("time,objective,bound\n")
        for t, obj, bnd in cb.trace:
            f.write(f"{t},{obj},{bnd}\n")
    with open(base + "_meta.csv", "w") as f:
        f.write("key,value\n")
        for k in ("instance", "num_workers", "time_limit", "seed", "status", "objective", "bound",
                  "build_time", "wall_time", "initial_makespan", "peak_rss_mb", "python", "ortools"):
            f.write(f"{k},{'' if out[k] is None else out[k]}\n")
    print(f"{data['name']} workers={a.workers}: {out['status']} objective={out['objective']} "
          f"bound={out['bound']} wall={out['wall_time']:.1f}s build={build_time:.2f}s "
          f"solutions={len(cb.trace)}")


if __name__ == "__main__":
    main()
