#!/usr/bin/env python3
"""stall_check.py — detect benchmark runs distorted by system sleep / stalls (wall-clock limits).

Usage:
    python3 scripts/stall_check.py <results folder> [--runs=inst__method,...] [--invalidate]
                                   [--reason="..."] [--quiet]

Scans every complete run folder <results folder>/runs/<instance>__<method>/ (DONE / done marker;
*.tmp, *.FAILED and runs_invalidated/ are ignored) and flags a run as STALLED if

  fix-and-optimize runs (iterations.csv):
    * a sub-problem with sub_time > 10 s
    * a setup_time > 5 s
    * a wall-clock gap > 10 s between consecutive iterations (iteration `time` stamps;
      the gap before the first iteration contains the model build and is not checked)
  all runs:
    * total runtime > 1210 s (CP-SAT runs that ended OPTIMAL early are not affected anyway)
    * the run's wall-clock interval overlaps a system sleep interval from `pmset -g log`
      ("Entering Sleep state" … next "Wake from" / "DarkWake from"). Needed for full-Gurobi and CP-SAT runs: their time limit is
      wall-clock, so a run that slept still ends after ~1200 s, and their traces contain only
      improvements (no regular log lines whose gaps could be checked).

Writes <results folder>/stall_scan.csv (one row per finding: instance, method, iteration, time,
sub_time, setup_time, wallclock, lost_s, reason) and stall_runs.csv (one row per scanned run:
stalled, total time lost = Σ max(0, sub_time − 7.5) + max(0, setup_time − 1), or the sleep
overlap for runs without iterations). With --runs only those runs are scanned and the two
files are written as stall_scan_<timestamp>.csv / stall_runs_<timestamp>.csv instead.
--scan-dir=<dir> writes these files into <dir> instead of the results folder.
--retry-note="..." appends "STILL STALLED after retry, kept" lines to repair_log.txt (no move).

--invalidate moves every flagged run folder to <results folder>/runs_invalidated/
<instance>__<method>__<timestamp>/ (kept, not deleted) and appends a line per run to
<results folder>/repair_log.txt.

Exit code: 0 = no stalled run, 2 = at least one stalled run, 1 = error.
Standard library only (no solver, no Julia).
"""

import csv
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys

SUB_MAX, SETUP_MAX, GAP_MAX, RUNTIME_MAX = 10.0, 5.0, 10.0, 1210.0
SUB_LIMIT, SETUP_OK = 7.5, 1.0


def parse_time(s):
    s = (s or "").strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return dt.datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def sleep_intervals():
    """(start, end) of every system sleep from `pmset -g log`: "Entering Sleep state" → next
    "Wake from" or "DarkWake from" (processes run again during a DarkWake)."""
    try:
        out = subprocess.run(["pmset", "-g", "log"], capture_output=True, text=True, timeout=120).stdout
    except Exception:
        return []
    ev = []
    pat = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) [+-]\d{4} (Sleep|Wake|DarkWake)\s+"
                     r"(Entering Sleep state|Wake from|DarkWake from)")
    for line in out.splitlines():
        m = pat.match(line)
        if m:
            ev.append((dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"), m.group(2)))
    res, start = [], None
    for t, kind in ev:
        if kind == "Sleep" and start is None:
            start = t
        elif kind in ("Wake", "DarkWake") and start is not None:
            res.append((start, t))
            start = None
    if start is not None:
        res.append((start, dt.datetime.now()))
    return res


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def run_interval(d, run):
    """Wall-clock (start, end) of a run."""
    runtime = fnum(run.get("runtime"))
    start = parse_time(run.get("started", ""))
    if start is None:                                   # CP-SAT folders: output written right after the solve
        for f in ("cpsat.json", "cpsat_meta.csv", "run.csv"):
            p = os.path.join(d, f)
            if os.path.isfile(p):
                start = dt.datetime.fromtimestamp(os.path.getmtime(p)) - dt.timedelta(seconds=runtime)
                break
    if start is None:
        return None, None
    return start, start + dt.timedelta(seconds=runtime)


def scan_run(d, inst, method, sleeps):
    rows, lost = [], 0.0
    run = read_csv(os.path.join(d, "run.csv"))[0]
    start, end = run_interval(d, run)
    runtime = fnum(run.get("runtime"))
    status = run.get("cpsat_status") or run.get("status") or ""

    def add(it, t, sub, setup, reason):
        wall = (start + dt.timedelta(seconds=fnum(t))).strftime("%Y-%m-%d %H:%M:%S") if start and t != "" else ""
        rows.append(dict(instance=inst, method=method, iteration=it, time=t, sub_time=sub,
                         setup_time=setup, wallclock=wall, lost_s="", reason=reason))
        return rows[-1]

    its_path = os.path.join(d, "iterations.csv")
    if os.path.isfile(its_path):
        its = read_csv(its_path)
        prev_t = None
        for r in its:
            t, sub, setup = fnum(r.get("time")), fnum(r.get("sub_time")), fnum(r.get("setup_time"))
            l = max(0.0, sub - SUB_LIMIT) + max(0.0, setup - SETUP_OK)
            lost += l
            reasons = []
            if sub > SUB_MAX:
                reasons.append(f"sub_time {sub:.1f} s > {SUB_MAX:g}")
            if setup > SETUP_MAX:
                reasons.append(f"setup_time {setup:.1f} s > {SETUP_MAX:g}")
            if prev_t is not None and t - prev_t > GAP_MAX:
                reasons.append(f"gap {t - prev_t:.1f} s since previous iteration > {GAP_MAX:g}")
            if reasons:
                add(r.get("iteration", ""), r.get("time", ""), r.get("sub_time", ""), r.get("setup_time", ""),
                    "; ".join(reasons))["lost_s"] = round(l, 1)
            prev_t = t
    if runtime > RUNTIME_MAX and not (status == "OPTIMAL" and runtime < RUNTIME_MAX):
        add("", "", "", "", f"runtime {runtime:.1f} s > {RUNTIME_MAX:g}")
    overlap = 0.0
    if start is not None:
        for s, e in sleeps:
            o = (min(e, end) - max(s, start)).total_seconds()
            if o > 0:
                overlap += o
                add("", "", "", "", f"system sleep {s:%Y-%m-%d %H:%M:%S}–{e:%H:%M:%S} overlaps run "
                    f"({start:%H:%M:%S}–{end:%H:%M:%S}), {o:.0f} s")["lost_s"] = round(o, 1)
    else:
        add("", "", "", "", "WARNING: run start unknown, sleep overlap not checked")
        rows[-1]["reason"] += " (not counted as stall)"
    stalled = any(not r["reason"].startswith("WARNING") for r in rows)
    total_lost = lost if os.path.isfile(its_path) else overlap
    summary = dict(instance=inst, method=method, stalled=stalled, runtime=round(runtime, 1),
                   start=f"{start:%Y-%m-%d %H:%M:%S}" if start else "", end=f"{end:%Y-%m-%d %H:%M:%S}" if end else "",
                   total_lost_s=round(total_lost, 1), sleep_overlap_s=round(overlap, 1),
                   reasons=(f"{len(rows)} finding(s): " + " | ".join(r["reason"] for r in rows[:6]) +
                            (" | …" if len(rows) > 6 else "")) if rows else "")
    return rows, summary


def main(argv):
    if len(argv) < 2 or argv[1].startswith("--"):
        print(__doc__)
        return 1
    folder = os.path.abspath(argv[1])
    opts = {a.split("=", 1)[0]: (a.split("=", 1)[1] if "=" in a else True) for a in argv[2:]}
    only = set(opts["--runs"].split(",")) if isinstance(opts.get("--runs"), str) else None
    runs_dir = os.path.join(folder, "runs")
    if not os.path.isdir(runs_dir):
        print(f"no runs/ folder in {folder}")
        return 1
    sleeps = sleep_intervals()
    findings, summaries = [], []
    for name in sorted(os.listdir(runs_dir)):
        d = os.path.join(runs_dir, name)
        if not os.path.isdir(d) or name.endswith(".tmp") or name.endswith(".FAILED") or "__" not in name:
            continue
        if only is not None and name not in only:
            continue
        if not (os.path.isfile(os.path.join(d, "DONE")) or os.path.isfile(os.path.join(d, "done"))):
            continue
        if not os.path.isfile(os.path.join(d, "run.csv")):
            continue
        inst, method = name.split("__", 1)
        rows, summ = scan_run(d, inst, method, sleeps)
        findings += rows
        summaries.append(summ)
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    suffix = "" if only is None else f"_{stamp}"
    scan_dir = os.path.abspath(opts["--scan-dir"]) if isinstance(opts.get("--scan-dir"), str) else folder
    os.makedirs(scan_dir, exist_ok=True)
    f1 = os.path.join(scan_dir, f"stall_scan{suffix}.csv")
    f2 = os.path.join(scan_dir, f"stall_runs{suffix}.csv")
    cols = ["instance", "method", "iteration", "time", "sub_time", "setup_time", "wallclock", "lost_s", "reason"]
    with open(f1, "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(findings)
    with open(f2, "w", newline="") as f:
        w = csv.DictWriter(f, ["instance", "method", "stalled", "runtime", "start", "end", "total_lost_s",
                               "sleep_overlap_s", "reasons"])
        w.writeheader()
        w.writerows(summaries)
    stalled = [s for s in summaries if s["stalled"]]
    if not opts.get("--quiet"):
        print(f"stall check {folder}: {len(summaries)} runs scanned, {len(stalled)} stalled; "
              f"{len(sleeps)} system sleep intervals known (pmset); → {os.path.basename(f1)}, {os.path.basename(f2)}")
        for s in stalled:
            print(f"  STALLED {s['instance']}__{s['method']}: lost {s['total_lost_s']} s; {s['reasons']}")
    if isinstance(opts.get("--retry-note"), str) and stalled:   # still stalled after the retry: keep, log
        with open(os.path.join(folder, "repair_log.txt"), "a") as log:
            for s in stalled:
                log.write(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  STILL STALLED after retry, kept: "
                          f"{s['instance']}__{s['method']} ({opts['--retry-note']}; time lost "
                          f"{s['total_lost_s']} s; {s['reasons']})\n")
    if opts.get("--invalidate") and stalled:
        inv = os.path.join(folder, "runs_invalidated")
        os.makedirs(inv, exist_ok=True)
        reason = opts.get("--reason") if isinstance(opts.get("--reason"), str) else "stall check"
        with open(os.path.join(folder, "repair_log.txt"), "a") as log:
            for s in stalled:
                name = f"{s['instance']}__{s['method']}"
                dest = os.path.join(inv, f"{name}__{stamp}")
                shutil.move(os.path.join(runs_dir, name), dest)
                log.write(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  INVALIDATED {name} → runs_invalidated/"
                          f"{os.path.basename(dest)}  ({reason}; run {s['start']}–{s['end']}, runtime "
                          f"{s['runtime']} s, time lost {s['total_lost_s']} s; {s['reasons']})\n")
                print(f"  moved {name} → runs_invalidated/{os.path.basename(dest)}")
    # machine-readable list of stalled runs on the last line
    print("STALLED_RUNS=" + ",".join(f"{s['instance']}__{s['method']}" for s in stalled))
    return 2 if stalled else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
