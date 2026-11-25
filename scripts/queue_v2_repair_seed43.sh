#!/bin/bash
# =============================================================================
# queue_v2_repair_seed43.sh — unattended chain:
#   0. wait until no benchmark process is running
#   A. stall check of results/benchmark_large_v2 → invalidate stalled runs → rerun them with the
#      v2 restart mechanism (seed 42) → stall check of the reruns (1 retry) → re-aggregate v2 and
#      benchmark_large_combined; then stall check (and if needed the same repair) of
#      benchmark_large_v1 and benchmark_large_v1_seed43
#   B. v2 with seed 43 → results/benchmark_large_v2_seed43 (smoke test on ta31 first, stall
#      check after every run), seed comparison, results/benchmark_large_combined_2seeds
# Start:
#   nohup caffeinate -ims bash scripts/queue_v2_repair_seed43.sh > results/benchmark_large_v2_seed43/queue.log 2>&1 &
# Status: results/benchmark_large_v2_seed43/STATUS.txt (WAITING / RUNNING …, finally DONE or FAILED).
# =============================================================================
set -u

ROOT="/Users/larshansmeier/Repos/OR Lab B"
JULIA="/Users/larshansmeier/.julia/juliaup/julia-1.13.4+0.aarch64.apple.darwin14/Julia-1.13.app/Contents/Resources/julia/bin/julia"
PY="$ROOT/.venv-cpsat/bin/python"
STALL="$ROOT/scripts/stall_check.py"
SEED=43
R="$ROOT/results"
V1="$R/benchmark_large_v1"; V1S="$R/benchmark_large_v1_seed43"; V2="$R/benchmark_large_v2"
OUT="$R/benchmark_large_v2_seed43"
V2SEED_SCRIPT="$ROOT/scripts/benchmark_large_v2_seed.jl"
SLEEP_NOTE="Mac asleep 2026-11-24 15:37-15:57 (lid closed on battery), wall-clock limits exceeded"

log()    { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*"; }
status() { echo "$*" > "$OUT/STATUS.txt"; log "STATUS: $*"; }
fail()   { status "FAILED — $*"; exit 1; }
jl()     { "$JULIA" --project="$ROOT" "$@"; }

cd "$ROOT" || { echo "cannot cd to $ROOT"; exit 1; }
mkdir -p "$OUT"
log "queue started (pid $$); julia: $JULIA"

# number of complete run folders (DONE marker; `done` is the same file on macOS) in <folder>/runs
done_count() {
    local n=0 d
    for d in "$1"/runs/*/; do
        d=${d%/}
        case "$d" in *.tmp|*.FAILED) continue ;; esac
        [ -f "$d/DONE" ] && n=$((n + 1))
    done
    echo "$n"
}

# run_until_complete <folder> <expected runs> <log file> <command…>: run the (restartable) driver;
# if it exits with an error or leaves runs incomplete, run it once more (it skips DONE runs)
run_until_complete() {
    local folder=$1 expected=$2 logfile=$3; shift 3
    local attempt rc n
    for attempt in 1 2; do
        echo "== started by queue_v2_repair_seed43.sh $(date '+%Y-%m-%dT%H:%M:%S') (attempt $attempt)" >> "$logfile"
        "$@" >> "$logfile" 2>&1 &
        local pid=$!
        [ "$folder" = "$V2" ] && echo "$pid" > "$V2/run.pid"
        wait "$pid"; rc=$?
        n=$(done_count "$folder")
        log "driver for $(basename "$folder") exited with code $rc; $n/$expected runs DONE"
        [ "$rc" -eq 0 ] && [ "$n" -ge "$expected" ] && return 0
    done
    return 1
}

# stall <folder> <stall_check options…>: runs the stall check, logs it, sets STALLED_LIST, returns its exit code
STALLED_LIST=""
stall() {
    local folder=$1; shift
    local out rc
    out=$("$PY" "$STALL" "$folder" "$@" 2>&1); rc=$?
    echo "$out" | sed 's/^/        /'
    STALLED_LIST=$(echo "$out" | sed -n 's/^STALLED_RUNS=//p' | tail -1)
    return $rc
}

# repair <label> <folder> <expected runs> <log file> <driver command…>
# stall check (full folder) → invalidate → rerun → check reruns (invalidate) → rerun → check (keep + note)
REPAIRED=""
repair() {
    local label=$1 folder=$2 expected=$3 logfile=$4; shift 4
    local list attempt
    REPAIRED=""
    log "$label: stall check of all runs"
    stall "$folder" --invalidate "--reason=$label stall check: $SLEEP_NOTE"
    [ $? -eq 1 ] && { log "$label: stall check error"; return 1; }
    list=$STALLED_LIST
    [ -z "$list" ] && { log "$label: no stalled runs, nothing to repair"; return 0; }
    REPAIRED=$list
    for attempt in 1 2; do
        log "$label: rerunning $list (attempt $attempt) with its own restart mechanism"
        run_until_complete "$folder" "$expected" "$logfile" "$@" || log "$label: WARNING driver did not complete all runs"
        if [ $attempt -eq 1 ]; then
            stall "$folder" "--runs=$list" --invalidate "--reason=$label: rerun stalled again" "--scan-dir=$folder/stall_checks"
        else
            stall "$folder" "--runs=$list" "--retry-note=$label: still stalled after one retry" "--scan-dir=$folder/stall_checks"
        fi
        if [ -z "$STALLED_LIST" ]; then
            log "$label: reruns are clean ($list)"
            echo "$(date '+%Y-%m-%d %H:%M:%S')  RERUN OK (not stalled): $list" >> "$folder/repair_log.txt"
            return 0
        fi
        [ $attempt -eq 2 ] && { log "$label: still stalled after retry, kept and marked: $STALLED_LIST"; return 0; }
        list=$STALLED_LIST
    done
}

# --- 0. wait until no other benchmark process is running ----------------------------------
other_benchmarks() {
    pgrep -f "scripts/benchmark_large|scripts/benchmark_seed|scripts/benchmark_cpsat|cpsat_jssp.py|queue_seed43.sh" |
        grep -v "^$$\$" | tr '\n' ' '
}
status "WAITING for running benchmark processes to end"
log "v1_seed43 STATUS: $(cat "$V1S/STATUS.txt" 2>/dev/null || echo 'none')"
while :; do
    p=$(other_benchmarks)
    [ -z "$p" ] && break
    log "benchmark process(es) still running: $p — waiting 60 s"
    sleep 60
done
log "no benchmark process running"

# --- A. repair v2 (seed 42) -----------------------------------------------------------------
status "RUNNING — Part A: stall check and repair of benchmark_large_v2 (seed 42)"
repair "v2 (seed 42)" "$V2" 28 "$V2/run.log" jl "$ROOT/scripts/benchmark_large_v2.jl"
V2_REPAIRED=$REPAIRED
[ "$(done_count "$V2")" -ge 28 ] || fail "Part A: benchmark_large_v2 incomplete after repair ($(done_count "$V2")/28 DONE)"
log "Part A: re-aggregating benchmark_large_v2 and benchmark_large_combined"
jl "$ROOT/scripts/benchmark_large_v2.jl" --aggregate >> "$V2/run.log" 2>&1 || fail "Part A: v2 aggregation failed (see $V2/run.log)"
if [ -n "$V2_REPAIRED" ]; then         # old values must be gone: run start times of invalidated runs
    "$PY" - "$V2" <<'EOF'
import csv, glob, os, sys
v2 = sys.argv[1]
runs = list(csv.DictReader(open(os.path.join(v2, "runs.csv"))))
started_now = {r.get("started", "") for r in runs}
for d in sorted(glob.glob(os.path.join(v2, "runs_invalidated", "*"))):
    old = list(csv.DictReader(open(os.path.join(d, "run.csv"))))[0]
    new = [r for r in runs if r["instance"] == old["instance"] and r["method"] == old["method"]]
    gone = old.get("started", "") not in started_now
    print(f"        {os.path.basename(d)}: old start {old.get('started')} incumbent {old['incumbent']} → "
          f"aggregated now: start {new[0].get('started') if new else '-'} incumbent {new[0]['incumbent'] if new else '-'} "
          f"→ old values {'GONE' if gone else 'STILL PRESENT'}")
EOF
fi
log "Part A: final stall check of benchmark_large_v2"
stall "$V2"

# --- A6. stall check of v1 and v1_seed43 (repair the same way if needed) --------------------
status "RUNNING — Part A: stall check of benchmark_large_v1 and benchmark_large_v1_seed43"
repair "v1 (seed 42)" "$V1" 20 "$V1/run.log" jl "$ROOT/scripts/benchmark_large.jl"
if [ -n "$REPAIRED" ]; then
    log "v1 repaired → re-aggregating v1-dependent outputs (CP-SAT v1 comparison, combined, v1 seed comparison)"
    jl "$ROOT/scripts/benchmark_large.jl" --aggregate >> "$V1/run.log" 2>&1
    jl "$ROOT/scripts/benchmark_cpsat.jl" --aggregate >> "$R/benchmark_large_cpsat_v1/run.log" 2>&1
    jl "$ROOT/scripts/benchmark_large_v2.jl" --aggregate >> "$V2/run.log" 2>&1
    jl "$ROOT/scripts/benchmark_seed.jl" --seed-compare --seed=43 >> "$V1S/run.log" 2>&1
fi
repair "v1_seed43" "$V1S" 20 "$V1S/run.log" jl "$ROOT/scripts/benchmark_seed.jl" --seed=43
if [ -n "$REPAIRED" ]; then
    log "v1_seed43 repaired → re-aggregating it and its seed comparison"
    jl "$ROOT/scripts/benchmark_seed.jl" --aggregate --seed=43 >> "$V1S/run.log" 2>&1
    jl "$ROOT/scripts/benchmark_seed.jl" --seed-compare --seed=43 >> "$V1S/run.log" 2>&1
fi

# --- B1. smoke test of v2 seed 43 (ta31, 60 s, all 7 methods) --------------------------------
status "RUNNING — Part B step 1: smoke test (ta31, 60 s per method, seed $SEED)"
mkdir -p "$OUT/smoke"
jl "$V2SEED_SCRIPT" --seed=$SEED --time=60 --instances=ta31 --out=benchmark_large_v2_seed43/smoke >> "$OUT/smoke/run.log" 2>&1
log "smoke driver exited with code $?"
check_out=$(jl "$V2SEED_SCRIPT" --check --seed=$SEED --time=60 --instances=ta31 --out=benchmark_large_v2_seed43/smoke 2>&1)
rc=$?
log "smoke check: $check_out"
[ $rc -eq 0 ] || fail "Part B step 1 (smoke test): $(echo "$check_out" | grep -E 'CHECK|ERROR' | head -3)"

# --- B2. full v2 benchmark with seed 43 ----------------------------------------------------------
status "RUNNING — Part B step 2: full v2 benchmark with seed $SEED (28 runs); see $OUT/progress.csv"
run_until_complete "$OUT" 28 "$OUT/run.log" jl "$V2SEED_SCRIPT" --seed=$SEED ||
    log "WARNING: v2 seed-43 driver did not complete all runs ($(done_count "$OUT")/28)"
log "aggregating v2 seed 43"
jl "$V2SEED_SCRIPT" --aggregate --seed=$SEED >> "$OUT/run.log" 2>&1 || log "WARNING: aggregation exited with an error"
check_out=$(jl "$V2SEED_SCRIPT" --check --seed=$SEED 2>&1)
check_rc=$?
log "full check: $check_out"
log "final stall check of benchmark_large_v2_seed43"
stall "$OUT"
stall_rc=$?

# --- B3. outputs -------------------------------------------------------------------------------
status "RUNNING — Part B step 3: seed comparison and combined 2-seed tables"
jl "$V2SEED_SCRIPT" --seed-compare --seed=$SEED >> "$OUT/run.log" 2>&1
cmp_rc=$?
jl "$V2SEED_SCRIPT" --combined-2seeds --seed=$SEED >> "$OUT/run.log" 2>&1
comb_rc=$?
log "seed comparison exit $cmp_rc, combined 2-seed exit $comb_rc"

# --- B4. finish ------------------------------------------------------------------------------------
[ $check_rc -eq 0 ] || fail "Part B step 2 (full benchmark check): $(echo "$check_out" | grep CHECK | head -1)"
[ $stall_rc -eq 0 ] || fail "Part B step 2: stalled run(s) kept after retry: $STALLED_LIST (see $OUT/repair_log.txt)"
[ $cmp_rc -eq 0 ] || fail "Part B step 3 (seed comparison): see $OUT/run.log"
[ $comb_rc -eq 0 ] || fail "Part B step 3 (combined 2-seed tables): see $OUT/run.log"
status "DONE — $(date '+%Y-%m-%d %H:%M:%S'): v2 repaired (${V2_REPAIRED:-nothing to repair}), v2 seed $SEED complete and checked; see seed_comparison.md and results/benchmark_large_combined_2seeds"
exit 0
