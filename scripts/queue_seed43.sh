#!/bin/bash
# =============================================================================
# queue_seed43.sh — unattended chain: wait for benchmark_large_v2, then the seed-43 repeat of
# benchmark_large_v1 (smoke test on ta41 first). Start with
#   nohup caffeinate -ims bash scripts/queue_seed43.sh > results/benchmark_large_v1_seed43/queue.log 2>&1 &
# Status: results/benchmark_large_v1_seed43/STATUS.txt (WAITING/RUNNING …, finally DONE or FAILED).
# =============================================================================
set -u

ROOT="/Users/larshansmeier/Repos/OR Lab B"
JULIA="/Users/larshansmeier/.julia/juliaup/julia-1.13.4+0.aarch64.apple.darwin14/Julia-1.13.app/Contents/Resources/julia/bin/julia"
SEED=43
OUT="$ROOT/results/benchmark_large_v1_seed43"
V2="$ROOT/results/benchmark_large_v2"
V2_EXPECTED=28                      # 4 instances × 7 methods
V2_WAIT_PID=7204                    # driver of the running v2 benchmark (checked at queue start, 2026-11-24)
SEED_SCRIPT="$ROOT/scripts/benchmark_seed.jl"

log()    { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*"; }
status() { echo "$*" > "$OUT/STATUS.txt"; log "STATUS: $*"; }
fail()   { status "FAILED — $*"; exit 1; }

cd "$ROOT" || { echo "cannot cd to $ROOT"; exit 1; }
mkdir -p "$OUT"
log "queue started (pid $$), julia: $JULIA"

# --- step 1: wait for benchmark_large_v2 -------------------------------------------------
is_v2_driver() { ps -p "$1" -o command= 2>/dev/null | grep -q "benchmark_large_v2.jl"; }
wait_for() {
    local pid=$1
    log "waiting for v2 driver pid $pid (poll every 60 s)"
    while is_v2_driver "$pid"; do sleep 60; done
    log "v2 driver pid $pid has ended"
}
v2_done_count() {
    local n=0 d
    for d in "$V2"/runs/*/; do
        d=${d%/}
        case "$d" in *.tmp|*.FAILED) continue ;; esac
        [ -f "$d/DONE" ] && n=$((n + 1))
    done
    echo "$n"
}

status "WAITING for benchmark_large_v2 (pid $V2_WAIT_PID)"
if is_v2_driver "$V2_WAIT_PID"; then
    wait_for "$V2_WAIT_PID"
else
    other=$(pgrep -f "scripts/benchmark_large_v2.jl" | head -1)
    if [ -n "$other" ]; then
        log "pid $V2_WAIT_PID is not the v2 driver; found running v2 driver pid $other instead"
        wait_for "$other"
    else
        log "no running v2 driver found"
    fi
fi

for resume in 1 2; do
    n=$(v2_done_count)
    log "benchmark_large_v2: $n / $V2_EXPECTED runs with DONE"
    [ "$n" -ge "$V2_EXPECTED" ] && break
    log "benchmark_large_v2 is INCOMPLETE → resuming it with its own restart mechanism (attempt $resume)"
    status "RUNNING — resuming benchmark_large_v2 ($n/$V2_EXPECTED done, attempt $resume)"
    echo "== resumed by queue_seed43.sh $(date '+%Y-%m-%dT%H:%M:%S')" >> "$V2/run.log"
    "$JULIA" --project="$ROOT" "$ROOT/scripts/benchmark_large_v2.jl" >> "$V2/run.log" 2>&1 &
    rpid=$!
    echo "$rpid" > "$V2/run.pid"
    wait "$rpid"
    log "v2 resume (pid $rpid) exited with code $?"
done
n=$(v2_done_count)
[ "$n" -ge "$V2_EXPECTED" ] || fail "step 1: benchmark_large_v2 still incomplete after 2 resumes ($n/$V2_EXPECTED DONE); seed 43 not started"

# final aggregation of v2 (normally already done by its driver): run it if runs.csv does not
# list all runs, or runs.csv / the combined summary are older than the newest DONE
newest_done=$(ls -t "$V2"/runs/*/DONE | head -1)
rows=$(( $(wc -l < "$V2/runs.csv" 2>/dev/null || echo 0) - 1 ))
COMB="$ROOT/results/benchmark_large_combined/summary.csv"
if [ "$rows" -ne "$V2_EXPECTED" ] || [ ! -f "$COMB" ] || [ "$newest_done" -nt "$V2/runs.csv" ] || [ "$newest_done" -nt "$COMB" ]; then
    log "v2 aggregation missing or outdated (runs.csv rows: $rows) → running --aggregate"
    "$JULIA" --project="$ROOT" "$ROOT/scripts/benchmark_large_v2.jl" --aggregate >> "$V2/run.log" 2>&1 ||
        log "WARNING: v2 aggregation exited with an error (see $V2/run.log); continuing with seed 43"
else
    log "v2 aggregation already complete (runs.csv has $rows rows, combined summary up to date)"
fi

# --- step 2: smoke test (ta41, 60 s, all five methods, seed 43) -------------------------
status "RUNNING — step 2: smoke test (ta41, 60 s per method, seed $SEED)"
log "smoke test → $OUT/smoke"
mkdir -p "$OUT/smoke"
"$JULIA" --project="$ROOT" "$SEED_SCRIPT" --seed=$SEED --time=60 --instances=ta41 \
    --out=benchmark_large_v1_seed43/smoke >> "$OUT/smoke/run.log" 2>&1
log "smoke driver exited with code $?"
check_out=$("$JULIA" --project="$ROOT" "$SEED_SCRIPT" --check --seed=$SEED --time=60 --instances=ta41 \
    --out=benchmark_large_v1_seed43/smoke 2>&1)
rc=$?
log "smoke check: $check_out"
[ $rc -eq 0 ] || fail "step 2 (smoke test): $(echo "$check_out" | grep -E 'CHECK|ERROR' | head -3)"

# --- step 3/4: full seed-43 benchmark (driver aggregates at the end) ----------------------
status "RUNNING — step 3: full seed-$SEED benchmark (20 runs × 20 min); see $OUT/progress.csv"
for attempt in 1 2; do                    # second attempt only if the driver itself crashed
    log "full benchmark, driver attempt $attempt"
    "$JULIA" --project="$ROOT" "$SEED_SCRIPT" --seed=$SEED >> "$OUT/run.log" 2>&1
    rc=$?
    log "driver exited with code $rc"
    [ $rc -eq 0 ] && break
done
log "aggregating (step 4)"
"$JULIA" --project="$ROOT" "$SEED_SCRIPT" --aggregate --seed=$SEED >> "$OUT/run.log" 2>&1 ||
    log "WARNING: aggregation exited with an error"
check_out=$("$JULIA" --project="$ROOT" "$SEED_SCRIPT" --check --seed=$SEED 2>&1)
check_rc=$?
log "full check: $check_out"

# --- step 5: seed comparison ----------------------------------------------------------------
status "RUNNING — step 5: seed comparison"
"$JULIA" --project="$ROOT" "$SEED_SCRIPT" --seed-compare --seed=$SEED >> "$OUT/run.log" 2>&1
cmp_rc=$?
log "seed comparison exited with code $cmp_rc"

# --- step 6: finish ----------------------------------------------------------------------------
[ $check_rc -eq 0 ] || fail "step 3/4 (full benchmark check): $(echo "$check_out" | grep CHECK | head -1)"
[ $cmp_rc -eq 0 ] || fail "step 5 (seed comparison): see $OUT/run.log"
status "DONE — $(date '+%Y-%m-%d %H:%M:%S'): 20 seed-$SEED runs complete and checked; see seed_comparison.md"
exit 0
