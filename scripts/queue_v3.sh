#!/bin/bash
# =============================================================================
# queue_v3.sh — unattended, restartable chain for the v3 benchmark (ta21, yn1, ta42, ta62, swv13):
#   0. wait until results/benchmark_large_v2_seed43 is DONE/FAILED and no benchmark process runs
#   1. smoke test (ta42, 60 s, 7 methods, seed 42) → results/benchmark_large_v3/smoke
#   2. full seed 42 → results/benchmark_large_v3            (35 runs) → STATUS.txt
#   3. full seed 43 → results/benchmark_large_v3_seed43     (35 runs) → seed comparison → STATUS.txt
#   4. combined evaluation → results/benchmark_large_combined_13x2
# Restartable: rerunning it skips every completed step (smoke marker, STATUS DONE) and every
# completed run (DONE folders); an interrupted run is redone from scratch.
# Start:
#   nohup caffeinate -ims bash scripts/queue_v3.sh > results/benchmark_large_v3/queue.log 2>&1 &
# =============================================================================
set -u

ROOT="/Users/larshansmeier/Repos/OR Lab B"
JULIA="/Users/larshansmeier/.julia/juliaup/julia-1.13.4+0.aarch64.apple.darwin14/Julia-1.13.app/Contents/Resources/julia/bin/julia"
PY="$ROOT/.venv-cpsat/bin/python"
DRV="$ROOT/scripts/benchmark_large_v3.jl"
R="$ROOT/results"
V3="$R/benchmark_large_v3"; V3S="$R/benchmark_large_v3_seed43"; V2S="$R/benchmark_large_v2_seed43"
EXPECTED=35                                    # 5 instances × 7 methods

log()     { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*"; }
status()  { echo "$2" > "$1/STATUS.txt"; log "STATUS $(basename "$1"): $2"; }
jl()      { "$JULIA" --project="$ROOT" "$@"; }
is_done() { grep -q "^DONE" "$1/STATUS.txt" 2>/dev/null; }

cd "$ROOT" || { echo "cannot cd to $ROOT"; exit 1; }
mkdir -p "$V3" "$V3S"
log "queue_v3 started (pid $$); julia: $JULIA"

done_count() {
    local n=0 d
    for d in "$1"/runs/*/; do
        d=${d%/}
        case "$d" in *.tmp|*.FAILED) continue ;; esac
        [ -f "$d/DONE" ] && n=$((n + 1))
    done
    echo "$n"
}

# run the restartable driver; once more if it errors or leaves runs incomplete (it skips DONE runs)
run_until_complete() {
    local folder=$1 expected=$2 logfile=$3; shift 3
    local attempt rc n pid
    for attempt in 1 2; do
        echo "== started by queue_v3.sh $(date '+%Y-%m-%dT%H:%M:%S') (attempt $attempt)" >> "$logfile"
        "$@" >> "$logfile" 2>&1 &
        pid=$!
        echo "$pid" > "$folder/run.pid"
        wait "$pid"; rc=$?
        n=$(done_count "$folder")
        log "driver for $(basename "$folder") exited with code $rc; $n/$expected runs DONE"
        [ "$rc" -eq 0 ] && [ "$n" -ge "$expected" ] && return 0
    done
    return 1
}

# --- 0. wait ------------------------------------------------------------------------------------
others() {
    pgrep -f "queue_v2_repair_seed43|scripts/benchmark_large|scripts/benchmark_seed|scripts/benchmark_cpsat|cpsat_jssp.py" |
        grep -v "^$$\$" | tr '\n' ' '
}
is_done "$V3" || status "$V3" "WAITING for benchmark_large_v2_seed43 to finish"
idle=0
while :; do
    st=$(cat "$V2S/STATUS.txt" 2>/dev/null || echo "missing")
    p=$(others)
    if [ -z "$p" ]; then
        case "$st" in
            DONE*|FAILED*) break ;;
        esac
        # v2_seed43 neither finished nor running (e.g. after a shutdown): resume its own queue
        idle=$((idle + 1))
        if [ $idle -ge 3 ]; then
            log "benchmark_large_v2_seed43 is not running and not finished (STATUS: $st) → resuming queue_v2_repair_seed43.sh"
            echo "== resumed by queue_v3.sh $(date '+%Y-%m-%d %H:%M:%S')" >> "$V2S/queue.log"
            caffeinate -ims bash "$ROOT/scripts/queue_v2_repair_seed43.sh" >> "$V2S/queue.log" 2>&1
            log "queue_v2_repair_seed43.sh exited with code $?"
            idle=0
            continue
        fi
    else
        idle=0
    fi
    log "waiting 60 s (v2_seed43 STATUS: $st; running: ${p:-none})"
    sleep 60
done
st=$(cat "$V2S/STATUS.txt")
case "$st" in FAILED*) log "NOTE: benchmark_large_v2_seed43 ended with FAILED ($st) — continuing with v3 as instructed" ;;
               *) log "benchmark_large_v2_seed43 finished: $st" ;; esac

# --- 1. smoke test ----------------------------------------------------------------------------
if [ -f "$V3/smoke/SMOKE_OK" ]; then
    log "smoke test already passed earlier — skipped"
else
    status "$V3" "RUNNING — step 1: smoke test (ta42, 60 s per method, seed 42)"
    mkdir -p "$V3/smoke"
    jl "$DRV" --seed=42 --time=60 --instances=ta42 --out=benchmark_large_v3/smoke >> "$V3/smoke/run.log" 2>&1
    log "smoke driver exited with code $?"
    out=$(jl "$DRV" --check --seed=42 --time=60 --instances=ta42 --out=benchmark_large_v3/smoke 2>&1); rc=$?
    log "smoke check: $out"
    sout=$("$PY" "$ROOT/scripts/stall_check.py" "$V3/smoke" 2>&1); src=$?
    echo "$sout" | sed 's/^/        /'
    if [ $rc -ne 0 ] || [ $src -ne 0 ]; then
        status "$V3" "FAILED — step 1 (smoke test): $(echo "$out" | grep -E 'CHECK|ERROR' | head -2) $( [ $src -ne 0 ] && echo "; stall check: $(echo "$sout" | tail -1)")"
        status "$V3S" "FAILED — not started (smoke test of benchmark_large_v3 failed)"
        exit 1
    fi
    date '+%Y-%m-%d %H:%M:%S' > "$V3/smoke/SMOKE_OK"
fi

# --- 2./3. full benchmarks ------------------------------------------------------------------------
full() {   # full <folder> <seed>
    local F=$1 S=$2 out rc sout src
    if is_done "$F"; then log "$(basename "$F") already DONE — skipped"; return 0; fi
    status "$F" "RUNNING — full benchmark, seed $S ($EXPECTED runs); see $F/progress.csv"
    run_until_complete "$F" $EXPECTED "$F/run.log" jl "$DRV" --seed=$S ||
        log "WARNING: $(basename "$F") driver did not complete all runs ($(done_count "$F")/$EXPECTED)"
    jl "$DRV" --aggregate --seed=$S >> "$F/run.log" 2>&1 || log "WARNING: aggregation of $(basename "$F") exited with an error"
    out=$(jl "$DRV" --check --seed=$S 2>&1); rc=$?
    log "check $(basename "$F"): $out"
    sout=$("$PY" "$ROOT/scripts/stall_check.py" "$F" 2>&1); src=$?
    echo "$sout" | sed 's/^/        /'
    if [ "$S" != "42" ]; then
        jl "$DRV" --seed-compare --seed=$S >> "$F/run.log" 2>&1 || { rc=1; out="$out; seed comparison failed (see run.log)"; }
    fi
    if [ $rc -eq 0 ] && [ $src -eq 0 ]; then
        status "$F" "DONE — $(date '+%Y-%m-%d %H:%M:%S'): $EXPECTED runs complete, checked, no stalled runs"
    else
        status "$F" "FAILED — full benchmark seed $S: $(echo "$out" | grep -E 'CHECK|failed' | head -2) $( [ $src -ne 0 ] && echo "; stalled (kept after retry): $(echo "$sout" | tail -1)")"
    fi
}
full "$V3" 42
is_done "$V3S" || status "$V3S" "QUEUED — waits for benchmark_large_v3 (seed 42)"
full "$V3S" 43

# --- 4. combined evaluation over 13 instances × 2 seeds ---------------------------------------------
log "combined evaluation → results/benchmark_large_combined_13x2"
jl "$DRV" --combined-13x2 >> "$V3/run.log" 2>&1
log "combined evaluation exited with code $?"
log "queue_v3 finished: v3 $(head -c 60 "$V3/STATUS.txt"); v3_seed43 $(head -c 60 "$V3S/STATUS.txt")"
exit 0
