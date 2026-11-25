#!/bin/bash
: '
-------------------------------------------------------------------------------
Script Name: run-inference-experiments.sh

Description:
    Runs the experiments of the article on Rao-Blackwellized inference in
    semiparametric Bayesian networks (inference_experiments.py) and their
    analyses (inference_analysis.py), with one log per experiment and stage.

    1. Reference stage (untimed) of all the experiments IN PARALLEL: builds the
       networks, tasks and reference posteriors. Each experiment uses
       REFERENCE_WORKERS processes, which share the GPUs.
    2. Timed stage of each experiment ONE AFTER ANOTHER, once all the
       references are ready: each one already uses one worker per GPU, pinned
       to its own CPU core, so running two timed stages at once, or a timed
       stage next to reference workers, would distort the runtimes.
    3. Analysis of each experiment in the background, right after its timed
       stage (CPU only, kept off the cores of the timed workers), while the
       next timed stage runs.

Usage (from anywhere):
    bash src/run-inference-experiments.sh [--quick] [EXPERIMENT ...]

    EXPERIMENT     Any of E1 E2 E3 E4 E5 E7 (default: all, in this order).
    --quick        Smoke run with tiny sizes (results in data/results/inference/quick).

    Environment variables:
    REFERENCE_WORKERS  Workers of the reference stage per experiment (default 16).
    EXTRA_ARGS         Extra arguments for inference_experiments.py of every
                       experiment, e.g. EXTRA_ARGS="--n_reps 10".

    Examples:
    bash src/run-inference-experiments.sh                 # everything
    bash src/run-inference-experiments.sh E1 E2           # only E1 and E2
    bash src/run-inference-experiments.sh --quick E2      # smoke run of E2
    nohup bash src/run-inference-experiments.sh > /dev/null 2>&1 &   # in the background

Logs:
    logs/inference/<start time>/
        main.log                  progress and timings of every stage
        <EXP>_references.log      reference stage of each experiment
        <EXP>_runs.log            timed stage of each experiment
        <EXP>_analysis.log        analysis of each experiment

Notes:
    - The runs resume: launching the script again skips the finished units.
    - A unit that fails is reported in its log ("FAILED") and in main.log; the
      other units go on.
-------------------------------------------------------------------------------
'

# region Configuration
set -u
ALL_EXPERIMENTS=(E1 E2 E4 E7 E3 E5)
REFERENCE_WORKERS="${REFERENCE_WORKERS:-16}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

QUICK_FLAG=""
EXPERIMENTS=()
for argument in "$@"; do
    case "$argument" in
        --quick) QUICK_FLAG="--quick" ;;
        E1|E2|E3|E4|E5|E7) EXPERIMENTS+=("$argument") ;;
        -h|--help) sed -n '2,52p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $argument (expected --quick or E1 E2 E3 E4 E5 E7)"; exit 1 ;;
    esac
done
if [ ${#EXPERIMENTS[@]} -eq 0 ]; then
    EXPERIMENTS=("${ALL_EXPERIMENTS[@]}")
fi

# The Python modules are run from the library root (the parent of this script's folder)
LIBRARY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LIBRARY_ROOT" || exit 1
LOG_DIR="$LIBRARY_ROOT/logs/inference/$(date +%Y-%m-%d_%H-%M-%S)${QUICK_FLAG:+_quick}"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/main.log"
# endregion

# region Helpers
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MAIN_LOG"
}

human_time() {
    printf "%02d:%02d:%02d" $(($1 / 3600)) $((($1 % 3600) / 60)) $(($1 % 60))
}

# run_stage EXPERIMENT STAGE: runs one stage of an experiment with its own log, and reports its time and failed units
run_stage() {
    local experiment="$1" stage="$2"
    local stage_log="$LOG_DIR/${experiment}_${stage}.log"
    local command="python -W ignore -m src.inference_experiments $experiment --stage $stage $QUICK_FLAG $EXTRA_ARGS"
    if [ "$stage" = "references" ]; then
        command="$command --reference_workers $REFERENCE_WORKERS"
    fi
    {
        echo "Command: $command"
        echo "Start: $(date)"
        echo ""
    } > "$stage_log"
    local start=$(date +%s)
    $command >> "$stage_log" 2>&1
    local status=$?
    local elapsed=$(( $(date +%s) - start ))
    local failed
    failed=$(grep -c "FAILED" "$stage_log")
    {
        echo ""
        echo "End: $(date) (exit code $status, $(human_time $elapsed))"
    } >> "$stage_log"
    log "$experiment $stage finished in $(human_time $elapsed) (exit code $status, $failed failed units)"
    return $status
}

# The timed workers are pinned to the first cores (one per GPU), so the background analyses use the others
N_CORES=$(nproc)
ANALYSIS_CORES=""
if command -v taskset > /dev/null && [ "$N_CORES" -gt 16 ]; then
    ANALYSIS_CORES="taskset -c 16-$((N_CORES - 1))"
fi

# run_analysis EXPERIMENT: builds the figures and tables of an experiment
run_analysis() {
    local experiment="$1"
    local analysis_log="$LOG_DIR/${experiment}_analysis.log"
    local start=$(date +%s)
    $ANALYSIS_CORES python -W ignore -m src.inference_analysis "$experiment" $QUICK_FLAG > "$analysis_log" 2>&1
    local status=$?
    log "$experiment analysis finished in $(human_time $(( $(date +%s) - start ))) (exit code $status)"
}
# endregion

# region Execution
TOTAL_START=$(date +%s)
log "Experiments: ${EXPERIMENTS[*]} ${QUICK_FLAG:+(quick)}"
log "Logs: $LOG_DIR"
log "GPUs: $(nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null | tr '\n' ';')"

# 1. Reference stages of all the experiments in parallel (untimed)
log "Reference stage of ${EXPERIMENTS[*]} in parallel ($REFERENCE_WORKERS workers each)"
declare -A REFERENCE_PIDS
for experiment in "${EXPERIMENTS[@]}"; do
    run_stage "$experiment" references &
    REFERENCE_PIDS[$experiment]=$!
done

# 2. Timed stages one after another, once ALL the reference stages have finished (reference workers running next to a
#    timed stage would compete for its GPUs and cores)
declare -A REFERENCE_STATUS
for experiment in "${EXPERIMENTS[@]}"; do
    wait "${REFERENCE_PIDS[$experiment]}"
    REFERENCE_STATUS[$experiment]=$?
done
log "Reference stage of all the experiments finished in $(human_time $(( $(date +%s) - TOTAL_START )))"

# 3. Each analysis runs in the background after its timed stage
ANALYSIS_PIDS=()
for experiment in "${EXPERIMENTS[@]}"; do
    if [ "${REFERENCE_STATUS[$experiment]}" -ne 0 ]; then
        log "$experiment: the reference stage failed, skipping its timed stage (see ${experiment}_references.log)"
        continue
    fi
    log "$experiment timed stage started"
    run_stage "$experiment" runs
    run_analysis "$experiment" &
    ANALYSIS_PIDS+=($!)
done
for pid in "${ANALYSIS_PIDS[@]}"; do
    wait "$pid"
done

log "All done in $(human_time $(( $(date +%s) - TOTAL_START )))"
# endregion

# NOTE: Useful commands to monitor the experiments:
# tail -f logs/inference/<start time>/main.log        # progress of the stages
# tail -f logs/inference/<start time>/E2_runs.log     # progress of the units of a stage
# nvidia-smi                                          # GPU use
# pkill -f run-inference-experiments.sh; pkill -f src.inference_experiments   # stop everything (resumable)
