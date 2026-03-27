#!/bin/bash
: '
-------------------------------------------------------------------------------
Script Name: full-grid-search.sh

Description:
    This script automates the execution of a grid search for machine learning
    experiments using the grid_search.py Python script. It supports running
    grid searches across all datasets and models, new datasets, or testing new
    models by configuring the COMMAND and LOG_FILE variables. The script logs
    the command execution details, including timestamps and total execution
    time, to a specified log file.
Usage:
    - Configure the COMMAND and LOG_FILE variables as needed by uncommenting
      the desired command section.
    - Run the script to execute the grid search and log the output.

Sections:
    1. Command definition:
        - Defines the COMMAND to execute and the LOG_FILE for logging.
        - Multiple command templates are provided for different use cases.
    2. Command execution:
        - Logs the command and timestamp to the log file.
        - Executes the grid search command and appends output to the log.
        - Logs the completion timestamp and total execution time.

Notes:
    - Only one COMMAND and LOG_FILE should be active at a time.
    - Output and errors are redirected to the specified log file.
-------------------------------------------------------------------------------
'

# region Command definition
# Command for all datasets and models
COMMAND="python3 grid_search.py \
    -c \
    --experiment_name random_forest \
    --data_source public \
    --n_splits 10 \
    --n_runs 5 \
    --max_workers 50"
LOG_FILE="grid_search_complete.log"

# TODO: Parameterize datasets used
# # Command for new datasets and models
# COMMAND="python3 grid_search.py \
#     -c \
#     --experiment_name bnc_single \
#     --data_source public \
#     --dataset_name iris \
#     --n_splits 2 \
#     --n_runs 1 \
#     --max_workers 25"
# LOG_FILE="grid_search_single.log"
# endregion

# region Command execution
# Print the command and timestamp to the log file
{
    echo "==================== GRID SEARCH EXECUTION ===================="
    echo "Timestamp: $(date)"
    echo "Command executed: $COMMAND"
    echo "=============================================================="
    echo ""
} > $LOG_FILE 2>&1

# Execute the command in the foreground and append to the log
START_TIME=$(date +%s)
$COMMAND >> $LOG_FILE 2>&1
END_TIME=$(date +%s)
EXEC_TIME=$((END_TIME - START_TIME))

{
    echo ""
    echo "==================== GRID SEARCH COMPLETED ===================="
    echo "End Timestamp: $(date)"
    echo "Total Execution Time: ${EXEC_TIME} seconds"
    printf "Total Execution Time (human readable): %02d:%02d:%02d (hh:mm:ss)\n" $((EXEC_TIME/3600)) $(( (EXEC_TIME%3600)/60 )) $((EXEC_TIME%60))
    echo "=============================================================="
} >> $LOG_FILE 2>&1

# endregion

# To run in background use:
# nohup bash full-grid-search.sh > output.log 2>&1 &

# Useful commands to monitor the process:
# ps aux | grep grid_search.py # This will show the process ID (PID) and other details
# kill <pid> # Replace <pid> with the actual process ID from the previous command
# pkill -f grid_search # This will kill all processes that match the name 'grid_search.py'