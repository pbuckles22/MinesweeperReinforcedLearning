#!/bin/bash

# Smart Training Wrapper
# Monitors training progress and kills if not improving

STATS_FILE="training_stats.txt"
MAX_NO_IMPROVEMENT=50
MIN_WIN_RATE=5
MIN_ITERATIONS=200
CHECK_INTERVAL=30  # Check every 30 seconds

# Function to check if training should be stopped
check_should_stop() {
    if [ ! -f "$STATS_FILE" ]; then
        return 1  # Don't stop if stats file doesn't exist yet
    fi
    
    # Get the last line
    LAST_LINE=$(tail -n 1 "$STATS_FILE" 2>/dev/null)
    if [ -z "$LAST_LINE" ]; then
        return 1
    fi
    
    # Parse CSV line
    IFS=',' read -r timestamp iteration timesteps win_rate avg_reward avg_length stage phase stage_time no_improvement <<< "$LAST_LINE"
    
    echo "🔍 Progress Check: Iter=$iteration, Win=${win_rate}%, NoImp=$no_improvement"
    
    # Check conditions for stopping
    if [ "$no_improvement" -gt "$MAX_NO_IMPROVEMENT" ]; then
        echo "🚨 STOPPING: No improvement for $no_improvement iterations (max: $MAX_NO_IMPROVEMENT)"
        return 0
    fi
    
    if (( $(echo "$win_rate < $MIN_WIN_RATE" | bc -l) )) && [ "$iteration" -gt "$MIN_ITERATIONS" ]; then
        echo "🚨 STOPPING: Win rate too low (${win_rate}%) after $iteration iterations (min: $MIN_WIN_RATE%)"
        return 0
    fi
    
    return 1  # Don't stop
}

# Function to monitor training in background
monitor_training() {
    local training_pid=$1
    
    echo "🔍 Starting progress monitor (PID: $training_pid)"
    echo "   Max no improvement: $MAX_NO_IMPROVEMENT iterations"
    echo "   Min win rate: $MIN_WIN_RATE%"
    echo "   Check interval: $CHECK_INTERVAL seconds"
    echo ""
    
    while kill -0 "$training_pid" 2>/dev/null; do
        sleep $CHECK_INTERVAL
        
        if check_should_stop; then
            echo "🛑 Stopping training process (PID: $training_pid)"
            kill "$training_pid"
            wait "$training_pid" 2>/dev/null
            echo "✅ Training stopped due to lack of progress"
            exit 1
        fi
    done
    
    echo "✅ Training completed normally"
}

# Main execution
echo "🚀 Smart Training Wrapper"
echo "=========================="

# Check if training command is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <training_command>"
    echo "Example: $0 './scripts/mac/quick_training.sh'"
    exit 1
fi

# Start training in background
echo "Starting training: $*"
"$@" &
TRAINING_PID=$!

echo "Training PID: $TRAINING_PID"
echo ""

# Start monitoring
monitor_training $TRAINING_PID 