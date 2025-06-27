#!/bin/bash

# Signal handling for graceful shutdown
cleanup() {
    echo ""
    echo "🛑 Received interrupt signal. Cleaning up..."
    if [ ! -z "$TRAINING_PID" ]; then
        echo "   Stopping training process (PID: $TRAINING_PID)..."
        kill $TRAINING_PID 2>/dev/null
        wait $TRAINING_PID 2>/dev/null
        echo "   ✅ Training process stopped"
    fi
    echo "✅ Cleanup completed"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Activate virtual environment
source venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH="src:$PYTHONPATH"

# Run quick training session (30-60 minutes)
echo "🚀 Starting Quick Training Test (100k timesteps, ~30-60 minutes)"
echo "Expected: Progression through Beginner (4x4) and Intermediate (6x6) stages"
echo "💡 Press Ctrl+C to stop training gracefully"
echo ""

# Start training in background and capture PID
python src/core/train_agent.py \
    --total_timesteps 100000 \
    --eval_freq 5000 \
    --n_eval_episodes 50 \
    --verbose 0 &
TRAINING_PID=$!

echo "Training started with PID: $TRAINING_PID"
echo "Waiting for training to complete..."

# Wait for training to complete
wait $TRAINING_PID
TRAINING_EXIT_CODE=$?

echo ""
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully"
else
    echo "⚠️  Training exited with code: $TRAINING_EXIT_CODE"
fi 