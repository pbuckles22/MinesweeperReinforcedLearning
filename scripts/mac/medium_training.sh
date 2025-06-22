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

# Run medium training session (2-4 hours)
echo "🚀 Starting Medium Training Test (500k timesteps, ~2-4 hours)"
echo "Expected: Progression through 4-5 curriculum stages (Beginner -> Intermediate -> Easy -> Normal)"
echo "💡 Press Ctrl+C to stop training gracefully"
echo ""

# Start training in background and capture PID
python src/core/train_agent.py \
    --total_timesteps 500000 \
    --eval_freq 10000 \
    --n_eval_episodes 100 \
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