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

# Run debug training with better early learning parameters
echo "🔧 Debug Training with Enhanced Early Learning"
echo "=============================================="
echo "This uses parameters optimized for early learning and exploration"
echo "💡 Press Ctrl+C to stop training gracefully"
echo ""

# Start training in background and capture PID
python src/core/train_agent.py \
    --total_timesteps 20000 \
    --eval_freq 1000 \
    --n_eval_episodes 50 \
    --learning_rate 0.001 \
    --ent_coef 0.1 \
    --batch_size 64 \
    --n_steps 1024 \
    --n_epochs 8 \
    --verbose 1 \
    --device mps &
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