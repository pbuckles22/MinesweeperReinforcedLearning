#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH="src:$PYTHONPATH"

# Run quick training session (30-60 minutes)
echo "🚀 Starting Quick Training Test (100k timesteps, ~30-60 minutes)"
echo "Expected: Progression through Beginner (4x4) and Intermediate (6x6) stages"
echo ""

python src/core/train_agent.py \
    --total_timesteps 100000 \
    --eval_freq 5000 \
    --n_eval_episodes 50 \
    --verbose 1 