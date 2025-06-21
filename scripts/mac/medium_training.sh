#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH="src:$PYTHONPATH"

# Run medium training session (2-4 hours)
echo "🚀 Starting Medium Training Test (500k timesteps, ~2-4 hours)"
echo "Expected: Progression through 4-5 curriculum stages (Beginner -> Intermediate -> Easy -> Normal)"
echo ""

python src/core/train_agent.py \
    --total_timesteps 500000 \
    --eval_freq 10000 \
    --n_eval_episodes 100 \
    --verbose 1 