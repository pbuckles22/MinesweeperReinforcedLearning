#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH="src:$PYTHONPATH"

# Run full training session
python src/core/train_agent.py --total_timesteps 1000000 --eval_freq 10000 --n_eval_episodes 100 --learning_rate 0.0003 --verbose 1 