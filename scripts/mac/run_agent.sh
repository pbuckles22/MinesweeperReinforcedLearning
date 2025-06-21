#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH="src:$PYTHONPATH"

# Run the agent visualization/demo script
python src/visualization/visualize_agent.py "$@" 