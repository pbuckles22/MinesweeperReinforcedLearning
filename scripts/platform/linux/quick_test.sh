#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH="src:$PYTHONPATH"

# Run quick subset of tests
pytest tests/unit/core tests/functional/game_flow -v --maxfail=5 --disable-warnings 