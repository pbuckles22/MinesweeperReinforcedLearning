#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH="src:$PYTHONPATH"

# Run medium test suite
pytest tests/unit/core tests/functional tests/integration/core -v --maxfail=10 --disable-warnings 