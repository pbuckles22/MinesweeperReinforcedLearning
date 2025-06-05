#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if Python is installed
if ! command_exists python; then
    echo "Python is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if (( $(echo "$python_version < 3.8" | bc -l) )); then
    echo "Python version $python_version is not supported. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Add src directory to Python path
export PYTHONPATH="src:$PYTHONPATH"

# Create logs directory
mkdir -p logs

# Run environment tests
echo "Running environment tests..."
python tests/test_environment.py

# Run training script
echo "Starting training..."
python src/core/train_agent.py \
    --board-size 8 \
    --max-mines 12 \
    --timesteps 1000000 \
    --learning-rate 0.0001 \
    --batch-size 64 \
    --n-steps 2048 \
    --n-epochs 10 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --clip-range 0.2 \
    --ent-coef 0.01 \
    --vf-coef 0.5 \
    --max-grad-norm 0.5 \
    --debug-level 2

# Deactivate virtual environment
deactivate 