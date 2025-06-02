#!/bin/bash
set -e

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run the training script
echo "Running training script..."
python train_agent.py

# Run the test suite
echo "Running tests..."
python -m unittest test_train_agent.py -v 