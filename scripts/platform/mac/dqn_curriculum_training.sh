#!/bin/bash

# DQN Curriculum Training Runner for Mac
# Progressive training from easy to hard configurations

set -e  # Exit on any error

echo "🚀 DQN Curriculum Training - Mac Runner"
echo "========================================"

# Check if we're in the right directory
if [ ! -f "src/core/dqn_agent.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: Virtual environment not detected"
    echo "   Activating virtual environment..."
    source venv/bin/activate
fi

# Check if required packages are installed
echo "🔍 Checking dependencies..."
python -c "import torch, numpy, gymnasium" 2>/dev/null || {
    echo "❌ Missing required packages. Installing..."
    pip install torch numpy gymnasium pygame
}

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p training_stats/dqn_curriculum
mkdir -p models

# Run DQN curriculum training
echo "🎯 Starting DQN curriculum training..."
echo "   This will train progressively from 1 mine to 5 mines"
echo "   Each stage builds on the previous one"
echo ""

python scripts/dqn_curriculum_training.py

echo ""
echo "✅ DQN curriculum training completed!"
echo "📊 Check results in: training_stats/dqn_curriculum/"
echo "💾 Models saved in: models/" 