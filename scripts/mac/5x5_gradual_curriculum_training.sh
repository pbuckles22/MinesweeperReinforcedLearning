#!/bin/bash

# 5x5 Gradual Curriculum Training Script for Mac
# Uses 5x5 board with very gradual mine count increases

set -e  # Exit on any error

echo "🎓 Starting 5x5 Gradual Curriculum Training for Minesweeper RL"
echo "=============================================================="

# Check if we're in the right directory
if [ ! -f "src/core/minesweeper_env.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run install_and_run.sh first."
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if required packages are installed
echo "📦 Checking dependencies..."
python -c "import stable_baselines3, gymnasium, numpy" 2>/dev/null || {
    echo "❌ Missing dependencies. Please run install_and_run.sh first."
    exit 1
}

# Create experiments directory if it doesn't exist
mkdir -p experiments

# Run the 5x5 gradual curriculum training
echo "🚀 Starting 5x5 gradual curriculum training..."
echo "   This uses 5x5 board with very gradual mine count increases."
echo "   Should avoid dramatic difficulty spikes and improve knowledge transfer."
echo ""

python scripts/curriculum_training_5x5_gradual.py

echo ""
echo "✅ 5x5 gradual curriculum training completed!"
echo "📊 Check the experiments/ directory for detailed results."
echo "🎯 The agent should show better knowledge transfer across mine counts." 