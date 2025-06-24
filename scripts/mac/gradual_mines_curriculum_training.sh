#!/bin/bash

# Gradual Mine Count Curriculum Training Script for Mac
# Keeps same board size but varies mine count gradually

set -e  # Exit on any error

echo "🎓 Starting Gradual Mine Count Curriculum Training for Minesweeper RL"
echo "===================================================================="

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

# Run the gradual mine count curriculum training
echo "🚀 Starting gradual mine count curriculum training..."
echo "   This keeps the same board size (4x4) but varies mine count gradually."
echo "   Should avoid observation space mismatches and overfitting."
echo ""

python scripts/curriculum_training_gradual_mines.py

echo ""
echo "✅ Gradual mine count curriculum training completed!"
echo "📊 Check the experiments/ directory for detailed results."
echo "🎯 The agent should show better knowledge transfer across mine counts." 