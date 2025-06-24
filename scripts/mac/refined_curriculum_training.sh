#!/bin/bash

# Refined Curriculum Training Script for Mac
# Runs the improved curriculum learning with gradual difficulty progression

set -e  # Exit on any error

echo "🎓 Starting Refined Curriculum Training for Minesweeper RL"
echo "=========================================================="

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

# Run the refined curriculum training
echo "🚀 Starting refined curriculum training..."
echo "   This will run multiple stages with gradual difficulty increases."
echo "   Each stage builds on the previous one for better knowledge transfer."
echo ""

python scripts/curriculum_training_refined.py

echo ""
echo "✅ Refined curriculum training completed!"
echo "📊 Check the experiments/ directory for detailed results."
echo "🎯 The agent should show better generalization across difficulty levels." 