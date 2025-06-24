#!/bin/bash

# Density-Based Curriculum Training Script for Mac
# Uses mine density and different board sizes for gradual progression

set -e  # Exit on any error

echo "🎓 Starting Density-Based Curriculum Training for Minesweeper RL"
echo "================================================================"

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

# Run the density-based curriculum training
echo "🚀 Starting density-based curriculum training..."
echo "   This uses mine density and different board sizes for gradual progression."
echo "   Should avoid overfitting to specific board configurations."
echo ""

python scripts/curriculum_training_density_based.py

echo ""
echo "✅ Density-based curriculum training completed!"
echo "📊 Check the experiments/ directory for detailed results."
echo "🎯 The agent should show better generalization across board sizes and densities." 