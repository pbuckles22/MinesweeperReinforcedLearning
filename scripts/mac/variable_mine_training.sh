#!/bin/bash

# Variable Mine Count Training Script for Mac
# Trains agent on variable mine counts for better generalization

set -e  # Exit on any error

echo "🎯 Starting Variable Mine Count Training for Minesweeper RL"
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

# Run the variable mine count training
echo "🚀 Starting variable mine count training..."
echo "   This trains the agent on variable mine counts (1-2, then 1-3, etc.)."
echo "   Should learn generalizable strategies instead of overfitting."
echo ""

python scripts/variable_mine_training.py

echo ""
echo "✅ Variable mine count training completed!"
echo "📊 Check the experiments/ directory for detailed results."
echo "🎯 The agent should show consistent performance across different mine counts." 