#!/bin/bash

# DQN 8x8 Overnight Training Runner
# 7-hour training session for 8x8 Minesweeper

set -e

echo "🌙 DQN 8x8 Overnight Training"
echo "=============================="
echo "⏱️  Expected duration: 6-8 hours"
echo "📋 8x8 board with progressive difficulty"
echo "🚀 Starting at: $(date)"
echo ""

# Check environment
if [ ! -f "src/core/dqn_agent.py" ]; then
    echo "❌ Error: Please run from project root"
    exit 1
fi

# Activate virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
fi

# Create directories
echo "📁 Creating directories..."
mkdir -p training_stats
mkdir -p models

# Start training
echo "🎯 Starting 8x8 curriculum training..."
echo "   This will run for several hours"
echo "   Progress will be saved automatically"
echo ""

python scripts/dqn_8x8_curriculum.py

echo ""
echo "✅ Overnight training completed!"
echo "📊 Check results in: training_stats/"
echo "💾 Models saved in: models/"
echo "🌅 Finished at: $(date)" 