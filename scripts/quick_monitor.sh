#!/bin/bash

# Quick Training Monitor - One-liner style
# Usage: ./scripts/quick_monitor.sh

echo "🔍 Quick Training Monitor"
echo "Press Ctrl+C to stop"
echo ""

# Show GPU info on startup
echo "🔧 Device:"
if python -c "import torch; print('MPS available:', torch.backends.mps.is_available() and torch.backends.mps.is_built())" 2>/dev/null | grep -q "True"; then
    echo "   ✅ M1 GPU (MPS) - Expected: 2-4x faster than CPU"
elif python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    echo "   ✅ NVIDIA GPU: $GPU_NAME - Expected: Fastest option"
else
    echo "   ⚠️  CPU only - Expected: Slowest option"
fi
echo ""

# Show the last line of stats file with tail -f
tail -f training_stats.txt | while read line; do
    # Skip header line
    if [[ $line == timestamp* ]]; then
        continue
    fi
    
    # Parse CSV and show formatted output
    IFS=',' read -r timestamp iteration timesteps win_rate avg_reward avg_length stage phase stage_time no_improvement <<< "$line"
    
    echo "[$timestamp] Iter:$iteration Win:${win_rate}% Reward:$avg_reward Stage:$stage ($phase) NoImp:$no_improvement"
    
    # Show warnings
    if [ "$no_improvement" -gt 20 ]; then
        echo "   ⚠️  WARNING: No improvement for $no_improvement iterations"
    fi
    if [ "$no_improvement" -gt 50 ]; then
        echo "   🚨 CRITICAL: Consider stopping training!"
    fi
    
    # Show performance hints based on device
    if [ "$iteration" -gt 100 ]; then
        if python -c "import torch; print('MPS available:', torch.backends.mps.is_available() and torch.backends.mps.is_built())" 2>/dev/null | grep -q "True"; then
            if (( $(echo "$win_rate < 5" | bc -l) )); then
                echo "   💡 M1 Hint: Low win rate - check hyperparameters or thermal throttling"
            fi
        elif python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
            if (( $(echo "$win_rate < 5" | bc -l) )); then
                echo "   💡 CUDA Hint: Low win rate - check GPU utilization or hyperparameters"
            fi
        else
            if (( $(echo "$win_rate < 5" | bc -l) )); then
                echo "   💡 CPU Hint: Low win rate - consider using GPU for faster training"
            fi
        fi
    fi
done 