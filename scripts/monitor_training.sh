#!/bin/bash

# Training Progress Monitor
# Run this in a separate terminal while training is running

STATS_FILE="training_stats.txt"
CHECK_INTERVAL=10  # Check every 10 seconds

echo "🔍 Training Progress Monitor"
echo "Monitoring: $STATS_FILE"
echo "Check interval: $CHECK_INTERVAL seconds"
echo "Press Ctrl+C to stop monitoring"
echo ""

# Function to detect GPU and performance
detect_gpu_info() {
    echo "🔧 Device Information:"
    
    # Check for M1 GPU (MPS)
    if python -c "import torch; print('MPS available:', torch.backends.mps.is_available() and torch.backends.mps.is_built())" 2>/dev/null | grep -q "True"; then
        echo "   ✅ Apple M1 GPU (MPS) detected"
        echo "   🚀 Expected: 2-4x faster than CPU"
        
        # Quick M1 performance test
        M1_PERF=$(python -c "
import torch
import time
device = torch.device('mps')
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)
start = time.time()
for _ in range(10):
    z = torch.mm(x, y)
end = time.time()
print(f'{(end-start)/10:.3f}')
" 2>/dev/null)
        
        if [ -n "$M1_PERF" ]; then
            if (( $(echo "$M1_PERF < 0.1" | bc -l) )); then
                echo "   ✅ M1 Performance: Excellent ($M1_PERF s/op)"
            elif (( $(echo "$M1_PERF < 0.2" | bc -l) )); then
                echo "   ✅ M1 Performance: Good ($M1_PERF s/op)"
            else
                echo "   ⚠️  M1 Performance: Slow ($M1_PERF s/op) - Check thermal throttling"
            fi
        fi
        
    # Check for CUDA GPU
    elif python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
        GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
        echo "   ✅ NVIDIA GPU detected: $GPU_NAME"
        echo "   🚀 Expected: Fastest option for NVIDIA GPUs"
        
        # Quick CUDA performance test
        CUDA_PERF=$(python -c "
import torch
import time
device = torch.device('cuda')
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)
start = time.time()
for _ in range(10):
    z = torch.mm(x, y)
end = time.time()
print(f'{(end-start)/10:.3f}')
" 2>/dev/null)
        
        if [ -n "$CUDA_PERF" ]; then
            if (( $(echo "$CUDA_PERF < 0.05" | bc -l) )); then
                echo "   ✅ CUDA Performance: Excellent ($CUDA_PERF s/op)"
            elif (( $(echo "$CUDA_PERF < 0.1" | bc -l) )); then
                echo "   ✅ CUDA Performance: Good ($CUDA_PERF s/op)"
            else
                echo "   ⚠️  CUDA Performance: Slow ($CUDA_PERF s/op) - Check GPU utilization"
            fi
        fi
        
    else
        echo "   ⚠️  No GPU detected - Using CPU"
        echo "   🐌 Expected: Slowest option, suitable for testing only"
        
        # Quick CPU performance test
        CPU_PERF=$(python -c "
import torch
import time
device = torch.device('cpu')
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)
start = time.time()
for _ in range(10):
    z = torch.mm(x, y)
end = time.time()
print(f'{(end-start)/10:.3f}')
" 2>/dev/null)
        
        if [ -n "$CPU_PERF" ]; then
            echo "   ℹ️  CPU Performance: $CPU_PERF s/op (as expected)"
        fi
    fi
    echo ""
}

# Function to check training performance
check_training_performance() {
    if [ -f "$STATS_FILE" ]; then
        # Get last few lines to calculate training speed
        LAST_LINES=$(tail -n 5 "$STATS_FILE" 2>/dev/null)
        if [ -n "$LAST_LINES" ]; then
            # Calculate iterations per second
            FIRST_LINE=$(echo "$LAST_LINES" | head -n 1)
            LAST_LINE=$(echo "$LAST_LINES" | tail -n 1)
            
            IFS=',' read -r timestamp1 iteration1 timesteps1 win_rate1 avg_reward1 avg_length1 stage1 phase1 stage_time1 no_improvement1 <<< "$FIRST_LINE"
            IFS=',' read -r timestamp2 iteration2 timesteps2 win_rate2 avg_reward2 avg_length2 stage2 phase2 stage_time2 no_improvement2 <<< "$LAST_LINE"
            
            if [ "$iteration1" != "$iteration2" ] && [ -n "$iteration1" ] && [ -n "$iteration2" ]; then
                ITER_DIFF=$((iteration2 - iteration1))
                TIMESTEPS_DIFF=$((timesteps2 - timesteps1))
                
                echo "📊 Training Performance:"
                echo "   Iterations: $ITER_DIFF iterations in last 5 measurements"
                echo "   Timesteps: $TIMESTEPS_DIFF timesteps in last 5 measurements"
                
                # Estimate speed based on device
                if python -c "import torch; print('MPS available:', torch.backends.mps.is_available() and torch.backends.mps.is_built())" 2>/dev/null | grep -q "True"; then
                    if [ "$ITER_DIFF" -ge 3 ]; then
                        echo "   ✅ M1 Training Speed: Good (${ITER_DIFF} iterations)"
                    else
                        echo "   ⚠️  M1 Training Speed: Slow (${ITER_DIFF} iterations) - Check thermal throttling"
                    fi
                elif python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
                    if [ "$ITER_DIFF" -ge 4 ]; then
                        echo "   ✅ CUDA Training Speed: Good (${ITER_DIFF} iterations)"
                    else
                        echo "   ⚠️  CUDA Training Speed: Slow (${ITER_DIFF} iterations) - Check GPU utilization"
                    fi
                else
                    if [ "$ITER_DIFF" -ge 1 ]; then
                        echo "   ℹ️  CPU Training Speed: As expected (${ITER_DIFF} iterations)"
                    else
                        echo "   ⚠️  CPU Training Speed: Very slow (${ITER_DIFF} iterations)"
                    fi
                fi
                echo ""
            fi
        fi
    fi
}

# Function to display current status
show_status() {
    if [ -f "$STATS_FILE" ]; then
        echo "=== $(date '+%H:%M:%S') ==="
        
        # Get the last line (most recent stats)
        LAST_LINE=$(tail -n 1 "$STATS_FILE" 2>/dev/null)
        
        if [ -n "$LAST_LINE" ]; then
            # Parse CSV line: timestamp,iteration,timesteps,win_rate,avg_reward,avg_length,stage,phase,stage_time,no_improvement
            IFS=',' read -r timestamp iteration timesteps win_rate avg_reward avg_length stage phase stage_time no_improvement <<< "$LAST_LINE"
            
            echo "📊 Current Status:"
            echo "   Iteration: $iteration"
            echo "   Timesteps: $timesteps"
            echo "   Win Rate: ${win_rate}%"
            echo "   Avg Reward: $avg_reward"
            echo "   Stage: $stage ($phase)"
            echo "   Stage Time: ${stage_time}s"
            echo "   No Improvement: $no_improvement iterations"
            
            # Warning indicators
            if [ "$no_improvement" -gt 20 ]; then
                echo "   ⚠️  WARNING: No improvement for $no_improvement iterations"
            fi
            if [ "$no_improvement" -gt 50 ]; then
                echo "   🚨 CRITICAL: No improvement for $no_improvement iterations - Consider stopping!"
            fi
            if (( $(echo "$win_rate < 5" | bc -l) )) && [ "$iteration" -gt 200 ]; then
                echo "   🚨 CRITICAL: Win rate too low (${win_rate}%) after $iteration iterations"
            fi
            
            echo ""
        else
            echo "Waiting for training to start..."
        fi
    else
        echo "Stats file not found. Waiting for training to start..."
    fi
}

# Function to show recent progress
show_recent() {
    if [ -f "$STATS_FILE" ]; then
        echo "📈 Recent Progress (last 5 iterations):"
        tail -n 5 "$STATS_FILE" | while IFS=',' read -r timestamp iteration timesteps win_rate avg_reward avg_length stage phase stage_time no_improvement; do
            echo "   $timestamp: Iter=$iteration, Win=${win_rate}%, Reward=$avg_reward, Stage=$stage"
        done
        echo ""
    fi
}

# Show GPU info on startup
detect_gpu_info

# Main monitoring loop
while true; do
    clear
    show_status
    show_recent
    check_training_performance
    
    # Wait for next check
    sleep $CHECK_INTERVAL
done 