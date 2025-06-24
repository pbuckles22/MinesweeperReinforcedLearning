#!/bin/bash

# GPU vs CPU Performance Benchmark for Minesweeper RL
# Determines optimal board size threshold for switching from CPU to GPU
#
# TODO: When ready to test on larger machines with CUDA or more powerful GPUs,
#       re-enable GPU benchmarking and add support for CUDA/MPS as needed.

set -e

echo "🚀 CPU Performance Benchmark"
echo "============================="
echo "This test will benchmark CPU training performance only."
echo "(GPU/MPS benchmarking is currently disabled.)"
echo ""

# Check if we're in the right directory
if [ ! -f "src/core/minesweeper_env.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: Virtual environment not detected"
    echo "   Make sure to activate your virtual environment first:"
    echo "   source venv/bin/activate"
    echo ""
fi

# Create benchmark directory
mkdir -p benchmark_results
timestamp=$(date +"%Y%m%d_%H%M%S")
results_file="benchmark_results/cpu_benchmark_${timestamp}.json"

echo "📊 Benchmark Configuration:"
echo "   - Board sizes: 4x4, 6x6, 8x8, 9x9, 12x12, 16x16"
echo "   - Training steps: 10,000 per test"
echo "   - Device: CPU only"
echo "   - Results file: $results_file"
echo ""

# Initialize results JSON
cat > "$results_file" << EOF
{
    "benchmark_info": {
        "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
        "description": "CPU performance benchmark for Minesweeper RL",
        "training_steps": 10000,
        "board_configurations": []
    },
    "results": []
}
EOF

# Board configurations to test
declare -a board_configs=(
    "4x4:2"    # 4x4 board, 2 mines
    "6x6:4"    # 6x6 board, 4 mines  
    "8x8:8"    # 8x8 board, 8 mines
    "9x9:10"   # 9x9 board, 10 mines
    "12x12:20" # 12x12 board, 20 mines
    "16x16:40" # 16x16 board, 40 mines
)

# Function to run benchmark for a specific configuration
run_benchmark() {
    local board_size=$1
    local mines=$2
    local device="cpu"
    
    echo "🧪 Testing ${board_size} board with ${mines} mines on ${device}..."
    
    # Create temporary benchmark script
    cat > "temp_benchmark.py" << EOF
#!/usr/bin/env python3
import os
import sys
import time
import json
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.core.minesweeper_env import MinesweeperEnv
from src.core.train_agent import ActionMaskingWrapper

# Configuration
board_size = tuple(map(int, "$board_size".split('x')))
mines = $mines
device_name = "$device"

device = torch.device("cpu")
print(f"Using device: {device}")

# Create environment
def make_env():
    return MinesweeperEnv(max_board_size=board_size, max_mines=mines)

env = DummyVecEnv([make_env])
env = ActionMaskingWrapper(env)

# Conservative parameters optimized for benchmarking
params = {
    'learning_rate': 1e-4,
    'n_steps': 1024,
    'batch_size': 32,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.1,
    'ent_coef': 0.005,
    'vf_coef': 0.5,
    'max_grad_norm': 0.3,
    'verbose': 0,
    'device': device
}

# Create model
model = PPO("MlpPolicy", env, **params)

# Benchmark training
print(f"Starting benchmark: {board_size[0]}x{board_size[1]} board, {mines} mines, {device_name}")
start_time = time.time()

model.learn(total_timesteps=10000, progress_bar=False)

end_time = time.time()
training_time = end_time - start_time

# Quick evaluation
wins = 0
total_reward = 0
n_eval = 20

for _ in range(n_eval):
    obs = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward[0]
        
        if info[0].get('won', False):
            wins += 1
    
    total_reward += episode_reward

win_rate = wins / n_eval
avg_reward = total_reward / n_eval

# Calculate performance metrics
board_area = board_size[0] * board_size[1]
steps_per_second = 10000 / training_time

result = {
    "board_size": f"{board_size[0]}x{board_size[1]}",
    "mines": mines,
    "board_area": int(board_area),
    "device": device_name,
    "training_time_seconds": float(training_time),
    "steps_per_second": float(steps_per_second),
    "win_rate": float(win_rate),
    "avg_reward": float(avg_reward),
    "evaluation_episodes": int(n_eval)
}

print(f"Results: {json.dumps(result, indent=2)}")

# Save to file
with open("temp_benchmark_result.json", "w") as f:
    json.dump(result, f, indent=2)

env.close()
EOF

    # Run the benchmark
    python temp_benchmark.py
    
    # Clean up
    rm -f temp_benchmark.py
}

# Function to update results JSON
update_results() {
    local result_file="temp_benchmark_result.json"
    if [ -f "$result_file" ]; then
        # Read the result and append to main results file
        result=$(cat "$result_file")
        
        # Use jq to append to the results array
        if command -v jq &> /dev/null; then
            jq --argjson result "$result" '.results += [$result]' "$results_file" > "${results_file}.tmp"
            mv "${results_file}.tmp" "$results_file"
        else
            # Fallback: simple append (less robust)
            echo ",\n$result" >> "$results_file"
        fi
        
        rm -f "$result_file"
    fi
}

# Run benchmarks
echo "🔄 Starting CPU-only benchmarks..."
echo ""

for config in "${board_configs[@]}"; do
    IFS=':' read -r board_size mines <<< "$config"
    
    echo "📋 Board: ${board_size}, Mines: ${mines}"
    echo "   Area: $(( $(echo $board_size | cut -d'x' -f1) * $(echo $board_size | cut -d'x' -f2) ))"
    echo ""
    
    # Test CPU only
    run_benchmark "$board_size" "$mines"
    update_results
    
    echo ""
done

# Generate analysis
echo "📈 Generating performance analysis..."
python -c "
import json
import sys

# Load results
with open('$results_file', 'r') as f:
    data = json.load(f)

print('\\n🎯 CPU Performance Analysis')
print('=' * 50)

# Group by board size
board_results = {}
for result in data['results']:
    board_size = result['board_size']
    if board_size not in board_results:
        board_results[board_size] = []
    board_results[board_size].append(result)

# Analyze each board size
for board_size in sorted(board_results.keys(), key=lambda x: int(x.split('x')[0]) * int(x.split('x')[1])):
    results = board_results[board_size]
    result = results[0]
    print(f'\\n📊 {board_size} Board:')
    print(f'   CPU: {result[\"steps_per_second\"]:.1f} steps/sec')
    print(f'   Win Rate: {result[\"win_rate\"]:.3f}')

print(f'\\n📄 Detailed results saved to: $results_file')
"

echo ""
echo "✅ CPU-only benchmark completed!"
echo "📄 Results saved to: $results_file"
echo ""
echo "💡 TODO: When ready to test on larger machines with CUDA or more powerful GPUs,"
echo "   re-enable GPU benchmarking and add support for CUDA/MPS as needed."
echo "" 