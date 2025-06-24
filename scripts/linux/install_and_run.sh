#!/bin/bash
set -e

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if Python is installed
if ! command_exists python; then
    echo "Python is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if (( $(echo "$python_version < 3.8" | bc -l) )); then
    echo "Python version $python_version is not supported. Please install Python 3.8 or higher."
    exit 1
fi

# Remove old venv if exists
if [ -d "venv" ]; then
  echo "Removing old venv..."
  rm -rf venv
fi

# Create new venv
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Check gymnasium version
GYMNASIUM_VERSION=$(python -c 'import gymnasium; print(gymnasium.__version__)')
echo "Installed gymnasium version: $GYMNASIUM_VERSION"

# Check torch and stable-baselines3
python -c 'import torch; import stable_baselines3; print("Torch and SB3 installed successfully.")'

# Add src directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

echo "✅ Environment setup complete. Activate with: source venv/bin/activate"

# Run a quick test to verify everything works
echo "🧪 Running quick test..."
python -c "
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

print('✅ All imports successful!')
print(f'PyTorch version: {torch.__version__}')
print(f'Device available: {torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")}')
"

echo "🎉 Setup complete! You can now run training scripts."

# Create logs directory
mkdir -p logs

# Run environment tests
echo "Running environment tests..."
python -m pytest tests/integration/test_environment.py -v

# Run tests
echo "Running tests..."
python -m pytest tests/unit/core tests/unit/agent tests/integration tests/functional tests/scripts -v

# Ask user if they want to run RL training test
echo ""
echo "============================================================"
echo "All tests passed! 🎉"
echo "============================================================"
echo ""
echo "Would you like to run a quick RL training test to verify early learning works?"
echo "This will run a short training session (10,000 timesteps) to test the fixes."
echo ""
read -p "Run RL test? (y/n): " run_rl_test

if [[ $run_rl_test =~ ^[Yy]$ ]] || [[ $run_rl_test =~ ^[Yy][Ee][Ss]$ ]]; then
    echo ""
    echo "Starting RL training test..."
    echo "This will run for ~1-2 minutes to verify early learning works correctly."
    echo ""
    
    # Run a short training test
    python src/core/train_agent.py \
        --total_timesteps 10000 \
        --eval_freq 2000 \
        --n_eval_episodes 20 \
        --learning_rate 0.0003 \
        --verbose 0
    
    echo ""
    echo "============================================================"
    echo "RL training test completed!"
    echo "Check the output above to verify early learning is working."
    echo "============================================================"
else
    echo ""
    echo "Skipping RL training test."
    echo ""
    echo "To run the RL training test later:"
    echo "1. Activate the virtual environment:"
    echo "   source venv/bin/activate"
    echo ""
    echo "2. Run a quick test (10k timesteps, ~1-2 minutes):"
    echo "   python src/core/train_agent.py --total_timesteps 10000 --eval_freq 2000 --n_eval_episodes 20 --verbose 0"
    echo ""
    echo "3. Or run a longer test (50k timesteps, ~5-10 minutes):"
    echo "   python src/core/train_agent.py --total_timesteps 50000 --eval_freq 5000 --n_eval_episodes 50 --verbose 0"
    echo ""
    echo "4. Or run the full training (1M timesteps, ~1-2 hours):"
    echo "   python src/core/train_agent.py --total_timesteps 1000000 --eval_freq 10000 --n_eval_episodes 100 --verbose 0"
    echo ""
    echo "The training will show progress including win rates and learning phases."
fi

echo ""
echo "============================================================"
echo "Installation and setup completed! 🎉"
echo "============================================================"
echo ""
echo "📝 Important: The virtual environment will be deactivated when this script ends."
echo "To use the project, you need to reactivate it:"
echo ""
echo "   source venv/bin/activate"
echo ""
echo "You should then see (venv) in your prompt, indicating the virtual environment is active."
echo ""

# Deactivate virtual environment
deactivate
echo "Virtual environment deactivated." 