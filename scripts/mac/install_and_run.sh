#!/bin/bash
set -e

# Function to check if a command exists
check_command() {
    if command -v "$1" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Check Python version
echo "🔍 Checking Python version..."
if check_command python3; then
    python_cmd="python3"
elif check_command python; then
    python_cmd="python"
else
    echo "❌ Python not found. Please install Python 3.8 or higher."
    exit 1
fi

python_version=$($python_cmd --version 2>&1 | cut -d' ' -f2)
echo "Found Python version: $python_version"

# Check if Python version is supported (3.8 or higher)
if [[ "$(printf '%s\n' "3.8" "$python_version" | sort -V | head -n1)" != "3.8" ]]; then
    echo "❌ Python version $python_version is not supported. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python version $python_version is supported!"

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
from src.core.gym_compatibility import wrap_for_sb3

print('✅ All imports successful!')
print(f'PyTorch version: {torch.__version__}')
print(f'Device available: {torch.device(\"mps\" if torch.backends.mps.is_available() else \"cpu\")}')

# Test compatibility wrapper
test_env = gym.make('CartPole-v1')
wrapped_env = wrap_for_sb3(test_env)
print('✅ Gymnasium compatibility wrapper works!')
test_env.close()
wrapped_env.close()
"

echo "🎉 Setup complete! You can now run training scripts." 