#!/bin/bash

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

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Fix permissions for virtual environment executables
echo "Fixing virtual environment permissions..."
chmod +x venv/bin/activate venv/bin/activate.csh venv/bin/activate.fish 2>/dev/null || true
chmod +x venv/bin/python* 2>/dev/null || true
chmod +x venv/bin/pip* 2>/dev/null || true
chmod +x venv/bin/* 2>/dev/null || true
echo "✅ Virtual environment permissions fixed!"

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated! (You should see (venv) in your prompt)"

# Verify we're using the right Python
echo "🔍 Verifying virtual environment..."
VENV_PYTHON=$(which python)
echo "   Using Python: $VENV_PYTHON"
if [[ "$VENV_PYTHON" == *"venv/bin/python"* ]]; then
    echo "   ✅ Correct virtual environment Python detected"
else
    echo "   ❌ Warning: Not using virtual environment Python"
    echo "   Expected: */venv/bin/python"
    echo "   Found: $VENV_PYTHON"
fi

# Upgrade pip to latest version
echo "Upgrading pip to latest version..."
python -m pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Add src directory to Python path
export PYTHONPATH="src:$PYTHONPATH"

# Permanently set PYTHONPATH in the virtual environment activation script
activate_script="venv/bin/activate"
if [ -f "$activate_script" ]; then
    # Check if PYTHONPATH is already set in the activation script
    if ! grep -q 'export PYTHONPATH="src:$PYTHONPATH"' "$activate_script"; then
        echo "Setting PYTHONPATH in virtual environment activation script..."
        echo "" >> "$activate_script"
        echo "# Set project PYTHONPATH" >> "$activate_script"
        echo 'export PYTHONPATH="src:$PYTHONPATH"' >> "$activate_script"
    fi
fi

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
        --verbose 1
    
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
    echo "   python src/core/train_agent.py --total_timesteps 10000 --eval_freq 2000 --n_eval_episodes 20 --verbose 1"
    echo ""
    echo "3. Or run a longer test (50k timesteps, ~5-10 minutes):"
    echo "   python src/core/train_agent.py --total_timesteps 50000 --eval_freq 5000 --n_eval_episodes 50 --verbose 1"
    echo ""
    echo "4. Or run the full training (1M timesteps, ~1-2 hours):"
    echo "   python src/core/train_agent.py --total_timesteps 1000000 --eval_freq 10000 --n_eval_episodes 100 --verbose 1"
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