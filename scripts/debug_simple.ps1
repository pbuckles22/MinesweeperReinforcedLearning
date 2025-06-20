#!/usr/bin/env pwsh

# Simple Debug Script
# Tests the evaluation function with minimal complexity

Write-Host "=== Simple Debug Test ===" -ForegroundColor Green

# Activate virtual environment
if (Test-Path "venv\Scripts\Activate.ps1") {
    & "venv\Scripts\Activate.ps1"
} else {
    Write-Host "Error: Virtual environment not found." -ForegroundColor Red
    exit 1
}

# Set PYTHONPATH
$env:PYTHONPATH = "src;$env:PYTHONPATH"

Write-Host "Testing basic evaluation..." -ForegroundColor Yellow

# Create a simple Python script to test evaluation
$testScript = @"
import sys
print('Starting evaluation test...')

try:
    from src.core.train_agent import evaluate_model, make_env
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    print('1. Imports successful')
    
    # Create environment
    env = DummyVecEnv([make_env(max_board_size=4, max_mines=2)])
    print('2. Environment created')
    
    # Create model
    model = PPO('MlpPolicy', env, verbose=0)
    print('3. Model created')
    
    # Test evaluation with just 1 episode
    print('4. Starting evaluation...')
    results = evaluate_model(model, env, n_episodes=1)
    print('5. Evaluation completed')
    print(f'Results: {results}')
    
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"@

# Write the script to a temporary file
$testScript | Out-File -FilePath "debug_test.py" -Encoding UTF8

# Run the test
try {
    python debug_test.py
    Write-Host "✅ Simple evaluation test completed!" -ForegroundColor Green
} catch {
    Write-Host "❌ Simple evaluation test failed: $_" -ForegroundColor Red
} finally {
    # Clean up
    if (Test-Path "debug_test.py") {
        Remove-Item "debug_test.py"
    }
} 