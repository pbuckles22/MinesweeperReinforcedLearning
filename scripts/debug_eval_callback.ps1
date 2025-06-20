#!/usr/bin/env pwsh

# EvalCallback Debug Script
# Tests just the EvalCallback to identify the hanging issue

Write-Host "=== EvalCallback Debug Test ===" -ForegroundColor Green

# Activate virtual environment
if (Test-Path "venv\Scripts\Activate.ps1") {
    & "venv\Scripts\Activate.ps1"
} else {
    Write-Host "Error: Virtual environment not found." -ForegroundColor Red
    exit 1
}

# Set PYTHONPATH
$env:PYTHONPATH = "src;$env:PYTHONPATH"

Write-Host "Testing EvalCallback step by step..." -ForegroundColor Yellow

# Create a Python script to test just the EvalCallback
$testScript = @"
import sys
print('Starting EvalCallback test...')

try:
    from src.core.train_agent import make_env
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback
    
    print('1. Imports successful')
    
    # Create environment
    env = DummyVecEnv([make_env(max_board_size=4, max_mines=2)])
    eval_env = DummyVecEnv([make_env(max_board_size=4, max_mines=2)])
    print('2. Environments created')
    
    # Create model
    model = PPO('MlpPolicy', env, verbose=0)
    print('3. Model created')
    
    # Test EvalCallback with minimal settings
    print('4. Creating EvalCallback with minimal settings...')
    eval_callback = EvalCallback(
        eval_env,
        eval_freq=25,  # Very frequent evaluation
        n_eval_episodes=1,  # Just 1 episode
        deterministic=True,
        render=False,
        verbose=1  # Add verbose output
    )
    print('5. EvalCallback created')
    
    # Test training with EvalCallback
    print('6. Starting training with EvalCallback...')
    model.learn(total_timesteps=50, callback=eval_callback, progress_bar=False)
    print('7. Training with EvalCallback completed')
    
    print('✅ EvalCallback test completed!')
    
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"@

# Write the script to a temporary file
$testScript | Out-File -FilePath "debug_eval_callback.py" -Encoding UTF8

# Run the test
try {
    python debug_eval_callback.py
    Write-Host "✅ EvalCallback test completed!" -ForegroundColor Green
} catch {
    Write-Host "❌ EvalCallback test failed: $_" -ForegroundColor Red
} finally {
    # Clean up
    if (Test-Path "debug_eval_callback.py") {
        Remove-Item "debug_eval_callback.py"
    }
} 