#!/usr/bin/env pwsh

# No EvalCallback Test Script
# Tests training without EvalCallback to see if that resolves hanging

Write-Host "=== No EvalCallback Test ===" -ForegroundColor Green

# Activate virtual environment
if (Test-Path "venv\Scripts\Activate.ps1") {
    & "venv\Scripts\Activate.ps1"
} else {
    Write-Host "Error: Virtual environment not found." -ForegroundColor Red
    exit 1
}

# Set PYTHONPATH
$env:PYTHONPATH = "src;$env:PYTHONPATH"

Write-Host "Testing training without EvalCallback..." -ForegroundColor Yellow

# Create a Python script to test training without EvalCallback
$testScript = @"
import sys

print('Starting training without EvalCallback...')

try:
    from src.core.train_agent import make_env, IterationCallback, ExperimentTracker
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    print('1. Imports successful')
    
    # Create environment
    env = DummyVecEnv([make_env(max_board_size=4, max_mines=2)])
    print('2. Environment created')
    
    # Create model
    model = PPO('MlpPolicy', env, verbose=0)
    print('3. Model created')
    
    # Create experiment tracker
    experiment_tracker = ExperimentTracker()
    print('4. Experiment tracker created')
    
    # Create iteration callback only (no EvalCallback)
    iteration_callback = IterationCallback(verbose=1, debug_level=2, experiment_tracker=experiment_tracker)
    print('5. Iteration callback created')
    
    # Test training with only iteration callback
    print('6. Starting training with iteration callback only...')
    model.learn(total_timesteps=200, callback=iteration_callback, progress_bar=False)
    print('7. Training completed successfully!')
    
    print('✅ Training without EvalCallback completed!')
    
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"@

# Write the script to a temporary file
$testScript | Out-File -FilePath "test_no_eval.py" -Encoding UTF8

# Run the test
try {
    python test_no_eval.py
    Write-Host "✅ Training without EvalCallback completed!" -ForegroundColor Green
} catch {
    Write-Host "❌ Training without EvalCallback failed: $_" -ForegroundColor Red
} finally {
    # Clean up
    if (Test-Path "test_no_eval.py") {
        Remove-Item "test_no_eval.py"
    }
} 