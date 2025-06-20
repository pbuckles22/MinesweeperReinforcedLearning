#!/usr/bin/env pwsh

# Training Loop Debug Script
# Tests the training loop step by step

Write-Host "=== Training Loop Debug Test ===" -ForegroundColor Green

# Activate virtual environment
if (Test-Path "venv\Scripts\Activate.ps1") {
    & "venv\Scripts\Activate.ps1"
} else {
    Write-Host "Error: Virtual environment not found." -ForegroundColor Red
    exit 1
}

# Set PYTHONPATH
$env:PYTHONPATH = "src;$env:PYTHONPATH"

Write-Host "Testing training loop step by step..." -ForegroundColor Yellow

# Create a Python script to test the training loop
$testScript = @"
import sys
print('Starting training loop test...')

try:
    from src.core.train_agent import make_env, IterationCallback, ExperimentTracker
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
    
    # Create experiment tracker
    experiment_tracker = ExperimentTracker()
    print('4. Experiment tracker created')
    
    # Create iteration callback
    iteration_callback = IterationCallback(verbose=1, debug_level=2, experiment_tracker=experiment_tracker)
    print('5. Iteration callback created')
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./debug_best_model',
        log_path='./debug_logs/',
        eval_freq=50,
        n_eval_episodes=3,
        deterministic=True,
        render=False
    )
    print('6. Evaluation callback created')
    
    # Test training with just iteration callback
    print('7. Testing training with iteration callback only...')
    model.learn(total_timesteps=50, callback=iteration_callback, progress_bar=False)
    print('8. Training with iteration callback completed')
    
    # Test training with just evaluation callback
    print('9. Testing training with evaluation callback only...')
    model.learn(total_timesteps=50, callback=eval_callback, progress_bar=False)
    print('10. Training with evaluation callback completed')
    
    # Test training with both callbacks
    print('11. Testing training with both callbacks...')
    model.learn(total_timesteps=50, callback=[eval_callback, iteration_callback], progress_bar=False)
    print('12. Training with both callbacks completed')
    
    print('✅ All training loop tests completed!')
    
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"@

# Write the script to a temporary file
$testScript | Out-File -FilePath "debug_training_loop.py" -Encoding UTF8

# Run the test
try {
    python debug_training_loop.py
    Write-Host "✅ Training loop test completed!" -ForegroundColor Green
} catch {
    Write-Host "❌ Training loop test failed: $_" -ForegroundColor Red
} finally {
    # Clean up
    if (Test-Path "debug_training_loop.py") {
        Remove-Item "debug_training_loop.py"
    }
} 