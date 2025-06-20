#!/usr/bin/env pwsh

# Minimal Step Debug Script
# Tests just the environment step function to identify the hanging issue

Write-Host "=== Minimal Step Debug Test ===" -ForegroundColor Green

# Activate virtual environment
if (Test-Path "venv\Scripts\Activate.ps1") {
    & "venv\Scripts\Activate.ps1"
} else {
    Write-Host "Error: Virtual environment not found." -ForegroundColor Red
    exit 1
}

# Set PYTHONPATH
$env:PYTHONPATH = "src;$env:PYTHONPATH"

Write-Host "Testing minimal step function..." -ForegroundColor Yellow

# Create a Python script to test just the step function
$testScript = @"
import sys
import numpy as np

print('Starting minimal step test...')

try:
    from src.core.train_agent import make_env
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    print('1. Imports successful')
    
    # Create environment
    env = DummyVecEnv([make_env(max_board_size=4, max_mines=2)])
    print('2. Environment created')
    
    # Test reset
    print('3. Testing reset...')
    obs = env.reset()
    print(f'4. Reset successful, obs shape: {obs.shape}')
    
    # Test single step
    print('5. Testing single step...')
    action = np.array([0])
    print(f'6. Action created: {action}')
    
    print('7. About to call env.step(action)...')
    result = env.step(action)
    print('8. env.step() completed!')
    
    print(f'9. Step result length: {len(result)}')
    if len(result) == 4:
        obs, reward, done, info = result
        print('10. Unpacked 4 values')
    elif len(result) == 5:
        obs, reward, done, truncated, info = result
        print('10. Unpacked 5 values')
    else:
        print(f'10. Unexpected result length: {len(result)}')
    
    print(f'11. Reward: {reward}')
    print(f'12. Done: {done}')
    print(f'13. Info: {info}')
    
    print('✅ Minimal step test completed!')
    
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"@

# Write the script to a temporary file
$testScript | Out-File -FilePath "debug_minimal_step.py" -Encoding UTF8

# Run the test
try {
    python debug_minimal_step.py
    Write-Host "✅ Minimal step test completed!" -ForegroundColor Green
} catch {
    Write-Host "❌ Minimal step test failed: $_" -ForegroundColor Red
} finally {
    # Clean up
    if (Test-Path "debug_minimal_step.py") {
        Remove-Item "debug_minimal_step.py"
    }
} 