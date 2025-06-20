#!/usr/bin/env pwsh

# Episode Completion Debug Script
# Tests if episodes complete properly without infinite loops

Write-Host "=== Episode Completion Debug Test ===" -ForegroundColor Green

# Activate virtual environment
if (Test-Path "venv\Scripts\Activate.ps1") {
    & "venv\Scripts\Activate.ps1"
} else {
    Write-Host "Error: Virtual environment not found." -ForegroundColor Red
    exit 1
}

# Set PYTHONPATH
$env:PYTHONPATH = "src;$env:PYTHONPATH"

Write-Host "Testing episode completion..." -ForegroundColor Yellow

# Create a Python script to test episode completion
$testScript = @"
import sys
import numpy as np

print('Starting episode completion test...')

try:
    from src.core.train_agent import make_env
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    print('1. Imports successful')
    
    # Create environment
    env = DummyVecEnv([make_env(max_board_size=4, max_mines=2)])
    print('2. Environment created')
    
    # Test multiple episodes to completion
    for episode in range(5):
        print(f'3.{episode+1}. Starting episode {episode+1}...')
        obs = env.reset()
        step_count = 0
        max_steps = 100  # Safety limit
        
        while step_count < max_steps:
            # Take random action
            action = np.array([np.random.randint(0, 16)])  # 4x4 board = 16 possible actions
            obs, reward, done, info = env.step(action)
            step_count += 1
            
            print(f'   Step {step_count}: reward={reward[0]:.2f}, done={done[0]}, won={info[0].get("won", False)}')
            
            if done[0]:
                print(f'   Episode {episode+1} completed in {step_count} steps')
                break
        
        if step_count >= max_steps:
            print(f'   Episode {episode+1} hit max steps limit!')
    
    print('✅ Episode completion test completed!')
    
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"@

# Write the script to a temporary file
$testScript | Out-File -FilePath "debug_episode_completion.py" -Encoding UTF8

# Run the test
try {
    python debug_episode_completion.py
    Write-Host "✅ Episode completion test completed!" -ForegroundColor Green
} catch {
    Write-Host "❌ Episode completion test failed: $_" -ForegroundColor Red
} finally {
    # Clean up
    if (Test-Path "debug_episode_completion.py") {
        Remove-Item "debug_episode_completion.py"
    }
} 