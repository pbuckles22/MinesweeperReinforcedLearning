#!/usr/bin/env pwsh

# Test Cascade Safety Implementation
# Verify that the first_cascade safety system works correctly

Write-Host "=== Testing Cascade Safety Implementation ===" -ForegroundColor Green

# Activate virtual environment
if (Test-Path "venv\Scripts\Activate.ps1") {
    & "venv\Scripts\Activate.ps1"
} else {
    Write-Host "Error: Virtual environment not found." -ForegroundColor Red
    exit 1
}

# Set PYTHONPATH
$env:PYTHONPATH = "src;$env:PYTHONPATH"

Write-Host "1. Testing cascade detection and safety..." -ForegroundColor Yellow
python -c "
from src.core.minesweeper_env import MinesweeperEnv
import numpy as np

# Create environment
env = MinesweeperEnv(max_board_size=4, max_mines=2)

print('Testing cascade safety system...')
print('=' * 50)

# Test multiple games to see cascade behavior
for game in range(3):
    print(f'\n--- Game {game + 1} ---')
    obs, info = env.reset()
    
    print(f'Initial cascade state: {info[\"first_cascade_occurred\"]}')
    print(f'Pre-cascade moves: {info[\"pre_cascade_moves\"]}')
    
    total_reward = 0
    steps = 0
    cascade_occurred = False
    
    for step in range(10):
        # Take random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        steps += 1
        
        print(f'Step {step}: action={action}, reward={reward}, cascade={info[\"first_cascade_occurred\"]}, pre_moves={info[\"pre_cascade_moves\"]}')
        
        if info['first_cascade_occurred'] and not cascade_occurred:
            cascade_occurred = True
            print(f'  *** FIRST CASCADE DETECTED at step {step} ***')
            print(f'  Cells revealed before cascade: {info[\"cells_revealed_before_cascade\"]}')
        
        if terminated or truncated:
            print(f'  Game ended: won={info[\"won\"]}')
            break
    
    print(f'Game Summary: steps={steps}, total_reward={total_reward}, cascade_occurred={cascade_occurred}')
    print('=' * 50)

print('\n✅ Cascade safety test completed!')
"

Write-Host "`n2. Testing reward structure..." -ForegroundColor Yellow
python -c "
from src.core.constants import *
from src.core.minesweeper_env import MinesweeperEnv

print('Reward Structure:')
print(f'  REWARD_PRE_CASCADE_SAFE: {REWARD_PRE_CASCADE_SAFE}')
print(f'  REWARD_SAFE_REVEAL: {REWARD_SAFE_REVEAL}')
print(f'  REWARD_WIN: {REWARD_WIN}')
print(f'  REWARD_HIT_MINE: {REWARD_HIT_MINE}')

print('\nReward Ratios:')
print(f'  Post-cascade / Pre-cascade: {REWARD_SAFE_REVEAL / REWARD_PRE_CASCADE_SAFE:.1f}x')
print(f'  Win / Post-cascade: {REWARD_WIN / REWARD_SAFE_REVEAL:.1f}x')
print(f'  Mine penalty / Post-cascade: {abs(REWARD_HIT_MINE) / REWARD_SAFE_REVEAL:.1f}x')
" 