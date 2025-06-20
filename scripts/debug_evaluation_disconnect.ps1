#!/usr/bin/env pwsh

# Debug Evaluation Disconnect
# Test to understand why training rewards are high but evaluation shows 0.00

Write-Host "=== Debugging Evaluation Disconnect ===" -ForegroundColor Green

# Activate virtual environment
if (Test-Path "venv\Scripts\Activate.ps1") {
    & "venv\Scripts\Activate.ps1"
} else {
    Write-Host "Error: Virtual environment not found." -ForegroundColor Red
    exit 1
}

# Set PYTHONPATH
$env:PYTHONPATH = "src;$env:PYTHONPATH"

Write-Host "1. Testing environment creation and reward values..." -ForegroundColor Yellow
python -c "
from src.core.train_agent import make_env
from src.core.constants import *
from stable_baselines3.common.vec_env import DummyVecEnv

# Create training environment
train_env = DummyVecEnv([make_env(max_board_size=4, max_mines=2)])

# Create evaluation environment  
eval_env = DummyVecEnv([make_env(max_board_size=4, max_mines=2)])

print('Training Environment:')
print(f'  invalid_action_penalty: {train_env.envs[0].env.reward_invalid_action}')
print(f'  mine_penalty: {train_env.envs[0].env.reward_hit_mine}')
print(f'  safe_reveal_base: {train_env.envs[0].env.reward_safe_reveal}')
print(f'  win_reward: {train_env.envs[0].env.reward_win}')

print('\nEvaluation Environment:')
print(f'  invalid_action_penalty: {eval_env.envs[0].env.reward_invalid_action}')
print(f'  mine_penalty: {eval_env.envs[0].env.reward_hit_mine}')
print(f'  safe_reveal_base: {eval_env.envs[0].env.reward_safe_reveal}')
print(f'  win_reward: {eval_env.envs[0].env.reward_win}')

print('\nConstants:')
print(f'  REWARD_INVALID_ACTION: {REWARD_INVALID_ACTION}')
print(f'  REWARD_HIT_MINE: {REWARD_HIT_MINE}')
print(f'  REWARD_SAFE_REVEAL: {REWARD_SAFE_REVEAL}')
print(f'  REWARD_WIN: {REWARD_WIN}')
"

Write-Host "`n2. Testing single episode evaluation..." -ForegroundColor Yellow
python -c "
from src.core.train_agent import make_env
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

# Create environment
env = DummyVecEnv([make_env(max_board_size=4, max_mines=2)])

# Reset environment
obs = env.reset()
print(f'Initial observation shape: {obs.shape}')

# Take a few random actions and track rewards
total_reward = 0
steps = 0
won = False

for step in range(10):
    # Take random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Handle vectorized environment
    if isinstance(reward, (list, np.ndarray)):
        reward = reward[0]
    if isinstance(info, (list, tuple)) and len(info) > 0:
        info = info[0]
    
    total_reward += reward
    steps += 1
    
    print(f'Step {step}: action={action}, reward={reward}, terminated={terminated}, won={info.get(\"won\", False)}')
    
    if terminated or truncated:
        won = info.get('won', False)
        break

print(f'\nEpisode Summary:')
print(f'  Total steps: {steps}')
print(f'  Total reward: {total_reward}')
print(f'  Won: {won}')
print(f'  Average reward per step: {total_reward/steps if steps > 0 else 0}')
"

Write-Host "`n3. Testing evaluation callback logic..." -ForegroundColor Yellow
python -c "
from src.core.train_agent import CustomEvalCallback, make_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

# Create environments
train_env = DummyVecEnv([make_env(max_board_size=4, max_mines=2)])
eval_env = DummyVecEnv([make_env(max_board_size=4, max_mines=2)])

# Create a simple model
model = PPO('MlpPolicy', train_env, verbose=0)

# Create evaluation callback
eval_callback = CustomEvalCallback(
    eval_env,
    eval_freq=1,
    n_eval_episodes=3,
    verbose=1
)

# Test evaluation manually
print('Testing evaluation callback manually...')
eval_callback.model = model
eval_callback._on_step()

print('\nEvaluation callback test completed.')
" 