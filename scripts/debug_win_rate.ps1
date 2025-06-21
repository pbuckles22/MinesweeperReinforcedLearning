#!/usr/bin/env pwsh

# Debug Win Rate Issue Script
# Diagnoses why win rate might be low after 200k iterations

Write-Host "🔍 Debugging Win Rate Issue" -ForegroundColor Green
Write-Host "==========================" -ForegroundColor Green

# Activate virtual environment
if (Test-Path "venv\Scripts\Activate.ps1") {
    & "venv\Scripts\Activate.ps1"
} else {
    Write-Host "⚠️  Virtual environment not found, using system Python" -ForegroundColor Yellow
}

Write-Host "`n1. Checking current training configuration..." -ForegroundColor Cyan

# Check the training agent configuration
python -c "
from src.core.train_agent import detect_optimal_device, get_optimal_hyperparameters
import torch

print('=== Device Detection ===')
device_info = detect_optimal_device()
print(f'Device: {device_info[\"device\"]}')
print(f'Description: {device_info[\"description\"]}')

print('\n=== Optimal Hyperparameters ===')
optimal_params = get_optimal_hyperparameters(device_info)
for key, value in optimal_params.items():
    print(f'{key}: {value}')

print('\n=== PyTorch Info ===')
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'CUDA available: {torch.cuda.is_available()}')
"

Write-Host "`n2. Testing environment with simple scenario..." -ForegroundColor Cyan

# Test with a very simple scenario that should learn quickly
python -c "
from src.core.minesweeper_env import MinesweeperEnv
import numpy as np

print('=== Simple Environment Test ===')
env = MinesweeperEnv(max_board_size=(4, 4), max_mines=2)
print(f'Board size: 4x4')
print(f'Mine count: 2')
print(f'Action space: {env.action_space}')
print(f'Observation space: {env.observation_space}')

# Test a few random episodes
wins = 0
total_episodes = 10
for i in range(total_episodes):
    obs, info = env.reset()
    done = False
    steps = 0
    while not done and steps < 20:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        if info.get('won', False):
            wins += 1
            break

print(f'Random play win rate: {wins/total_episodes*100:.1f}%')
print(f'Average steps per episode: {steps/total_episodes:.1f}')
"

Write-Host "`n3. Running focused training test..." -ForegroundColor Cyan

# Run a focused training test with better parameters for learning
Write-Host "Running 20k timestep test with optimized parameters..." -ForegroundColor Yellow

python -m src.core.train_agent `
    --total_timesteps 20000 `
    --eval_freq 2000 `
    --n_eval_episodes 20 `
    --learning_rate 0.0003 `
    --batch_size 64 `
    --n_steps 1024 `
    --n_epochs 8 `
    --gamma 0.99 `
    --gae_lambda 0.95 `
    --clip_range 0.2 `
    --ent_coef 0.01 `
    --vf_coef 0.5 `
    --max_grad_norm 0.5 `
    --verbose 2 `
    --device auto

Write-Host "`n4. Checking evaluation logs..." -ForegroundColor Cyan

# Check if evaluation logs exist and show recent results
if (Test-Path "logs\eval_log.txt") {
    Write-Host "Recent evaluation results:" -ForegroundColor Yellow
    Get-Content "logs\eval_log.txt" -Tail 10
} else {
    Write-Host "No evaluation logs found" -ForegroundColor Red
}

Write-Host "`n5. Recommendations for improving win rate:" -ForegroundColor Cyan
Write-Host "   - Start with smaller boards (4x4, 6x6) to build confidence" -ForegroundColor White
Write-Host "   - Use curriculum learning (starts easy, gets harder)" -ForegroundColor White
Write-Host "   - Increase evaluation frequency to monitor progress" -ForegroundColor White
Write-Host "   - Check if early learning mode is working" -ForegroundColor White
Write-Host "   - Verify reward signals are appropriate" -ForegroundColor White

Write-Host "`n🎯 Next steps:" -ForegroundColor Green
Write-Host "   1. Run: python scripts/quick_test.ps1 (10k timesteps)" -ForegroundColor White
Write-Host "   2. Run: python scripts/medium_test.ps1 (50k timesteps)" -ForegroundColor White
Write-Host "   3. Check MLflow for learning curves" -ForegroundColor White
Write-Host "      mlflow ui" -ForegroundColor White
Write-Host "      Then open http://127.0.0.1:5000 in your browser" -ForegroundColor White
Write-Host "   4. Monitor logs in 'logs' directory" -ForegroundColor White 