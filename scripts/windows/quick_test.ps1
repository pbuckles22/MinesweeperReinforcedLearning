#!/usr/bin/env pwsh

# Quick RL Training Test
# Runs a fast training session to verify the RL system is working
# Duration: ~1-2 minutes

Write-Host "Starting Quick RL Training Test..." -ForegroundColor Green
Write-Host "Duration: ~1-2 minutes" -ForegroundColor Yellow
Write-Host ""

# Activate virtual environment
if (Test-Path "venv\Scripts\Activate.ps1") {
    & "venv\Scripts\Activate.ps1"
} else {
    Write-Host "Error: Virtual environment not found. Run install_and_run.ps1 first." -ForegroundColor Red
    exit 1
}

# Run quick training
python src/core/train_agent.py `
    --total_timesteps 10000 `
    --eval_freq 2000 `
    --n_eval_episodes 20 `
    --verbose 1

Write-Host ""
Write-Host "Quick test completed!" -ForegroundColor Green 