#!/usr/bin/env pwsh

# Medium RL Training Test
# Runs a moderate training session to see learning progress
# Duration: ~5-10 minutes

Write-Host "Starting Medium RL Training Test..." -ForegroundColor Green
Write-Host "Duration: ~5-10 minutes" -ForegroundColor Yellow
Write-Host ""

# Activate virtual environment
if (Test-Path "venv\Scripts\Activate.ps1") {
    & "venv\Scripts\Activate.ps1"
} else {
    Write-Host "Error: Virtual environment not found. Run install_and_run.ps1 first." -ForegroundColor Red
    exit 1
}

# Run medium training
python src/core/train_agent.py `
    --total_timesteps 50000 `
    --eval_freq 5000 `
    --n_eval_episodes 50 `
    --verbose 1

Write-Host ""
Write-Host "Medium test completed!" -ForegroundColor Green 