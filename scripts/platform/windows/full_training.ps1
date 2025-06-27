#!/usr/bin/env pwsh

# Full RL Training
# Runs a complete training session for production-ready model
# Duration: ~1-2 hours

Write-Host "Starting Full RL Training..." -ForegroundColor Green
Write-Host "Duration: ~1-2 hours" -ForegroundColor Yellow
Write-Host "This will train a production-ready model." -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment
if (Test-Path "venv\Scripts\Activate.ps1") {
    & "venv\Scripts\Activate.ps1"
} else {
    Write-Host "Error: Virtual environment not found. Run install_and_run.ps1 first." -ForegroundColor Red
    exit 1
}

# Run full training
python src/core/train_agent.py `
    --total_timesteps 1000000 `
    --eval_freq 10000 `
    --n_eval_episodes 100 `
    --verbose 0

Write-Host ""
Write-Host "Full training completed!" -ForegroundColor Green
Write-Host "Check the 'best_model' directory for the trained model." -ForegroundColor Cyan 