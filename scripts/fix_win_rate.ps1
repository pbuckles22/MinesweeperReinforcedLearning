#!/usr/bin/env pwsh

# Fix Win Rate Issue Script
# Runs training with optimized parameters to improve win rate

Write-Host "🔧 Fixing Win Rate Issue" -ForegroundColor Green
Write-Host "=======================" -ForegroundColor Green

# Activate virtual environment
if (Test-Path "venv\Scripts\Activate.ps1") {
    & "venv\Scripts\Activate.ps1"
} else {
    Write-Host "⚠️  Virtual environment not found, using system Python" -ForegroundColor Yellow
}

Write-Host "`n🎯 Running optimized training to fix win rate..." -ForegroundColor Cyan
Write-Host "   - Using early learning mode" -ForegroundColor White
Write-Host "   - Starting with small boards (4x4)" -ForegroundColor White
Write-Host "   - More frequent evaluation" -ForegroundColor White
Write-Host "   - Better hyperparameters" -ForegroundColor White

# Run training with optimized parameters for better learning
python -m src.core.train_agent `
    --total_timesteps 100000 `
    --eval_freq 5000 `
    --n_eval_episodes 50 `
    --learning_rate 0.0003 `
    --batch_size 64 `
    --n_steps 1024 `
    --n_epochs 10 `
    --gamma 0.99 `
    --gae_lambda 0.95 `
    --clip_range 0.2 `
    --ent_coef 0.01 `
    --vf_coef 0.5 `
    --max_grad_norm 0.5 `
    --verbose 2 `
    --device auto

Write-Host "`n📊 Checking results..." -ForegroundColor Cyan

# Check evaluation logs
if (Test-Path "logs\eval_log.txt") {
    Write-Host "Recent evaluation results:" -ForegroundColor Yellow
    Get-Content "logs\eval_log.txt" -Tail 15
} else {
    Write-Host "No evaluation logs found" -ForegroundColor Red
}

Write-Host "`n🎯 If win rate is still low, try these steps:" -ForegroundColor Green
Write-Host "   1. Run with even smaller boards: 4x4 with 2 mines" -ForegroundColor White
Write-Host "   2. Monitor training progress with MLflow" -ForegroundColor White
Write-Host "      mlflow ui" -ForegroundColor White
Write-Host "      Then open http://127.0.0.1:5000 in your browser" -ForegroundColor White
Write-Host "   3. Verify early learning mode is working" -ForegroundColor White
Write-Host "   4. Check for any error messages in the console" -ForegroundColor White

Write-Host "`n📈 To monitor training progress:" -ForegroundColor Cyan
Write-Host "   mlflow ui" -ForegroundColor White
Write-Host "   Then open http://127.0.0.1:5000 in your browser" -ForegroundColor White 