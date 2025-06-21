#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Test M1 GPU training functionality for Minesweeper RL

.DESCRIPTION
    This script runs a quick training session to verify M1 GPU support
    and ensure the training pipeline is working correctly.

.PARAMETER QuickTest
    Run a very quick test (10k timesteps) for fast verification

.PARAMETER FullTest
    Run a full test (100k timesteps) for comprehensive verification

.EXAMPLE
    .\test_m1_training.ps1 -QuickTest
    Runs a quick 10k timestep test

.EXAMPLE
    .\test_m1_training.ps1 -FullTest
    Runs a full 100k timestep test
#>

param(
    [switch]$QuickTest,
    [switch]$FullTest
)

# Set error action preference
$ErrorActionPreference = "Stop"

Write-Host "🚀 M1 GPU Training Test" -ForegroundColor Green
Write-Host "=======================" -ForegroundColor Green

# Check if we're in the right directory
if (-not (Test-Path "src\core\train_agent.py")) {
    Write-Host "❌ Error: Please run this script from the project root directory" -ForegroundColor Red
    exit 1
}

# Activate virtual environment if it exists
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "🔧 Activating virtual environment..." -ForegroundColor Yellow
    & "venv\Scripts\Activate.ps1"
} else {
    Write-Host "⚠️  Virtual environment not found, using system Python" -ForegroundColor Yellow
}

# Test M1 GPU detection first
Write-Host "`n🔍 Testing M1 GPU detection..." -ForegroundColor Cyan
try {
    python test_m1_gpu.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ M1 GPU test failed" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "❌ Error running M1 GPU test: $_" -ForegroundColor Red
    exit 1
}

# Determine timesteps based on parameters
$timesteps = 10000  # Default quick test
if ($FullTest) {
    $timesteps = 100000
    Write-Host "`n🎯 Running FULL test with $timesteps timesteps" -ForegroundColor Green
} else {
    Write-Host "`n🎯 Running QUICK test with $timesteps timesteps" -ForegroundColor Green
}

# Create test directories
$testDirs = @("logs", "mlruns", "best_model", "experiments")
foreach ($dir in $testDirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "📁 Created directory: $dir" -ForegroundColor Yellow
    }
}

# Run training test
Write-Host "`n🏋️  Starting training test..." -ForegroundColor Cyan
Write-Host "   Timesteps: $timesteps" -ForegroundColor White
Write-Host "   Evaluation frequency: 2000" -ForegroundColor White
Write-Host "   Evaluation episodes: 10" -ForegroundColor White

try {
    $startTime = Get-Date
    
    # Run the training script
    python -m src.core.train_agent `
        --total_timesteps $timesteps `
        --eval_freq 2000 `
        --n_eval_episodes 10 `
        --save_freq 10000 `
        --verbose 1 `
        --device auto
    
    $endTime = Get-Date
    $duration = $endTime - $startTime
    
    Write-Host "`n✅ Training test completed successfully!" -ForegroundColor Green
    Write-Host "   Duration: $($duration.ToString('mm\:ss'))" -ForegroundColor White
    Write-Host "   Timesteps: $timesteps" -ForegroundColor White
    
    # Check for output files
    Write-Host "`n📊 Checking output files..." -ForegroundColor Cyan
    
    $outputFiles = @(
        @{Path="logs\eval_log.txt"; Description="Evaluation log"},
        @{Path="best_model\best_model.zip"; Description="Best model"},
        @{Path="experiments\*\metrics.json"; Description="Experiment metrics"}
    )
    
    foreach ($file in $outputFiles) {
        if (Test-Path $file.Path) {
            Write-Host "   ✅ $($file.Description): Found" -ForegroundColor Green
        } else {
            Write-Host "   ⚠️  $($file.Description): Not found" -ForegroundColor Yellow
        }
    }
    
    # Check MLflow logs
    $mlflowLogs = Get-ChildItem "mlruns" -Recurse -Filter "*.yaml" -ErrorAction SilentlyContinue
    if ($mlflowLogs.Count -gt 0) {
        Write-Host "   ✅ MLflow logs: $($mlflowLogs.Count) files found" -ForegroundColor Green
    } else {
        Write-Host "   ⚠️  MLflow logs: No files found" -ForegroundColor Yellow
    }
    
    Write-Host "`n🎉 M1 GPU training test PASSED!" -ForegroundColor Green
    Write-Host "   Your M1 MacBook is ready for full training!" -ForegroundColor White
    
} catch {
    Write-Host "`n❌ Training test failed: $_" -ForegroundColor Red
    Write-Host "   Check the error messages above for details" -ForegroundColor Yellow
    exit 1
}

Write-Host "`n📚 Next steps:" -ForegroundColor Cyan
Write-Host "   1. Run full training: python -m src.core.train_agent --total_timesteps 1000000" -ForegroundColor White
Write-Host "   2. Monitor with MLflow: mlflow ui" -ForegroundColor White
Write-Host "   3. Open http://127.0.0.1:5000 in your browser" -ForegroundColor White
Write-Host "   4. Check logs in the 'logs' directory" -ForegroundColor White
Write-Host "   5. Find best model in 'best_model' directory" -ForegroundColor White 