#!/usr/bin/env pwsh

# Quick Test Script for Windows
# Runs a quick subset of tests to verify the system is working
# Duration: ~30-60 seconds

Write-Host "Starting Quick Test..." -ForegroundColor Green
Write-Host "Duration: ~30-60 seconds" -ForegroundColor Yellow
Write-Host ""

# Activate virtual environment
if (Test-Path "venv\Scripts\Activate.ps1") {
    & "venv\Scripts\Activate.ps1"
} else {
    Write-Host "Error: Virtual environment not found. Run install_and_run.ps1 first." -ForegroundColor Red
    exit 1
}

# Set PYTHONPATH
$env:PYTHONPATH = "src;$env:PYTHONPATH"

# Run quick subset of tests (same as Mac/Linux)
Write-Host "Running quick test suite..." -ForegroundColor Cyan
python -m pytest tests/unit/core tests/functional/game_flow -v --maxfail=5 --disable-warnings

Write-Host ""
Write-Host "Quick test completed!" -ForegroundColor Green 