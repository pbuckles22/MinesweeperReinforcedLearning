#!/usr/bin/env pwsh

# Medium Test Script for Windows
# Runs a medium subset of tests to verify the system is working
# Duration: ~2-3 minutes

Write-Host "Starting Medium Test..." -ForegroundColor Green
Write-Host "Duration: ~2-3 minutes" -ForegroundColor Yellow
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

# Run medium test suite (same as Mac/Linux)
Write-Host "Running medium test suite..." -ForegroundColor Cyan
python -m pytest tests/unit/core tests/functional tests/integration/core -v --maxfail=10 --disable-warnings

Write-Host ""
Write-Host "Medium test completed!" -ForegroundColor Green 