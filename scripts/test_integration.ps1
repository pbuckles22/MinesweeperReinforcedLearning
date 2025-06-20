#!/usr/bin/env pwsh

# Integration Tests Script
# Runs comprehensive integration tests to catch RL system issues

Write-Host "=== Running Comprehensive Integration Tests ===" -ForegroundColor Green

# Activate virtual environment
if (Test-Path "venv\Scripts\Activate.ps1") {
    & "venv\Scripts\Activate.ps1"
} else {
    Write-Host "Error: Virtual environment not found." -ForegroundColor Red
    exit 1
}

# Set PYTHONPATH
$env:PYTHONPATH = "src;$env:PYTHONPATH"

Write-Host "Running RL integration tests..." -ForegroundColor Yellow

# Run the integration tests
try {
    python -m pytest tests/integration/rl/test_rl_integration.py -v --tb=short
    $integrationExitCode = $LASTEXITCODE
} catch {
    Write-Host "❌ Integration tests failed: $_" -ForegroundColor Red
    $integrationExitCode = 1
}

Write-Host "Running evaluation unit tests..." -ForegroundColor Yellow

# Run the evaluation unit tests specifically
try {
    python -m pytest tests/unit/rl/test_evaluation_unit.py -v --tb=short
    $evalExitCode = $LASTEXITCODE
} catch {
    Write-Host "❌ Evaluation unit tests failed: $_" -ForegroundColor Red
    $evalExitCode = 1
}

Write-Host "Running comprehensive RL tests..." -ForegroundColor Yellow

# Run the comprehensive RL tests
try {
    python -m pytest tests/unit/rl/test_rl_comprehensive_unit.py -v --tb=short
    $comprehensiveExitCode = $LASTEXITCODE
} catch {
    Write-Host "❌ Comprehensive RL tests failed: $_" -ForegroundColor Red
    $comprehensiveExitCode = 1
}

# Summary
Write-Host "`n=== Integration Test Summary ===" -ForegroundColor Cyan
Write-Host "RL Integration Tests: $($integrationExitCode -eq 0 ? '✅ PASSED' : '❌ FAILED')" -ForegroundColor $(if ($integrationExitCode -eq 0) { 'Green' } else { 'Red' })
Write-Host "Evaluation Unit Tests: $($evalExitCode -eq 0 ? '✅ PASSED' : '❌ FAILED')" -ForegroundColor $(if ($evalExitCode -eq 0) { 'Green' } else { 'Red' })
Write-Host "Comprehensive RL Tests: $($comprehensiveExitCode -eq 0 ? '✅ PASSED' : '❌ FAILED')" -ForegroundColor $(if ($comprehensiveExitCode -eq 0) { 'Green' } else { 'Red' })

$overallExitCode = $integrationExitCode -bor $evalExitCode -bor $comprehensiveExitCode

if ($overallExitCode -eq 0) {
    Write-Host "`n🎉 All integration tests passed!" -ForegroundColor Green
    Write-Host "The RL system is working correctly and should not have the hanging issues we encountered." -ForegroundColor Green
} else {
    Write-Host "`n⚠️  Some integration tests failed. Please review the output above." -ForegroundColor Yellow
    Write-Host "These tests are designed to catch the specific issues we encountered." -ForegroundColor Yellow
}

exit $overallExitCode 