# Test Version Switcher Script (PowerShell)
# Helps switch between different testing versions for comparison

param(
    [Parameter(Position=0)]
    [ValidateSet("with", "without", "test", "status", "both")]
    [string]$Command = "status"
)

Write-Host "🧪 Minesweeper RL Test Version Switcher" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Check current branch
$CURRENT_BRANCH = git branch --show-current
Write-Host "Current branch: $CURRENT_BRANCH" -ForegroundColor Yellow

# Function to switch to a specific test version
function Switch-ToVersion {
    param([string]$Version)
    
    $branchName = "test/$Version"
    Write-Host "🔄 Switching to $Version version..." -ForegroundColor Green
    
    # Check if branch exists locally
    $branchExists = git show-ref --verify --quiet refs/heads/$branchName
    if ($LASTEXITCODE -eq 0) {
        git checkout $branchName
    } else {
        # Try to fetch and checkout from remote
        git fetch origin $branchName
        git checkout $branchName
    }
    
    Write-Host "✅ Switched to $branchName" -ForegroundColor Green
    Write-Host "Current branch: $(git branch --show-current)" -ForegroundColor Yellow
    
    # Show the key difference
    if ($Version -eq "without-learnable-filtering") {
        Write-Host "🔧 Learnable filtering: DISABLED (learnable_only=False)" -ForegroundColor Red
    } else {
        Write-Host "🔧 Learnable filtering: ENABLED (learnable_only=True)" -ForegroundColor Green
    }
}

# Function to run the comprehensive test
function Run-Test {
    Write-Host "🚀 Running comprehensive DQN test..." -ForegroundColor Cyan
    
    # Activate virtual environment if it exists
    if (Test-Path "venv") {
        Write-Host "📦 Activating virtual environment..." -ForegroundColor Yellow
        & "venv\Scripts\Activate.ps1"
    }
    
    # Run the test script
    python scripts/training/train_dqn_comprehensive_test.py
}

# Function to show status
function Show-Status {
    Write-Host "📊 Current Status:" -ForegroundColor Cyan
    Write-Host "   Branch: $(git branch --show-current)" -ForegroundColor Yellow
    
    $learnableSetting = (Select-String -Path "src/core/minesweeper_env.py" -Pattern "learnable_only=[^,]*").Matches[0].Value
    $learnableValue = $learnableSetting.Split('=')[1]
    Write-Host "   Learnable filtering: $learnableValue" -ForegroundColor Yellow
    
    $scriptType = (Select-String -Path "scripts/training/train_dqn_comprehensive_test.py" -Pattern "Without Learnable Filtering|Comprehensive DQN Training Test Script").Matches[0].Value
    Write-Host "   Test script: $scriptType" -ForegroundColor Yellow
}

# Main logic
switch ($Command) {
    "with" {
        Switch-ToVersion "with-learnable-filtering"
    }
    "without" {
        Switch-ToVersion "without-learnable-filtering"
    }
    "test" {
        Run-Test
    }
    "status" {
        Show-Status
    }
    "both" {
        Write-Host "🔄 Running tests on both versions..." -ForegroundColor Cyan
        
        # Test with learnable filtering
        Write-Host "📊 Testing WITH learnable filtering..." -ForegroundColor Green
        Switch-ToVersion "with-learnable-filtering"
        Run-Test
        
        Write-Host ""
        Write-Host "📊 Testing WITHOUT learnable filtering..." -ForegroundColor Green
        Switch-ToVersion "without-learnable-filtering"
        Run-Test
        
        Write-Host "✅ Both tests completed!" -ForegroundColor Green
    }
    default {
        Write-Host "Usage: .\test_version_switcher.ps1 {with|without|test|status|both}" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Commands:" -ForegroundColor Cyan
        Write-Host "  with    - Switch to version WITH learnable filtering" -ForegroundColor White
        Write-Host "  without - Switch to version WITHOUT learnable filtering" -ForegroundColor White
        Write-Host "  test    - Run the comprehensive test on current version" -ForegroundColor White
        Write-Host "  status  - Show current version status" -ForegroundColor White
        Write-Host "  both    - Run tests on both versions sequentially" -ForegroundColor White
        Write-Host ""
        Show-Status
    }
} 