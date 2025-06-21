# Test Clean Requirements Script
# This script creates a fresh virtual environment and tests the cleaned requirements.txt

Write-Host "🧪 Testing Clean Requirements Installation..." -ForegroundColor Cyan

# Step 1: Create a fresh test directory
$test_dir = "test_clean_env"
if (Test-Path $test_dir) {
    Write-Host "🗑️  Removing existing test directory..." -ForegroundColor Yellow
    Remove-Item $test_dir -Recurse -Force
}

Write-Host "📁 Creating fresh test directory: $test_dir" -ForegroundColor Green
New-Item -ItemType Directory -Path $test_dir | Out-Null
Set-Location $test_dir

# Step 2: Copy requirements.txt to test directory
Write-Host "📋 Copying requirements.txt to test directory..." -ForegroundColor Green
Copy-Item "../requirements.txt" "."

# Step 3: Create fresh virtual environment
Write-Host "🐍 Creating fresh virtual environment..." -ForegroundColor Green
python -m venv venv_test

# Step 4: Activate virtual environment
Write-Host "🔧 Activating virtual environment..." -ForegroundColor Green
& "./venv_test/Scripts/Activate.ps1"

# Step 5: Upgrade pip
Write-Host "⬆️  Upgrading pip..." -ForegroundColor Green
python -m pip install --upgrade pip

# Step 6: Install requirements
Write-Host "📦 Installing requirements from requirements.txt..." -ForegroundColor Green
pip install -r requirements.txt

# Step 7: Test basic imports
Write-Host "🧪 Testing basic imports..." -ForegroundColor Green
python -c "
import sys
print('✅ Python version:', sys.version)

# Test core dependencies
try:
    import gymnasium
    print('✅ gymnasium imported successfully')
except ImportError as e:
    print('❌ gymnasium import failed:', e)

try:
    import numpy
    print('✅ numpy imported successfully')
except ImportError as e:
    print('❌ numpy import failed:', e)

try:
    import pygame
    print('✅ pygame imported successfully')
except ImportError as e:
    print('❌ pygame import failed:', e)

try:
    import torch
    print('✅ torch imported successfully')
except ImportError as e:
    print('❌ torch import failed:', e)

try:
    import stable_baselines3
    print('✅ stable_baselines3 imported successfully')
except ImportError as e:
    print('❌ stable_baselines3 import failed:', e)

try:
    import mlflow
    print('✅ mlflow imported successfully')
except ImportError as e:
    print('❌ mlflow import failed:', e)

try:
    import matplotlib
    print('✅ matplotlib imported successfully')
except ImportError as e:
    print('❌ matplotlib import failed:', e)

try:
    import pytest
    print('✅ pytest imported successfully')
except ImportError as e:
    print('❌ pytest import failed:', e)

# Test that unused dependencies are NOT installed
unused_deps = ['scipy', 'pandas', 'cloudpickle', 'cv2', 'psutil', 'rich', 'tqdm']
for dep in unused_deps:
    try:
        __import__(dep)
        print(f'⚠️  {dep} is installed but should not be needed')
    except ImportError:
        print(f'✅ {dep} correctly not installed')
"

# Step 8: Test project imports (if we copy the source)
Write-Host "📁 Copying source code for import testing..." -ForegroundColor Green
Copy-Item "../src" "." -Recurse

Write-Host "🧪 Testing project imports..." -ForegroundColor Green
python -c "
import sys
sys.path.append('.')

try:
    from src.core.minesweeper_env import MinesweeperEnv
    print('✅ MinesweeperEnv imported successfully')
except ImportError as e:
    print('❌ MinesweeperEnv import failed:', e)

try:
    from src.core.constants import REWARD_WIN
    print('✅ Constants imported successfully')
except ImportError as e:
    print('❌ Constants import failed:', e)

try:
    from src.core.train_agent import ExperimentTracker
    print('✅ ExperimentTracker imported successfully')
except ImportError as e:
    print('❌ ExperimentTracker import failed:', e)
"

# Step 9: Summary
Write-Host "`n📊 Installation Summary:" -ForegroundColor Cyan
Write-Host "✅ Fresh virtual environment created" -ForegroundColor Green
Write-Host "✅ Requirements installed from cleaned requirements.txt" -ForegroundColor Green
Write-Host "✅ Basic dependency imports tested" -ForegroundColor Green
Write-Host "✅ Project imports tested" -ForegroundColor Green

Write-Host "`n🎉 Clean requirements test completed!" -ForegroundColor Green
Write-Host "If all tests passed, your requirements.txt is clean and complete." -ForegroundColor Cyan

# Step 10: Cleanup option
Write-Host "`n🧹 To clean up the test environment, run: Remove-Item '$test_dir' -Recurse -Force" -ForegroundColor Yellow 