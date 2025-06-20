#!/usr/bin/env pwsh

# Debug Environment Script
# Tests the Minesweeper environment step by step

Write-Host "=== Environment Debug Test ===" -ForegroundColor Green

# Activate virtual environment
if (Test-Path "venv\Scripts\Activate.ps1") {
    & "venv\Scripts\Activate.ps1"
} else {
    Write-Host "Error: Virtual environment not found." -ForegroundColor Red
    exit 1
}

# Set PYTHONPATH
$env:PYTHONPATH = "src;$env:PYTHONPATH"

Write-Host "1. Testing environment import..." -ForegroundColor Yellow
try {
    python -c "from src.core.minesweeper_env import MinesweeperEnv; print('✅ Import successful')"
} catch {
    Write-Host "❌ Import failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host "2. Testing environment creation..." -ForegroundColor Yellow
try {
    python -c "from src.core.minesweeper_env import MinesweeperEnv; env = MinesweeperEnv(initial_board_size=4, initial_mines=2); print('✅ Environment created')"
} catch {
    Write-Host "❌ Environment creation failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host "3. Testing environment reset..." -ForegroundColor Yellow
try {
    python -c "from src.core.minesweeper_env import MinesweeperEnv; env = MinesweeperEnv(initial_board_size=4, initial_mines=2); state, info = env.reset(seed=42); print('✅ Reset successful'); print(f'State shape: {state[0].shape}'); print(f'Info: {info}')"
} catch {
    Write-Host "❌ Reset failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host "4. Testing environment step..." -ForegroundColor Yellow
try {
    python -c "from src.core.minesweeper_env import MinesweeperEnv; env = MinesweeperEnv(initial_board_size=4, initial_mines=2); state, info = env.reset(seed=42); next_state, reward, terminated, truncated, info = env.step(0); print('✅ Step successful'); print(f'Reward: {reward}'); print(f'Terminated: {terminated}'); print(f'Info: {info}')"
} catch {
    Write-Host "❌ Step failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host "5. Testing vectorized environment..." -ForegroundColor Yellow
try {
    python -c "from src.core.train_agent import make_env; from stable_baselines3.common.vec_env import DummyVecEnv; env = DummyVecEnv([make_env(max_board_size=4, max_mines=2)]); obs = env.reset(); print('✅ Vectorized environment created'); print(f'Observation shape: {obs.shape}')"
} catch {
    Write-Host "❌ Vectorized environment failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host "6. Testing vectorized environment step..." -ForegroundColor Yellow
try {
    python -c "from src.core.train_agent import make_env; from stable_baselines3.common.vec_env import DummyVecEnv; import numpy as np; env = DummyVecEnv([make_env(max_board_size=4, max_mines=2)]); obs = env.reset(); action = np.array([0]); obs, reward, terminated, truncated, info = env.step(action); print('✅ Vectorized step successful'); print(f'Reward: {reward}'); print(f'Info: {info}')"
} catch {
    Write-Host "❌ Vectorized step failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host "✅ All environment tests passed!" -ForegroundColor Green 