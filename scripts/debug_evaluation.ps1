#!/usr/bin/env pwsh

# Debug Evaluation Script
# Tests the evaluation function step by step

Write-Host "=== Evaluation Debug Test ===" -ForegroundColor Green

# Activate virtual environment
if (Test-Path "venv\Scripts\Activate.ps1") {
    & "venv\Scripts\Activate.ps1"
} else {
    Write-Host "Error: Virtual environment not found." -ForegroundColor Red
    exit 1
}

# Set PYTHONPATH
$env:PYTHONPATH = "src;$env:PYTHONPATH"

Write-Host "1. Testing evaluation function import..." -ForegroundColor Yellow
try {
    python -c "from src.core.train_agent import evaluate_model; print('✅ Evaluation function imported')"
} catch {
    Write-Host "❌ Evaluation function import failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host "2. Testing evaluation with single environment..." -ForegroundColor Yellow
try {
    python -c "from src.core.train_agent import evaluate_model, make_env; from stable_baselines3 import PPO; from stable_baselines3.common.vec_env import DummyVecEnv;

# Create environment
env = DummyVecEnv([make_env(max_board_size=4, max_mines=2)]);

# Create a simple model
model = PPO('MlpPolicy', env, verbose=0);

# Test evaluation
print('Starting evaluation...');
results = evaluate_model(model, env, n_episodes=5);
print('✅ Evaluation completed');
print(f'Results: {results}');"
} catch {
    Write-Host "❌ Single environment evaluation failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host "3. Testing evaluation with multiple episodes..." -ForegroundColor Yellow
try {
    python -c "
from src.core.train_agent import evaluate_model, make_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Create environment
env = DummyVecEnv([make_env(max_board_size=4, max_mines=2)])

# Create a simple model
model = PPO('MlpPolicy', env, verbose=0)

# Test evaluation with more episodes
print('Starting multi-episode evaluation...')
results = evaluate_model(model, env, n_episodes=10)
print('✅ Multi-episode evaluation completed')
print(f'Win rate: {results[\"win_rate\"]}%')
print(f'Avg reward: {results[\"avg_reward\"]}')
"
} catch {
    Write-Host "❌ Multi-episode evaluation failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host "4. Testing evaluation callback..." -ForegroundColor Yellow
try {
    python -c "
from src.core.train_agent import make_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

# Create environments
env = DummyVecEnv([make_env(max_board_size=4, max_mines=2)])
eval_env = DummyVecEnv([make_env(max_board_size=4, max_mines=2)])

# Create model
model = PPO('MlpPolicy', env, verbose=0)

# Create evaluation callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./debug_best_model',
    log_path='./debug_logs/',
    eval_freq=100,
    n_eval_episodes=5,
    deterministic=True,
    render=False
)

print('Starting training with evaluation callback...')
model.learn(total_timesteps=200, callback=eval_callback, progress_bar=False)
print('✅ Training with evaluation callback completed')
"
} catch {
    Write-Host "❌ Evaluation callback test failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host "✅ All evaluation tests passed!" -ForegroundColor Green 