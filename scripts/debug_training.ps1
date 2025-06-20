#!/usr/bin/env pwsh

# Debug Training Script
# Tests the training pipeline step by step

Write-Host "=== Training Debug Test ===" -ForegroundColor Green

# Activate virtual environment
if (Test-Path "venv\Scripts\Activate.ps1") {
    & "venv\Scripts\Activate.ps1"
} else {
    Write-Host "Error: Virtual environment not found." -ForegroundColor Red
    exit 1
}

# Set PYTHONPATH
$env:PYTHONPATH = "src;$env:PYTHONPATH"

Write-Host "1. Testing training imports..." -ForegroundColor Yellow
try {
    python -c "from src.core.train_agent import main, make_env; from stable_baselines3 import PPO; print('✅ Training imports successful')"
} catch {
    Write-Host "❌ Training imports failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host "2. Testing environment creation..." -ForegroundColor Yellow
try {
    python -c "
from src.core.train_agent import make_env
from stable_baselines3.common.vec_env import DummyVecEnv

# Test environment creation
env = DummyVecEnv([make_env(max_board_size=4, max_mines=2)])
print('✅ Environment creation successful')
print(f'Environment observation space: {env.observation_space}')
print(f'Environment action space: {env.action_space}')
"
} catch {
    Write-Host "❌ Environment creation failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host "3. Testing model creation..." -ForegroundColor Yellow
try {
    python -c "
from src.core.train_agent import make_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Create environment and model
env = DummyVecEnv([make_env(max_board_size=4, max_mines=2)])
model = PPO('MlpPolicy', env, verbose=0)
print('✅ Model creation successful')
"
} catch {
    Write-Host "❌ Model creation failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host "4. Testing basic training (no callbacks)..." -ForegroundColor Yellow
try {
    python -c "
from src.core.train_agent import make_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Create environment and model
env = DummyVecEnv([make_env(max_board_size=4, max_mines=2)])
model = PPO('MlpPolicy', env, verbose=0)

print('Starting basic training...')
model.learn(total_timesteps=100, progress_bar=False)
print('✅ Basic training completed')
"
} catch {
    Write-Host "❌ Basic training failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host "5. Testing training with iteration callback only..." -ForegroundColor Yellow
try {
    python -c "
from src.core.train_agent import make_env, IterationCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Create environment and model
env = DummyVecEnv([make_env(max_board_size=4, max_mines=2)])
model = PPO('MlpPolicy', env, verbose=0)

# Create iteration callback only
iteration_callback = IterationCallback(verbose=1, debug_level=2)

print('Starting training with iteration callback...')
model.learn(total_timesteps=100, callback=iteration_callback, progress_bar=False)
print('✅ Training with iteration callback completed')
"
} catch {
    Write-Host "❌ Training with iteration callback failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host "6. Testing training with eval callback only..." -ForegroundColor Yellow
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

# Create evaluation callback only
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./debug_best_model',
    log_path='./debug_logs/',
    eval_freq=50,
    n_eval_episodes=3,
    deterministic=True,
    render=False
)

print('Starting training with eval callback...')
model.learn(total_timesteps=100, callback=eval_callback, progress_bar=False)
print('✅ Training with eval callback completed')
"
} catch {
    Write-Host "❌ Training with eval callback failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host "✅ All training tests passed!" -ForegroundColor Green 