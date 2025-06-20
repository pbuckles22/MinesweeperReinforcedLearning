#!/usr/bin/env pwsh

# Custom EvalCallback Debug Script
# Tests with a custom evaluation callback that handles our environment properly

Write-Host "=== Custom EvalCallback Debug Test ===" -ForegroundColor Green

# Activate virtual environment
if (Test-Path "venv\Scripts\Activate.ps1") {
    & "venv\Scripts\Activate.ps1"
} else {
    Write-Host "Error: Virtual environment not found." -ForegroundColor Red
    exit 1
}

# Set PYTHONPATH
$env:PYTHONPATH = "src;$env:PYTHONPATH"

Write-Host "Testing custom EvalCallback..." -ForegroundColor Yellow

# Create a Python script with a custom evaluation callback
$testScript = @"
import sys
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

print('Starting custom EvalCallback test...')

try:
    from src.core.train_agent import make_env
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    print('1. Imports successful')
    
    # Create a custom evaluation callback that doesn't hang
    class CustomEvalCallback(BaseCallback):
        def __init__(self, eval_env, eval_freq=1000, n_eval_episodes=5, verbose=1):
            super().__init__(verbose)
            self.eval_env = eval_env
            self.eval_freq = eval_freq
            self.n_eval_episodes = n_eval_episodes
            self.best_mean_reward = -np.inf
            
        def _on_step(self):
            if self.n_calls % self.eval_freq == 0:
                print(f'Evaluating at step {self.n_calls}...')
                
                # Simple evaluation without accessing problematic attributes
                rewards = []
                for _ in range(self.n_eval_episodes):
                    obs = self.eval_env.reset()
                    done = False
                    episode_reward = 0
                    
                    while not done:
                        action = self.model.predict(obs, deterministic=True)[0]
                        obs, reward, done, info = self.eval_env.step(action)
                        episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
                    
                    rewards.append(episode_reward)
                
                mean_reward = np.mean(rewards)
                print(f'Evaluation completed. Mean reward: {mean_reward:.2f}')
                
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    print(f'New best mean reward: {self.best_mean_reward:.2f}')
                
            return True
    
    # Create environment
    env = DummyVecEnv([make_env(max_board_size=4, max_mines=2)])
    eval_env = DummyVecEnv([make_env(max_board_size=4, max_mines=2)])
    print('2. Environments created')
    
    # Create model
    model = PPO('MlpPolicy', env, verbose=0)
    print('3. Model created')
    
    # Create custom evaluation callback
    print('4. Creating custom EvalCallback...')
    eval_callback = CustomEvalCallback(
        eval_env,
        eval_freq=25,
        n_eval_episodes=2,
        verbose=1
    )
    print('5. Custom EvalCallback created')
    
    # Test training with custom EvalCallback
    print('6. Starting training with custom EvalCallback...')
    model.learn(total_timesteps=100, callback=eval_callback, progress_bar=False)
    print('7. Training with custom EvalCallback completed')
    
    print('✅ Custom EvalCallback test completed!')
    
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"@

# Write the script to a temporary file
$testScript | Out-File -FilePath "debug_custom_eval.py" -Encoding UTF8

# Run the test
try {
    python debug_custom_eval.py
    Write-Host "✅ Custom EvalCallback test completed!" -ForegroundColor Green
} catch {
    Write-Host "❌ Custom EvalCallback test failed: $_" -ForegroundColor Red
} finally {
    # Clean up
    if (Test-Path "debug_custom_eval.py") {
        Remove-Item "debug_custom_eval.py"
    }
} 