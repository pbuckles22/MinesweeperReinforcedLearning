import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from minesweeper_env import MinesweeperEnv

# Create logs directory
log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
def make_env():
    env = MinesweeperEnv(board_size=5, num_mines=4)  # Start with a small board
    env = Monitor(env, log_dir)
    return env

# Create vectorized environment
env = DummyVecEnv([make_env])

# Create evaluation environment
eval_env = DummyVecEnv([make_env])

# Create evaluation callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=log_dir,
    log_path=log_dir,
    eval_freq=1000,
    deterministic=True,
    render=False
)

# Initialize the agent
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    tensorboard_log=log_dir
)

# Train the agent
total_timesteps = 100000  # Adjust based on your needs
model.learn(
    total_timesteps=total_timesteps,
    callback=eval_callback,
    progress_bar=True
)

# Save the final model
model.save(os.path.join(log_dir, "final_model"))

# Test the trained agent
obs = env.reset()
for _ in range(100):  # Run 100 test episodes
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    if dones:
        obs = env.reset()

print("Training completed! Model saved in", log_dir) 