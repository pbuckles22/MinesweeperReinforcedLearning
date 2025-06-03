import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from minesweeper_env import MinesweeperEnv
import argparse

def make_env():
    """Create and wrap the Minesweeper environment"""
    env = MinesweeperEnv(board_size=5, num_mines=4)
    return env

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a PPO agent on Minesweeper')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU for training (not recommended for MLP policies)')
    parser.add_argument('--timesteps', type=int, default=100, help='Number of timesteps to train for (default: 100)')
    args = parser.parse_args()

    # Set device based on argument
    device = 'cuda' if args.use_gpu else 'cpu'
    print(f"Using {device} device")

    # Create logs directory
    log_dir = "logs/"
    os.makedirs(log_dir, exist_ok=True)

    # Create and wrap the environment
    env = DummyVecEnv([make_env])

    # Create evaluation environment with Monitor wrapper
    eval_env = DummyVecEnv([lambda: Monitor(make_env(), log_dir)])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=10,  # Reduced for testing
        deterministic=True,
        render=False
    )

    # Initialize the PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=64,  # Reduced for testing
        batch_size=32,  # Reduced for testing
        n_epochs=2,  # Reduced for testing
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=log_dir,
        device=device
    )

    # Train the agent
    model.learn(
        total_timesteps=args.timesteps,
        callback=eval_callback,
        progress_bar=True
    )

    # Save the final model
    model.save(os.path.join(log_dir, "final_model"))

    # Test the trained agent
    obs = env.reset()
    for _ in range(3):  # Reduced number of test episodes
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()

if __name__ == "__main__":
    main() 