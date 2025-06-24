#!/usr/bin/env python3
"""
Modular Minesweeper RL Training Agent

This combines the simplicity of the working approach with the flexibility
of parameter adjustment from the main train_agent.py script.

Key Features:
- Simple, proven training approach (22% win rates achieved)
- Flexible parameter adjustment
- Clean, readable output
- Modular design for easy extension
"""

import os
import sys
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper
from stable_baselines3.common.callbacks import BaseCallback
import argparse
from datetime import datetime
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.minesweeper_env import MinesweeperEnv

class ActionMaskingWrapper(VecEnvWrapper):
    """Wrapper that enforces action masking to prevent invalid actions."""
    
    def __init__(self, venv):
        super().__init__(venv)
        self.action_space = venv.action_space
    
    def reset(self):
        return self.venv.reset()
    
    def step(self, actions):
        # Get action masks from the environment
        action_masks = []
        for env in self.venv.envs:
            if hasattr(env, 'get_action_mask'):
                mask = env.get_action_mask()
            elif hasattr(env, 'action_masks'):
                mask = env.action_masks
            else:
                # If no action mask available, allow all actions
                mask = np.ones(self.action_space.n, dtype=bool)
            action_masks.append(mask)
        
        # Convert to numpy array
        action_masks = np.array(action_masks)
        
        # Mask out invalid actions by setting their probability to 0
        # For now, we'll replace invalid actions with random valid actions
        masked_actions = actions.copy()
        
        for i, (action, mask) in enumerate(zip(actions, action_masks)):
            if not mask[action]:
                # Find a valid action
                valid_actions = np.where(mask)[0]
                if len(valid_actions) > 0:
                    masked_actions[i] = np.random.choice(valid_actions)
                else:
                    # If no valid actions, keep the original (will be penalized)
                    pass
        
        # Take the step with masked actions
        return self.venv.step(masked_actions)
    
    def step_wait(self):
        """Required by VecEnvWrapper, but not used in this wrapper."""
        return self.venv.step_wait()
    
    def get_action_mask(self):
        """Get action mask for the current state."""
        masks = []
        for env in self.venv.envs:
            if hasattr(env, 'get_action_mask'):
                mask = env.get_action_mask()
            elif hasattr(env, 'action_masks'):
                mask = env.action_masks
            else:
                mask = np.ones(self.action_space.n, dtype=bool)
            masks.append(mask)
        return np.array(masks)

class ModularProgressCallback(BaseCallback):
    """Simple callback to monitor training progress with flexible parameters."""
    
    def __init__(self, verbose=1, board_size=None, max_mines=None):
        super().__init__(verbose)
        self.total_episodes = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.wins = 0
        self.current_episode_length = 0
        self.board_size = board_size
        self.max_mines = max_mines
        
    def _on_step(self):
        # Track current episode length
        self.current_episode_length += 1
        
        # Get episode info from the model's environment
        if hasattr(self.model, 'get_env'):
            training_env = self.model.get_env()
            if hasattr(training_env, 'buf_rews') and len(training_env.buf_rews) > 0:
                for i, (rew, done, info) in enumerate(zip(training_env.buf_rews, training_env.buf_dones, training_env.buf_infos)):
                    if done:
                        self.total_episodes += 1
                        self.episode_rewards.append(rew)
                        self.episode_lengths.append(self.current_episode_length)
                        
                        # Check if episode was won
                        if info.get('won', False):
                            self.wins += 1
                        
                        self.current_episode_length = 0
        
        # Log progress every 100 episodes
        if self.total_episodes % 100 == 0 and self.total_episodes > 0:
            avg_reward = np.mean(self.episode_rewards[-100:])
            win_rate = self.wins / self.total_episodes
            avg_length = np.mean(self.episode_lengths[-100:])
            
            # Display compact single-line format
            board_info = f"Board={self.board_size}x{self.board_size}x{self.max_mines}" if self.board_size and self.max_mines else ""
            compact_line = f"Episode {self.total_episodes}: Avg Reward={avg_reward:.2f}, Win Rate={win_rate:.3f}, Avg Length={avg_length:.1f}"
            if board_info:
                compact_line += f", {board_info}"
            
            print(compact_line)
        
        return True

def make_modular_env(board_size, max_mines):
    """Create a modular environment with flexible parameters."""
    def _init():
        env = MinesweeperEnv(
            max_board_size=board_size,
            max_mines=max_mines,
            render_mode=None,
            early_learning_mode=False,
            early_learning_threshold=200,
            early_learning_corner_safe=False,
            early_learning_edge_safe=False,
            mine_spacing=1,
            initial_board_size=board_size,
            initial_mines=max_mines
        )
        return env
    
    return _init

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def get_conservative_params():
    """Get conservative hyperparameters that work well."""
    return {
        'learning_rate': 1e-4,      # Conservative learning rate
        'n_steps': 1024,            # Smaller steps for more frequent updates
        'batch_size': 32,           # Smaller batch size
        'n_epochs': 15,             # More epochs for better learning
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.1,          # Tighter clipping
        'ent_coef': 0.005,          # Lower entropy for focused learning
        'vf_coef': 0.5,
        'max_grad_norm': 0.3,       # Lower gradient norm
    }

def train_modular(board_size, max_mines, total_timesteps, device="cpu", **kwargs):
    """
    Modular training function with flexible parameters.
    
    Args:
        board_size: Board size (height, width)
        max_mines: Maximum number of mines
        total_timesteps: Total training timesteps
        device: Training device (cpu, cuda, mps)
        **kwargs: Additional PPO parameters to override defaults
    """
    print(f"🚀 Starting Modular Training")
    print(f"==================================================")
    print(f"Board: {board_size}x{board_size} with {max_mines} mines")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Device: {device}")
    print(f"")
    
    # Create environment
    env = DummyVecEnv([make_modular_env(board_size, max_mines)])
    env = ActionMaskingWrapper(env)
    
    # Get hyperparameters (conservative defaults + overrides)
    params = get_conservative_params()
    params.update(kwargs)  # Allow parameter overrides
    
    print(f"📊 Hyperparameters:")
    for key, value in params.items():
        print(f"   {key}: {value}")
    print(f"")
    
    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        device=device,
        **params
    )
    
    # Create callback
    callback = ModularProgressCallback(
        verbose=1,
        board_size=board_size,
        max_mines=max_mines
    )
    
    # Train
    print(f"🚀 Starting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    # Evaluate
    print(f"🔍 Evaluating final performance...")
    rewards = []
    wins = 0
    n_eval_episodes = 100
    
    for episode in range(n_eval_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_won = False
        
        while not done:
            action = model.predict(obs, deterministic=True)[0]
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            
            if info and isinstance(info, list) and len(info) > 0:
                if info[0].get('won', False):
                    episode_won = True
            elif info and isinstance(info, dict):
                if info.get('won', False):
                    episode_won = True
        
        rewards.append(episode_reward)
        if episode_won:
            wins += 1
        
        if (episode + 1) % 20 == 0:
            print(f"   Evaluated {episode + 1}/{n_eval_episodes} episodes...")
    
    # Calculate final metrics
    mean_reward = np.mean(rewards)
    win_rate = wins / n_eval_episodes
    
    print(f"📈 Final Results for {board_size}x{board_size} board:")
    print(f"   Win Rate: {win_rate:.3f} ({wins}/{n_eval_episodes})")
    print(f"   Average Reward: {mean_reward:.2f}")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/modular_model_{board_size}x{board_size}x{max_mines}_{timestamp}"
    model.save(model_path)
    print(f"💾 Model saved to: {model_path}")
    
    # Save results
    results = {
        'board_size': board_size,
        'max_mines': max_mines,
        'total_timesteps': total_timesteps,
        'device': device,
        'hyperparameters': params,
        'final_win_rate': float(win_rate),  # Convert numpy types to native Python
        'final_mean_reward': float(mean_reward),
        'wins': int(wins),
        'total_episodes': n_eval_episodes,
        'timestamp': timestamp
    }
    
    results_dir = Path("experiments")
    results_dir.mkdir(exist_ok=True)
    results_filename = f"modular_results_{timestamp}.json"
    results_path = results_dir / results_filename
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert_numpy_types)
    print(f"📊 Results saved to: {results_path}")
    
    return model, win_rate, mean_reward

def parse_args():
    parser = argparse.ArgumentParser(description='Modular Minesweeper RL Training')
    parser.add_argument('--board_size', type=int, default=4, help='Board size (height=width)')
    parser.add_argument('--max_mines', type=int, default=2, help='Maximum number of mines')
    parser.add_argument('--total_timesteps', type=int, default=10000, help='Total training timesteps')
    parser.add_argument('--device', type=str, default='cpu', help='Training device (cpu, cuda, mps)')
    
    # PPO hyperparameters (optional overrides)
    parser.add_argument('--learning_rate', type=float, help='Learning rate override')
    parser.add_argument('--batch_size', type=int, help='Batch size override')
    parser.add_argument('--n_steps', type=int, help='Steps per update override')
    parser.add_argument('--n_epochs', type=int, help='Epochs override')
    parser.add_argument('--ent_coef', type=float, help='Entropy coefficient override')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Build kwargs for parameter overrides
    kwargs = {}
    if args.learning_rate is not None:
        kwargs['learning_rate'] = args.learning_rate
    if args.batch_size is not None:
        kwargs['batch_size'] = args.batch_size
    if args.n_steps is not None:
        kwargs['n_steps'] = args.n_steps
    if args.n_epochs is not None:
        kwargs['n_epochs'] = args.n_epochs
    if args.ent_coef is not None:
        kwargs['ent_coef'] = args.ent_coef
    
    # Train
    model, win_rate, mean_reward = train_modular(
        board_size=args.board_size,
        max_mines=args.max_mines,
        total_timesteps=args.total_timesteps,
        device=args.device,
        **kwargs
    )
    
    print(f"🎉 Training completed! Final win rate: {win_rate:.1%}")

if __name__ == "__main__":
    main() 