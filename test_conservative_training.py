#!/usr/bin/env python3
"""
Conservative Training Test - Fix for Performance Degradation

This script runs training with more conservative hyperparameters to address
the issue where agent performance gets worse during training.

Key Changes:
- Lower learning rate (1e-4 instead of 3e-4)
- Smaller batch size (32 instead of 128)
- More epochs (15 instead of 12)
- Lower entropy coefficient (0.005 instead of 0.01)
"""

import os
import sys
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from src.core.minesweeper_env import MinesweeperEnv
from src.core.constants import REWARD_INVALID_ACTION, REWARD_HIT_MINE, REWARD_SAFE_REVEAL, REWARD_WIN

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class ConservativeTrainingCallback(BaseCallback):
    """Callback to monitor conservative training progress."""
    
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.wins = 0
        self.total_episodes = 0
        
    def _on_step(self):
        # Get episode info
        if len(self.training_env.buf_rews) > 0:
            for i, (rew, done, info) in enumerate(zip(self.training_env.buf_rews, self.training_env.buf_dones, self.training_env.buf_infos)):
                if done:
                    self.total_episodes += 1
                    self.episode_rewards.append(rew)
                    self.episode_lengths.append(info.get('episode', {}).get('l', 0))
                    
                    # Check if it's a win
                    if info.get('won', False):
                        self.wins += 1
                    
                    # Print progress every 100 episodes
                    if self.total_episodes % 100 == 0:
                        avg_reward = np.mean(self.episode_rewards[-100:])
                        win_rate = self.wins / max(1, self.total_episodes)
                        avg_length = np.mean(self.episode_lengths[-100:])
                        
                        print(f"Episode {self.total_episodes}: Avg Reward={avg_reward:.2f}, "
                              f"Win Rate={win_rate:.3f}, Avg Length={avg_length:.1f}")
        
        return True

def make_conservative_env():
    """Create environment with conservative settings."""
    def _init():
        env = MinesweeperEnv(max_board_size=(4, 4), max_mines=2)
        return env
    return _init

def main():
    print("🚀 Starting Conservative Training Test")
    print("=" * 50)
    
    # Conservative hyperparameters
    conservative_params = {
        'learning_rate': 1e-4,      # Much lower learning rate
        'n_steps': 1024,            # Smaller steps for more frequent updates
        'batch_size': 32,           # Smaller batch size
        'n_epochs': 15,             # More epochs for better learning
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.1,          # Tighter clipping
        'ent_coef': 0.005,          # Lower entropy for more focused learning
        'vf_coef': 0.5,
        'max_grad_norm': 0.3,       # Lower gradient norm
        'verbose': 1
    }
    
    print("📊 Conservative Hyperparameters:")
    for key, value in conservative_params.items():
        print(f"   {key}: {value}")
    
    # Create environment
    env = DummyVecEnv([make_conservative_env()])
    
    # Create model with conservative parameters
    model = PPO(
        "MlpPolicy",
        env,
        **conservative_params,
        tensorboard_log="./conservative_training_logs/"
    )
    
    # Create callback
    callback = ConservativeTrainingCallback(verbose=1)
    
    print("\n🎯 Training for 50,000 timesteps with conservative parameters...")
    print("Expected improvements:")
    print("   - More stable learning")
    print("   - Gradual improvement in win rate")
    print("   - Better average rewards")
    
    # Train
    try:
        model.learn(
            total_timesteps=50000,
            callback=callback,
            progress_bar=True
        )
        
        print("\n✅ Training completed!")
        
        # Evaluate
        print("\n🔍 Evaluating final performance...")
        wins = 0
        total_reward = 0
        n_eval_episodes = 100
        
        for episode in range(n_eval_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward[0]
                
                if info[0].get('won', False):
                    wins += 1
            
            total_reward += episode_reward
        
        final_win_rate = wins / n_eval_episodes
        final_avg_reward = total_reward / n_eval_episodes
        
        print(f"\n📈 Final Results:")
        print(f"   Win Rate: {final_win_rate:.3f} ({wins}/{n_eval_episodes})")
        print(f"   Average Reward: {final_avg_reward:.2f}")
        
        if final_win_rate >= 0.15:
            print("🎉 SUCCESS: Agent achieved 15% win rate target!")
        else:
            print("⚠️  Agent needs more training time")
            
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
    
    env.close()

if __name__ == "__main__":
    main() 