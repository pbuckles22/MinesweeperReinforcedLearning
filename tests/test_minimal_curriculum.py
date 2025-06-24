#!/usr/bin/env python3
"""
Minimal Curriculum Test - Remove All Complexity

This test removes all wrappers and complexity to isolate the learning problem.
Matches the successful conservative test exactly, but with curriculum progression.
"""

import os
import sys
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from src.core.minesweeper_env import MinesweeperEnv
from src.core.train_agent import ActionMaskingWrapper

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class MinimalCallback(BaseCallback):
    """Minimal callback to monitor progress without complexity."""
    
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
                    
                    # Print progress every 50 episodes (less frequent)
                    if self.total_episodes % 50 == 0:
                        avg_reward = np.mean(self.episode_rewards[-50:])
                        win_rate = self.wins / max(1, self.total_episodes)
                        avg_length = np.mean(self.episode_lengths[-50:])
                        
                        print(f"Episode {self.total_episodes}: Avg Reward={avg_reward:.2f}, "
                              f"Win Rate={win_rate:.3f}, Avg Length={avg_length:.1f}")
        
        return True

def get_conservative_params():
    """Get the exact hyperparameters from the successful conservative test."""
    return {
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
        'verbose': 0                 # No verbose output
    }

def main():
    print("🚀 Starting Minimal Curriculum Test")
    print("=" * 50)
    print("Key changes:")
    print("   - No MultiBoardTrainingWrapper")
    print("   - No DeterministicTrainingCallback")
    print("   - No complex evaluation callbacks")
    print("   - Simple environment setup")
    print("   - Conservative hyperparameters (from successful test)")
    print("")
    
    # Board configuration (same as successful test)
    board_size = (4, 4)
    max_mines = 2
    
    # Force CPU usage
    device = torch.device("cpu")
    print(f"🎯 Using CPU: {device}")
    
    # Get conservative hyperparameters
    conservative_params = get_conservative_params()
    
    print("📊 Conservative Hyperparameters:")
    for key, value in conservative_params.items():
        print(f"   {key}: {value}")
    
    # Create simple environment (no wrappers except action masking)
    def make_env():
        return MinesweeperEnv(max_board_size=board_size, max_mines=max_mines)
    
    env = DummyVecEnv([make_env])
    
    # Apply ONLY action masking wrapper (like successful test)
    env = ActionMaskingWrapper(env)
    print(f"🔒 Applied action masking wrapper only")
    
    # Create model with conservative parameters
    model = PPO(
        "MlpPolicy",
        env,
        **conservative_params,
        tensorboard_log="./minimal_training_logs/",
        device=device
    )
    
    # Create minimal callback
    callback = MinimalCallback(verbose=1)
    
    print(f"\n🎯 Training for 50,000 timesteps on {board_size[0]}x{board_size[1]} board...")
    print("Expected: Should match the 30% win rate from conservative test")
    
    # Train
    try:
        model.learn(
            total_timesteps=50000,
            callback=callback,
            progress_bar=True
        )
        
        print("\n✅ Training completed!")
        
        # Create evaluation environment (identical to training)
        eval_env = DummyVecEnv([make_env])
        eval_env = ActionMaskingWrapper(eval_env)
        
        # Evaluate
        print("\n🔍 Evaluating final performance...")
        wins = 0
        total_reward = 0
        n_eval_episodes = 100
        
        for episode in range(n_eval_episodes):
            obs = eval_env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                episode_reward += reward[0]
                
                if info[0].get('won', False):
                    wins += 1
            
            total_reward += episode_reward
            
            # Progress indicator
            if (episode + 1) % 10 == 0:
                print(f"   Evaluated {episode + 1}/{n_eval_episodes} episodes...")
        
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
    if 'eval_env' in locals():
        eval_env.close()

if __name__ == "__main__":
    main() 