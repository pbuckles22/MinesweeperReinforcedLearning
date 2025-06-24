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
- M1 GPU acceleration with MPS
- Action masking to prevent invalid actions
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
from src.core.train_agent import ActionMaskingWrapper

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def detect_device(board_size=(4, 4)):
    """Detect the best available device for training based on board size."""
    board_area = board_size[0] * board_size[1]
    
    # For small boards (4x4, 6x6), CPU is often faster due to MPS overhead
    if board_area <= 36:  # 6x6 or smaller
        device = torch.device("cpu")
        print(f"🎯 Using CPU for small board ({board_size[0]}x{board_size[1]}): GPU overhead not worth it")
        return device, "cpu"
    
    # For medium boards (9x9), check GPU availability
    elif board_area <= 81:  # 9x9
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f"🚀 Using M1 GPU (MPS) for medium board ({board_size[0]}x{board_size[1]}): {device}")
            return device, "mps"
        else:
            device = torch.device("cpu")
            print(f"⚠️  Using CPU for medium board ({board_size[0]}x{board_size[1]}): No GPU available")
            return device, "cpu"
    
    # For large boards (16x16+), definitely use GPU
    else:  # 16x16 or larger
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f"🚀 Using M1 GPU (MPS) for large board ({board_size[0]}x{board_size[1]}): {device}")
            return device, "mps"
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"🚀 Using CUDA GPU for large board ({board_size[0]}x{board_size[1]}): {device}")
            return device, "cuda"
        else:
            device = torch.device("cpu")
            print(f"⚠️  Using CPU for large board ({board_size[0]}x{board_size[1]}): No GPU available")
            return device, "cpu"

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

def get_device_optimized_params(device_type):
    """Get hyperparameters optimized for the detected device."""
    base_params = {
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
    
    if device_type == "mps":
        # M1 GPU optimizations
        optimized_params = base_params.copy()
        optimized_params.update({
            'batch_size': 64,       # M1 can handle larger batches
            'n_steps': 2048,        # More steps for M1 efficiency
            'n_epochs': 12,         # Optimized for M1
        })
        print("🔧 Applied M1 GPU optimizations")
    elif device_type == "cuda":
        # CUDA GPU optimizations
        optimized_params = base_params.copy()
        optimized_params.update({
            'batch_size': 128,      # CUDA can handle very large batches
            'n_steps': 2048,        # More steps for CUDA efficiency
            'n_epochs': 10,         # Optimized for CUDA
        })
        print("🔧 Applied CUDA GPU optimizations")
    else:
        # CPU optimizations
        optimized_params = base_params.copy()
        print("🔧 Using CPU-optimized parameters")
    
    return optimized_params

def main():
    print("🚀 Starting Conservative Training Test")
    print("=" * 50)
    
    # Board configuration
    board_size = (4, 4)
    max_mines = 2
    
    # Detect device based on board size
    device, device_type = detect_device(board_size)
    
    # Get device-optimized hyperparameters
    conservative_params = get_device_optimized_params(device_type)
    
    print("📊 Conservative Hyperparameters:")
    for key, value in conservative_params.items():
        print(f"   {key}: {value}")
    
    # Create environment
    def make_env():
        return MinesweeperEnv(max_board_size=board_size, max_mines=max_mines)
    
    env = DummyVecEnv([make_env])
    
    # Apply action masking to prevent invalid actions
    env = ActionMaskingWrapper(env)
    
    # Create model with conservative parameters and device specification
    model = PPO(
        "MlpPolicy",
        env,
        **conservative_params,
        tensorboard_log="./conservative_training_logs/",
        device=device
    )
    
    # Create callback
    callback = ConservativeTrainingCallback(verbose=1)
    
    print(f"\n🎯 Training for 50,000 timesteps on {board_size[0]}x{board_size[1]} board with {device_type.upper()}...")
    print("Expected improvements:")
    print("   - More stable learning")
    print("   - Gradual improvement in win rate")
    print("   - Better average rewards")
    print(f"   - Optimized for {board_size[0]}x{board_size[1]} board size")
    
    # Train
    try:
        model.learn(
            total_timesteps=50000,
            callback=callback,
            progress_bar=True
        )
        
        print("\n✅ Training completed!")
        
        # Create evaluation environment with action masking
        def make_eval_env():
            return MinesweeperEnv(max_board_size=board_size, max_mines=max_mines)
        
        eval_env = DummyVecEnv([make_eval_env])
        eval_env = ActionMaskingWrapper(eval_env)
        
        # Evaluate
        print("\n🔍 Evaluating final performance...")
        wins = 0
        total_reward = 0
        n_eval_episodes = 100  # Back to full evaluation
        
        for episode in range(n_eval_episodes):
            try:
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
                    
            except Exception as e:
                print(f"❌ Error in episode {episode}: {e}")
                continue
        
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