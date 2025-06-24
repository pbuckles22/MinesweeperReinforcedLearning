#!/usr/bin/env python3
"""
Progressive Training - Master One Board at a Time

This approach trains the agent to master each board size completely before moving to the next.
Based on the successful minimal test that achieved 19% win rate.
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

class ProgressiveCallback(BaseCallback):
    """Callback to monitor progressive training progress."""
    
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.wins = 0
        self.total_episodes = 0
        self.current_episode_length = 0
        
    def _on_step(self):
        # Track current episode length
        self.current_episode_length += 1
        
        # Get episode info
        if len(self.training_env.buf_rews) > 0:
            for i, (rew, done, info) in enumerate(zip(self.training_env.buf_rews, self.training_env.buf_dones, self.training_env.buf_infos)):
                if done:
                    self.total_episodes += 1
                    self.episode_rewards.append(rew)
                    self.episode_lengths.append(self.current_episode_length)
                    
                    # Check if it's a win
                    if info.get('won', False):
                        self.wins += 1
                    
                    # Reset episode length for next episode
                    self.current_episode_length = 0
                    
                    # Print progress every 100 episodes
                    if self.total_episodes % 100 == 0:
                        avg_reward = np.mean(self.episode_rewards[-100:])
                        win_rate = self.wins / max(1, self.total_episodes)
                        avg_length = np.mean(self.episode_lengths[-100:])
                        
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

def train_board_size(board_size, max_mines, target_win_rate, max_timesteps=100000):
    """
    Train on a specific board size until target win rate is achieved or max timesteps reached.
    Returns the trained model and final win rate.
    """
    print(f"\n🎯 Training on {board_size[0]}x{board_size[1]} board with {max_mines} mines")
    print(f"   Target win rate: {target_win_rate*100:.0f}%")
    print(f"   Max timesteps: {max_timesteps:,}")
    
    # Force CPU usage
    device = torch.device("cpu")
    
    # Get conservative hyperparameters
    conservative_params = get_conservative_params()
    
    # Create simple environment (no wrappers except action masking)
    def make_env():
        return MinesweeperEnv(max_board_size=board_size, max_mines=max_mines)
    
    env = DummyVecEnv([make_env])
    env = ActionMaskingWrapper(env)
    
    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        **conservative_params,
        tensorboard_log=f"./progressive_logs/{board_size[0]}x{board_size[1]}_{max_mines}mines/",
        device=device
    )
    
    # Create callback
    callback = ProgressiveCallback(verbose=1)
    
    # Train
    print(f"🚀 Starting training...")
    model.learn(
        total_timesteps=max_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    # Evaluate
    print(f"🔍 Evaluating final performance...")
    eval_env = DummyVecEnv([make_env])
    eval_env = ActionMaskingWrapper(eval_env)
    
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
        if (episode + 1) % 20 == 0:
            print(f"   Evaluated {episode + 1}/{n_eval_episodes} episodes...")
    
    final_win_rate = wins / n_eval_episodes
    final_avg_reward = total_reward / n_eval_episodes
    
    print(f"📈 Final Results for {board_size[0]}x{board_size[1]} board:")
    print(f"   Win Rate: {final_win_rate:.3f} ({wins}/{n_eval_episodes})")
    print(f"   Average Reward: {final_avg_reward:.2f}")
    
    if final_win_rate >= target_win_rate:
        print(f"🎉 SUCCESS: Achieved {final_win_rate*100:.1f}% win rate (target: {target_win_rate*100:.0f}%)")
    else:
        print(f"⚠️  Did not reach target win rate ({final_win_rate*100:.1f}% vs {target_win_rate*100:.0f}%)")
    
    env.close()
    eval_env.close()
    
    return model, final_win_rate

def main():
    print("🚀 Starting Progressive Training")
    print("=" * 50)
    print("Approach: Master one board size at a time")
    print("Based on successful minimal test (19% win rate)")
    print("")
    
    # Progressive training stages
    stages = [
        {
            'name': 'Beginner',
            'board_size': (4, 4),
            'max_mines': 2,
            'target_win_rate': 0.15,  # 15% target
            'max_timesteps': 75000    # More time to master
        },
        {
            'name': 'Intermediate',
            'board_size': (4, 4),
            'max_mines': 3,
            'target_win_rate': 0.12,  # 12% target
            'max_timesteps': 75000
        },
        {
            'name': 'Easy',
            'board_size': (6, 6),
            'max_mines': 4,
            'target_win_rate': 0.10,  # 10% target
            'max_timesteps': 100000
        },
        {
            'name': 'Normal',
            'board_size': (6, 6),
            'max_mines': 6,
            'target_win_rate': 0.08,  # 8% target
            'max_timesteps': 100000
        },
        {
            'name': 'Hard',
            'board_size': (8, 8),
            'max_mines': 8,
            'target_win_rate': 0.05,  # 5% target
            'max_timesteps': 150000
        }
    ]
    
    # Train each stage
    results = []
    current_model = None
    
    for i, stage in enumerate(stages):
        print(f"\n{'='*60}")
        print(f"STAGE {i+1}: {stage['name']}")
        print(f"{'='*60}")
        
        try:
            # Train on this board size
            model, win_rate = train_board_size(
                board_size=stage['board_size'],
                max_mines=stage['max_mines'],
                target_win_rate=stage['target_win_rate'],
                max_timesteps=stage['max_timesteps']
            )
            
            # Save results
            results.append({
                'stage': stage['name'],
                'board_size': stage['board_size'],
                'max_mines': stage['max_mines'],
                'target_win_rate': stage['target_win_rate'],
                'achieved_win_rate': win_rate,
                'success': win_rate >= stage['target_win_rate']
            })
            
            # Save model
            model_path = f"models/progressive_{stage['name'].lower()}_model"
            model.save(model_path)
            print(f"💾 Model saved to: {model_path}")
            
            current_model = model
            
            # Check if we should continue
            if win_rate < stage['target_win_rate']:
                print(f"⚠️  Stage {i+1} did not meet target. Stopping progression.")
                break
                
        except KeyboardInterrupt:
            print(f"\n⏹️  Training interrupted at stage {i+1}")
            break
        except Exception as e:
            print(f"\n❌ Error in stage {i+1}: {e}")
            break
    
    # Print final summary
    print(f"\n{'='*60}")
    print(f"PROGRESSIVE TRAINING SUMMARY")
    print(f"{'='*60}")
    
    for result in results:
        status = "✅ PASSED" if result['success'] else "❌ FAILED"
        print(f"{status} {result['stage']}: {result['achieved_win_rate']*100:.1f}% "
              f"({result['board_size'][0]}x{result['board_size'][1]}, {result['max_mines']} mines)")
    
    successful_stages = sum(1 for r in results if r['success'])
    print(f"\n📊 Overall: {successful_stages}/{len(results)} stages completed successfully")

if __name__ == "__main__":
    main() 