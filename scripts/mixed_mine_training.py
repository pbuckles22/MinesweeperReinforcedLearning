#!/usr/bin/env python3
"""
Mixed Mine Count Training with Experience Replay for Minesweeper RL

Trains the agent on multiple mine counts simultaneously using experience replay
to prevent catastrophic forgetting and maintain skills across difficulty levels.
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.train_agent_modular import make_modular_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


def convert_to_json_serializable(obj):
    """Convert numpy types and other non-serializable objects to JSON-compatible types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj


class MixedMineCallback(BaseCallback):
    """Callback to track performance across different mine counts during training."""
    
    def __init__(self, eval_freq=10000, n_eval_episodes=50, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_results = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Evaluate on different mine counts
            mine_counts = [1, 2, 3, 4, 5]
            results = {}
            
            for mine_count in mine_counts:
                win_rate = self._evaluate_mine_count(mine_count)
                results[mine_count] = win_rate
                
            self.eval_results.append({
                'timestep': self.n_calls,
                'results': results
            })
            
            if self.verbose > 0:
                print(f"\n📊 Evaluation at {self.n_calls:,} timesteps:")
                for mine_count, win_rate in results.items():
                    print(f"   {mine_count} mines: {win_rate:.1%}")
                    
        return True
    
    def _evaluate_mine_count(self, mine_count):
        """Evaluate model on specific mine count."""
        env = DummyVecEnv([make_modular_env((5, 5), mine_count)])
        wins = 0
        
        for _ in range(self.n_eval_episodes):
            obs = env.reset()
            done = False
            step_count = 0
            max_steps = 200
            
            while not done and step_count < max_steps:
                action = self.model.predict(obs, deterministic=True)[0]
                obs, reward, done, info = env.step(action)
                step_count += 1
                
                # Check for win
                if info and isinstance(info, list) and len(info) > 0:
                    if info[0].get('won', False):
                        wins += 1
                        break
                elif info and isinstance(info, dict):
                    if info.get('won', False):
                        wins += 1
                        break
                        
        return wins / self.n_eval_episodes


def create_mixed_env(board_size, mine_counts, n_envs_per_mine=2):
    """Create a vectorized environment with multiple mine counts."""
    envs = []
    
    for mine_count in mine_counts:
        for _ in range(n_envs_per_mine):
            envs.append(make_modular_env(board_size, mine_count))
    
    return SubprocVecEnv(envs) if len(envs) > 1 else DummyVecEnv(envs)


def evaluate_mixed_model(model, board_size, mine_counts, n_episodes=100):
    """Evaluate model across different mine counts."""
    print(f"🔍 Evaluating on {board_size[0]}x{board_size[1]} with mine counts: {mine_counts}")
    
    results = {}
    total_wins = 0
    total_episodes = 0
    
    for mine_count in mine_counts:
        print(f"   Testing with {mine_count} mines...")
        
        env = DummyVecEnv([make_modular_env(board_size, mine_count)])
        wins = 0
        rewards = []
        
        for episode in range(n_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            step_count = 0
            max_steps = 200
            
            while not done and step_count < max_steps:
                try:
                    action = model.predict(obs, deterministic=True)[0]
                    obs, reward, done, info = env.step(action)
                    episode_reward += reward[0] if isinstance(reward, list) else reward
                    step_count += 1
                    
                    # Check for win condition
                    if info and isinstance(info, list) and len(info) > 0:
                        if info[0].get('won', False):
                            wins += 1
                            break
                    elif info and isinstance(info, dict):
                        if info.get('won', False):
                            wins += 1
                            break
                            
                except Exception as e:
                    print(f"⚠️  Error in episode {episode}, step {step_count}: {e}")
                    break
            
            rewards.append(episode_reward)
        
        win_rate = wins / n_episodes
        mean_reward = np.mean(rewards)
        
        results[mine_count] = {
            "win_rate": float(win_rate),
            "mean_reward": float(mean_reward),
            "wins": int(wins),
            "total_episodes": int(n_episodes)
        }
        
        total_wins += wins
        total_episodes += n_episodes
        
        print(f"     {mine_count} mines: {win_rate:.1%} win rate, {mean_reward:.1f} avg reward")
    
    overall_win_rate = total_wins / total_episodes
    print(f"📊 Overall performance: {overall_win_rate:.1%} win rate")
    
    return results, overall_win_rate


def mixed_mine_training():
    """Train agent with mixed mine counts and experience replay."""
    
    print("🎯 Mixed Mine Count Training with Experience Replay")
    print("=" * 60)
    
    # Training configuration
    board_size = (5, 5)
    
    # Training phases with mixed mine counts
    training_phases = [
        # Phase 1: Learn basics (1-2 mines mixed)
        {"mine_counts": [1, 2], "timesteps": 300000, "target_win_rate": 0.60},
        
        # Phase 2: Expand to 1-3 mines mixed
        {"mine_counts": [1, 2, 3], "timesteps": 400000, "target_win_rate": 0.50},
        
        # Phase 3: Expand to 1-4 mines mixed
        {"mine_counts": [1, 2, 3, 4], "timesteps": 500000, "target_win_rate": 0.40},
        
        # Phase 4: Master 1-5 mines mixed
        {"mine_counts": [1, 2, 3, 4, 5], "timesteps": 600000, "target_win_rate": 0.35},
    ]
    
    results = []
    current_model = None
    
    for phase_num, phase_config in enumerate(training_phases, 1):
        mine_counts = phase_config["mine_counts"]
        timesteps = phase_config["timesteps"]
        target_win_rate = phase_config["target_win_rate"]
        
        print(f"\n🎯 Phase {phase_num}: Mixed training on {mine_counts}")
        print(f"   Board: {board_size[0]}x{board_size[1]}")
        print(f"   Mine Counts: {mine_counts}")
        print(f"   Timesteps: {timesteps:,}")
        print(f"   Target Win Rate: {target_win_rate:.1%}")
        print("-" * 60)
        
        try:
            # Create mixed environment with multiple mine counts
            env = create_mixed_env(board_size, mine_counts, n_envs_per_mine=3)
            
            if current_model is None:
                # First phase: train from scratch
                print(f"🚀 Starting fresh mixed training...")
                current_model = PPO(
                    "MlpPolicy",
                    env,
                    verbose=1,
                    learning_rate=0.0003,
                    batch_size=64,
                    n_steps=1024,
                    n_epochs=15,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.1,
                    ent_coef=0.01,
                    vf_coef=0.5,
                    max_grad_norm=0.3,
                    device="cpu"
                )
            else:
                # Continue training with existing model
                print(f"🔄 Continuing mixed training...")
                current_model.set_env(env)
            
            # Create callback for monitoring
            callback = MixedMineCallback(eval_freq=50000, n_eval_episodes=30)
            
            # Train the model
            current_model.learn(
                total_timesteps=timesteps,
                progress_bar=True,
                callback=callback
            )
            
            # Evaluate performance across all mine counts
            mine_results, overall_win_rate = evaluate_mixed_model(
                current_model, board_size, mine_counts, n_episodes=50
            )
            
            # Save phase results
            result = {
                "phase": phase_num,
                "board_size": board_size,
                "mine_counts": mine_counts,
                "total_timesteps": timesteps,
                "overall_win_rate": float(overall_win_rate),
                "target_win_rate": target_win_rate,
                "mine_results": mine_results,
                "eval_history": callback.eval_results,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            
            results.append(result)
            
            print(f"📊 Phase {phase_num} Results:")
            print(f"   Overall Win Rate: {overall_win_rate:.3f}")
            print(f"   Target: {target_win_rate:.1%}")
            
            # Check if we should continue
            if overall_win_rate < target_win_rate:
                print(f"⚠️  Performance below target ({overall_win_rate:.1%} < {target_win_rate:.1%}).")
                response = input("Continue to next phase anyway? (y/n): ").lower().strip()
                if response != 'y':
                    print("Stopping training.")
                    break
            else:
                print(f"✅ Target achieved! ({overall_win_rate:.1%} >= {target_win_rate:.1%})")
                
        except Exception as e:
            print(f"❌ Error in phase {phase_num}: {e}")
            break
    
    # Save training results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"experiments/mixed_mine_training_results_{timestamp}.json"
    
    # Ensure experiments directory exists
    os.makedirs("experiments", exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(convert_to_json_serializable({
            "training_summary": {
                "total_phases": len(training_phases),
                "completed_phases": len(results),
                "timestamp": timestamp,
                "approach": "mixed_mine_count_with_experience_replay"
            },
            "results": results
        }), f, indent=2)
    
    print(f"\n💾 Training results saved to: {results_file}")
    
    # Print final summary
    print(f"\n📈 Mixed Mine Training Summary")
    print("=" * 60)
    for result in results:
        phase = result["phase"]
        mine_counts = result["mine_counts"]
        win_rate = result["overall_win_rate"]
        target = result["target_win_rate"]
        status = "✅ PASS" if win_rate >= target else "⚠️  BELOW"
        
        print(f"Phase {phase}: {mine_counts} | {win_rate:.1%} | {status}")
    
    return results, current_model


if __name__ == "__main__":
    mixed_mine_training() 