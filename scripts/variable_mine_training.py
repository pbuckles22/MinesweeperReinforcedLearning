#!/usr/bin/env python3
"""
Variable Mine Count Training for Minesweeper RL

Trains the agent on variable mine counts to learn generalizable strategies
instead of overfitting to specific configurations.
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
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO


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


def evaluate_variable_mine_model(model, board_size, min_mines, max_mines, n_episodes=100):
    """Evaluate model across different mine counts."""
    print(f"🔍 Evaluating on {board_size[0]}x{board_size[1]} with {min_mines}-{max_mines} mines...")
    
    mine_results = {}
    total_wins = 0
    total_episodes = 0
    
    for mine_count in range(min_mines, max_mines + 1):
        print(f"   Testing with {mine_count} mines...")
        
        # Create environment for specific mine count
        env = DummyVecEnv([make_modular_env(board_size, mine_count)])
        
        wins = 0
        rewards = []
        
        for episode in range(n_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_won = False
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
                            episode_won = True
                            done = True
                    elif info and isinstance(info, dict):
                        if info.get('won', False):
                            episode_won = True
                            done = True
                            
                except Exception as e:
                    print(f"⚠️  Error in episode {episode}, step {step_count}: {e}")
                    done = True
                    break
            
            rewards.append(episode_reward)
            if episode_won:
                wins += 1
        
        win_rate = wins / n_episodes
        mean_reward = np.mean(rewards)
        
        mine_results[mine_count] = {
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
    
    return mine_results, overall_win_rate


def variable_mine_training():
    """Train agent with variable mine counts."""
    
    print("🎯 Variable Mine Count Training for Minesweeper RL")
    print("=" * 60)
    
    # Training configuration
    board_size = (5, 5)
    
    # Training phases - each phase trains on a range of mine counts
    training_phases = [
        # Phase 1: Learn basics (1-2 mines)
        {"mine_range": (1, 2), "timesteps": 300000, "target_win_rate": 0.60},
        
        # Phase 2: Expand to 1-3 mines
        {"mine_range": (1, 3), "timesteps": 400000, "target_win_rate": 0.50},
        
        # Phase 3: Expand to 1-4 mines
        {"mine_range": (1, 4), "timesteps": 500000, "target_win_rate": 0.40},
        
        # Phase 4: Master 1-5 mines
        {"mine_range": (1, 5), "timesteps": 600000, "target_win_rate": 0.35},
    ]
    
    results = []
    current_model = None
    
    for phase_num, phase_config in enumerate(training_phases, 1):
        mine_range = phase_config["mine_range"]
        timesteps = phase_config["timesteps"]
        target_win_rate = phase_config["target_win_rate"]
        
        print(f"\n🎯 Phase {phase_num}: {mine_range[0]}-{mine_range[1]} mines")
        print(f"   Board: {board_size[0]}x{board_size[1]}")
        print(f"   Mine Range: {mine_range[0]}-{mine_range[1]} mines")
        print(f"   Timesteps: {timesteps:,}")
        print(f"   Target Win Rate: {target_win_rate:.1%}")
        print("-" * 60)
        
        try:
            # For each phase, we'll train on the maximum mine count in the range
            # This ensures the agent can handle the hardest case in the range
            max_mines = mine_range[1]
            
            # Create environment with max mines in range
            env = DummyVecEnv([make_modular_env(board_size, max_mines)])
            
            if current_model is None:
                # First phase: train from scratch
                print(f"🚀 Starting fresh training on {max_mines} mines...")
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
                print(f"🔄 Continuing training on {max_mines} mines...")
                current_model.set_env(env)
            
            # Train the model
            current_model.learn(
                total_timesteps=timesteps,
                progress_bar=True
            )
            
            # Evaluate performance across the entire mine range
            mine_results, overall_win_rate = evaluate_variable_mine_model(
                current_model, board_size, mine_range[0], mine_range[1], n_episodes=50
            )
            
            # Save phase results
            result = {
                "phase": phase_num,
                "board_size": board_size,
                "mine_range": mine_range,
                "total_timesteps": timesteps,
                "overall_win_rate": float(overall_win_rate),
                "target_win_rate": target_win_rate,
                "mine_results": mine_results,
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
    results_file = f"experiments/variable_mine_training_results_{timestamp}.json"
    
    # Ensure experiments directory exists
    os.makedirs("experiments", exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(convert_to_json_serializable({
            "training_summary": {
                "total_phases": len(training_phases),
                "completed_phases": len(results),
                "timestamp": timestamp,
                "approach": "progressive_mine_count_training"
            },
            "results": results
        }), f, indent=2)
    
    print(f"\n💾 Training results saved to: {results_file}")
    
    # Print final summary
    print(f"\n📈 Progressive Mine Count Training Summary")
    print("=" * 60)
    for result in results:
        phase = result["phase"]
        mine_range = result["mine_range"]
        win_rate = result["overall_win_rate"]
        target = result["target_win_rate"]
        status = "✅ PASS" if win_rate >= target else "⚠️  BELOW"
        
        print(f"Phase {phase}: {mine_range[0]}-{mine_range[1]} mines | {win_rate:.1%} | {status}")
    
    return results, current_model


if __name__ == "__main__":
    variable_mine_training() 