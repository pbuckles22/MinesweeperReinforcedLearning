#!/usr/bin/env python3
"""
Curriculum Learning for Minesweeper RL

Starts with easier boards and gradually increases difficulty to help the agent
learn transferable strategies rather than board-specific patterns.
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.train_agent_modular import train_modular, make_modular_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
import numpy as np


def curriculum_training():
    """Run curriculum learning with gradually increasing difficulty."""
    
    print("🎓 Curriculum Learning for Minesweeper RL")
    print("=" * 50)
    
    # Curriculum stages: (board_size, max_mines, timesteps, description)
    curriculum_stages = [
        # Stage 1: Easiest possible (4x4, 1 mine)
        ((4, 4), 1, 100000, "Stage 1: 4x4 with 1 mine (6.25% density)"),
        
        # Stage 2: Slightly harder (4x4, 2 mines) 
        ((4, 4), 2, 150000, "Stage 2: 4x4 with 2 mines (12.5% density)"),
        
        # Stage 3: Medium difficulty (6x6, 3 mines)
        ((6, 6), 3, 200000, "Stage 3: 6x6 with 3 mines (8.3% density)"),
        
        # Stage 4: Harder (6x6, 4 mines)
        ((6, 6), 4, 250000, "Stage 4: 6x6 with 4 mines (11.1% density)"),
        
        # Stage 5: Challenge (6x6, 6 mines)
        ((6, 6), 6, 300000, "Stage 5: 6x6 with 6 mines (16.7% density)"),
    ]
    
    results = []
    current_model = None
    
    for stage_num, (board_size, max_mines, timesteps, description) in enumerate(curriculum_stages, 1):
        print(f"\n🎯 {description}")
        print(f"   Board: {board_size[0]}x{board_size[1]}, Mines: {max_mines}")
        print(f"   Timesteps: {timesteps:,}")
        print(f"   Stage: {stage_num}/{len(curriculum_stages)}")
        print("-" * 50)
        
        try:
            # Train the agent
            if current_model is None:
                # First stage: train from scratch
                model, win_rate, mean_reward = train_modular(
                    board_size=board_size,
                    max_mines=max_mines,
                    total_timesteps=timesteps,
                    device="cpu",
                    learning_rate=0.0003,
                    batch_size=64,
                    ent_coef=0.01
                )
                # Calculate wins from win_rate for first stage
                wins = int(win_rate * 100)
            else:
                # Subsequent stages: continue training with existing model
                print("🔄 Continuing training with previous model...")
                
                # Create environment for current stage
                env = DummyVecEnv([make_modular_env(board_size, max_mines)])
                
                # Continue training the existing model
                model = current_model
                model.set_env(env)
                
                # Train for additional timesteps
                model.learn(
                    total_timesteps=timesteps,
                    progress_bar=True
                )
                
                # Evaluate the model
                print("🔍 Evaluating performance...")
                rewards = []
                wins = 0
                n_eval_episodes = 100
                
                for episode in range(n_eval_episodes):
                    obs = env.reset()
                    done = False
                    episode_reward = 0
                    episode_won = False
                    step_count = 0
                    max_steps = 100  # Safety limit
                    
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
                    
                    # Safety check
                    if step_count >= max_steps:
                        print(f"⚠️  Episode {episode} hit max steps ({max_steps})")
                        done = True
                    
                    rewards.append(episode_reward)
                    if episode_won:
                        wins += 1
                    
                    if (episode + 1) % 20 == 0:
                        print(f"   Evaluated {episode + 1}/{n_eval_episodes} episodes...")
                
                win_rate = wins / n_eval_episodes
                mean_reward = np.mean(rewards)
            
            # Save stage results
            result = {
                "stage": stage_num,
                "board_size": board_size,
                "max_mines": max_mines,
                "total_timesteps": timesteps,
                "description": description,
                "win_rate": float(win_rate),
                "mean_reward": float(mean_reward),
                "wins": int(wins),
                "total_episodes": 100,
                "mine_density": max_mines / (board_size[0] * board_size[1]),
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            
            results.append(result)
            current_model = model
            
            print(f"📊 Stage {stage_num} Results:")
            print(f"   Win Rate: {win_rate:.3f} ({wins}/100)")
            print(f"   Mean Reward: {mean_reward:.2f}")
            print(f"   Mine Density: {result['mine_density']:.1%}")
            
            # Check if we should continue
            if win_rate < 0.15:  # Less than 15% win rate
                print(f"⚠️  Performance too low ({win_rate:.1%}). Stopping curriculum.")
                break
                
        except Exception as e:
            print(f"❌ Error in stage {stage_num}: {e}")
            break
    
    # Save curriculum results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"experiments/curriculum_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            "curriculum_summary": {
                "total_stages": len(curriculum_stages),
                "completed_stages": len(results),
                "timestamp": timestamp
            },
            "results": results
        }, f, indent=2)
    
    print(f"\n💾 Curriculum results saved to: {results_file}")
    
    # Print final summary
    print(f"\n📈 Curriculum Learning Summary")
    print("=" * 50)
    for result in results:
        stage = result["stage"]
        desc = result["description"]
        win_rate = result["win_rate"]
        density = result["mine_density"]
        print(f"Stage {stage}: {desc:35} | {win_rate:.1%} | {density:.1%} density")
    
    return results, current_model


if __name__ == "__main__":
    curriculum_training() 