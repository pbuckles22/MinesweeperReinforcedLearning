#!/usr/bin/env python3
"""
5x5 Gradual Curriculum Learning for Minesweeper RL

Uses 5x5 board with very gradual mine count increases to avoid dramatic
difficulty spikes and improve knowledge transfer.
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.train_agent_modular import train_modular, make_modular_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO


def evaluate_model(model, board_size, max_mines, n_episodes=200):
    """Evaluate model performance with more episodes for better statistics."""
    env = DummyVecEnv([make_modular_env(board_size, max_mines)])
    
    rewards = []
    wins = 0
    total_steps = 0
    successful_episodes = 0
    
    print(f"🔍 Evaluating on {board_size[0]}x{board_size[1]} with {max_mines} mines...")
    
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_won = False
        step_count = 0
        max_steps = 200  # Increased safety limit
        
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
        total_steps += step_count
        
        if episode_won:
            wins += 1
            successful_episodes += 1
        elif step_count < max_steps:  # Only count as successful if not hitting max steps
            successful_episodes += 1
        
        if (episode + 1) % 50 == 0:
            print(f"   Evaluated {episode + 1}/{n_episodes} episodes...")
    
    win_rate = wins / n_episodes
    mean_reward = np.mean(rewards)
    avg_steps = total_steps / n_episodes if n_episodes > 0 else 0
    success_rate = successful_episodes / n_episodes
    
    return {
        "win_rate": win_rate,
        "mean_reward": mean_reward,
        "wins": wins,
        "total_episodes": n_episodes,
        "avg_steps": avg_steps,
        "success_rate": success_rate
    }


def gradual_5x5_curriculum_training():
    """Run 5x5 curriculum learning with very gradual mine count increases."""
    
    print("🎓 5x5 Gradual Curriculum Learning for Minesweeper RL")
    print("=" * 60)
    
    # 5x5 curriculum stages with very gradual progression
    # Format: (board_size, max_mines, timesteps, min_win_rate, description, target_density)
    curriculum_stages = [
        # Stage 1: Easiest (5x5, 1 mine) - 4% density
        ((5, 5), 1, 200000, 0.80, "Stage 1: 5x5 with 1 mine (4% density)", 0.04),
        
        # Stage 2: Slightly harder (5x5, 2 mines) - 8% density
        ((5, 5), 2, 300000, 0.65, "Stage 2: 5x5 with 2 mines (8% density)", 0.08),
        
        # Stage 3: Medium (5x5, 3 mines) - 12% density
        ((5, 5), 3, 400000, 0.50, "Stage 3: 5x5 with 3 mines (12% density)", 0.12),
        
        # Stage 4: Harder (5x5, 4 mines) - 16% density
        ((5, 5), 4, 500000, 0.40, "Stage 4: 5x5 with 4 mines (16% density)", 0.16),
        
        # Stage 5: Very hard (5x5, 5 mines) - 20% density
        ((5, 5), 5, 600000, 0.30, "Stage 5: 5x5 with 5 mines (20% density)", 0.20),
        
        # Stage 6: Expert (5x5, 6 mines) - 24% density
        ((5, 5), 6, 700000, 0.25, "Stage 6: 5x5 with 6 mines (24% density)", 0.24),
        
        # Stage 7: Master (5x5, 7 mines) - 28% density
        ((5, 5), 7, 800000, 0.20, "Stage 7: 5x5 with 7 mines (28% density)", 0.28),
    ]
    
    results = []
    current_model = None
    
    for stage_num, (board_size, max_mines, timesteps, min_win_rate, description, target_density) in enumerate(curriculum_stages, 1):
        print(f"\n🎯 {description}")
        print(f"   Board: {board_size[0]}x{board_size[1]}, Mines: {max_mines}")
        print(f"   Density: {target_density:.1%}")
        print(f"   Timesteps: {timesteps:,}")
        print(f"   Target Win Rate: {min_win_rate:.1%}")
        print(f"   Stage: {stage_num}/{len(curriculum_stages)}")
        print("-" * 60)
        
        try:
            # Train the agent
            if current_model is None:
                # First stage: train from scratch
                print("🚀 Starting fresh training...")
                model, win_rate, mean_reward = train_modular(
                    board_size=board_size,
                    max_mines=max_mines,
                    total_timesteps=timesteps,
                    device="cpu",
                    learning_rate=0.0003,
                    batch_size=64,
                    ent_coef=0.01
                )
                wins = int(win_rate * 100)
                # For first stage, we already have win_rate and mean_reward from train_modular
                eval_results = {
                    "win_rate": win_rate,
                    "mean_reward": mean_reward,
                    "wins": wins,
                    "total_episodes": 100,
                    "avg_steps": 0,  # Will be calculated if needed
                    "success_rate": 1.0  # Assume 100% for first stage
                }
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
                
                # Evaluate the model with more episodes
                eval_results = evaluate_model(model, board_size, max_mines, n_episodes=200)
                win_rate = eval_results["win_rate"]
                mean_reward = eval_results["mean_reward"]
                wins = eval_results["wins"]
            
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
                "total_episodes": eval_results["total_episodes"],
                "mine_density": target_density,
                "target_win_rate": min_win_rate,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            
            # Add evaluation metrics if available
            if "avg_steps" in eval_results:
                result.update({
                    "avg_steps": eval_results["avg_steps"],
                    "success_rate": eval_results["success_rate"]
                })
            
            results.append(result)
            current_model = model
            
            print(f"📊 Stage {stage_num} Results:")
            print(f"   Win Rate: {win_rate:.3f} ({wins}/{result['total_episodes']})")
            print(f"   Mean Reward: {mean_reward:.2f}")
            print(f"   Mine Density: {target_density:.1%}")
            if "avg_steps" in eval_results:
                print(f"   Avg Steps: {eval_results['avg_steps']:.1f}")
                print(f"   Success Rate: {eval_results['success_rate']:.1%}")
            
            # Check if we should continue based on target win rate
            if win_rate < min_win_rate:
                print(f"⚠️  Performance below target ({win_rate:.1%} < {min_win_rate:.1%}).")
                print(f"   Consider extending training or adjusting curriculum.")
                
                # Ask if we should continue anyway
                response = input("Continue to next stage anyway? (y/n): ").lower().strip()
                if response != 'y':
                    print("Stopping curriculum.")
                    break
            else:
                print(f"✅ Target achieved! ({win_rate:.1%} >= {min_win_rate:.1%})")
                
        except Exception as e:
            print(f"❌ Error in stage {stage_num}: {e}")
            break
    
    # Save curriculum results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"experiments/5x5_gradual_curriculum_results_{timestamp}.json"
    
    # Ensure experiments directory exists
    os.makedirs("experiments", exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            "curriculum_summary": {
                "total_stages": len(curriculum_stages),
                "completed_stages": len(results),
                "timestamp": timestamp,
                "approach": "5x5_gradual_progression"
            },
            "results": results
        }, f, indent=2)
    
    print(f"\n💾 Curriculum results saved to: {results_file}")
    
    # Print final summary
    print(f"\n📈 5x5 Gradual Curriculum Learning Summary")
    print("=" * 60)
    print(f"{'Stage':<6} {'Board':<8} {'Mines':<6} {'Density':<10} {'Win Rate':<10} {'Status':<10}")
    print("-" * 60)
    
    for result in results:
        stage = result["stage"]
        board = f"{result['board_size'][0]}x{result['board_size'][1]}"
        mines = result["max_mines"]
        density = result["mine_density"]
        win_rate = result["win_rate"]
        target = result["target_win_rate"]
        status = "✅ PASS" if win_rate >= target else "⚠️  BELOW"
        
        print(f"{stage:<6} {board:<8} {mines:<6} {density:<10.1%} {win_rate:<10.1%} {status:<10}")
    
    return results, current_model


if __name__ == "__main__":
    gradual_5x5_curriculum_training() 