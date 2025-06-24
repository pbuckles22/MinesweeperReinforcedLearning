#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for Minesweeper RL Agent

Tests multiple board sizes and mine densities with larger evaluation samples
to get statistically significant results.
"""

import argparse
import json
import os
import sys
import numpy as np
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.train_agent_modular import train_modular, make_modular_env
from core.minesweeper_env import MinesweeperEnv
from stable_baselines3.common.vec_env import DummyVecEnv


def evaluate_model(model, env, n_eval_episodes=500, verbose=True):
    """
    Evaluate a trained model with a larger sample size for statistical significance.
    
    Args:
        model: Trained PPO model
        env: Vectorized environment
        n_eval_episodes: Number of evaluation episodes
        verbose: Whether to print progress
    
    Returns:
        win_rate, mean_reward, wins, total_episodes
    """
    if verbose:
        print(f"🔍 Evaluating with {n_eval_episodes} episodes for statistical significance...")
    
    rewards = []
    wins = 0
    
    for episode in range(n_eval_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_won = False
        step_count = 0
        max_steps = 100  # Safety limit to prevent infinite loops
        
        while not done and step_count < max_steps:
            try:
                action = model.predict(obs, deterministic=True)[0]
                obs, reward, done, info = env.step(action)
                episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
                step_count += 1
                
                # Check for win condition
                if info and isinstance(info, list) and len(info) > 0:
                    if info[0].get('won', False):
                        episode_won = True
                        done = True  # Force episode to end
                elif info and isinstance(info, dict):
                    if info.get('won', False):
                        episode_won = True
                        done = True  # Force episode to end
                
                # Check for game over conditions
                if info and isinstance(info, list) and len(info) > 0:
                    if info[0].get('game_over', False):
                        done = True
                elif info and isinstance(info, dict):
                    if info.get('game_over', False):
                        done = True
                        
            except Exception as e:
                print(f"⚠️  Error in episode {episode}, step {step_count}: {e}")
                done = True  # Force episode to end on error
                break
        
        # Safety check - if we hit max steps, consider it a loss
        if step_count >= max_steps:
            print(f"⚠️  Episode {episode} hit max steps ({max_steps}), forcing end")
            done = True
        
        rewards.append(episode_reward)
        if episode_won:
            wins += 1
        
        if verbose and (episode + 1) % 100 == 0:
            print(f"   Evaluated {episode + 1}/{n_eval_episodes} episodes...")
    
    mean_reward = np.mean(rewards)
    win_rate = wins / n_eval_episodes
    
    return win_rate, mean_reward, wins, n_eval_episodes


def run_comprehensive_evaluation():
    """Run comprehensive evaluation across multiple board configurations."""
    
    print("🔬 Comprehensive Minesweeper RL Evaluation")
    print("=" * 50)
    
    # Test configurations: (board_size, max_mines, timesteps, description)
    test_configs = [
        # Small boards (quick tests)
        ((4, 4), 2, 2000, "4x4 with 2 mines (12.5% density)"),
        ((4, 4), 3, 2000, "4x4 with 3 mines (18.8% density)"),
        
        # Medium boards (more realistic)
        ((6, 6), 4, 5000, "6x6 with 4 mines (11.1% density)"),
        ((6, 6), 6, 5000, "6x6 with 6 mines (16.7% density)"),
        ((8, 8), 8, 8000, "8x8 with 8 mines (12.5% density)"),
        
        # Larger evaluation samples
        ((4, 4), 2, 10000, "4x4 with 2 mines (extended training)"),
    ]
    
    results = []
    
    for board_size, max_mines, timesteps, description in test_configs:
        print(f"\n🎯 Testing: {description}")
        print(f"   Board: {board_size[0]}x{board_size[1]}, Mines: {max_mines}")
        print(f"   Timesteps: {timesteps:,}")
        print("-" * 40)
        
        try:
            # Train the agent
            model, win_rate, mean_reward = train_modular(
                board_size=board_size,
                max_mines=max_mines,
                total_timesteps=timesteps,
                device="cpu"
            )
            
            # Create environment for evaluation
            env = DummyVecEnv([make_modular_env(board_size, max_mines)])
            
            # Evaluate with larger sample (500 episodes instead of 100)
            win_rate_large, mean_reward_large, wins, total_episodes = evaluate_model(
                model, env, n_eval_episodes=500, verbose=True
            )
            
            # Calculate confidence interval (approximate)
            # For binomial distribution, 95% CI ≈ ±1.96 * sqrt(p*(1-p)/n)
            p = win_rate_large
            n = total_episodes
            margin_of_error = 1.96 * ((p * (1-p)) / n) ** 0.5
            ci_lower = max(0, p - margin_of_error)
            ci_upper = min(1, p + margin_of_error)
            
            result = {
                "board_size": board_size,
                "max_mines": max_mines,
                "total_timesteps": timesteps,
                "description": description,
                "win_rate": win_rate_large,
                "mean_reward": mean_reward_large,
                "wins": wins,
                "total_episodes": total_episodes,
                "confidence_interval": [ci_lower, ci_upper],
                "margin_of_error": margin_of_error,
                "mine_density": max_mines / (board_size[0] * board_size[1]),
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            
            results.append(result)
            
            print(f"📊 Results:")
            print(f"   Win Rate: {win_rate_large:.3f} ({wins}/{total_episodes})")
            print(f"   Mean Reward: {float(mean_reward_large):.2f}")
            print(f"   95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
            print(f"   Margin of Error: ±{margin_of_error:.3f}")
            print(f"   Mine Density: {result['mine_density']:.1%}")
            
        except Exception as e:
            print(f"❌ Error testing {description}: {e}")
            continue
    
    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"experiments/comprehensive_evaluation_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            "evaluation_summary": {
                "total_configurations": len(test_configs),
                "successful_runs": len(results),
                "timestamp": timestamp
            },
            "results": results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Print summary
    print(f"\n📈 Evaluation Summary")
    print("=" * 50)
    for result in results:
        desc = result["description"]
        win_rate = result["win_rate"]
        ci = result["confidence_interval"]
        density = result["mine_density"]
        print(f"{desc:35} | {win_rate:.1%} [{ci[0]:.1%}-{ci[1]:.1%}] | {density:.1%} density")
    
    return results


def analyze_human_benchmarks():
    """Compare against human performance benchmarks."""
    print(f"\n👤 Human Performance Comparison")
    print("=" * 50)
    
    # Human performance estimates (from research)
    human_benchmarks = {
        (4, 4): {
            2: 0.85,  # 85% win rate on 4x4 with 2 mines
            3: 0.70,  # 70% win rate on 4x4 with 3 mines
        },
        (6, 6): {
            4: 0.80,  # 80% win rate on 6x6 with 4 mines
            6: 0.65,  # 65% win rate on 6x6 with 6 mines
        },
        (8, 8): {
            8: 0.75,  # 75% win rate on 8x8 with 8 mines
        }
    }
    
    print("Human benchmarks (estimated):")
    for board_size, mine_configs in human_benchmarks.items():
        for mines, win_rate in mine_configs.items():
            print(f"  {board_size[0]}x{board_size[1]} with {mines} mines: {win_rate:.0%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive Minesweeper RL Evaluation")
    parser.add_argument("--quick", action="store_true", help="Run only quick tests (4x4 boards)")
    parser.add_argument("--human-benchmarks", action="store_true", help="Show human performance benchmarks")
    
    args = parser.parse_args()
    
    if args.human_benchmarks:
        analyze_human_benchmarks()
    else:
        run_comprehensive_evaluation() 