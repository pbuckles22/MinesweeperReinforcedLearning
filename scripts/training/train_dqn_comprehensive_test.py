#!/usr/bin/env python3
"""
Comprehensive DQN Training Test Script (Without Learnable Filtering)
Tests multiple training durations on a single board size and compares results.

This script runs short, medium, and long training sessions to determine optimal training length
and compare performance to past results. This version DISABLES learnable filtering to test
if that's causing training instability.
"""

import sys
import os
import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.core.train_agent import (
    make_env, detect_optimal_device, get_optimal_hyperparameters,
    evaluate_model, ExperimentTracker, CustomEvalCallback, IterationCallback
)
from src.core.dqn_agent import DQNAgent, train_dqn_agent, evaluate_dqn_agent
import torch
import mlflow

def get_dqn_hyperparameters(device_info):
    """
    Get optimal DQN hyperparameters based on the detected device.
    Conservative settings optimized for CPU stability.
    """
    base_params = {
        'learning_rate': 0.000005,  # Even lower for stability (was 0.00001)
        'discount_factor': 0.99,
        'epsilon': 1.0,
        'epsilon_decay': 0.99995,  # Extremely slow decay for sustained exploration
        'epsilon_min': 0.15,       # Higher minimum for continued exploration
        'replay_buffer_size': 100000,
        'batch_size': 16,          # Small batch for stability
        'target_update_freq': 100  # More frequent target updates (was 300)
    }
    
    # Force CPU for stability (MPS was causing issues)
    optimized_params = base_params.copy()
    optimized_params.update({
        'device': 'cpu',           # Force CPU usage
        'batch_size': 16,          # Small batch for CPU
        'learning_rate': 0.000005,  # Very conservative learning rate
        'epsilon_decay': 0.99997,  # Very slow decay
    })
    print("🔧 Applied CPU optimizations (Conservative):")
    print(f"   - Forced CPU usage for stability")
    print(f"   - Small batch size: {optimized_params['batch_size']}")
    print(f"   - Very conservative learning rate: {optimized_params['learning_rate']}")
    print(f"   - Very slow epsilon decay: {optimized_params['epsilon_decay']}")
    print(f"   - Higher epsilon min: {optimized_params['epsilon_min']}")
    print(f"   - Frequent target updates: {optimized_params['target_update_freq']}")
    print(f"   - Reward normalization: [-0.5, 0.5] range")
    print(f"   - No learning rate scheduling (constant LR)")
    print(f"   - Learnable filtering: DISABLED (testing without)")
    return optimized_params

def run_training_session(board_size, mine_count, timesteps, session_name, device_info):
    """Run a single training session with specified parameters."""
    
    print(f"\n🚀 Starting {session_name} Training Session")
    print(f"   Board: {board_size[0]}×{board_size[1]}, Mines: {mine_count}")
    print(f"   Timesteps: {timesteps:,}")
    print(f"   Device: {device_info['device']} ({device_info['description']})")
    print("=" * 60)
    
    # Create environment
    env_fn = make_env(max_board_size=board_size, max_mines=mine_count)
    env = env_fn()
    
    # Get optimal hyperparameters for device
    hyperparams = get_dqn_hyperparameters(device_info)
    
    # Create DQN agent
    agent = DQNAgent(
        board_size=board_size,
        action_size=env.action_space.n,
        device='cpu',  # Force CPU usage
        **{k: v for k, v in hyperparams.items() if k != 'device'}  # Exclude device from hyperparams
    )
    
    # Convert timesteps to episodes (rough estimate: 1 episode ≈ 30 timesteps for 4x4)
    episodes = max(200, timesteps // 30)  # More episodes for better learning
    eval_freq = max(100, episodes // 10)  # Evaluate every 10% of episodes
    
    print(f"   Episodes: {episodes:,}")
    print(f"   Evaluation frequency: every {eval_freq} episodes")
    
    # Start training
    start_time = time.time()
    
    try:
        # Train the agent using DQN training function
        training_stats = train_dqn_agent(
            env=env,
            agent=agent,
            episodes=episodes,
            mine_count=mine_count,
            eval_freq=eval_freq
        )
        
        training_time = time.time() - start_time
        
        # Final evaluation
        print(f"\n📊 Final Evaluation for {session_name}")
        final_results = evaluate_dqn_agent(agent, env, 200)
        
        results = {
            'session_name': session_name,
            'board_size': board_size,
            'mine_count': mine_count,
            'timesteps': timesteps,
            'episodes': episodes,
            'training_time': training_time,
            'final_results': final_results,
            'training_stats': training_stats,
            'device_info': device_info,
            'hyperparams': hyperparams
        }
        
        print(f"   ✅ Training completed in {training_time/60:.1f} minutes")
        print(f"   📈 Final Win Rate: {final_results['win_rate']:.1f}%")
        print(f"   📊 Final Avg Reward: {final_results['mean_reward']:.2f}")
        
        return results
        
    except Exception as e:
        print(f"   ❌ Training failed: {e}")
        return {
            'session_name': session_name,
            'error': str(e),
            'board_size': board_size,
            'mine_count': mine_count,
            'timesteps': timesteps
        }

def compare_with_past_results(current_results, past_results_file="training_stats/past_results.json"):
    """Compare current results with past training results."""
    
    if not os.path.exists(past_results_file):
        print("   📝 No past results found for comparison")
        return
    
    try:
        with open(past_results_file, 'r') as f:
            past_results = json.load(f)
        
        print(f"\n📊 Comparison with Past Results")
        print("=" * 50)
        
        for result in current_results:
            if 'error' in result:
                continue
                
            session_name = result['session_name']
            current_win_rate = result['final_results']['win_rate']
            current_avg_reward = result['final_results']['mean_reward']
            
            print(f"\n{session_name.upper()} SESSION:")
            print(f"   Current Win Rate: {current_win_rate:.1f}%")
            print(f"   Current Avg Reward: {current_avg_reward:.2f}")
            
            # Find matching past result
            for past_result in past_results:
                if (past_result.get('board_size') == result['board_size'] and 
                    past_result.get('mine_count') == result['mine_count']):
                    
                    past_win_rate = past_result.get('win_rate', 0)
                    past_avg_reward = past_result.get('mean_reward', 0)
                    
                    print(f"   Past Win Rate: {past_win_rate:.1f}%")
                    print(f"   Past Avg Reward: {past_avg_reward:.2f}")
                    
                    win_rate_change = current_win_rate - past_win_rate
                    reward_change = current_avg_reward - past_avg_reward
                    
                    print(f"   Win Rate Change: {win_rate_change:+.1f}%")
                    print(f"   Reward Change: {reward_change:+.2f}")
                    
                    if win_rate_change > 0:
                        print(f"   🎉 Improved performance!")
                    elif win_rate_change < 0:
                        print(f"   📉 Performance declined")
                    else:
                        print(f"   ➡️  Similar performance")
                    
                    break
            else:
                print(f"   📝 No past results for this configuration")
                
    except Exception as e:
        print(f"   ❌ Error comparing with past results: {e}")

def analyze_training_progression(results):
    """Analyze how training duration affects performance."""
    
    print(f"\n📈 Training Duration Analysis")
    print("=" * 50)
    
    # Sort by timesteps
    sorted_results = sorted([r for r in results if 'error' not in r], key=lambda x: x['timesteps'])
    
    if len(sorted_results) < 2:
        print("   📝 Need at least 2 successful sessions for analysis")
        return
    
    print(f"\nPerformance by Training Duration:")
    
    for i, result in enumerate(sorted_results):
        session_name = result['session_name']
        timesteps = result['timesteps']
        win_rate = result['final_results']['win_rate']
        avg_reward = result['final_results']['mean_reward']
        training_time = result['training_time']
        
        print(f"\n{session_name.upper()}:")
        print(f"   Timesteps: {timesteps:,}")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Avg Reward: {avg_reward:.2f}")
        print(f"   Training Time: {training_time/60:.1f} minutes")
        print(f"   Efficiency: {win_rate/(training_time/60):.2f}% win rate per minute")
    
    # Calculate improvements
    print(f"\n📊 Improvement Analysis:")
    
    for i in range(1, len(sorted_results)):
        prev = sorted_results[i-1]
        curr = sorted_results[i]
        
        win_rate_improvement = curr['final_results']['win_rate'] - prev['final_results']['win_rate']
        reward_improvement = curr['final_results']['mean_reward'] - prev['final_results']['mean_reward']
        time_increase = curr['training_time'] - prev['training_time']
        
        print(f"\n{prev['session_name'].upper()} → {curr['session_name'].upper()}:")
        print(f"   Win Rate Change: {win_rate_improvement:+.1f}%")
        print(f"   Reward Change: {reward_improvement:+.2f}")
        print(f"   Time Increase: {time_increase/60:.1f} minutes")
        
        if time_increase > 0:
            efficiency = win_rate_improvement / (time_increase / 60)
            print(f"   Efficiency: {efficiency:+.2f}% win rate per additional minute")

def main():
    """Main function to run comprehensive DQN training test."""
    
    print("🚀 Comprehensive DQN Training Test")
    print("Testing multiple training durations on single board size")
    print("=" * 80)
    
    # Configuration
    board_size = (4, 4)
    mine_count = 2
    
    # Training durations (short, medium, long)
    training_configs = [
        ("short", 100000),   # 100K timesteps (was 50K)
        ("medium", 500000),  # 500K timesteps (was 200K)  
        ("long", 1000000),   # 1M timesteps (was 500K)
    ]
    
    # Detect optimal device
    device_info = detect_optimal_device()
    print(f"🎯 Using device: {device_info['device']} ({device_info['description']})")
    
    # Create results directory
    results_dir = Path("training_stats")
    results_dir.mkdir(exist_ok=True)
    
    # Run training sessions
    all_results = []
    
    for session_name, timesteps in training_configs:
        result = run_training_session(
            board_size=board_size,
            mine_count=mine_count,
            timesteps=timesteps,
            session_name=session_name,
            device_info=device_info
        )
        all_results.append(result)
        
        # Save intermediate results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"comprehensive_test_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
    
    # Analyze results
    print(f"\n🎉 All Training Sessions Complete!")
    print("=" * 80)
    
    # Compare with past results
    compare_with_past_results(all_results)
    
    # Analyze training progression
    analyze_training_progression(all_results)
    
    # Final summary
    successful_results = [r for r in all_results if 'error' not in r]
    
    if successful_results:
        best_result = max(successful_results, key=lambda x: x['final_results']['win_rate'])
        
        print(f"\n🏆 Best Performance:")
        print(f"   Session: {best_result['session_name'].upper()}")
        print(f"   Win Rate: {best_result['final_results']['win_rate']:.1f}%")
        print(f"   Avg Reward: {best_result['final_results']['mean_reward']:.2f}")
        print(f"   Timesteps: {best_result['timesteps']:,}")
        print(f"   Training Time: {best_result['training_time']/60:.1f} minutes")
    
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_results_file = results_dir / f"comprehensive_test_final_{timestamp}.json"
    with open(final_results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n📁 Results saved to: {final_results_file}")
    print(f"🎯 Ready for next stage training!")

if __name__ == "__main__":
    main() 