#!/usr/bin/env python3
"""
Quick DQN Hyperparameter Optimization

Focused optimization of the most impactful hyperparameters for DQN performance.
Tests fewer configurations with fewer episodes for quick results.
"""

import sys
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.minesweeper_env import MinesweeperEnv
from src.core.dqn_agent import DQNAgent, train_dqn_agent, evaluate_dqn_agent


def test_config(config: Dict[str, Any], board_size: Tuple[int, int] = (4, 4), 
                mine_count: int = 1, episodes: int = 50) -> Dict[str, Any]:
    """Test a single hyperparameter configuration."""
    print(f"\n🔬 Testing: {config['name']}")
    print(f"   LR: {config['learning_rate']}, Batch: {config['batch_size']}, "
          f"Epsilon: {config['epsilon_decay']}")
    
    # Create environment and agent
    env = MinesweeperEnv(initial_board_size=board_size, initial_mines=mine_count)
    action_size = board_size[0] * board_size[1]
    
    agent = DQNAgent(
        board_size=board_size,
        action_size=action_size,
        learning_rate=config['learning_rate'],
        discount_factor=config['discount_factor'],
        epsilon=config['epsilon'],
        epsilon_decay=config['epsilon_decay'],
        epsilon_min=config['epsilon_min'],
        replay_buffer_size=config['replay_buffer_size'],
        batch_size=config['batch_size'],
        target_update_freq=config['target_update_freq'],
        device='cpu'
    )
    
    # Train agent
    start_time = time.time()
    training_stats = train_dqn_agent(env, agent, episodes, mine_count, eval_freq=10)
    training_time = time.time() - start_time
    
    # Evaluate agent
    eval_stats = evaluate_dqn_agent(agent, env, n_episodes=20)
    
    # Calculate metrics
    final_win_rate = training_stats['win_rate']
    eval_win_rate = eval_stats['win_rate']
    mean_reward = eval_stats['mean_reward']
    mean_length = eval_stats['mean_length']
    
    # Stability metric (lower is better)
    win_rate_stability = abs(final_win_rate - eval_win_rate)
    
    # Efficiency metric (higher is better)
    efficiency = eval_win_rate / mean_length if mean_length > 0 else 0
    
    result = {
        'config': config,
        'training_win_rate': final_win_rate,
        'eval_win_rate': eval_win_rate,
        'mean_reward': mean_reward,
        'mean_length': mean_length,
        'win_rate_stability': win_rate_stability,
        'efficiency': efficiency,
        'training_time': training_time,
        'final_epsilon': agent.epsilon
    }
    
    print(f"   Results: Train WR: {final_win_rate:.3f}, Eval WR: {eval_win_rate:.3f}, "
          f"Stability: {win_rate_stability:.3f}, Efficiency: {efficiency:.3f}")
    
    return result


def main():
    """Run quick hyperparameter optimization."""
    print("🚀 Quick DQN Hyperparameter Optimization")
    print("=" * 60)
    
    board_size = (4, 4)
    mine_count = 1
    episodes_per_config = 50
    
    print(f"   Board size: {board_size}")
    print(f"   Mine count: {mine_count}")
    print(f"   Episodes per config: {episodes_per_config}")
    
    # Define focused configurations to test
    configs = [
        # Current baseline
        {
            'name': 'Baseline',
            'learning_rate': 0.001,
            'discount_factor': 0.99,
            'epsilon': 1.0,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.01,
            'replay_buffer_size': 100000,
            'batch_size': 64,
            'target_update_freq': 1000
        },
        
        # Lower learning rate (more stable)
        {
            'name': 'Lower_LR',
            'learning_rate': 0.0005,
            'discount_factor': 0.99,
            'epsilon': 1.0,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.01,
            'replay_buffer_size': 100000,
            'batch_size': 64,
            'target_update_freq': 1000
        },
        
        # Much lower learning rate (very stable)
        {
            'name': 'Very_Low_LR',
            'learning_rate': 0.0003,
            'discount_factor': 0.99,
            'epsilon': 1.0,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.01,
            'replay_buffer_size': 100000,
            'batch_size': 64,
            'target_update_freq': 1000
        },
        
        # Smaller batch size (more frequent updates)
        {
            'name': 'Small_Batch',
            'learning_rate': 0.001,
            'discount_factor': 0.99,
            'epsilon': 1.0,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.01,
            'replay_buffer_size': 100000,
            'batch_size': 32,
            'target_update_freq': 1000
        },
        
        # Slower epsilon decay (more exploration)
        {
            'name': 'Slow_Epsilon',
            'learning_rate': 0.001,
            'discount_factor': 0.99,
            'epsilon': 1.0,
            'epsilon_decay': 0.999,
            'epsilon_min': 0.01,
            'replay_buffer_size': 100000,
            'batch_size': 64,
            'target_update_freq': 1000
        },
        
        # Very slow epsilon decay
        {
            'name': 'Very_Slow_Epsilon',
            'learning_rate': 0.001,
            'discount_factor': 0.99,
            'epsilon': 1.0,
            'epsilon_decay': 0.9995,
            'epsilon_min': 0.01,
            'replay_buffer_size': 100000,
            'batch_size': 64,
            'target_update_freq': 1000
        },
        
        # Less frequent target updates
        {
            'name': 'Slow_Target',
            'learning_rate': 0.001,
            'discount_factor': 0.99,
            'epsilon': 1.0,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.01,
            'replay_buffer_size': 100000,
            'batch_size': 64,
            'target_update_freq': 2000
        },
        
        # Best combination (educated guess)
        {
            'name': 'Best_Guess',
            'learning_rate': 0.0005,
            'discount_factor': 0.99,
            'epsilon': 1.0,
            'epsilon_decay': 0.999,
            'epsilon_min': 0.01,
            'replay_buffer_size': 100000,
            'batch_size': 32,
            'target_update_freq': 2000
        },
        
        # Conservative approach
        {
            'name': 'Conservative',
            'learning_rate': 0.0003,
            'discount_factor': 0.99,
            'epsilon': 1.0,
            'epsilon_decay': 0.9995,
            'epsilon_min': 0.05,
            'replay_buffer_size': 200000,
            'batch_size': 16,
            'target_update_freq': 5000
        }
    ]
    
    results = []
    
    # Test each configuration
    for i, config in enumerate(configs):
        print(f"\n📊 Configuration {i+1}/{len(configs)}")
        result = test_config(config, board_size, mine_count, episodes_per_config)
        results.append(result)
    
    # Sort results by evaluation win rate
    results.sort(key=lambda x: x['eval_win_rate'], reverse=True)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🏆 OPTIMIZATION SUMMARY")
    print("=" * 60)
    
    # Top 3 configurations
    print("\n🥇 TOP 3 CONFIGURATIONS:")
    for i, result in enumerate(results[:3]):
        config = result['config']
        print(f"{i+1}. {config['name']}: Eval WR: {result['eval_win_rate']:.3f}, "
              f"Stability: {result['win_rate_stability']:.3f}, "
              f"Efficiency: {result['efficiency']:.3f}")
        print(f"   LR: {config['learning_rate']}, Batch: {config['batch_size']}, "
              f"Epsilon: {config['epsilon_decay']}, Target: {config['target_update_freq']}")
    
    # Best configuration
    best = results[0]
    print(f"\n🎯 BEST CONFIGURATION: {best['config']['name']}")
    print(f"   Evaluation Win Rate: {best['eval_win_rate']:.3f}")
    print(f"   Training Win Rate: {best['training_win_rate']:.3f}")
    print(f"   Stability: {best['win_rate_stability']:.3f}")
    print(f"   Efficiency: {best['efficiency']:.3f}")
    print(f"   Mean Reward: {best['mean_reward']:.2f}")
    print(f"   Mean Length: {best['mean_length']:.1f} steps")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dqn_quick_optimization_{timestamp}.json"
    
    # Convert numpy types for JSON serialization
    serializable_results = []
    for result in results:
        serializable_result = {}
        for key, value in result.items():
            if key == 'config':
                serializable_result[key] = value
            elif isinstance(value, (np.integer, np.floating)):
                serializable_result[key] = float(value)
            elif isinstance(value, np.ndarray):
                serializable_result[key] = value.tolist()
            else:
                serializable_result[key] = value
        serializable_results.append(serializable_result)
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n💾 Results saved to {filename}")
    print(f"✅ Quick optimization completed! Tested {len(configs)} configurations.")


if __name__ == "__main__":
    main() 