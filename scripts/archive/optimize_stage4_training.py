#!/usr/bin/env python3
"""
Stage 4 Optimization Training Script
Focused training on 4x4 board with 4 mines to improve performance.
"""

import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from src.core.minesweeper_env import MinesweeperEnv
from src.core.dqn_agent import DQNAgent, train_dqn_agent, evaluate_dqn_agent

def optimize_stage4():
    """Optimize training for stage 4 (4 mines on 4x4 board)."""
    
    print("🎯 Stage 4 Optimization Training")
    print("=" * 50)
    print("📋 Board Size: (4, 4)")
    print("📋 Mine Count: 4")
    print("📋 Target: Improve from 0% to >5% win rate")
    print("=" * 50)
    
    # Create environment
    board_size = (4, 4)
    mine_count = 4
    env = MinesweeperEnv(initial_board_size=board_size, initial_mines=mine_count)
    
    # Create optimized DQN agent for stage 4
    action_size = board_size[0] * board_size[1]
    
    # Try different hyperparameter configurations
    configs = [
        {
            'name': 'Conservative Learning',
            'learning_rate': 0.0001,
            'epsilon_decay': 0.9995,
            'epsilon_min': 0.15,
            'batch_size': 16,
            'target_update_freq': 100
        },
        {
            'name': 'Balanced Learning',
            'learning_rate': 0.0003,
            'epsilon_decay': 0.999,
            'epsilon_min': 0.12,
            'batch_size': 32,
            'target_update_freq': 200
        },
        {
            'name': 'Aggressive Learning',
            'learning_rate': 0.0005,
            'epsilon_decay': 0.998,
            'epsilon_min': 0.10,
            'batch_size': 64,
            'target_update_freq': 300
        }
    ]
    
    results = []
    
    for config_idx, config in enumerate(configs, 1):
        print(f"\n🔧 Testing Configuration {config_idx}: {config['name']}")
        print("-" * 40)
        
        # Create agent with current config
        agent = DQNAgent(
            board_size=board_size,
            action_size=action_size,
            learning_rate=config['learning_rate'],
            epsilon=1.0,
            epsilon_decay=config['epsilon_decay'],
            epsilon_min=config['epsilon_min'],
            batch_size=config['batch_size'],
            target_update_freq=config['target_update_freq'],
            device='cpu'
        )
        
        print(f"🤖 Agent created with {config['name']}")
        print(f"   Learning rate: {config['learning_rate']}")
        print(f"   Epsilon decay: {config['epsilon_decay']}")
        print(f"   Epsilon min: {config['epsilon_min']}")
        print(f"   Batch size: {config['batch_size']}")
        
        # Train with longer episodes
        episodes = 5000  # Longer training for optimization
        print(f"\n🎯 Training for {episodes} episodes...")
        
        start_time = time.time()
        training_stats = train_dqn_agent(env, agent, episodes, mine_count, eval_freq=200)
        training_time = time.time() - start_time
        
        # Evaluate performance
        print(f"\n🔍 Evaluating performance...")
        eval_stats = evaluate_dqn_agent(agent, env, n_episodes=200)  # More evaluation episodes
        
        # Store results
        result = {
            'config_name': config['name'],
            'config': config,
            'training_stats': training_stats,
            'evaluation_stats': eval_stats,
            'training_time': training_time,
            'episodes': episodes,
            'win_rate': eval_stats['win_rate'],
            'mean_reward': eval_stats['mean_reward']
        }
        
        results.append(result)
        
        print(f"\n📊 Configuration {config_idx} Results:")
        print(f"   Win Rate: {eval_stats['win_rate']:.1%}")
        print(f"   Mean Reward: {eval_stats['mean_reward']:.2f}")
        print(f"   Training Time: {training_time:.1f}s")
        print(f"   Final Epsilon: {agent.epsilon:.3f}")
        
        # Save agent checkpoint
        checkpoint_path = f"models/dqn_stage4_optimized_{config_idx}.pth"
        os.makedirs("models", exist_ok=True)
        agent.save_model(checkpoint_path)
        print(f"💾 Saved checkpoint: {checkpoint_path}")
    
    # Find best configuration
    best_result = max(results, key=lambda x: x['win_rate'])
    
    print(f"\n🏆 OPTIMIZATION RESULTS")
    print("=" * 50)
    print(f"Best Configuration: {best_result['config_name']}")
    print(f"Best Win Rate: {best_result['win_rate']:.1%}")
    print(f"Best Mean Reward: {best_result['mean_reward']:.2f}")
    
    print(f"\n📋 All Configurations:")
    for i, result in enumerate(results, 1):
        print(f"   {i}. {result['config_name']}: {result['win_rate']:.1%} win rate")
    
    # Save optimization results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"training_stats/stage4_optimization_{timestamp}.json"
    os.makedirs("training_stats", exist_ok=True)
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    serializable_results = convert_numpy(results)
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    return best_result

if __name__ == "__main__":
    best_result = optimize_stage4()
    print(f"\n🎯 Stage 4 optimization completed!")
    print(f"Best win rate achieved: {best_result['win_rate']:.1%}") 