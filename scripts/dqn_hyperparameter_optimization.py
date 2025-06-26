#!/usr/bin/env python3
"""
DQN Hyperparameter Optimization Script

Systematically tests different hyperparameter combinations to find optimal settings
for the DQN agent. Based on the conv128x4_dense512x2 architecture.
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


class HyperparameterOptimizer:
    """Systematic hyperparameter optimization for DQN."""
    
    def __init__(self, board_size: Tuple[int, int] = (4, 4), mine_count: int = 1):
        self.board_size = board_size
        self.mine_count = mine_count
        self.action_size = board_size[0] * board_size[1]
        self.results = []
        
    def create_agent(self, config: Dict[str, Any]) -> DQNAgent:
        """Create DQN agent with given configuration."""
        return DQNAgent(
            board_size=self.board_size,
            action_size=self.action_size,
            learning_rate=config['learning_rate'],
            discount_factor=config['discount_factor'],
            epsilon=config['epsilon'],
            epsilon_decay=config['epsilon_decay'],
            epsilon_min=config['epsilon_min'],
            replay_buffer_size=config['replay_buffer_size'],
            batch_size=config['batch_size'],
            target_update_freq=config['target_update_freq'],
            device='cpu'  # Use CPU for consistent benchmarking
        )
    
    def evaluate_config(self, config: Dict[str, Any], episodes: int = 100) -> Dict[str, Any]:
        """Evaluate a single hyperparameter configuration."""
        print(f"\n🔬 Testing config: {config['name']}")
        print(f"   LR: {config['learning_rate']}, Batch: {config['batch_size']}, "
              f"Epsilon: {config['epsilon_decay']}")
        
        # Create environment and agent
        env = MinesweeperEnv(initial_board_size=self.board_size, initial_mines=self.mine_count)
        agent = self.create_agent(config)
        
        # Train agent
        start_time = time.time()
        training_stats = train_dqn_agent(env, agent, episodes, self.mine_count, eval_freq=20)
        training_time = time.time() - start_time
        
        # Evaluate agent
        eval_stats = evaluate_dqn_agent(agent, env, n_episodes=50)
        
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
            'final_epsilon': agent.epsilon,
            'replay_buffer_size': len(agent.replay_buffer)
        }
        
        print(f"   Results: Train WR: {final_win_rate:.3f}, Eval WR: {eval_win_rate:.3f}, "
              f"Stability: {win_rate_stability:.3f}, Efficiency: {efficiency:.3f}")
        
        return result
    
    def run_optimization(self, episodes_per_config: int = 100) -> List[Dict[str, Any]]:
        """Run systematic hyperparameter optimization."""
        print(f"🚀 Starting DQN Hyperparameter Optimization")
        print(f"   Board size: {self.board_size}")
        print(f"   Mine count: {self.mine_count}")
        print(f"   Episodes per config: {episodes_per_config}")
        print("=" * 80)
        
        # Define hyperparameter configurations to test
        configs = self._generate_configs()
        
        # Test each configuration
        for i, config in enumerate(configs):
            print(f"\n📊 Configuration {i+1}/{len(configs)}")
            result = self.evaluate_config(config, episodes_per_config)
            self.results.append(result)
            
            # Save intermediate results
            self._save_results()
        
        # Sort results by evaluation win rate
        self.results.sort(key=lambda x: x['eval_win_rate'], reverse=True)
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _generate_configs(self) -> List[Dict[str, Any]]:
        """Generate hyperparameter configurations to test."""
        configs = []
        
        # Learning rate optimization
        learning_rates = [0.0001, 0.0003, 0.0005, 0.001, 0.003]
        
        # Batch size optimization
        batch_sizes = [16, 32, 64, 128]
        
        # Epsilon decay optimization
        epsilon_decays = [0.995, 0.999, 0.9995, 0.9999]
        
        # Target network update frequency
        target_update_freqs = [500, 1000, 2000, 5000]
        
        # Replay buffer sizes
        replay_buffer_sizes = [10000, 50000, 100000, 200000]
        
        config_id = 1
        
        # Test learning rate impact
        for lr in learning_rates:
            configs.append({
                'name': f'LR_{lr}',
                'learning_rate': lr,
                'discount_factor': 0.99,
                'epsilon': 1.0,
                'epsilon_decay': 0.995,
                'epsilon_min': 0.01,
                'replay_buffer_size': 100000,
                'batch_size': 64,
                'target_update_freq': 1000
            })
            config_id += 1
        
        # Test batch size impact
        for batch_size in batch_sizes:
            configs.append({
                'name': f'Batch_{batch_size}',
                'learning_rate': 0.001,
                'discount_factor': 0.99,
                'epsilon': 1.0,
                'epsilon_decay': 0.995,
                'epsilon_min': 0.01,
                'replay_buffer_size': 100000,
                'batch_size': batch_size,
                'target_update_freq': 1000
            })
            config_id += 1
        
        # Test epsilon decay impact
        for epsilon_decay in epsilon_decays:
            configs.append({
                'name': f'Epsilon_{epsilon_decay}',
                'learning_rate': 0.001,
                'discount_factor': 0.99,
                'epsilon': 1.0,
                'epsilon_decay': epsilon_decay,
                'epsilon_min': 0.01,
                'replay_buffer_size': 100000,
                'batch_size': 64,
                'target_update_freq': 1000
            })
            config_id += 1
        
        # Test target network update frequency
        for target_freq in target_update_freqs:
            configs.append({
                'name': f'Target_{target_freq}',
                'learning_rate': 0.001,
                'discount_factor': 0.99,
                'epsilon': 1.0,
                'epsilon_decay': 0.995,
                'epsilon_min': 0.01,
                'replay_buffer_size': 100000,
                'batch_size': 64,
                'target_update_freq': target_freq
            })
            config_id += 1
        
        # Test replay buffer size
        for buffer_size in replay_buffer_sizes:
            configs.append({
                'name': f'Buffer_{buffer_size}',
                'learning_rate': 0.001,
                'discount_factor': 0.99,
                'epsilon': 1.0,
                'epsilon_decay': 0.995,
                'epsilon_min': 0.01,
                'replay_buffer_size': buffer_size,
                'batch_size': 64,
                'target_update_freq': 1000
            })
            config_id += 1
        
        # Best combinations (based on initial results)
        best_combinations = [
            {
                'name': 'Best_LR_Batch',
                'learning_rate': 0.0005,  # Lower learning rate
                'discount_factor': 0.99,
                'epsilon': 1.0,
                'epsilon_decay': 0.999,  # Slower decay
                'epsilon_min': 0.01,
                'replay_buffer_size': 100000,
                'batch_size': 32,  # Smaller batch
                'target_update_freq': 2000  # Less frequent updates
            },
            {
                'name': 'Conservative',
                'learning_rate': 0.0003,  # Very conservative
                'discount_factor': 0.99,
                'epsilon': 1.0,
                'epsilon_decay': 0.9995,  # Very slow decay
                'epsilon_min': 0.05,  # Higher minimum
                'replay_buffer_size': 200000,  # Larger buffer
                'batch_size': 16,  # Small batch
                'target_update_freq': 5000  # Very infrequent updates
            }
        ]
        
        configs.extend(best_combinations)
        
        return configs
    
    def _save_results(self):
        """Save optimization results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dqn_optimization_results_{timestamp}.json"
        
        # Convert numpy types to native Python types for JSON serialization
        serializable_results = []
        for result in self.results:
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
        
        print(f"💾 Results saved to {filename}")
    
    def _print_summary(self):
        """Print optimization summary."""
        print("\n" + "=" * 80)
        print("🏆 OPTIMIZATION SUMMARY")
        print("=" * 80)
        
        if not self.results:
            print("❌ No results to summarize")
            return
        
        # Top 5 configurations
        print("\n🥇 TOP 5 CONFIGURATIONS:")
        for i, result in enumerate(self.results[:5]):
            config = result['config']
            print(f"{i+1}. {config['name']}: Eval WR: {result['eval_win_rate']:.3f}, "
                  f"Stability: {result['win_rate_stability']:.3f}, "
                  f"Efficiency: {result['efficiency']:.3f}")
            print(f"   LR: {config['learning_rate']}, Batch: {config['batch_size']}, "
                  f"Epsilon: {config['epsilon_decay']}, Target: {config['target_update_freq']}")
        
        # Best configuration
        best = self.results[0]
        print(f"\n🎯 BEST CONFIGURATION: {best['config']['name']}")
        print(f"   Evaluation Win Rate: {best['eval_win_rate']:.3f}")
        print(f"   Training Win Rate: {best['training_win_rate']:.3f}")
        print(f"   Stability: {best['win_rate_stability']:.3f}")
        print(f"   Efficiency: {best['efficiency']:.3f}")
        print(f"   Mean Reward: {best['mean_reward']:.2f}")
        print(f"   Mean Length: {best['mean_length']:.1f} steps")
        
        # Parameter recommendations
        print(f"\n📋 PARAMETER RECOMMENDATIONS:")
        print(f"   Learning Rate: {best['config']['learning_rate']}")
        print(f"   Batch Size: {best['config']['batch_size']}")
        print(f"   Epsilon Decay: {best['config']['epsilon_decay']}")
        print(f"   Target Update Frequency: {best['config']['target_update_freq']}")
        print(f"   Replay Buffer Size: {best['config']['replay_buffer_size']}")


def main():
    """Run hyperparameter optimization."""
    print("🧪 DQN Hyperparameter Optimization")
    print("=" * 50)
    
    # Create optimizer
    optimizer = HyperparameterOptimizer(board_size=(4, 4), mine_count=1)
    
    # Run optimization
    results = optimizer.run_optimization(episodes_per_config=100)
    
    print(f"\n✅ Optimization completed! Tested {len(results)} configurations.")
    print(f"📊 Results saved to JSON file for further analysis.")


if __name__ == "__main__":
    main() 