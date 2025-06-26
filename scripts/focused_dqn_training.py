#!/usr/bin/env python3
"""
Focused DQN Training

Targeted training approach to address core learning issues:
- Simplified architecture for better learning
- Improved exploration strategies
- Focused training on specific board sizes
- Better hyperparameter tuning
- Comprehensive analysis

Target: 85-90% win rate on 4x4 boards
"""

import sys
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.minesweeper_env import MinesweeperEnv
from src.core.dqn_agent_enhanced import EnhancedDQNAgent, train_enhanced_dqn_agent, evaluate_enhanced_dqn_agent


class FocusedDQNTrainer:
    """Focused trainer for DQN with targeted learning strategies."""
    
    def __init__(self):
        self.training_results = []
        
    def create_focused_agent(self, board_size: Tuple[int, int], config: Dict[str, Any]) -> EnhancedDQNAgent:
        """Create enhanced DQN agent with focused hyperparameters."""
        action_size = board_size[0] * board_size[1]
        
        return EnhancedDQNAgent(
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
            device='cpu',
            use_double_dqn=config['use_double_dqn'],
            use_dueling=config['use_dueling'],
            use_prioritized_replay=config['use_prioritized_replay']
        )
    
    def train_focused_4x4(self, episodes: int = 1000) -> Dict[str, Any]:
        """Focused training on 4x4 board with optimized strategy."""
        print("🎯 Focused DQN Training on 4x4 Board")
        print("=" * 60)
        print("Strategy: Direct training with optimized hyperparameters")
        print("Target: 85-90% win rate")
        print("=" * 60)
        
        # Multiple configurations to test
        configurations = [
            {
                'name': 'Conservative Learning',
                'config': {
                    'learning_rate': 0.0001,  # Very low LR for stable learning
                    'discount_factor': 0.99,
                    'epsilon': 1.0,
                    'epsilon_decay': 0.9995,  # Very slow decay
                    'epsilon_min': 0.05,
                    'replay_buffer_size': 100000,
                    'batch_size': 32,
                    'target_update_freq': 1000,
                    'use_double_dqn': True,
                    'use_dueling': True,
                    'use_prioritized_replay': True
                }
            },
            {
                'name': 'Aggressive Learning',
                'config': {
                    'learning_rate': 0.001,  # Higher LR for faster learning
                    'discount_factor': 0.99,
                    'epsilon': 1.0,
                    'epsilon_decay': 0.998,  # Faster decay
                    'epsilon_min': 0.1,
                    'replay_buffer_size': 50000,
                    'batch_size': 64,
                    'target_update_freq': 500,
                    'use_double_dqn': True,
                    'use_dueling': False,  # Simpler architecture
                    'use_prioritized_replay': False  # Simpler replay
                }
            },
            {
                'name': 'Balanced Learning',
                'config': {
                    'learning_rate': 0.0003,  # Balanced LR
                    'discount_factor': 0.99,
                    'epsilon': 1.0,
                    'epsilon_decay': 0.999,  # Balanced decay
                    'epsilon_min': 0.02,
                    'replay_buffer_size': 200000,
                    'batch_size': 64,
                    'target_update_freq': 2000,
                    'use_double_dqn': True,
                    'use_dueling': True,
                    'use_prioritized_replay': True
                }
            }
        ]
        
        results = []
        best_agent = None
        best_win_rate = 0.0
        
        for i, config_info in enumerate(configurations):
            print(f"\n🔬 Testing Configuration {i+1}/{len(configurations)}: {config_info['name']}")
            print(f"   Learning Rate: {config_info['config']['learning_rate']}")
            print(f"   Epsilon Decay: {config_info['config']['epsilon_decay']}")
            print(f"   Double DQN: {config_info['config']['use_double_dqn']}")
            print(f"   Dueling: {config_info['config']['use_dueling']}")
            print(f"   Prioritized Replay: {config_info['config']['use_prioritized_replay']}")
            print("-" * 60)
            
            # Create environment
            env = MinesweeperEnv(initial_board_size=(4, 4), initial_mines=1)
            
            # Create agent
            agent = self.create_focused_agent((4, 4), config_info['config'])
            
            # Train agent
            start_time = time.time()
            training_stats = train_enhanced_dqn_agent(
                env, agent, episodes, 1, eval_freq=20
            )
            training_time = time.time() - start_time
            
            # Extended evaluation
            eval_stats = evaluate_enhanced_dqn_agent(agent, env, n_episodes=200)
            
            # Calculate metrics
            final_win_rate = training_stats['win_rate']
            eval_win_rate = eval_stats['win_rate']
            
            config_result = {
                'config_name': config_info['name'],
                'config': config_info['config'],
                'training_stats': training_stats,
                'eval_stats': eval_stats,
                'training_time': training_time,
                'final_win_rate': final_win_rate,
                'eval_win_rate': eval_win_rate
            }
            
            results.append(config_result)
            
            print(f"\n✅ Configuration Results:")
            print(f"   Training Win Rate: {final_win_rate:.3f}")
            print(f"   Evaluation Win Rate: {eval_win_rate:.3f}")
            print(f"   Training Time: {training_time:.2f}s")
            print(f"   Final Epsilon: {agent.epsilon:.3f}")
            print(f"   Mean Reward: {eval_stats['mean_reward']:.2f}")
            print(f"   Mean Length: {eval_stats['mean_length']:.1f}")
            
            # Track best agent
            if eval_win_rate > best_win_rate:
                best_win_rate = eval_win_rate
                best_agent = agent
                print(f"   🏆 NEW BEST: {eval_win_rate:.3f} win rate!")
            
            # Save agent
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            agent_filename = f"focused_dqn_config_{i+1}_{timestamp}.pth"
            agent.save_model(agent_filename)
            print(f"💾 Agent saved to {agent_filename}")
        
        # Final comprehensive evaluation of best agent
        if best_agent:
            final_evaluation = self._comprehensive_evaluation(best_agent)
        else:
            final_evaluation = {'mean_win_rate': 0.0}
        
        # Print summary
        self._print_focused_summary(results, final_evaluation)
        
        return {
            'config_results': results,
            'final_evaluation': final_evaluation,
            'best_agent': best_agent,
            'best_win_rate': best_win_rate
        }
    
    def _comprehensive_evaluation(self, agent: EnhancedDQNAgent) -> Dict[str, Any]:
        """Comprehensive evaluation of the best agent."""
        print(f"\n🎯 Comprehensive Final Evaluation")
        print("=" * 50)
        
        # Create environment
        env = MinesweeperEnv(initial_board_size=(4, 4), initial_mines=1)
        
        # Multiple evaluation runs for statistical significance
        eval_runs = []
        for run in range(15):  # More runs for better statistics
            eval_stats = evaluate_enhanced_dqn_agent(agent, env, n_episodes=50)
            eval_runs.append(eval_stats)
        
        # Calculate statistics
        win_rates = [run['win_rate'] for run in eval_runs]
        mean_rewards = [run['mean_reward'] for run in eval_runs]
        mean_lengths = [run['mean_length'] for run in eval_runs]
        
        comprehensive_results = {
            'mean_win_rate': np.mean(win_rates),
            'std_win_rate': np.std(win_rates),
            'mean_reward': np.mean(mean_rewards),
            'std_reward': np.std(mean_rewards),
            'mean_length': np.mean(mean_lengths),
            'std_length': np.std(mean_lengths),
            'min_win_rate': np.min(win_rates),
            'max_win_rate': np.max(win_rates),
            'all_runs': eval_runs
        }
        
        print(f"📊 Statistical Results (15 runs of 50 episodes each):")
        print(f"   Mean Win Rate: {comprehensive_results['mean_win_rate']:.3f} ± {comprehensive_results['std_win_rate']:.3f}")
        print(f"   Win Rate Range: {comprehensive_results['min_win_rate']:.3f} - {comprehensive_results['max_win_rate']:.3f}")
        print(f"   Mean Reward: {comprehensive_results['mean_reward']:.2f} ± {comprehensive_results['std_reward']:.2f}")
        print(f"   Mean Length: {comprehensive_results['mean_length']:.1f} ± {comprehensive_results['std_length']:.1f}")
        
        # Performance assessment
        if comprehensive_results['mean_win_rate'] >= 0.90:
            performance_level = "🎉 EXCEPTIONAL"
        elif comprehensive_results['mean_win_rate'] >= 0.85:
            performance_level = "🏆 EXCELLENT"
        elif comprehensive_results['mean_win_rate'] >= 0.80:
            performance_level = "✅ VERY GOOD"
        elif comprehensive_results['mean_win_rate'] >= 0.75:
            performance_level = "👍 GOOD"
        else:
            performance_level = "⚠️  NEEDS IMPROVEMENT"
        
        print(f"   Performance Level: {performance_level}")
        
        return comprehensive_results
    
    def _print_focused_summary(self, results: List[Dict[str, Any]], final_evaluation: Dict[str, Any]):
        """Print comprehensive focused training summary."""
        print("\n" + "=" * 70)
        print("🏆 FOCUSED DQN TRAINING SUMMARY")
        print("=" * 70)
        
        total_training_time = sum(r['training_time'] for r in results)
        total_episodes = len(results) * 1000  # Assuming 1000 episodes per config
        
        print(f"\n📊 Overall Training Results:")
        print(f"   Configurations Tested: {len(results)}")
        print(f"   Total Episodes: {total_episodes}")
        print(f"   Total Training Time: {total_training_time:.2f}s")
        
        print(f"\n📈 Configuration-by-Configuration Results:")
        for i, result in enumerate(results):
            config_name = result['config_name']
            eval_win_rate = result['eval_win_rate']
            training_time = result['training_time']
            config = result['config']
            
            print(f"   {i+1}. {config_name}")
            print(f"      Win Rate: {eval_win_rate:.3f}")
            print(f"      Training Time: {training_time:.2f}s")
            print(f"      LR: {config['learning_rate']}")
            print(f"      Epsilon Decay: {config['epsilon_decay']}")
            print(f"      Architecture: Double={config['use_double_dqn']}, Dueling={config['use_dueling']}, Prioritized={config['use_prioritized_replay']}")
        
        # Final performance assessment
        final_win_rate = final_evaluation['mean_win_rate']
        print(f"\n🎯 Final Performance Assessment:")
        print(f"   Mean Win Rate: {final_win_rate:.3f} ± {final_evaluation['std_win_rate']:.3f}")
        print(f"   Win Rate Range: {final_evaluation['min_win_rate']:.3f} - {final_evaluation['max_win_rate']:.3f}")
        print(f"   Mean Reward: {final_evaluation['mean_reward']:.2f} ± {final_evaluation['std_reward']:.2f}")
        print(f"   Mean Length: {final_evaluation['mean_length']:.1f} ± {final_evaluation['std_length']:.1f}")
        
        # Goal achievement
        if final_win_rate >= 0.90:
            print(f"   🎉 TARGET EXCEEDED: Achieved 90%+ win rate!")
        elif final_win_rate >= 0.85:
            print(f"   🏆 TARGET ACHIEVED: Achieved 85%+ win rate!")
        elif final_win_rate >= 0.80:
            print(f"   ✅ CLOSE TO TARGET: 80%+ win rate achieved")
        else:
            print(f"   ⚠️  TARGET NOT REACHED: Below 80% win rate")
        
        # Save comprehensive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"focused_dqn_results_{timestamp}.json"
        
        # Convert results to serializable format
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
        
        comprehensive_summary = {
            'timestamp': timestamp,
            'total_configs': len(results),
            'total_episodes': total_episodes,
            'total_training_time': total_training_time,
            'final_win_rate': final_win_rate,
            'config_results': serializable_results,
            'final_evaluation': final_evaluation
        }
        
        with open(filename, 'w') as f:
            json.dump(comprehensive_summary, f, indent=2)
        
        print(f"\n💾 Comprehensive results saved to {filename}")


def main():
    """Run focused DQN training."""
    print("🧪 Focused DQN Training")
    print("=" * 50)
    
    try:
        # Create focused trainer
        trainer = FocusedDQNTrainer()
        
        # Run focused training
        results = trainer.train_focused_4x4(episodes=1000)
        
        print(f"\n✅ Focused DQN training completed!")
        print(f"📊 Best win rate: {results['best_win_rate']:.3f}")
        print(f"📊 Final mean win rate: {results['final_evaluation']['mean_win_rate']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during focused training: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Focused training completed successfully!")
    else:
        print("\n💥 Focused training failed. Please check the error messages above.")
        sys.exit(1) 