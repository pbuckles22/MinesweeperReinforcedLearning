#!/usr/bin/env python3
"""
Optimized Multi-Step DQN Training

Extended training with optimized hyperparameters for multi-step DQN:
- Longer training episodes (200-300)
- Optimized learning rates for multi-step learning
- Curriculum learning integration
- Advanced hyperparameter tuning
- Comprehensive evaluation and analysis

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
from src.core.dqn_agent_multistep import MultiStepDQNAgent, train_multistep_dqn_agent, evaluate_multistep_dqn_agent


class OptimizedMultiStepTrainer:
    """Optimized trainer for multi-step DQN with extended training."""
    
    def __init__(self):
        self.optimization_results = []
        
    def create_optimized_agent(self, board_size: Tuple[int, int], config: Dict[str, Any]) -> MultiStepDQNAgent:
        """Create multi-step DQN agent with optimized hyperparameters."""
        action_size = board_size[0] * board_size[1]
        
        return MultiStepDQNAgent(
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
            use_double_dqn=True,
            use_dueling=True,
            use_prioritized_replay=True,
            n_steps=config['n_steps'],
            max_n_steps=config['max_n_steps']
        )
    
    def train_with_curriculum(self, episodes_per_stage: int = 200) -> Dict[str, Any]:
        """Train multi-step DQN with curriculum learning."""
        print("🚀 Optimized Multi-Step DQN Training with Curriculum")
        print("=" * 70)
        print("Target: 85-90% win rate on 4x4 boards")
        print("=" * 70)
        
        # Curriculum stages
        curriculum_stages = [
            {
                'name': 'Stage 1: 2x2 Board (Foundation)',
                'board_size': (2, 2),
                'mine_count': 1,
                'episodes': episodes_per_stage // 2,  # Shorter for simple board
                'target_win_rate': 0.8,
                'config': {
                    'learning_rate': 0.0005,  # Higher LR for simple tasks
                    'discount_factor': 0.99,
                    'epsilon': 1.0,
                    'epsilon_decay': 0.995,
                    'epsilon_min': 0.05,
                    'replay_buffer_size': 50000,
                    'batch_size': 32,
                    'target_update_freq': 500,
                    'n_steps': 3,
                    'max_n_steps': 5
                }
            },
            {
                'name': 'Stage 2: 3x3 Board (Intermediate)',
                'board_size': (3, 3),
                'mine_count': 1,
                'episodes': episodes_per_stage,
                'target_win_rate': 0.7,
                'config': {
                    'learning_rate': 0.0003,  # Optimal LR from previous tests
                    'discount_factor': 0.99,
                    'epsilon': 0.8,  # Lower epsilon for transfer learning
                    'epsilon_decay': 0.997,
                    'epsilon_min': 0.03,
                    'replay_buffer_size': 100000,
                    'batch_size': 64,
                    'target_update_freq': 1000,
                    'n_steps': 5,  # Best from previous tests
                    'max_n_steps': 7
                }
            },
            {
                'name': 'Stage 3: 4x4 Board (Advanced)',
                'board_size': (4, 4),
                'mine_count': 1,
                'episodes': episodes_per_stage * 2,  # Longer for complex board
                'target_win_rate': 0.85,
                'config': {
                    'learning_rate': 0.0002,  # Lower LR for fine-tuning
                    'discount_factor': 0.99,
                    'epsilon': 0.6,  # Even lower epsilon
                    'epsilon_decay': 0.998,
                    'epsilon_min': 0.02,
                    'replay_buffer_size': 200000,
                    'batch_size': 64,
                    'target_update_freq': 2000,
                    'n_steps': 5,
                    'max_n_steps': 7
                }
            }
        ]
        
        results = []
        previous_agent = None
        
        for i, stage in enumerate(curriculum_stages):
            print(f"\n📊 Stage {i+1}/{len(curriculum_stages)}: {stage['name']}")
            print(f"   Board size: {stage['board_size']}")
            print(f"   Episodes: {stage['episodes']}")
            print(f"   Target win rate: {stage['target_win_rate']:.1%}")
            print(f"   N-steps: {stage['config']['n_steps']}")
            print("-" * 60)
            
            # Create environment
            env = MinesweeperEnv(
                initial_board_size=stage['board_size'],
                initial_mines=stage['mine_count']
            )
            
            # Create agent
            agent = self.create_optimized_agent(stage['board_size'], stage['config'])
            
            # Train agent
            start_time = time.time()
            training_stats = train_multistep_dqn_agent(
                env, agent, stage['episodes'], stage['mine_count'], eval_freq=20
            )
            training_time = time.time() - start_time
            
            # Evaluate agent
            eval_stats = evaluate_multistep_dqn_agent(agent, env, n_episodes=50)
            
            # Calculate stage metrics
            final_win_rate = training_stats['win_rate']
            eval_win_rate = eval_stats['win_rate']
            target_achieved = eval_win_rate >= stage['target_win_rate']
            
            stage_result = {
                'stage': stage,
                'training_stats': training_stats,
                'eval_stats': eval_stats,
                'training_time': training_time,
                'target_achieved': target_achieved,
                'final_epsilon': agent.epsilon,
                'replay_buffer_size': len(agent.replay_buffer)
            }
            
            results.append(stage_result)
            
            print(f"\n✅ Stage Results:")
            print(f"   Training Win Rate: {final_win_rate:.3f}")
            print(f"   Evaluation Win Rate: {eval_win_rate:.3f}")
            print(f"   Target Achieved: {'✅' if target_achieved else '❌'}")
            print(f"   Training Time: {training_time:.2f}s")
            print(f"   Final Epsilon: {agent.epsilon:.3f}")
            print(f"   Avg N-step Return: {training_stats['avg_n_step_return']:.2f}")
            
            # Save intermediate agent
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            agent_filename = f"optimized_multistep_stage_{i+1}_{timestamp}.pth"
            agent.save_model(agent_filename)
            print(f"💾 Agent saved to {agent_filename}")
            
            previous_agent = agent
        
        # Final comprehensive evaluation
        final_evaluation = self._comprehensive_evaluation(previous_agent, curriculum_stages[-1])
        
        # Print summary
        self._print_optimization_summary(results, final_evaluation)
        
        return {
            'curriculum_results': results,
            'final_evaluation': final_evaluation,
            'final_agent': previous_agent
        }
    
    def _comprehensive_evaluation(self, agent: MultiStepDQNAgent, final_stage: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive evaluation of the final agent."""
        print(f"\n🎯 Comprehensive Final Evaluation")
        print("=" * 50)
        
        board_size = final_stage['board_size']
        mine_count = final_stage['mine_count']
        
        # Create environment
        env = MinesweeperEnv(initial_board_size=board_size, initial_mines=mine_count)
        
        # Multiple evaluation runs for statistical significance
        eval_runs = []
        for run in range(5):
            eval_stats = evaluate_multistep_dqn_agent(agent, env, n_episodes=20)
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
        
        print(f"📊 Statistical Results (5 runs of 20 episodes each):")
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
    
    def _print_optimization_summary(self, results: List[Dict[str, Any]], final_evaluation: Dict[str, Any]):
        """Print comprehensive optimization summary."""
        print("\n" + "=" * 70)
        print("🏆 OPTIMIZED MULTI-STEP DQN TRAINING SUMMARY")
        print("=" * 70)
        
        total_training_time = sum(r['training_time'] for r in results)
        total_episodes = sum(r['stage']['episodes'] for r in results)
        stages_completed = len(results)
        targets_achieved = sum(1 for r in results if r['target_achieved'])
        
        print(f"\n📊 Overall Training Results:")
        print(f"   Stages Completed: {stages_completed}/{len(results)}")
        print(f"   Targets Achieved: {targets_achieved}/{stages_completed}")
        print(f"   Total Episodes: {total_episodes}")
        print(f"   Total Training Time: {total_training_time:.2f}s")
        
        print(f"\n📈 Stage-by-Stage Results:")
        for i, result in enumerate(results):
            stage = result['stage']
            eval_stats = result['eval_stats']
            target_achieved = result['target_achieved']
            
            print(f"   {i+1}. {stage['name']}")
            print(f"      Win Rate: {eval_stats['win_rate']:.3f} (Target: {stage['target_win_rate']:.1%})")
            print(f"      Status: {'✅ Achieved' if target_achieved else '❌ Not Achieved'}")
            print(f"      Training Time: {result['training_time']:.2f}s")
            print(f"      N-steps: {stage['config']['n_steps']}")
        
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
        filename = f"optimized_multistep_results_{timestamp}.json"
        
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            serializable_result = {}
            for key, value in result.items():
                if key == 'stage':
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
            'total_stages': len(results),
            'stages_completed': stages_completed,
            'targets_achieved': targets_achieved,
            'total_episodes': total_episodes,
            'total_training_time': total_training_time,
            'final_win_rate': final_win_rate,
            'stage_results': serializable_results,
            'final_evaluation': final_evaluation
        }
        
        with open(filename, 'w') as f:
            json.dump(comprehensive_summary, f, indent=2)
        
        print(f"\n💾 Comprehensive results saved to {filename}")


def main():
    """Run optimized multi-step DQN training."""
    print("🧪 Optimized Multi-Step DQN Training")
    print("=" * 50)
    
    try:
        # Create optimized trainer
        trainer = OptimizedMultiStepTrainer()
        
        # Run optimized training with curriculum
        results = trainer.train_with_curriculum(episodes_per_stage=200)
        
        print(f"\n✅ Optimized multi-step DQN training completed!")
        print(f"📊 Final win rate: {results['final_evaluation']['mean_win_rate']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during optimized training: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Optimized training completed successfully!")
    else:
        print("\n💥 Optimized training failed. Please check the error messages above.")
        sys.exit(1) 