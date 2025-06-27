#!/usr/bin/env python3
"""
Extended Curriculum Learning for Minesweeper DQN

Adaptive curriculum with early stopping and slower progression for 10-hour training:
- Stage 1 (4x4): Target 90% (mastery before progression)
- Stage 2 (5x5): Target 70% (strong foundation)
- Stage 3 (6x6): Target 50% (challenging but achievable)
- Stage 4 (8x8): Target 20% (ultimate challenge)

Features:
- Adaptive progression with early stopping
- Periodic evaluation and best model saving
- Slower, more thorough training per stage
- Transfer learning between stages
- MLflow tracking (optional)
"""

import sys
import os
import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.minesweeper_env import MinesweeperEnv
from core.dqn_agent_enhanced import EnhancedDQNAgent, train_enhanced_dqn_agent, evaluate_enhanced_dqn_agent

# Try to import MLflow for tracking
try:
    import mlflow
    MLFLOW_AVAILABLE = True
    print("✅ MLflow available for experiment tracking")
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️  MLflow not available - running without tracking")


class AdaptiveCurriculumTrainer:
    """Adaptive curriculum trainer with early stopping and best model saving."""
    
    def __init__(self, use_mlflow: bool = False):
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        self.curriculum_results = []
        self.current_agent = None
        self.current_model_path = None
        self.best_models = {}  # Store best model per stage
        
        # Adaptive curriculum stages for 10-hour training
        self.curriculum_stages = [
            {
                'name': 'Stage 1: 4x4 Board Mastery',
                'board_size': (4, 4),
                'mines': 1,
                'min_episodes': 5000,
                'max_episodes': 20000,
                'target_win_rate': 0.90,  # High target for mastery
                'eval_freq': 500,  # Evaluate every 500 episodes
                'eval_episodes': 50,
                'eval_runs': 5,
                'early_stop_consecutive': 3,  # Stop after 3 consecutive target achievements
                'description': 'Mastery training on small board'
            },
            {
                'name': 'Stage 2: 5x5 Board Foundation', 
                'board_size': (5, 5),
                'mines': 2,
                'min_episodes': 7500,
                'max_episodes': 25000,
                'target_win_rate': 0.70,  # Strong foundation target
                'eval_freq': 500,
                'eval_episodes': 50,
                'eval_runs': 5,
                'early_stop_consecutive': 3,
                'description': 'Transfer learning to medium board'
            },
            {
                'name': 'Stage 3: 6x6 Board Challenge',
                'board_size': (6, 6), 
                'mines': 3,
                'min_episodes': 10000,
                'max_episodes': 30000,
                'target_win_rate': 0.50,  # Challenging but achievable
                'eval_freq': 500,
                'eval_episodes': 50,
                'eval_runs': 5,
                'early_stop_consecutive': 3,
                'description': 'Advanced training on larger board'
            },
            {
                'name': 'Stage 4: 8x8 Board Ultimate',
                'board_size': (8, 8),
                'mines': 6,
                'min_episodes': 15000,
                'max_episodes': 40000,
                'target_win_rate': 0.20,  # Very challenging target
                'eval_freq': 500,
                'eval_episodes': 50,
                'eval_runs': 5,
                'early_stop_consecutive': 3,
                'description': 'Ultimate challenge on large board'
            }
        ]
        
        # Enhanced configuration with regularization
        self.agent_config = {
            'learning_rate': 0.0001,          # Conservative learning rate
            'epsilon_decay': 0.9995,          # Slow exploration decay
            'epsilon_min': 0.05,              # Higher minimum exploration
            'replay_buffer_size': 100000,     # Large buffer
            'batch_size': 32,                 # Smaller batches for stability
            'target_update_freq': 1000,       # Standard target updates
            'use_double_dqn': True,           # Advanced features
            'use_dueling': True,              # Advanced features
            'use_prioritized_replay': True    # Advanced features
        }
    
    def create_agent(self, board_size: Tuple[int, int]) -> EnhancedDQNAgent:
        """Create DQN agent with enhanced configuration."""
        action_size = board_size[0] * board_size[1]
        
        return EnhancedDQNAgent(
            board_size=board_size,
            action_size=action_size,
            learning_rate=self.agent_config['learning_rate'],
            discount_factor=0.99,
            epsilon=1.0,
            epsilon_decay=self.agent_config['epsilon_decay'],
            epsilon_min=self.agent_config['epsilon_min'],
            replay_buffer_size=self.agent_config['replay_buffer_size'],
            batch_size=self.agent_config['batch_size'],
            target_update_freq=self.agent_config['target_update_freq'],
            device='cpu',
            use_double_dqn=self.agent_config['use_double_dqn'],
            use_dueling=self.agent_config['use_dueling'],
            use_prioritized_replay=self.agent_config['use_prioritized_replay']
        )
    
    def comprehensive_evaluation(self, agent: EnhancedDQNAgent, env, 
                               n_episodes: int = 50, n_runs: int = 5) -> Dict[str, Any]:
        """Comprehensive evaluation with multiple runs."""
        print(f"   🔍 Running Evaluation ({n_runs} runs of {n_episodes} episodes each)...")
        
        all_results = []
        
        for run in range(n_runs):
            wins = 0
            total_rewards = []
            episode_lengths = []
            
            for episode in range(n_episodes):
                state, info = env.reset()
                done = False
                total_reward = 0
                steps = 0
                max_steps = 200
                
                while not done and steps < max_steps:
                    action = agent.choose_action(state, training=False)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    state = next_state
                    total_reward += reward
                    steps += 1
                
                # Check if episode was won
                won = False
                if info and isinstance(info, dict):
                    won = info.get('won', False)
                elif info and isinstance(info, list) and len(info) > 0:
                    won = info[0].get('won', False)
                
                if won:
                    wins += 1
                
                total_rewards.append(total_reward)
                episode_lengths.append(steps)
            
            # Calculate run statistics
            run_win_rate = wins / n_episodes
            run_mean_reward = np.mean(total_rewards)
            run_mean_length = np.mean(episode_lengths)
            
            all_results.append({
                'win_rate': run_win_rate,
                'mean_reward': run_mean_reward,
                'mean_length': run_mean_length
            })
        
        # Calculate overall statistics
        win_rates = [r['win_rate'] for r in all_results]
        mean_rewards = [r['mean_reward'] for r in all_results]
        mean_lengths = [r['mean_length'] for r in all_results]
        
        overall_stats = {
            'win_rate': np.mean(win_rates),
            'win_rate_std': np.std(win_rates),
            'win_rate_range': (np.min(win_rates), np.max(win_rates)),
            'mean_reward': np.mean(mean_rewards),
            'mean_reward_std': np.std(mean_rewards),
            'mean_length': np.mean(mean_lengths),
            'mean_length_std': np.std(mean_lengths),
            'all_results': all_results
        }
        
        print(f"   📊 Evaluation: {overall_stats['win_rate']:.3f} ± {overall_stats['win_rate_std']:.3f}")
        
        return overall_stats
    
    def train_stage_adaptive(self, stage: Dict[str, Any], stage_num: int) -> Dict[str, Any]:
        """Train a single curriculum stage with adaptive progression and early stopping."""
        print(f"\n🎯 {stage['name']}")
        print(f"   {stage['description']}")
        print(f"   Board: {stage['board_size']}, Mines: {stage['mines']}")
        print(f"   Target: {stage['target_win_rate']:.1%}")
        print(f"   Episodes: {stage['min_episodes']:,} - {stage['max_episodes']:,}")
        print(f"   Early Stop: After {stage['early_stop_consecutive']} consecutive target achievements")
        print("-" * 70)
        
        # Create environment
        env = MinesweeperEnv(
            initial_board_size=stage['board_size'],
            initial_mines=stage['mines']
        )
        
        # Create agent
        agent = self.create_agent(stage['board_size'])
        
        # Try to load previous model for transfer learning
        transfer_success = False
        if self.current_model_path and os.path.exists(self.current_model_path):
            print(f"   🔄 Attempting transfer from: {self.current_model_path}")
            transfer_success = agent.load_model(self.current_model_path)
        
        # Training variables
        start_time = time.time()
        best_eval_win_rate = 0.0
        best_model_path = None
        consecutive_target_achievements = 0
        eval_history = []
        
        # Training loop with periodic evaluation
        for episode in range(0, stage['max_episodes'], stage['eval_freq']):
            # Train for eval_freq episodes
            training_stats = self._train_episodes(
                env, agent, stage['eval_freq'], stage['mines']
            )
            
            # Evaluate
            eval_stats = self.comprehensive_evaluation(
                agent, env, stage['eval_episodes'], stage['eval_runs']
            )
            
            eval_win_rate = eval_stats['win_rate']
            eval_history.append(eval_win_rate)
            
            # Check if this is the best model so far
            if eval_win_rate > best_eval_win_rate:
                best_eval_win_rate = eval_win_rate
                # Save best model
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                best_model_path = f"curriculum_stage_{stage_num}_best_{timestamp}.pth"
                agent.save_model(best_model_path)
                print(f"   💾 New best model saved: {eval_win_rate:.3f}")
            
            # Check early stopping conditions
            target_achieved = eval_win_rate >= stage['target_win_rate']
            min_episodes_reached = (episode + stage['eval_freq']) >= stage['min_episodes']
            
            if target_achieved and min_episodes_reached:
                consecutive_target_achievements += 1
                print(f"   🎯 Target achieved! ({consecutive_target_achievements}/{stage['early_stop_consecutive']})")
                
                if consecutive_target_achievements >= stage['early_stop_consecutive']:
                    print(f"   ✅ Early stopping: Target achieved {stage['early_stop_consecutive']} times consecutively")
                    break
            else:
                consecutive_target_achievements = 0
            
            # Progress reporting
            total_episodes = episode + stage['eval_freq']
            elapsed_time = time.time() - start_time
            episodes_per_second = total_episodes / elapsed_time
            
            print(f"   📊 Episode {total_episodes:,}: Train {training_stats['win_rate']:.3f}, "
                  f"Eval {eval_win_rate:.3f}, Epsilon {agent.epsilon:.3f}, "
                  f"Speed {episodes_per_second:.1f} ep/s")
        
        training_time = time.time() - start_time
        total_episodes = len(eval_history) * stage['eval_freq']
        
        # Load best model for final evaluation
        if best_model_path and os.path.exists(best_model_path):
            agent.load_model(best_model_path)
            print(f"   🔄 Loaded best model for final evaluation")
        
        # Final comprehensive evaluation
        final_eval_stats = self.comprehensive_evaluation(
            agent, env, stage['eval_episodes'] * 2, stage['eval_runs'] * 2
        )
        
        # Performance assessment
        target_win_rate = stage['target_win_rate']
        achieved_win_rate = final_eval_stats['win_rate']
        
        if achieved_win_rate >= target_win_rate:
            status = "✅ TARGET ACHIEVED"
        elif achieved_win_rate >= target_win_rate * 0.8:
            status = "✅ CLOSE"
        elif achieved_win_rate >= target_win_rate * 0.6:
            status = "⚠️  PROGRESS"
        else:
            status = "❌ NEEDS WORK"
        
        print(f"\n   🎯 Stage {stage_num} Final Assessment:")
        print(f"      Training Episodes: {total_episodes:,}")
        print(f"      Training Time: {training_time:.1f}s ({training_time/60:.1f} minutes)")
        print(f"      Best Evaluation Win Rate: {best_eval_win_rate:.3f}")
        print(f"      Final Evaluation Win Rate: {achieved_win_rate:.3f} ± {final_eval_stats['win_rate_std']:.3f}")
        print(f"      Transfer Success: {transfer_success}")
        print(f"      Status: {status} ({achieved_win_rate:.1%} of {target_win_rate:.1%} target)")
        
        # Save final agent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_agent_filename = f"curriculum_stage_{stage_num}_final_{timestamp}.pth"
        agent.save_model(final_agent_filename)
        print(f"   💾 Final agent saved to {final_agent_filename}")
        
        # Store results
        stage_result = {
            'stage_num': stage_num,
            'stage_name': stage['name'],
            'board_size': stage['board_size'],
            'mines': stage['mines'],
            'total_episodes': total_episodes,
            'target_win_rate': target_win_rate,
            'training_stats': {
                'win_rate': np.mean(eval_history),
                'final_epsilon': agent.epsilon,
                'replay_buffer_size': len(agent.replay_buffer)
            },
            'eval_stats': final_eval_stats,
            'best_eval_win_rate': best_eval_win_rate,
            'training_time': training_time,
            'transfer_success': transfer_success,
            'achieved_win_rate': achieved_win_rate,
            'status': status,
            'final_agent_filename': final_agent_filename,
            'best_model_filename': best_model_path,
            'eval_history': eval_history
        }
        
        self.curriculum_results.append(stage_result)
        self.current_agent = agent
        self.current_model_path = final_agent_filename
        self.best_models[stage_num] = best_model_path
        
        return stage_result
    
    def _train_episodes(self, env, agent: EnhancedDQNAgent, episodes: int, mine_count: int) -> Dict[str, Any]:
        """Train agent for a specific number of episodes."""
        episode_rewards = []
        episode_lengths = []
        losses = []
        wins = 0
        
        for episode in range(episodes):
            state, info = env.reset()
            done = False
            total_reward = 0
            steps = 0
            max_steps = 200
            
            while not done and steps < max_steps:
                # Choose action
                action = agent.choose_action(state, training=True)
                
                # Take action
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Store experience
                agent.store_experience(state, action, reward, next_state, done)
                
                # Train the network
                loss = agent.train()
                if loss is not None:
                    losses.append(loss)
                
                state = next_state
                total_reward += reward
                steps += 1
            
            # Update statistics
            agent.training_stats['episodes'] += 1
            agent.training_stats['total_reward'] += total_reward
            
            # Check if episode was won
            won = False
            if info and isinstance(info, dict):
                won = info.get('won', False)
            elif info and isinstance(info, list) and len(info) > 0:
                won = info[0].get('won', False)
            
            if won:
                agent.training_stats['wins'] += 1
                wins += 1
            else:
                agent.training_stats['losses'] += 1
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            
            # Update epsilon
            agent.update_epsilon()
        
        # Calculate statistics
        win_rate = wins / episodes
        mean_loss = np.mean(losses) if losses else 0
        
        return {
            'win_rate': win_rate,
            'mean_loss': mean_loss,
            'final_epsilon': agent.epsilon,
            'replay_buffer_size': len(agent.replay_buffer)
        }
    
    def run_curriculum(self) -> Dict[str, Any]:
        """Run the complete adaptive curriculum learning process."""
        print("🚀 Adaptive Curriculum Learning for Minesweeper DQN")
        print("=" * 70)
        print("Strategy: Adaptive progression with early stopping")
        print("Features: Best model saving, periodic evaluation, transfer learning")
        print("=" * 70)
        
        if self.use_mlflow:
            mlflow.set_experiment("minesweeper_adaptive_curriculum")
            mlflow.start_run()
            mlflow.log_params(self.agent_config)
        
        start_time = time.time()
        
        for i, stage in enumerate(self.curriculum_stages, 1):
            stage_result = self.train_stage_adaptive(stage, i)
            
            if self.use_mlflow:
                mlflow.log_metrics({
                    f"stage_{i}_final_eval_win_rate": stage_result['achieved_win_rate'],
                    f"stage_{i}_best_eval_win_rate": stage_result['best_eval_win_rate'],
                    f"stage_{i}_training_time": stage_result['training_time'],
                    f"stage_{i}_total_episodes": stage_result['total_episodes']
                })
        
        total_time = time.time() - start_time
        
        if self.use_mlflow:
            mlflow.end_run()
        
        # Print final summary
        self._print_curriculum_summary(total_time)
        
        return {
            'curriculum_results': self.curriculum_results,
            'total_time': total_time,
            'final_agent': self.current_agent,
            'best_models': self.best_models
        }
    
    def _print_curriculum_summary(self, total_time: float):
        """Print comprehensive curriculum summary."""
        print(f"\n🎉 Adaptive Curriculum Learning Complete!")
        print("=" * 70)
        print(f"Total Time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        print(f"Stages Completed: {len(self.curriculum_results)}")
        print("-" * 70)
        
        for result in self.curriculum_results:
            stage_num = result['stage_num']
            stage_name = result['stage_name']
            board_size = result['board_size']
            mines = result['mines']
            target = result['target_win_rate']
            achieved = result['achieved_win_rate']
            best = result['best_eval_win_rate']
            status = result['status']
            transfer = "✅" if result['transfer_success'] else "❌"
            episodes = result['total_episodes']
            
            print(f"Stage {stage_num}: {stage_name}")
            print(f"   Board: {board_size}, Mines: {mines}")
            print(f"   Episodes: {episodes:,}")
            print(f"   Target: {target:.1%}, Achieved: {achieved:.1%}, Best: {best:.1%}")
            print(f"   Status: {status}, Transfer: {transfer}")
            print()
        
        # Overall performance
        avg_achievement = np.mean([r['achieved_win_rate'] / r['target_win_rate'] for r in self.curriculum_results])
        print(f"Overall Performance: {avg_achievement:.1%} of targets achieved")
        
        if avg_achievement >= 0.8:
            print("🎯 EXCELLENT: Most targets achieved!")
        elif avg_achievement >= 0.6:
            print("✅ GOOD: Most targets close to achieved")
        elif avg_achievement >= 0.4:
            print("⚠️  ACCEPTABLE: Some targets achieved")
        else:
            print("❌ NEEDS IMPROVEMENT: Most targets not achieved")


def main():
    """Main function to run adaptive curriculum learning."""
    # Check if MLflow should be used
    use_mlflow = len(sys.argv) > 1 and sys.argv[1] == '--mlflow'
    
    # Create trainer
    trainer = AdaptiveCurriculumTrainer(use_mlflow=use_mlflow)
    
    # Run curriculum
    results = trainer.run_curriculum()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"adaptive_curriculum_results_{timestamp}.json"
    
    # Convert results to JSON-serializable format
    json_results = {
        'total_time': results['total_time'],
        'curriculum_results': []
    }
    
    for result in results['curriculum_results']:
        json_result = {
            'stage_num': result['stage_num'],
            'stage_name': result['stage_name'],
            'board_size': result['board_size'],
            'mines': result['mines'],
            'total_episodes': result['total_episodes'],
            'target_win_rate': result['target_win_rate'],
            'training_stats': result['training_stats'],
            'eval_stats': {
                'win_rate': result['eval_stats']['win_rate'],
                'win_rate_std': result['eval_stats']['win_rate_std'],
                'win_rate_range': result['eval_stats']['win_rate_range'],
                'mean_reward': result['eval_stats']['mean_reward'],
                'mean_reward_std': result['eval_stats']['mean_reward_std'],
                'mean_length': result['eval_stats']['mean_length'],
                'mean_length_std': result['eval_stats']['mean_length_std']
            },
            'best_eval_win_rate': result['best_eval_win_rate'],
            'training_time': result['training_time'],
            'transfer_success': result['transfer_success'],
            'achieved_win_rate': result['achieved_win_rate'],
            'status': result['status'],
            'final_agent_filename': result['final_agent_filename'],
            'best_model_filename': result['best_model_filename'],
            'eval_history': result['eval_history']
        }
        json_results['curriculum_results'].append(json_result)
    
    with open(filename, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 Results saved to {filename}")


if __name__ == "__main__":
    main() 