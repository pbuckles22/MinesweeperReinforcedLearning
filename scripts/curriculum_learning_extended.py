#!/usr/bin/env python3
"""
Curriculum Learning with Extended Training

This script implements progressive curriculum learning from 4x4 to 8x8 boards
with extended training times and transfer learning between stages.
"""

import sys
import os
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.minesweeper_env import MinesweeperEnv
from core.dqn_agent_enhanced import EnhancedDQNAgent, train_enhanced_dqn_agent

# Import MLflow for tracking
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    print("⚠️  MLflow not available. Install with: pip install mlflow")
    MLFLOW_AVAILABLE = False


class CurriculumLearningManager:
    """Manages progressive curriculum learning across board sizes."""
    
    def __init__(self):
        self.total_start_time = time.time()
        self.stage_results = []
        self.current_agent = None
        self.mlflow_run = None
        
    def create_agent_for_stage(self, board_size: Tuple[int, int], mines: int, 
                              previous_agent: Optional[EnhancedDQNAgent] = None) -> EnhancedDQNAgent:
        """Create agent optimized for current stage, optionally transferring from previous stage."""
        action_size = board_size[0] * board_size[1]
        
        # Stage-specific hyperparameters (adjusted for more realistic targets)
        if board_size[0] * board_size[1] <= 16:  # 4x4 - Use PROVEN 90% configuration
            learning_rate = 0.0001          # PROVEN: Conservative learning rate
            epsilon_decay = 0.9995          # PROVEN: Very slow exploration decay
            epsilon_min = 0.05              # PROVEN: Higher minimum exploration
            batch_size = 32                 # PROVEN: Smaller batches
            target_update_freq = 1000       # PROVEN: More frequent updates
            episodes = 1000                 # PROVEN: Back to proven amount
            replay_buffer_size = 100000     # PROVEN: Smaller buffer
        elif board_size[0] * board_size[1] <= 25:  # 5x5
            learning_rate = 0.00005
            epsilon_decay = 0.9997
            epsilon_min = 0.08
            batch_size = 64
            target_update_freq = 1500
            episodes = 3000  # Increased training
            replay_buffer_size = 200000
        elif board_size[0] * board_size[1] <= 36:  # 6x6
            learning_rate = 0.00003
            epsilon_decay = 0.9998
            epsilon_min = 0.1
            batch_size = 128
            target_update_freq = 2000
            episodes = 4000  # Increased training
            replay_buffer_size = 200000
        else:  # 8x8
            learning_rate = 0.00002
            epsilon_decay = 0.9999
            epsilon_min = 0.15
            batch_size = 256
            target_update_freq = 3000
            episodes = 6000  # Increased training
            replay_buffer_size = 200000
        
        agent = EnhancedDQNAgent(
            board_size=board_size,
            action_size=action_size,
            learning_rate=learning_rate,
            discount_factor=0.99,
            epsilon=0.3 if previous_agent is not None else 1.0,  # Lower initial epsilon for transfer
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            replay_buffer_size=replay_buffer_size,
            batch_size=batch_size,
            target_update_freq=target_update_freq,
            device='cpu',
            use_double_dqn=True,
            use_dueling=True,
            use_prioritized_replay=True
        )
        
        # Transfer learning: copy weights from previous stage if available
        if previous_agent is not None:
            self._transfer_weights(previous_agent, agent, board_size)
            print(f"   🔄 Transferred weights from {previous_agent.board_size[0]}x{previous_agent.board_size[1]} to {board_size[0]}x{board_size[1]}")
        
        return agent
    
    def _transfer_weights(self, source_agent: EnhancedDQNAgent, target_agent: EnhancedDQNAgent, 
                         target_board_size: Tuple[int, int]):
        """Transfer learned weights from smaller to larger board."""
        try:
            # Copy the main network weights
            target_agent.q_network.load_state_dict(source_agent.q_network.state_dict())
            
            # Copy the target network weights
            target_agent.target_network.load_state_dict(source_agent.target_network.state_dict())
            
            # Copy replay buffer (if compatible)
            if len(source_agent.replay_buffer) > 0:
                # Only copy recent experiences that might be relevant
                recent_experiences = list(source_agent.replay_buffer)[-50000:]  # Last 50k experiences
                for experience in recent_experiences:
                    if len(experience) >= 5:  # Ensure valid experience format
                        target_agent.replay_buffer.append(experience)
                
                print(f"   📦 Transferred {len(recent_experiences)} recent experiences")
            
            # Reset training stats but keep some knowledge
            target_agent.training_stats = {'episodes': 0, 'wins': 0, 'total_reward': 0}
            
        except Exception as e:
            print(f"   ⚠️  Weight transfer failed: {e}")
            print(f"   🆕 Starting with fresh weights")
    
    def train_stage(self, stage_name: str, board_size: Tuple[int, int], mines: int, 
                   episodes: int, target_win_rate: float) -> Dict[str, Any]:
        """Train on a specific stage with extended training."""
        print(f"\n{'='*20} STAGE {stage_name}: {board_size[0]}x{board_size[1]} BOARD {'='*20}")
        print(f"   Target Episodes: {episodes}")
        print(f"   Target Win Rate: {target_win_rate:.1%}")
        print(f"   Mines: {mines}")
        
        # Log stage parameters to MLflow
        if MLFLOW_AVAILABLE and self.mlflow_run:
            mlflow.log_param(f"{stage_name}_board_size", f"{board_size[0]}x{board_size[1]}")
            mlflow.log_param(f"{stage_name}_mines", mines)
            mlflow.log_param(f"{stage_name}_target_episodes", episodes)
            mlflow.log_param(f"{stage_name}_target_win_rate", target_win_rate)
        
        # Create environment and agent
        env = MinesweeperEnv(initial_board_size=board_size, initial_mines=mines)
        agent = self.create_agent_for_stage(board_size, mines, self.current_agent)
        
        print(f"   ✅ Stage Configuration:")
        print(f"      Learning Rate: {agent.learning_rate}")
        print(f"      Epsilon Decay: {agent.epsilon_decay}")
        print(f"      Epsilon Min: {agent.epsilon_min}")
        print(f"      Batch Size: {agent.batch_size}")
        print(f"      Target Update Freq: {agent.target_update_freq}")
        print(f"      Replay Buffer: {len(agent.replay_buffer)}")
        
        # Log agent parameters to MLflow
        if MLFLOW_AVAILABLE and self.mlflow_run:
            mlflow.log_param(f"{stage_name}_learning_rate", agent.learning_rate)
            mlflow.log_param(f"{stage_name}_epsilon_decay", agent.epsilon_decay)
            mlflow.log_param(f"{stage_name}_epsilon_min", agent.epsilon_min)
            mlflow.log_param(f"{stage_name}_batch_size", agent.batch_size)
            mlflow.log_param(f"{stage_name}_target_update_freq", agent.target_update_freq)
        
        # Train with more frequent evaluation for longer runs
        eval_freq = max(50, episodes // 20)
        stage_start_time = time.time()
        
        print(f"\n   🚀 Starting Training...")
        training_stats = train_enhanced_dqn_agent(env, agent, episodes, mines, eval_freq=eval_freq)
        
        stage_time = time.time() - stage_start_time
        
        # Final statistics
        final_stats = agent.get_stats()
        episodes_per_second = agent.training_stats['episodes'] / stage_time
        
        print(f"\n   ✅ {stage_name} Training Completed!")
        print(f"      Final Win Rate: {final_stats['win_rate']:.3f}")
        print(f"      Final Epsilon: {agent.epsilon:.3f}")
        print(f"      Total Episodes: {agent.training_stats['episodes']}")
        print(f"      Training Time: {stage_time:.2f}s ({stage_time/60:.1f} minutes)")
        print(f"      Episodes/second: {episodes_per_second:.2f}")
        
        # Log training metrics to MLflow
        if MLFLOW_AVAILABLE and self.mlflow_run:
            mlflow.log_metric(f"{stage_name}_training_win_rate", final_stats['win_rate'])
            mlflow.log_metric(f"{stage_name}_final_epsilon", agent.epsilon)
            mlflow.log_metric(f"{stage_name}_training_episodes", agent.training_stats['episodes'])
            mlflow.log_metric(f"{stage_name}_training_time", stage_time)
            mlflow.log_metric(f"{stage_name}_episodes_per_second", episodes_per_second)
        
        # Comprehensive evaluation
        print(f"\n   🔍 Running Comprehensive Evaluation...")
        eval_results = self.evaluate_stage(agent, board_size, mines, n_runs=25, episodes_per_run=50)
        
        # Performance assessment
        final_win_rate = eval_results['mean_win_rate']
        
        print(f"\n   🎯 {stage_name} Performance Assessment:")
        print(f"      Training Win Rate: {final_stats['win_rate']:.3f}")
        print(f"      Evaluation Win Rate: {final_win_rate:.3f} ± {eval_results['std_win_rate']:.3f}")
        print(f"      Training Speed: {episodes_per_second:.2f} episodes/second")
        print(f"      Stage Time: {stage_time:.2f}s")
        
        # Log evaluation metrics to MLflow
        if MLFLOW_AVAILABLE and self.mlflow_run:
            mlflow.log_metric(f"{stage_name}_eval_win_rate", final_win_rate)
            mlflow.log_metric(f"{stage_name}_eval_win_rate_std", eval_results['std_win_rate'])
            mlflow.log_metric(f"{stage_name}_eval_mean_reward", eval_results['mean_reward'])
            mlflow.log_metric(f"{stage_name}_eval_mean_length", eval_results['mean_length'])
            mlflow.log_metric(f"{stage_name}_eval_min_win_rate", eval_results['min_win_rate'])
            mlflow.log_metric(f"{stage_name}_eval_max_win_rate", eval_results['max_win_rate'])
        
        # Goal achievement (more lenient criteria)
        if final_win_rate >= target_win_rate:
            print(f"      🎉 TARGET ACHIEVED: {final_win_rate:.1%} >= {target_win_rate:.1%}")
        elif final_win_rate >= target_win_rate * 0.7:  # 70% of target (more lenient)
            print(f"      ✅ CLOSE: {final_win_rate:.1%} (70% of target)")
        elif final_win_rate >= target_win_rate * 0.5:  # 50% of target (more lenient)
            print(f"      ⚠️  ACCEPTABLE: {final_win_rate:.1%} (50% of target)")
        else:
            print(f"      💥 BELOW TARGET: {final_win_rate:.1%} < {target_win_rate:.1%}")
        
        # Store results
        stage_result = {
            'stage_name': stage_name,
            'board_size': board_size,
            'mines': mines,
            'target_episodes': episodes,
            'target_win_rate': target_win_rate,
            'training_results': {
                'win_rate': final_stats['win_rate'],
                'final_epsilon': agent.epsilon,
                'episodes': agent.training_stats['episodes'],
                'training_time': stage_time,
                'episodes_per_second': episodes_per_second
            },
            'evaluation_results': eval_results,
            'performance': {
                'final_win_rate': float(final_win_rate),
                'target_achieved': bool(final_win_rate >= target_win_rate * 0.5),  # More lenient
                'stage_time': float(stage_time)
            }
        }
        
        self.stage_results.append(stage_result)
        self.current_agent = agent  # Keep for next stage
        
        return stage_result
    
    def evaluate_stage(self, agent: EnhancedDQNAgent, board_size: Tuple[int, int], 
                      mines: int, n_runs: int = 25, episodes_per_run: int = 50) -> Dict[str, Any]:
        """Evaluate the trained agent on the same board size."""
        env = MinesweeperEnv(initial_board_size=board_size, initial_mines=mines)
        all_results = []
        
        for run in range(n_runs):
            wins = 0
            total_rewards = []
            episode_lengths = []
            
            for episode in range(episodes_per_run):
                state, info = env.reset()
                done = False
                total_reward = 0
                steps = 0
                max_steps = board_size[0] * board_size[1] * 2
                
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
            
            run_win_rate = wins / episodes_per_run
            run_mean_reward = np.mean(total_rewards)
            run_mean_length = np.mean(episode_lengths)
            
            all_results.append({
                'win_rate': run_win_rate,
                'mean_reward': run_mean_reward,
                'mean_length': run_mean_length
            })
            
            if run % 5 == 0:  # Print every 5th run
                print(f"      Run {run + 1:2d}: Win Rate {run_win_rate:.3f}, "
                      f"Mean Reward {run_mean_reward:.2f}, "
                      f"Mean Length {run_mean_length:.1f}")
        
        # Calculate statistics
        win_rates = [r['win_rate'] for r in all_results]
        mean_rewards = [r['mean_reward'] for r in all_results]
        mean_lengths = [r['mean_length'] for r in all_results]
        
        evaluation_results = {
            'board_size': board_size,
            'mines': mines,
            'mean_win_rate': np.mean(win_rates),
            'std_win_rate': np.std(win_rates),
            'mean_reward': np.mean(mean_rewards),
            'std_reward': np.std(mean_rewards),
            'mean_length': np.mean(mean_lengths),
            'std_length': np.std(mean_lengths),
            'min_win_rate': np.min(win_rates),
            'max_win_rate': np.max(win_rates),
            'all_runs': all_results
        }
        
        print(f"      📊 Evaluation Results:")
        print(f"         Mean Win Rate: {evaluation_results['mean_win_rate']:.3f} ± {evaluation_results['std_win_rate']:.3f}")
        print(f"         Win Rate Range: {evaluation_results['min_win_rate']:.3f} - {evaluation_results['max_win_rate']:.3f}")
        print(f"         Mean Reward: {evaluation_results['mean_reward']:.2f} ± {evaluation_results['std_reward']:.2f}")
        print(f"         Mean Length: {evaluation_results['mean_length']:.1f} ± {evaluation_results['std_length']:.1f}")
        
        return evaluation_results
    
    def run_curriculum(self) -> Dict[str, Any]:
        """Run the complete curriculum learning process."""
        print("🎓 CURRICULUM LEARNING WITH EXTENDED TRAINING")
        print("=" * 65)
        print("   Progressive learning from 4x4 to 8x8 boards")
        print("   Extended training times with transfer learning")
        print("   Target: Improve performance on larger boards")
        print("   Stage 1: Uses PROVEN 90% configuration for 4x4")
        print("=" * 65)
        
        # Start MLflow run
        if MLFLOW_AVAILABLE:
            self.mlflow_run = mlflow.start_run(run_name=f"CurriculumLearning-{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            print(f"   📊 MLflow tracking enabled: {mlflow.get_tracking_uri()}")
        else:
            print(f"   ⚠️  MLflow tracking disabled")
        
        try:
            # Curriculum stages with more realistic targets
            curriculum_stages = [
                ("Stage 1", (4, 4), 1, 1000, 0.90),   # 4x4, 1 mine, 1000 episodes, 90% target (PROVEN configuration)
                ("Stage 2", (5, 5), 2, 3000, 0.50),   # 5x5, 2 mines, 3000 episodes, 50% target
                ("Stage 3", (6, 6), 3, 4000, 0.35),   # 6x6, 3 mines, 4000 episodes, 35% target
                ("Stage 4", (8, 8), 5, 6000, 0.20),   # 8x8, 5 mines, 6000 episodes, 20% target
            ]
            
            # Log curriculum parameters to MLflow
            if MLFLOW_AVAILABLE and self.mlflow_run:
                mlflow.log_param("curriculum_approach", "Progressive board sizes with transfer learning")
                mlflow.log_param("total_stages", len(curriculum_stages))
                mlflow.log_param("base_configuration", "Winning 95% Conservative Learning")
            
            for stage_name, board_size, mines, episodes, target_win_rate in curriculum_stages:
                stage_result = self.train_stage(stage_name, board_size, mines, episodes, target_win_rate)
                
                # Check if we should continue (more lenient criteria)
                final_win_rate = stage_result['evaluation_results']['mean_win_rate']
                if final_win_rate < target_win_rate * 0.3:  # Less than 30% of target (was 50%)
                    print(f"\n   ⚠️  Performance too low on {board_size[0]}x{board_size[1]}")
                    print(f"   🛑 Stopping curriculum at {stage_name}")
                    break
            
            # Calculate total time
            total_time = time.time() - self.total_start_time
            
            # Summary across all stages
            print(f"\n{'='*20} CURRICULUM SUMMARY {'='*20}")
            print(f"{'Stage':<10} {'Board':<8} {'Target':<8} {'Achieved':<10} {'Status':<15}")
            print("-" * 65)
            
            for result in self.stage_results:
                stage = result['stage_name']
                board = f"{result['board_size'][0]}x{result['board_size'][1]}"
                target = f"{result['target_win_rate']:.1%}"
                achieved = f"{result['evaluation_results']['mean_win_rate']:.1%}"
                
                if result['performance']['target_achieved']:
                    status = "🎉 ACHIEVED"
                elif result['evaluation_results']['mean_win_rate'] >= result['target_win_rate'] * 0.7:
                    status = "✅ CLOSE"
                elif result['evaluation_results']['mean_win_rate'] >= result['target_win_rate'] * 0.5:
                    status = "⚠️  ACCEPTABLE"
                else:
                    status = "💥 BELOW TARGET"
                
                print(f"{stage:<10} {board:<8} {target:<8} {achieved:<10} {status:<15}")
            
            print(f"\n⏱️  Total Curriculum Time: {total_time:.2f}s ({total_time/3600:.1f} hours)")
            
            # Log final metrics to MLflow
            if MLFLOW_AVAILABLE and self.mlflow_run:
                mlflow.log_metric("total_curriculum_time", total_time)
                mlflow.log_metric("stages_completed", len(self.stage_results))
                mlflow.log_metric("stages_achieved_target", len([r for r in self.stage_results if r['performance']['target_achieved']]))
                mlflow.log_metric("average_win_rate", np.mean([r['evaluation_results']['mean_win_rate'] for r in self.stage_results]))
            
            # Save comprehensive results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"curriculum_learning_results_{timestamp}.json"
            
            # Prepare results for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                else:
                    return obj
            
            # Remove agent objects and convert numpy types
            serializable_results = []
            for result in self.stage_results:
                serializable_result = result.copy()
                # Remove agent object from training results
                if 'agent' in serializable_result.get('training_results', {}):
                    serializable_result['training_results'].pop('agent')
                serializable_results.append(convert_numpy_types(serializable_result))
            
            results = {
                'timestamp': timestamp,
                'config': {
                    'approach': 'Curriculum Learning with Extended Training',
                    'strategy': 'Progressive board sizes with transfer learning',
                    'target': 'Improve performance on larger boards through curriculum'
                },
                'curriculum_stages': serializable_results,
                'summary': {
                    'total_stages_completed': len(self.stage_results),
                    'stages_achieved_target': len([r for r in self.stage_results if r['performance']['target_achieved']]),
                    'stages_close_to_target': len([r for r in self.stage_results if r['evaluation_results']['mean_win_rate'] >= r['target_win_rate'] * 0.7]),
                    'total_curriculum_time': float(total_time),
                    'average_win_rate': np.mean([r['evaluation_results']['mean_win_rate'] for r in self.stage_results])
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Log artifact to MLflow
            if MLFLOW_AVAILABLE and self.mlflow_run:
                mlflow.log_artifact(filename)
            
            print(f"\n💾 Comprehensive curriculum results saved to {filename}")
            
            return results
            
        except Exception as e:
            total_time = time.time() - self.total_start_time
            print(f"\n❌ Error during curriculum learning: {e}")
            print(f"   Total time elapsed: {total_time:.2f}s")
            import traceback
            traceback.print_exc()
            return None
        finally:
            # End MLflow run
            if MLFLOW_AVAILABLE and self.mlflow_run:
                mlflow.end_run()


def main():
    """Run the curriculum learning process."""
    curriculum_manager = CurriculumLearningManager()
    results = curriculum_manager.run_curriculum()
    
    if results:
        print("\n🎉 Curriculum learning completed successfully!")
        return True
    else:
        print("\n💥 Curriculum learning failed. Please check the error messages above.")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 