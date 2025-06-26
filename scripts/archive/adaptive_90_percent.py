#!/usr/bin/env python3
"""
Adaptive 90%+ Win Rate Script

This script uses adaptive training strategies to push toward 90%+ while avoiding
overfitting and performance regression.
"""

import sys
import os
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple, List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.minesweeper_env import MinesweeperEnv
from core.dqn_agent_enhanced import EnhancedDQNAgent


def create_adaptive_agent(board_size: Tuple[int, int]) -> EnhancedDQNAgent:
    """Create agent with adaptive training configuration."""
    return EnhancedDQNAgent(
        board_size=board_size,
        action_size=board_size[0] * board_size[1],
        learning_rate=0.0003,  # Start with proven LR
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.999,  # Start with proven decay
        epsilon_min=0.02,  # Start with proven minimum
        replay_buffer_size=200000,  # Proven buffer size
        batch_size=64,  # Proven batch size
        target_update_freq=2000,  # Proven update frequency
        device='cpu',
        use_double_dqn=True,
        use_dueling=True,
        use_prioritized_replay=True
    )


def adaptive_training(max_episodes: int = 2000, eval_freq: int = 100) -> Dict[str, Any]:
    """Adaptive training that monitors performance and adjusts parameters."""
    print(f"🎯 Adaptive Training for 90%+ Win Rate")
    print(f"   Max Episodes: {max_episodes}")
    print(f"   Evaluation Frequency: {eval_freq}")
    print("=" * 60)
    
    # Create environment and agent
    env = MinesweeperEnv(initial_board_size=(4, 4), initial_mines=1)
    agent = create_adaptive_agent((4, 4))
    
    print(f"✅ Starting with PROVEN configuration:")
    print(f"   Learning Rate: {agent.learning_rate}")
    print(f"   Epsilon Decay: {agent.epsilon_decay}")
    print(f"   Epsilon Min: {agent.epsilon_min}")
    print(f"   Replay Buffer: {len(agent.replay_buffer)}")
    print(f"   Batch Size: {agent.batch_size}")
    print("-" * 60)
    
    start_time = time.time()
    losses = []
    win_rates = []
    recent_win_rates = []
    best_win_rate = 0.0
    episodes_without_improvement = 0
    
    # Training loop with adaptive strategies
    for episode in range(max_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        max_steps = 200
        
        while not done and steps < max_steps:
            # Choose action with exploration
            action = agent.choose_action(state, training=True)
            
            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store experience
            agent.store_experience(state, action, reward, next_state, done)
            
            # Train the agent
            if len(agent.replay_buffer) > agent.batch_size:
                loss = agent.train()
                if loss is not None:
                    losses.append(loss)
            
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
            agent.training_stats['wins'] += 1
        agent.training_stats['episodes'] += 1
        
        # Update epsilon
        agent.update_epsilon()
        
        # Evaluation and adaptive adjustments
        if (episode + 1) % eval_freq == 0:
            stats = agent.get_stats()
            recent_win_rate = stats['win_rate']
            recent_win_rates.append(recent_win_rate)
            
            avg_loss = np.mean(losses[-eval_freq:]) if losses else 0
            
            print(f"Episode {episode + 1:4d}: Win Rate {stats['win_rate']:.3f} "
                  f"(Recent: {recent_win_rate:.3f}), Epsilon: {agent.epsilon:.3f}, "
                  f"Loss: {avg_loss:.4f}")
            
            # Track best performance
            if recent_win_rate > best_win_rate:
                best_win_rate = recent_win_rate
                episodes_without_improvement = 0
                print(f"   🏆 NEW BEST: {recent_win_rate:.3f} win rate!")
            else:
                episodes_without_improvement += 1
            
            # Adaptive strategies based on performance
            if episodes_without_improvement >= 3:
                print(f"   ⚠️  No improvement for {episodes_without_improvement} evaluations")
                
                # Strategy 1: Increase exploration if performance is low
                if recent_win_rate < 0.6:
                    old_epsilon = agent.epsilon
                    agent.epsilon = min(1.0, agent.epsilon * 1.1)  # Increase exploration
                    print(f"   🔄 Increasing exploration: {old_epsilon:.3f} → {agent.epsilon:.3f}")
                
                # Strategy 2: Slow down learning if overfitting
                elif recent_win_rate < 0.7:
                    old_lr = agent.learning_rate
                    agent.learning_rate *= 0.9  # Reduce learning rate
                    print(f"   🔄 Reducing learning rate: {old_lr:.6f} → {agent.learning_rate:.6f}")
                
                # Strategy 3: Early stopping if performance is very low
                elif recent_win_rate < 0.4:
                    print(f"   🛑 Early stopping: Performance too low ({recent_win_rate:.3f})")
                    break
            
            # Early stopping if we achieve target
            if recent_win_rate >= 0.90:
                print(f"🎉 Target achieved: {recent_win_rate:.3f} win rate!")
                break
    
    training_time = time.time() - start_time
    
    # Final statistics
    final_stats = agent.get_stats()
    mean_loss = np.mean(losses) if losses else 0
    episodes_per_second = agent.training_stats['episodes'] / training_time
    
    print(f"\n✅ Adaptive training completed!")
    print(f"   Final Win Rate: {final_stats['win_rate']:.3f}")
    print(f"   Best Win Rate: {best_win_rate:.3f}")
    print(f"   Final Epsilon: {agent.epsilon:.3f}")
    print(f"   Total Episodes: {agent.training_stats['episodes']}")
    print(f"   Training Time: {training_time:.2f}s")
    print(f"   Episodes/second: {episodes_per_second:.2f}")
    print(f"   Mean Loss: {mean_loss:.4f}")
    
    return {
        'win_rate': final_stats['win_rate'],
        'best_win_rate': best_win_rate,
        'mean_loss': mean_loss,
        'final_epsilon': agent.epsilon,
        'episodes': agent.training_stats['episodes'],
        'training_time': training_time,
        'episodes_per_second': episodes_per_second,
        'agent': agent,
        'recent_win_rates': recent_win_rates
    }


def comprehensive_evaluation(agent: EnhancedDQNAgent, n_runs: int = 25, episodes_per_run: int = 100) -> Dict[str, Any]:
    """Comprehensive evaluation with more runs for statistical significance."""
    print(f"🔍 Comprehensive Evaluation")
    print(f"   {n_runs} runs of {episodes_per_run} episodes each")
    print("=" * 50)
    
    env = MinesweeperEnv(initial_board_size=(4, 4), initial_mines=1)
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
            max_steps = 200
            
            while not done and steps < max_steps:
                # Choose action (no exploration during evaluation)
                action = agent.choose_action(state, training=False)
                
                # Take action
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
        
        print(f"Run {run + 1:2d}: Win Rate {run_win_rate:.3f}, "
              f"Mean Reward {run_mean_reward:.2f}, "
              f"Mean Length {run_mean_length:.1f}")
    
    # Calculate statistics
    win_rates = [r['win_rate'] for r in all_results]
    mean_rewards = [r['mean_reward'] for r in all_results]
    mean_lengths = [r['mean_length'] for r in all_results]
    
    comprehensive_results = {
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
    
    print(f"\n📊 Statistical Results:")
    print(f"   Mean Win Rate: {comprehensive_results['mean_win_rate']:.3f} ± {comprehensive_results['std_win_rate']:.3f}")
    print(f"   Win Rate Range: {comprehensive_results['min_win_rate']:.3f} - {comprehensive_results['max_win_rate']:.3f}")
    print(f"   Mean Reward: {comprehensive_results['mean_reward']:.2f} ± {comprehensive_results['std_reward']:.2f}")
    print(f"   Mean Length: {comprehensive_results['mean_length']:.1f} ± {comprehensive_results['std_length']:.1f}")
    
    return comprehensive_results


def main():
    """Run adaptive training to push toward 90%+ win rate."""
    print("🏆 Adaptive 90%+ Win Rate Push")
    print("=" * 55)
    
    try:
        # Run adaptive training
        training_results = adaptive_training(max_episodes=2000, eval_freq=100)
        
        # Comprehensive evaluation
        eval_results = comprehensive_evaluation(training_results['agent'], n_runs=25, episodes_per_run=100)
        
        # Performance assessment
        final_win_rate = eval_results['mean_win_rate']
        training_speed = training_results['episodes_per_second']
        
        print(f"\n🎯 Adaptive Performance Assessment:")
        print(f"   Training Win Rate: {training_results['win_rate']:.3f}")
        print(f"   Best Training Win Rate: {training_results['best_win_rate']:.3f}")
        print(f"   Evaluation Win Rate: {final_win_rate:.3f} ± {eval_results['std_win_rate']:.3f}")
        print(f"   Training Speed: {training_speed:.2f} episodes/second")
        print(f"   Training Time: {training_results['training_time']:.2f}s")
        
        # Goal achievement
        if final_win_rate >= 0.95:
            print(f"   🎉 TARGET EXCEEDED: Achieved 95%+ win rate!")
        elif final_win_rate >= 0.90:
            print(f"   🏆 TARGET ACHIEVED: Achieved 90%+ win rate!")
        elif final_win_rate >= 0.85:
            print(f"   ✅ VERY CLOSE: Achieved 85%+ win rate!")
        elif final_win_rate >= 0.80:
            print(f"   🎯 GOOD PROGRESS: Achieved 80%+ win rate!")
        else:
            print(f"   ⚠️  NEED MORE TRAINING: Below 80% win rate")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"adaptive_90_percent_results_{timestamp}.json"
        
        # Create a copy of training_results without the agent object for JSON serialization
        training_results_serializable = training_results.copy()
        training_results_serializable.pop('agent', None)  # Remove agent object
        
        # Convert numpy types to native Python types for JSON serialization
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
        
        training_results_serializable = convert_numpy_types(training_results_serializable)
        eval_results_serializable = convert_numpy_types(eval_results)
        
        results = {
            'timestamp': timestamp,
            'config': {
                'max_episodes': 2000,
                'eval_freq': 100,
                'target': '90%+ win rate (adaptive training)'
            },
            'training_results': training_results_serializable,
            'evaluation_results': eval_results_serializable,
            'performance': {
                'final_win_rate': float(final_win_rate),
                'training_speed': float(training_speed),
                'target_achieved': bool(final_win_rate >= 0.90)
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to {filename}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during adaptive training: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Adaptive training completed successfully!")
    else:
        print("\n💥 Adaptive training failed. Please check the error messages above.")
        sys.exit(1) 