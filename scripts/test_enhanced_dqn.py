#!/usr/bin/env python3
"""
Test script for Enhanced DQN agent

Compares the enhanced DQN (with Double DQN, Dueling DQN, and Prioritized Replay)
against the baseline DQN using the optimal hyperparameters discovered.
"""

import sys
import os
import time
from datetime import datetime
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.minesweeper_env import MinesweeperEnv
from src.core.dqn_agent import DQNAgent, train_dqn_agent, evaluate_dqn_agent
from src.core.dqn_agent_enhanced import EnhancedDQNAgent, train_enhanced_dqn_agent, evaluate_enhanced_dqn_agent


def test_baseline_dqn():
    """Test baseline DQN with optimal hyperparameters."""
    print("🧪 Testing Baseline DQN (Optimal Config)")
    print("=" * 50)
    
    # Optimal configuration from optimization results
    board_size = (4, 4)
    mine_count = 1
    action_size = board_size[0] * board_size[1]
    
    # Create environment
    env = MinesweeperEnv(initial_board_size=board_size, initial_mines=mine_count)
    
    # Create baseline DQN with optimal hyperparameters
    baseline_agent = DQNAgent(
        board_size=board_size,
        action_size=action_size,
        learning_rate=0.0003,  # Optimal from optimization
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        replay_buffer_size=100000,
        batch_size=64,  # Optimal from optimization
        target_update_freq=1000,
        device='cpu'
    )
    
    print(f"   Board size: {board_size}")
    print(f"   Action size: {action_size}")
    print(f"   Learning rate: {baseline_agent.learning_rate}")
    print(f"   Batch size: {baseline_agent.batch_size}")
    print(f"   Epsilon decay: {baseline_agent.epsilon_decay}")
    
    # Train baseline agent
    episodes = 100
    print(f"\n🎯 Training baseline DQN for {episodes} episodes...")
    
    start_time = time.time()
    baseline_stats = train_dqn_agent(env, baseline_agent, episodes, mine_count, eval_freq=20)
    baseline_training_time = time.time() - start_time
    
    # Evaluate baseline agent
    print(f"\n🔍 Evaluating baseline DQN...")
    baseline_eval_stats = evaluate_dqn_agent(baseline_agent, env, n_episodes=50)
    
    print(f"\n✅ Baseline DQN Results:")
    print(f"   Training Win Rate: {baseline_stats['win_rate']:.3f}")
    print(f"   Evaluation Win Rate: {baseline_eval_stats['win_rate']:.3f}")
    print(f"   Training Time: {baseline_training_time:.2f}s")
    print(f"   Mean Reward: {baseline_eval_stats['mean_reward']:.2f}")
    print(f"   Mean Length: {baseline_eval_stats['mean_length']:.1f} steps")
    
    return {
        'agent': baseline_agent,
        'training_stats': baseline_stats,
        'eval_stats': baseline_eval_stats,
        'training_time': baseline_training_time
    }


def test_enhanced_dqn():
    """Test enhanced DQN with advanced techniques."""
    print("\n🚀 Testing Enhanced DQN (Advanced Techniques)")
    print("=" * 50)
    
    # Same configuration as baseline
    board_size = (4, 4)
    mine_count = 1
    action_size = board_size[0] * board_size[1]
    
    # Create environment
    env = MinesweeperEnv(initial_board_size=board_size, initial_mines=mine_count)
    
    # Create enhanced DQN with optimal hyperparameters + advanced techniques
    enhanced_agent = EnhancedDQNAgent(
        board_size=board_size,
        action_size=action_size,
        learning_rate=0.0003,  # Same optimal learning rate
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        replay_buffer_size=100000,
        batch_size=64,  # Same optimal batch size
        target_update_freq=1000,
        device='cpu',
        use_double_dqn=True,
        use_dueling=True,
        use_prioritized_replay=True
    )
    
    print(f"   Board size: {board_size}")
    print(f"   Action size: {action_size}")
    print(f"   Learning rate: {enhanced_agent.learning_rate}")
    print(f"   Batch size: {enhanced_agent.batch_size}")
    print(f"   Epsilon decay: {enhanced_agent.epsilon_decay}")
    print(f"   Double DQN: {enhanced_agent.use_double_dqn}")
    print(f"   Dueling DQN: {enhanced_agent.use_dueling}")
    print(f"   Prioritized Replay: {enhanced_agent.use_prioritized_replay}")
    
    # Train enhanced agent
    episodes = 100
    print(f"\n🎯 Training enhanced DQN for {episodes} episodes...")
    
    start_time = time.time()
    enhanced_stats = train_enhanced_dqn_agent(env, enhanced_agent, episodes, mine_count, eval_freq=20)
    enhanced_training_time = time.time() - start_time
    
    # Evaluate enhanced agent
    print(f"\n🔍 Evaluating enhanced DQN...")
    enhanced_eval_stats = evaluate_enhanced_dqn_agent(enhanced_agent, env, n_episodes=50)
    
    print(f"\n✅ Enhanced DQN Results:")
    print(f"   Training Win Rate: {enhanced_stats['win_rate']:.3f}")
    print(f"   Evaluation Win Rate: {enhanced_eval_stats['win_rate']:.3f}")
    print(f"   Training Time: {enhanced_training_time:.2f}s")
    print(f"   Mean Reward: {enhanced_eval_stats['mean_reward']:.2f}")
    print(f"   Mean Length: {enhanced_eval_stats['mean_length']:.1f} steps")
    
    return {
        'agent': enhanced_agent,
        'training_stats': enhanced_stats,
        'eval_stats': enhanced_eval_stats,
        'training_time': enhanced_training_time
    }


def compare_results(baseline_results, enhanced_results):
    """Compare baseline and enhanced DQN results."""
    print("\n" + "=" * 60)
    print("🏆 COMPARISON RESULTS")
    print("=" * 60)
    
    baseline_eval = baseline_results['eval_stats']
    enhanced_eval = enhanced_results['eval_stats']
    
    print(f"\n📊 Performance Comparison:")
    print(f"   Metric              | Baseline | Enhanced | Improvement")
    print(f"   --------------------|----------|----------|------------")
    print(f"   Training Win Rate   | {baseline_results['training_stats']['win_rate']:.3f}    | {enhanced_results['training_stats']['win_rate']:.3f}    | {enhanced_results['training_stats']['win_rate'] - baseline_results['training_stats']['win_rate']:+.3f}")
    print(f"   Evaluation Win Rate | {baseline_eval['win_rate']:.3f}    | {enhanced_eval['win_rate']:.3f}    | {enhanced_eval['win_rate'] - baseline_eval['win_rate']:+.3f}")
    print(f"   Mean Reward         | {baseline_eval['mean_reward']:.1f}    | {enhanced_eval['mean_reward']:.1f}    | {enhanced_eval['mean_reward'] - baseline_eval['mean_reward']:+.1f}")
    print(f"   Mean Length         | {baseline_eval['mean_length']:.1f}    | {enhanced_eval['mean_length']:.1f}    | {enhanced_eval['mean_length'] - baseline_eval['mean_length']:+.1f}")
    print(f"   Training Time       | {baseline_results['training_time']:.1f}s   | {enhanced_results['training_time']:.1f}s   | {enhanced_results['training_time'] - baseline_results['training_time']:+.1f}s")
    
    # Calculate improvement percentages
    win_rate_improvement = ((enhanced_eval['win_rate'] - baseline_eval['win_rate']) / baseline_eval['win_rate']) * 100
    reward_improvement = ((enhanced_eval['mean_reward'] - baseline_eval['mean_reward']) / baseline_eval['mean_reward']) * 100
    
    print(f"\n📈 Improvement Analysis:")
    print(f"   Win Rate Improvement: {win_rate_improvement:+.1f}%")
    print(f"   Reward Improvement: {reward_improvement:+.1f}%")
    
    if enhanced_eval['win_rate'] > baseline_eval['win_rate']:
        print(f"   🎉 Enhanced DQN performs better!")
    elif enhanced_eval['win_rate'] < baseline_eval['win_rate']:
        print(f"   ⚠️  Baseline DQN performs better")
    else:
        print(f"   🤝 Both agents perform similarly")
    
    # Save comparison results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dqn_comparison_results_{timestamp}.json"
    
    comparison_data = {
        'timestamp': timestamp,
        'baseline': {
            'training_stats': baseline_results['training_stats'],
            'eval_stats': baseline_results['eval_stats'],
            'training_time': baseline_results['training_time']
        },
        'enhanced': {
            'training_stats': enhanced_results['training_stats'],
            'eval_stats': enhanced_results['eval_stats'],
            'training_time': enhanced_results['training_time'],
            'techniques': {
                'double_dqn': enhanced_results['agent'].use_double_dqn,
                'dueling_dqn': enhanced_results['agent'].use_dueling,
                'prioritized_replay': enhanced_results['agent'].use_prioritized_replay
            }
        },
        'improvements': {
            'win_rate_improvement_pct': win_rate_improvement,
            'reward_improvement_pct': reward_improvement
        }
    }
    
    import json
    with open(filename, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\n💾 Comparison results saved to {filename}")


def main():
    """Run the complete comparison test."""
    print("🧪 Enhanced DQN vs Baseline DQN Comparison")
    print("=" * 60)
    
    try:
        # Test baseline DQN
        baseline_results = test_baseline_dqn()
        
        # Test enhanced DQN
        enhanced_results = test_enhanced_dqn()
        
        # Compare results
        compare_results(baseline_results, enhanced_results)
        
        print(f"\n✅ Comparison completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 All tests completed successfully!")
    else:
        print("\n💥 Tests failed. Please check the error messages above.")
        sys.exit(1) 