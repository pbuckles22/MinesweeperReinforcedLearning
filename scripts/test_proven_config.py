#!/usr/bin/env python3
"""
Test Proven 90% Configuration

Simple test to verify the proven Conservative Learning configuration
achieves 90%+ win rate as expected.
"""

import sys
import os
import time
import json
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.minesweeper_env import MinesweeperEnv
from core.dqn_agent_enhanced import EnhancedDQNAgent, train_enhanced_dqn_agent


def test_proven_config():
    """Test the exact proven 90% configuration."""
    print("🧪 Testing Proven 90% Configuration")
    print("=" * 50)
    
    # PROVEN Conservative Learning configuration
    config = {
        'learning_rate': 0.0001,          # PROVEN: Conservative learning rate
        'epsilon_decay': 0.9995,          # PROVEN: Very slow exploration decay
        'epsilon_min': 0.05,              # PROVEN: Higher minimum exploration
        'replay_buffer_size': 100000,     # PROVEN: Smaller buffer
        'batch_size': 32,                 # PROVEN: Smaller batches
        'target_update_freq': 1000,       # PROVEN: More frequent updates
        'use_double_dqn': True,
        'use_dueling': True,
        'use_prioritized_replay': True
    }
    
    print(f"✅ Using PROVEN Conservative Learning configuration:")
    print(f"   Learning Rate: {config['learning_rate']}")
    print(f"   Epsilon Decay: {config['epsilon_decay']}")
    print(f"   Epsilon Min: {config['epsilon_min']}")
    print(f"   Batch Size: {config['batch_size']}")
    print(f"   Target Update Freq: {config['target_update_freq']}")
    print(f"   Replay Buffer: {config['replay_buffer_size']}")
    print(f"   Double DQN: {config['use_double_dqn']}")
    print(f"   Dueling: {config['use_dueling']}")
    print(f"   Prioritized Replay: {config['use_prioritized_replay']}")
    print("-" * 50)
    
    # Create environment and agent
    env = MinesweeperEnv(initial_board_size=(4, 4), initial_mines=1)
    agent = EnhancedDQNAgent(
        board_size=(4, 4),
        action_size=16,
        learning_rate=config['learning_rate'],
        discount_factor=0.99,
        epsilon=1.0,
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
    
    # Train with proven configuration
    start_time = time.time()
    training_stats = train_enhanced_dqn_agent(env, agent, episodes=1000, mine_count=1, eval_freq=50)
    training_time = time.time() - start_time
    
    # Comprehensive evaluation (same as proven runs)
    print(f"\n🔍 Running Comprehensive Evaluation (30 runs of 50 episodes each)...")
    all_results = []
    
    for run in range(30):
        wins = 0
        total_rewards = []
        episode_lengths = []
        
        for episode in range(50):
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
        
        run_win_rate = wins / 50
        run_mean_reward = np.mean(total_rewards)
        run_mean_length = np.mean(episode_lengths)
        
        all_results.append({
            'win_rate': run_win_rate,
            'mean_reward': run_mean_reward,
            'mean_length': run_mean_length
        })
        
        if run % 5 == 0:  # Print every 5th run
            print(f"Run {run + 1:2d}: Win Rate {run_win_rate:.3f}, "
                  f"Mean Reward {run_mean_reward:.2f}, "
                  f"Mean Length {run_mean_length:.1f}")
    
    # Calculate final statistics
    win_rates = [r['win_rate'] for r in all_results]
    mean_rewards = [r['mean_reward'] for r in all_results]
    mean_lengths = [r['mean_length'] for r in all_results]
    
    final_win_rate = np.mean(win_rates)
    final_std = np.std(win_rates)
    
    print(f"\n📊 Final Results:")
    print(f"   Training Win Rate: {training_stats['win_rate']:.3f}")
    print(f"   Evaluation Win Rate: {final_win_rate:.3f} ± {final_std:.3f}")
    print(f"   Win Rate Range: {np.min(win_rates):.3f} - {np.max(win_rates):.3f}")
    print(f"   Mean Reward: {np.mean(mean_rewards):.2f} ± {np.std(mean_rewards):.2f}")
    print(f"   Mean Length: {np.mean(mean_lengths):.1f} ± {np.std(mean_lengths):.1f}")
    print(f"   Training Time: {training_time:.2f}s")
    
    # Performance assessment
    if final_win_rate >= 0.90:
        print(f"   🎉 SUCCESS: Achieved 90%+ win rate ({final_win_rate:.1%})")
    elif final_win_rate >= 0.85:
        print(f"   ✅ GOOD: Achieved 85%+ win rate ({final_win_rate:.1%})")
    elif final_win_rate >= 0.80:
        print(f"   ⚠️  ACCEPTABLE: Achieved 80%+ win rate ({final_win_rate:.1%})")
    else:
        print(f"   💥 FAILED: Below 80% win rate ({final_win_rate:.1%})")
    
    return {
        'training_win_rate': training_stats['win_rate'],
        'eval_win_rate': final_win_rate,
        'eval_std': final_std,
        'training_time': training_time,
        'all_results': all_results
    }


if __name__ == "__main__":
    results = test_proven_config()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"proven_config_test_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {filename}") 