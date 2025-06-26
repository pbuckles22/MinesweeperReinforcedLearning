#!/usr/bin/env python3
"""
Test Winning 95% Configuration on Larger Boards

This script tests the winning Conservative Learning configuration (93.5% on 4x4)
on larger boards to see how it scales: 5x5, 6x6, and 8x8.
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
from core.dqn_agent_enhanced import EnhancedDQNAgent, train_enhanced_dqn_agent


def create_winning_agent(board_size: Tuple[int, int], mines: int) -> EnhancedDQNAgent:
    """Create agent with the WINNING Conservative Learning configuration, adapted for larger boards."""
    action_size = board_size[0] * board_size[1]
    
    # Adapt hyperparameters for larger boards
    if board_size[0] * board_size[1] <= 16:  # 4x4 and smaller
        learning_rate = 0.0001
        epsilon_decay = 0.9995
        epsilon_min = 0.05
        batch_size = 32
        target_update_freq = 1000
    elif board_size[0] * board_size[1] <= 25:  # 5x5
        learning_rate = 0.00005  # More conservative for larger boards
        epsilon_decay = 0.9997   # Even slower decay
        epsilon_min = 0.08       # Higher minimum exploration
        batch_size = 64          # Larger batches for more complex state
        target_update_freq = 1500
    elif board_size[0] * board_size[1] <= 36:  # 6x6
        learning_rate = 0.00003
        epsilon_decay = 0.9998
        epsilon_min = 0.1
        batch_size = 128
        target_update_freq = 2000
    else:  # 8x8 and larger
        learning_rate = 0.00002
        epsilon_decay = 0.9999
        epsilon_min = 0.15
        batch_size = 256
        target_update_freq = 3000
    
    return EnhancedDQNAgent(
        board_size=board_size,
        action_size=action_size,
        learning_rate=learning_rate,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        replay_buffer_size=200000,  # Larger buffer for larger boards
        batch_size=batch_size,
        target_update_freq=target_update_freq,
        device='cpu',
        use_double_dqn=True,
        use_dueling=True,
        use_prioritized_replay=True
    )


def train_on_board_size(board_size: Tuple[int, int], mines: int, episodes: int = 1000) -> Dict[str, Any]:
    """Train the winning configuration on a specific board size."""
    print(f"🎯 Training on {board_size[0]}x{board_size[1]} board with {mines} mines")
    print(f"   Target Episodes: {episodes}")
    print("=" * 60)
    
    # Create environment and agent
    env = MinesweeperEnv(initial_board_size=board_size, initial_mines=mines)
    agent = create_winning_agent(board_size, mines)
    
    print(f"✅ Adapted WINNING configuration for {board_size[0]}x{board_size[1]}:")
    print(f"   Learning Rate: {agent.learning_rate}")
    print(f"   Epsilon Decay: {agent.epsilon_decay}")
    print(f"   Epsilon Min: {agent.epsilon_min}")
    print(f"   Batch Size: {agent.batch_size}")
    print(f"   Target Update Freq: {agent.target_update_freq}")
    print(f"   Replay Buffer: {len(agent.replay_buffer)}")
    print("-" * 60)
    
    start_time = time.time()
    
    # Train with more frequent evaluation for larger boards
    eval_freq = max(25, episodes // 20)  # More frequent evaluation for larger boards
    training_stats = train_enhanced_dqn_agent(env, agent, episodes, mines, eval_freq=eval_freq)
    
    training_time = time.time() - start_time
    
    # Final statistics
    final_stats = agent.get_stats()
    episodes_per_second = agent.training_stats['episodes'] / training_time
    
    print(f"\n✅ {board_size[0]}x{board_size[1]} training completed!")
    print(f"   Final Win Rate: {final_stats['win_rate']:.3f}")
    print(f"   Final Epsilon: {agent.epsilon:.3f}")
    print(f"   Total Episodes: {agent.training_stats['episodes']}")
    print(f"   Training Time: {training_time:.2f}s")
    print(f"   Episodes/second: {episodes_per_second:.2f}")
    
    return {
        'board_size': board_size,
        'mines': mines,
        'win_rate': final_stats['win_rate'],
        'final_epsilon': agent.epsilon,
        'episodes': agent.training_stats['episodes'],
        'training_time': training_time,
        'episodes_per_second': episodes_per_second,
        'agent': agent,
        'training_stats': training_stats
    }


def evaluate_on_board_size(agent: EnhancedDQNAgent, board_size: Tuple[int, int], 
                          mines: int, n_runs: int = 20, episodes_per_run: int = 50) -> Dict[str, Any]:
    """Evaluate the trained agent on the same board size."""
    print(f"🔍 Evaluating on {board_size[0]}x{board_size[1]} board")
    print(f"   {n_runs} runs of {episodes_per_run} episodes each")
    print("=" * 50)
    
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
            max_steps = board_size[0] * board_size[1] * 2  # More steps for larger boards
            
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
    
    print(f"\n📊 {board_size[0]}x{board_size[1]} Statistical Results:")
    print(f"   Mean Win Rate: {evaluation_results['mean_win_rate']:.3f} ± {evaluation_results['std_win_rate']:.3f}")
    print(f"   Win Rate Range: {evaluation_results['min_win_rate']:.3f} - {evaluation_results['max_win_rate']:.3f}")
    print(f"   Mean Reward: {evaluation_results['mean_reward']:.2f} ± {evaluation_results['std_reward']:.2f}")
    print(f"   Mean Length: {evaluation_results['mean_length']:.1f} ± {evaluation_results['std_length']:.1f}")
    
    return evaluation_results


def main():
    """Test the winning configuration on larger boards."""
    print("🏆 Testing Winning 95% Configuration on Larger Boards")
    print("=" * 65)
    
    # Board configurations to test
    board_configs = [
        ((5, 5), 2),   # 5x5 with 2 mines
        ((6, 6), 3),   # 6x6 with 3 mines
        ((8, 8), 5),   # 8x8 with 5 mines
    ]
    
    all_results = []
    
    try:
        for board_size, mines in board_configs:
            print(f"\n{'='*20} {board_size[0]}x{board_size[1]} BOARD {'='*20}")
            
            # Train on this board size
            training_results = train_on_board_size(board_size, mines, episodes=1000)
            
            # Evaluate on this board size
            eval_results = evaluate_on_board_size(training_results['agent'], board_size, mines, 
                                                n_runs=20, episodes_per_run=50)
            
            # Performance assessment
            final_win_rate = eval_results['mean_win_rate']
            training_speed = training_results['episodes_per_second']
            
            print(f"\n🎯 {board_size[0]}x{board_size[1]} Performance Assessment:")
            print(f"   Training Win Rate: {training_results['win_rate']:.3f}")
            print(f"   Evaluation Win Rate: {final_win_rate:.3f} ± {eval_results['std_win_rate']:.3f}")
            print(f"   Training Speed: {training_speed:.2f} episodes/second")
            print(f"   Training Time: {training_results['training_time']:.2f}s")
            
            # Goal achievement
            if final_win_rate >= 0.90:
                print(f"   🎉 EXCELLENT: Achieved 90%+ win rate!")
            elif final_win_rate >= 0.80:
                print(f"   🏆 VERY GOOD: Achieved 80%+ win rate!")
            elif final_win_rate >= 0.70:
                print(f"   ✅ GOOD: Achieved 70%+ win rate!")
            elif final_win_rate >= 0.60:
                print(f"   ⚠️  ACCEPTABLE: Achieved 60%+ win rate!")
            else:
                print(f"   💥 NEEDS IMPROVEMENT: Below 60% win rate")
            
            # Store results
            all_results.append({
                'board_size': board_size,
                'mines': mines,
                'training_results': training_results,
                'evaluation_results': eval_results,
                'performance': {
                    'final_win_rate': float(final_win_rate),
                    'training_speed': float(training_speed),
                    'target_achieved': bool(final_win_rate >= 0.70)  # 70%+ is good for larger boards
                }
            })
        
        # Summary across all board sizes
        print(f"\n{'='*20} SUMMARY ACROSS ALL BOARDS {'='*20}")
        print(f"{'Board':<10} {'Mines':<6} {'Win Rate':<12} {'Speed':<12} {'Status':<15}")
        print("-" * 65)
        
        for result in all_results:
            board_str = f"{result['board_size'][0]}x{result['board_size'][1]}"
            mines = result['mines']
            win_rate = result['evaluation_results']['mean_win_rate']
            speed = result['training_results']['episodes_per_second']
            
            if win_rate >= 0.90:
                status = "🎉 EXCELLENT"
            elif win_rate >= 0.80:
                status = "🏆 VERY GOOD"
            elif win_rate >= 0.70:
                status = "✅ GOOD"
            elif win_rate >= 0.60:
                status = "⚠️  ACCEPTABLE"
            else:
                status = "💥 NEEDS WORK"
            
            print(f"{board_str:<10} {mines:<6} {win_rate:.3f}±{result['evaluation_results']['std_win_rate']:.3f}  {speed:<12.2f} {status:<15}")
        
        # Save comprehensive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"larger_boards_results_{timestamp}.json"
        
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
        for result in all_results:
            serializable_result = result.copy()
            # Remove agent object from training results
            if 'agent' in serializable_result['training_results']:
                serializable_result['training_results'].pop('agent')
            serializable_results.append(convert_numpy_types(serializable_result))
        
        results = {
            'timestamp': timestamp,
            'config': {
                'base_configuration': 'Winning 95% Conservative Learning',
                'adaptation_strategy': 'Adapted hyperparameters for larger boards',
                'target': 'Test scalability of winning configuration'
            },
            'board_results': serializable_results,
            'summary': {
                'total_boards_tested': len(all_results),
                'boards_above_80_percent': len([r for r in all_results if r['evaluation_results']['mean_win_rate'] >= 0.80]),
                'boards_above_70_percent': len([r for r in all_results if r['evaluation_results']['mean_win_rate'] >= 0.70]),
                'average_win_rate': np.mean([r['evaluation_results']['mean_win_rate'] for r in all_results])
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Comprehensive results saved to {filename}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during larger boards testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Larger boards testing completed successfully!")
    else:
        print("\n💥 Larger boards testing failed. Please check the error messages above.")
        sys.exit(1) 