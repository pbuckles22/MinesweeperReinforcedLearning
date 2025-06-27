#!/usr/bin/env python3
"""
Focused Next Level Training Script

This script focuses on improving performance on the next level (5×5) 
using gradual progression and optimized hyperparameters.
"""

import json
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.dqn_agent_enhanced import DQNAgent
from core.minesweeper_env import MinesweeperEnv
from core.constants import REWARD_SAFE, REWARD_MINE, REWARD_WIN

def create_environment(board_size, mines):
    """Create environment with specific board size and mine count."""
    return MinesweeperEnv(
        board_size=board_size,
        num_mines=mines,
        max_steps=board_size[0] * board_size[1] * 2
    )

def evaluate_agent(agent, env, episodes=100, runs=5):
    """Evaluate agent with multiple runs for stability."""
    win_rates = []
    
    for run in range(runs):
        wins = 0
        for episode in range(episodes):
            state = env.reset()
            done = False
            
            while not done:
                action = agent.select_action(state, epsilon=0.0)  # Greedy
                state, reward, done, _ = env.step(action)
                
                if reward == REWARD_WIN:
                    wins += 1
                    break
                elif reward == REWARD_MINE:
                    break
        
        win_rate = wins / episodes
        win_rates.append(win_rate)
    
    return np.mean(win_rates), np.std(win_rates)

def train_stage(name, board_size, mines, target_win_rate, 
                min_episodes=5000, max_episodes=15000,
                load_model_path=None, save_model_path=None):
    """Train a single stage with focused parameters."""
    
    print(f"\n🎯 {name}")
    print(f"Board: {board_size[0]}×{board_size[1]}, Mines: {mines}")
    print(f"Target: {target_win_rate*100:.1f}%")
    print("-" * 50)
    
    # Create environment and agent
    env = create_environment(board_size, mines)
    
    # Enhanced agent configuration for better learning
    agent_config = {
        'learning_rate': 0.0001,
        'epsilon_decay': 0.9997,  # Slower decay for more exploration
        'epsilon_min': 0.1,       # Higher minimum for more exploration
        'replay_buffer_size': 100000,
        'batch_size': 32,
        'target_update_freq': 1000,
        'use_double_dqn': True,
        'use_dueling': True,
        'use_prioritized_replay': True,
        'dropout_rate': 0.4,
        'weight_decay': 1e-4
    }
    
    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        **agent_config
    )
    
    # Load previous model if available
    if load_model_path and Path(load_model_path).exists():
        print(f"📥 Loading model from {load_model_path}")
        agent.load_model(load_model_path)
    
    # Training loop with early stopping
    best_win_rate = 0.0
    consecutive_targets = 0
    eval_freq = 500
    target_consecutive = 2  # Reduced from 3 for faster progression
    
    training_stats = {
        'episodes': [],
        'win_rates': [],
        'mean_rewards': [],
        'epsilon_values': []
    }
    
    start_time = time.time()
    
    for episode in range(max_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.store_experience(state, action, reward, next_state, done)
            agent.train()
            
            total_reward += reward
            state = next_state
            steps += 1
            
            if done:
                break
        
        # Periodic evaluation
        if (episode + 1) % eval_freq == 0:
            win_rate, win_std = evaluate_agent(agent, env, episodes=50, runs=3)
            
            training_stats['episodes'].append(episode + 1)
            training_stats['win_rates'].append(win_rate)
            training_stats['mean_rewards'].append(total_reward)
            training_stats['epsilon_values'].append(agent.epsilon)
            
            print(f"Episode {episode + 1:5d} | Win Rate: {win_rate*100:5.1f}% ± {win_std*100:.1f}% | "
                  f"Epsilon: {agent.epsilon:.3f} | Reward: {total_reward:6.1f}")
            
            # Check for target achievement
            if win_rate >= target_win_rate:
                consecutive_targets += 1
                print(f"🎯 Target achieved! ({consecutive_targets}/{target_consecutive})")
                
                if consecutive_targets >= target_consecutive:
                    print(f"✅ Stage completed! Final win rate: {win_rate*100:.1f}%")
                    break
            else:
                consecutive_targets = 0
            
            # Save best model
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                if save_model_path:
                    agent.save_model(save_model_path)
                    print(f"💾 Best model saved: {win_rate*100:.1f}%")
        
        # Early stopping if no progress
        if episode > min_episodes and episode % 2000 == 0:
            recent_win_rates = training_stats['win_rates'][-4:]  # Last 4 evaluations
            if len(recent_win_rates) >= 4:
                if max(recent_win_rates) < target_win_rate * 0.5:  # Less than 50% of target
                    print(f"⚠️ Early stopping - no progress toward target")
                    break
    
    training_time = time.time() - start_time
    
    # Final evaluation
    final_win_rate, final_std = evaluate_agent(agent, env, episodes=100, runs=5)
    
    results = {
        'stage_name': name,
        'board_size': board_size,
        'mines': mines,
        'target_win_rate': target_win_rate,
        'final_win_rate': final_win_rate,
        'best_win_rate': best_win_rate,
        'training_episodes': episode + 1,
        'training_time': training_time,
        'training_stats': training_stats,
        'model_path': save_model_path
    }
    
    print(f"\n📊 Stage Results:")
    print(f"Final Win Rate: {final_win_rate*100:.1f}% ± {final_std*100:.1f}%")
    print(f"Best Win Rate: {best_win_rate*100:.1f}%")
    print(f"Training Time: {training_time/60:.1f} minutes")
    print(f"Episodes: {episode + 1}")
    
    return results, agent

def main():
    """Main training function with gradual progression."""
    
    print("🚀 Focused Next Level Training")
    print("=" * 50)
    
    # Create results directory
    results_dir = Path("focused_next_level_results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"focused_next_level_{timestamp}.json"
    
    # Gradual progression stages
    stages = [
        {
            'name': 'Stage 1: 4×4 Mastery (Baseline)',
            'board_size': (4, 4),
            'mines': 1,
            'target_win_rate': 0.80,  # Realistic target
            'min_episodes': 3000,
            'max_episodes': 10000
        },
        {
            'name': 'Stage 2: 4×5 Extension (Width)',
            'board_size': (4, 5),
            'mines': 1,
            'target_win_rate': 0.70,
            'min_episodes': 5000,
            'max_episodes': 12000
        },
        {
            'name': 'Stage 3: 5×4 Extension (Height)',
            'board_size': (5, 4),
            'mines': 1,
            'target_win_rate': 0.70,
            'min_episodes': 5000,
            'max_episodes': 12000
        },
        {
            'name': 'Stage 4: 5×5 Foundation',
            'board_size': (5, 5),
            'mines': 1,
            'target_win_rate': 0.60,
            'min_episodes': 8000,
            'max_episodes': 15000
        },
        {
            'name': 'Stage 5: 5×5 Challenge',
            'board_size': (5, 5),
            'mines': 2,
            'target_win_rate': 0.45,
            'min_episodes': 10000,
            'max_episodes': 20000
        }
    ]
    
    all_results = []
    previous_model_path = None
    
    for i, stage in enumerate(stages):
        stage_num = i + 1
        
        # Create model path for this stage
        model_path = results_dir / f"stage_{stage_num}_{stage['board_size'][0]}x{stage['board_size'][1]}_{stage['mines']}mines.pth"
        
        print(f"\n{'='*60}")
        print(f"🎯 STAGE {stage_num}/{len(stages)}")
        print(f"{'='*60}")
        
        # Train this stage
        results, agent = train_stage(
            name=stage['name'],
            board_size=stage['board_size'],
            mines=stage['mines'],
            target_win_rate=stage['target_win_rate'],
            min_episodes=stage['min_episodes'],
            max_episodes=stage['max_episodes'],
            load_model_path=previous_model_path,
            save_model_path=str(model_path)
        )
        
        all_results.append(results)
        previous_model_path = str(model_path)
        
        # Save intermediate results
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"💾 Results saved to {results_file}")
        
        # Check if we should continue
        if results['final_win_rate'] < stage['target_win_rate'] * 0.5:
            print(f"⚠️ Performance too low ({results['final_win_rate']*100:.1f}% vs {stage['target_win_rate']*100:.1f}% target)")
            print("Consider adjusting hyperparameters or targets")
            break
    
    # Final summary
    print(f"\n{'='*60}")
    print("📊 TRAINING SUMMARY")
    print(f"{'='*60}")
    
    for i, result in enumerate(all_results):
        stage_num = i + 1
        print(f"Stage {stage_num}: {result['stage_name']}")
        print(f"  Board: {result['board_size'][0]}×{result['board_size'][1]}, Mines: {result['mines']}")
        print(f"  Target: {result['target_win_rate']*100:.1f}%, Achieved: {result['final_win_rate']*100:.1f}%")
        print(f"  Status: {'✅' if result['final_win_rate'] >= result['target_win_rate'] else '❌'}")
        print()
    
    print(f"📁 Results saved to: {results_file}")
    print(f"📁 Models saved to: {results_dir}/")

if __name__ == "__main__":
    main() 