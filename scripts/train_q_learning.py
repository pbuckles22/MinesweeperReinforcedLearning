#!/usr/bin/env python3
"""
Q-Learning Training Script for Minesweeper

Trains Q-learning agent with experience replay and curriculum learning
to address catastrophic forgetting and improve performance.
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.q_learning_agent import QLearningAgent, train_q_learning_agent, evaluate_q_agent
from core.minesweeper_env import MinesweeperEnv
from stable_baselines3.common.vec_env import DummyVecEnv


def create_q_env(board_size, max_mines):
    """Create environment for Q-learning."""
    return DummyVecEnv([lambda: MinesweeperEnv(max_board_size=board_size, max_mines=max_mines)])


def q_learning_curriculum_training():
    """Train Q-learning agent with curriculum learning."""
    
    print("🎯 Q-Learning Curriculum Training for Minesweeper")
    print("=" * 60)
    
    # Training configuration
    board_size = (4, 4)
    
    # Curriculum stages
    curriculum_stages = [
        # Stage 1: Learn basics (1 mine)
        {"mine_count": 1, "episodes": 2000, "target_win_rate": 0.70},
        
        # Stage 2: Build complexity (2 mines)
        {"mine_count": 2, "episodes": 3000, "target_win_rate": 0.50},
        
        # Stage 3: Challenge (3 mines)
        {"mine_count": 3, "episodes": 4000, "target_win_rate": 0.30},
    ]
    
    results = []
    current_agent = None
    
    for stage_num, stage_config in enumerate(curriculum_stages, 1):
        mine_count = stage_config["mine_count"]
        episodes = stage_config["episodes"]
        target_win_rate = stage_config["target_win_rate"]
        
        print(f"\n🎯 Stage {stage_num}: {mine_count} mines")
        print(f"   Board: {board_size[0]}x{board_size[1]}")
        print(f"   Episodes: {episodes:,}")
        print(f"   Target Win Rate: {target_win_rate:.1%}")
        print("-" * 60)
        
        try:
            # Create environment
            env = create_q_env(board_size, mine_count)
            
            if current_agent is None:
                # First stage: create new agent
                print(f"🚀 Creating new Q-learning agent...")
                current_agent = QLearningAgent(
                    board_size=board_size,
                    max_mines=mine_count,
                    learning_rate=0.1,
                    epsilon=0.3,
                    epsilon_decay=0.9995,
                    replay_buffer_size=5000,
                    batch_size=32
                )
            else:
                # Continue with existing agent (experience replay will help)
                print(f"🔄 Continuing with existing agent...")
                current_agent.max_mines = mine_count  # Update for new mine count
            
            # Train the agent
            training_stats = train_q_learning_agent(
                env, current_agent, episodes=episodes, 
                mine_count=mine_count, eval_freq=200
            )
            
            # Evaluate performance
            eval_results = evaluate_q_agent(current_agent, env, n_episodes=100)
            
            # Save stage results
            result = {
                "stage": stage_num,
                "board_size": board_size,
                "mine_count": mine_count,
                "episodes": episodes,
                "training_stats": training_stats,
                "eval_results": eval_results,
                "target_win_rate": target_win_rate,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            
            results.append(result)
            
            print(f"📊 Stage {stage_num} Results:")
            print(f"   Win Rate: {eval_results['win_rate']:.3f}")
            print(f"   Target: {target_win_rate:.1%}")
            
            # Check if we should continue
            if eval_results['win_rate'] < target_win_rate:
                print(f"⚠️  Performance below target ({eval_results['win_rate']:.1%} < {target_win_rate:.1%}).")
                response = input("Continue to next stage anyway? (y/n): ").lower().strip()
                if response != 'y':
                    print("Stopping training.")
                    break
            else:
                print(f"✅ Target achieved! ({eval_results['win_rate']:.1%} >= {target_win_rate:.1%})")
                
        except Exception as e:
            print(f"❌ Error in stage {stage_num}: {e}")
            break
    
    # Save training results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"experiments/q_learning_curriculum_results_{timestamp}.json"
    
    # Ensure experiments directory exists
    os.makedirs("experiments", exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            "training_summary": {
                "total_stages": len(curriculum_stages),
                "completed_stages": len(results),
                "timestamp": timestamp,
                "approach": "q_learning_with_experience_replay"
            },
            "results": results
        }, f, indent=2)
    
    print(f"\n💾 Training results saved to: {results_file}")
    
    # Save final model
    if current_agent:
        model_file = f"models/q_learning_final_{timestamp}.pkl"
        os.makedirs("models", exist_ok=True)
        current_agent.save_model(model_file)
        print(f"💾 Final model saved to: {model_file}")
    
    # Print final summary
    print(f"\n📈 Q-Learning Curriculum Training Summary")
    print("=" * 60)
    for result in results:
        stage = result["stage"]
        mine_count = result["mine_count"]
        win_rate = result["eval_results"]["win_rate"]
        target = result["target_win_rate"]
        status = "✅ PASS" if win_rate >= target else "⚠️  BELOW"
        
        print(f"Stage {stage}: {mine_count} mines | {win_rate:.1%} | {status}")
    
    return results, current_agent


def quick_q_learning_test():
    """Quick test of Q-learning agent."""
    
    print("🧪 Quick Q-Learning Test")
    print("=" * 40)
    
    # Test configuration
    board_size = (4, 4)
    mine_count = 1
    episodes = 500
    
    # Create environment and agent
    env = create_q_env(board_size, mine_count)
    agent = QLearningAgent(
        board_size=board_size,
        max_mines=mine_count,
        learning_rate=0.1,
        epsilon=0.3,
        epsilon_decay=0.999
    )
    
    # Train for a few episodes
    print(f"Training on {mine_count} mine for {episodes} episodes...")
    training_stats = train_q_learning_agent(env, agent, episodes=episodes, mine_count=mine_count, eval_freq=100)
    
    # Evaluate
    eval_results = evaluate_q_agent(agent, env, n_episodes=50)
    
    print(f"\n✅ Quick test completed!")
    print(f"   Final Win Rate: {eval_results['win_rate']:.3f}")
    print(f"   Q-table size: {agent.get_stats()['q_table_size']}")
    
    return eval_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Q-learning agent for Minesweeper")
    parser.add_argument("--quick", action="store_true", help="Run quick test")
    parser.add_argument("--curriculum", action="store_true", help="Run curriculum training")
    
    args = parser.parse_args()
    
    if args.quick:
        quick_q_learning_test()
    elif args.curriculum:
        q_learning_curriculum_training()
    else:
        # Default to quick test
        quick_q_learning_test() 