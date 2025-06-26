#!/usr/bin/env python3
"""
DQN 8x8 Curriculum Training - Overnight Run
Simplified version for 7-hour training session.
"""

import sys
import os
import json
import time
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.core.minesweeper_env import MinesweeperEnv
from src.core.dqn_agent import DQNAgent, train_dqn_agent, evaluate_dqn_agent

def check_existing_progress():
    """Check if there's existing progress to resume from."""
    for stage in range(1, 5):
        progress_file = f"training_stats/progress_stage{stage}.json"
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                progress = json.load(f)
            print(f"📋 Found existing progress: Stage {progress['stage']}")
            print(f"   Episodes completed: {progress['episodes_completed']:,}/{progress['total_episodes']:,}")
            return progress
    return None

def main():
    print("🚀 DQN 8x8 Curriculum Training - Overnight Run")
    print("=" * 60)
    
    # Check for existing progress
    existing_progress = check_existing_progress()
    if existing_progress:
        response = input("Resume from existing progress? (y/n): ").lower().strip()
        if response != 'y':
            print("Starting fresh...")
            existing_progress = None
    
    # 8x8 curriculum stages
    stages = [
        (5, 0.25, 10000),   # Stage 1: 5 mines, 25% target (realistic)
        (8, 0.15, 15000),   # Stage 2: 8 mines, 15% target (challenging)
        (12, 0.10, 20000),  # Stage 3: 12 mines, 10% target (difficult)
        (16, 0.05, 25000),  # Stage 4: 16 mines, 5% target (very hard)
    ]
    
    board_size = (8, 8)
    action_size = 64
    current_agent = None
    results = []
    
    for stage_idx, (mines, target, episodes) in enumerate(stages, 1):
        print(f"\n🎯 Stage {stage_idx}: {mines} mines on 8x8 board")
        print(f"   Target: {target:.1%}, Episodes: {episodes:,}")
        print(f"   Mine density: {mines/64:.1%}")
        
        # Create environment and agent
        env = MinesweeperEnv(initial_board_size=board_size, initial_mines=mines)
        
        if current_agent is None:
            current_agent = DQNAgent(
                board_size=board_size,
                action_size=action_size,
                learning_rate=0.0003,
                epsilon=1.0,
                epsilon_decay=0.9995,
                epsilon_min=0.15,
                batch_size=64,
                device='cpu'
            )
        
        # Train
        start_time = time.time()
        
        # Save progress every 2000 episodes
        checkpoint_freq = 2000
        for i in range(0, episodes, checkpoint_freq):
            batch_episodes = min(checkpoint_freq, episodes - i)
            print(f"   Training episodes {i+1}-{i+batch_episodes}...")
            
            train_stats = train_dqn_agent(env, current_agent, batch_episodes, mines, eval_freq=500)
            
            # Save checkpoint every batch
            current_agent.save_model(f"models/dqn_8x8_stage{stage_idx}_checkpoint.pth")
            
            # Save progress
            progress = {
                'stage': stage_idx,
                'mines': mines,
                'episodes_completed': i + batch_episodes,
                'total_episodes': episodes,
                'epsilon': current_agent.epsilon,
                'timestamp': datetime.now().isoformat()
            }
            with open(f"training_stats/progress_stage{stage_idx}.json", 'w') as f:
                json.dump(progress, f)
        
        train_time = time.time() - start_time
        
        # Evaluate
        eval_stats = evaluate_dqn_agent(current_agent, env, n_episodes=200)
        
        # Save results
        result = {
            'stage': stage_idx,
            'mines': mines,
            'target': target,
            'episodes': episodes,
            'win_rate': eval_stats['win_rate'],
            'train_time': train_time,
            'epsilon': current_agent.epsilon
        }
        results.append(result)
        
        print(f"   ✅ Win rate: {eval_stats['win_rate']:.1%}")
        print(f"   ⏱️  Time: {train_time/60:.1f} minutes")
        
        # Save checkpoint
        os.makedirs("models", exist_ok=True)
        current_agent.save_model(f"models/dqn_8x8_stage{stage_idx}.pth")
        
        # Continue to next stage regardless of performance
        print(f"   🚀 Continuing to next stage...")
    
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"training_stats/dqn_8x8_results_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎉 8x8 curriculum completed!")
    print(f"📊 Stages completed: {len(results)}")
    for r in results:
        print(f"   Stage {r['stage']}: {r['win_rate']:.1%} win rate")

if __name__ == "__main__":
    main() 