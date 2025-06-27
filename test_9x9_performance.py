#!/usr/bin/env python3
"""Quick performance test for 9×9 boards with 7 mines"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from core.minesweeper_env import MinesweeperEnv

def test_9x9_performance():
    """Test performance of 9×9 boards with 7 mines"""
    
    print("🧪 Testing 9×9 Board Performance")
    print("=" * 50)
    
    # Create environment
    env = MinesweeperEnv(
        initial_board_size=(9, 9),
        initial_mines=7,
        learnable_only=True
    )
    
    print(f"Board: 9×9 with 7 mines")
    print(f"Board area: {9*9} cells")
    print(f"Mine density: {7/(9*9)*100:.1f}%")
    print(f"Action space: {env.action_space.n} actions")
    
    # Performance test
    start_time = time.time()
    wins = 0
    episodes = 100
    
    for i in range(episodes):
        state, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 200:
            action = env.action_space.sample()  # Random actions
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            
            if info.get('won', False):
                wins += 1
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nPerformance Results:")
    print(f"  {episodes} episodes in {total_time:.1f}s")
    print(f"  Win rate: {wins/episodes*100:.1f}%")
    print(f"  Episodes per second: {episodes/total_time:.1f}")
    print(f"  Average time per episode: {total_time/episodes*1000:.1f}ms")
    
    # Estimate training time
    episodes_per_second = episodes / total_time
    target_episodes = 50000  # Typical training episodes
    
    estimated_training_time = target_episodes / episodes_per_second
    
    print(f"\nTraining Time Estimates:")
    print(f"  For {target_episodes:,} episodes: {estimated_training_time/60:.1f} minutes")
    print(f"  For 100,000 episodes: {100000/episodes_per_second/60:.1f} minutes")
    print(f"  For 200,000 episodes: {200000/episodes_per_second/60:.1f} minutes")

if __name__ == "__main__":
    test_9x9_performance() 