#!/usr/bin/env python3

from src.core.minesweeper_env import MinesweeperEnv
from src.core.constants import *
import numpy as np

def test_cascade_safety():
    """Test the cascade safety implementation"""
    print("=== Testing Cascade Safety Implementation ===")
    
    # Create environment
    env = MinesweeperEnv(max_board_size=4, max_mines=2)
    
    print("Reward Structure:")
    print(f"  REWARD_FIRST_CASCADE_SAFE: {REWARD_FIRST_CASCADE_SAFE}")
    print(f"  REWARD_SAFE_REVEAL: {REWARD_SAFE_REVEAL}")
    print(f"  REWARD_WIN: {REWARD_WIN}")
    print(f"  REWARD_HIT_MINE: {REWARD_HIT_MINE}")
    
    print("\nReward Ratios:")
    if REWARD_FIRST_CASCADE_SAFE != 0:
        print(f"  Post-cascade / Pre-cascade: {REWARD_SAFE_REVEAL / REWARD_FIRST_CASCADE_SAFE:.1f}x")
    else:
        print(f"  Post-cascade / Pre-cascade: ∞ (pre-cascade reward is 0)")
    print(f"  Win / Post-cascade: {REWARD_WIN / REWARD_SAFE_REVEAL:.1f}x")
    print(f"  Mine penalty / Post-cascade: {abs(REWARD_HIT_MINE) / REWARD_SAFE_REVEAL:.1f}x")
    
    print("\n" + "=" * 50)
    
    # Test multiple games to see cascade behavior
    for game in range(3):
        print(f"\n--- Game {game + 1} ---")
        obs, info = env.reset()
        
        total_reward = 0
        steps = 0
        cascade_occurred = False
        
        for step in range(10):
            # Take random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            print(f"Step {step}: action={action}, reward={reward}, is_first_cascade={env.is_first_cascade}")
            
            if not env.is_first_cascade and not cascade_occurred:
                cascade_occurred = True
                print(f"  *** CASCADE OCCURRED at step {step} ***")
            
            if terminated or truncated:
                print(f"  Game ended: won={info['won']}")
                break
        
        print(f"Game Summary: steps={steps}, total_reward={total_reward}, cascade_occurred={cascade_occurred}")
        print("=" * 50)
    
    print("\n✅ Cascade safety test completed!")

if __name__ == "__main__":
    test_cascade_safety() 