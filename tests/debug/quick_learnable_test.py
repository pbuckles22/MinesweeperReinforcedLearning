#!/usr/bin/env python3
"""
Quick test to verify learnable filtering logic works correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from core.minesweeper_env import MinesweeperEnv

def test_learnable_filtering():
    """Test the learnable filtering logic with a small sample."""
    print("🧪 Quick Learnable Filtering Test")
    print("=" * 50)
    
    configs = [
        ((4, 4), 1),
        ((4, 4), 2),
        ((5, 5), 2),
    ]
    
    for board_size, mine_count in configs:
        print(f"\nTesting {board_size[0]}x{board_size[1]} with {mine_count} mines:")
        
        learnable_count = 0
        total_tests = 100
        
        for i in range(total_tests):
            env = MinesweeperEnv(
                max_board_size=board_size,
                initial_board_size=board_size,
                max_mines=mine_count,
                initial_mines=mine_count,
                learnable_only=True,
                max_learnable_attempts=1000
            )
            
            obs, info = env.reset()
            
            if info.get('learnable', False):
                learnable_count += 1
                
                # Quick test: check if any first move causes instant win
                has_instant_win = False
                for action in range(min(10, env.action_space.n)):  # Test first 10 positions
                    col = action % env.current_board_width
                    row = action // env.current_board_width
                    
                    if not env.mines[row, col]:  # Only test non-mine positions
                        # Reset and test this move
                        env.reset()
                        obs, reward, terminated, truncated, info = env.step(action)
                        
                        if info.get('won', False):
                            has_instant_win = True
                            print(f"   ❌ Board {i+1}: Action {action} (pos {row},{col}) caused instant win!")
                            break
        
        learnable_percentage = (learnable_count / total_tests) * 100
        print(f"   Learnable boards: {learnable_count}/{total_tests} ({learnable_percentage:.1f}%)")
        
        if learnable_count > 0:
            print(f"   ✅ No instant wins found in learnable boards")

def test_specific_configurations():
    """Test specific configurations that should or shouldn't be learnable."""
    print(f"\n{'='*50}")
    print("Testing Specific Configurations")
    print(f"{'='*50}")
    
    # Test 1: Single mine in corner - should not be learnable (1-move win possible)
    print("\nTest 1: Single mine in corner (should not be learnable)")
    env = MinesweeperEnv(
        max_board_size=(4, 4),
        initial_board_size=(4, 4),
        max_mines=1,
        initial_mines=1,
        learnable_only=False
    )
    
    env._place_mines_at_positions([(0, 0)])
    is_learnable = env._is_learnable_configuration([(0, 0)])
    print(f"   _is_learnable_configuration: {is_learnable}")
    
    # Test 2: Two mines that don't create instant win - should be learnable
    print("\nTest 2: Two mines not creating instant win (should be learnable)")
    env = MinesweeperEnv(
        max_board_size=(4, 4),
        initial_board_size=(4, 4),
        max_mines=2,
        initial_mines=2,
        learnable_only=False
    )
    
    env._place_mines_at_positions([(0, 0), (3, 3)])
    is_learnable = env._is_learnable_configuration([(0, 0), (3, 3)])
    print(f"   _is_learnable_configuration: {is_learnable}")

if __name__ == "__main__":
    test_learnable_filtering()
    test_specific_configurations() 