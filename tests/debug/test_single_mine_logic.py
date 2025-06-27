#!/usr/bin/env python3
"""
Test script to verify single-mine learnable logic.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from core.minesweeper_env import MinesweeperEnv

def test_single_mine_logic():
    """Test the single-mine learnable logic."""
    print("🧪 Testing Single-Mine Learnable Logic")
    print("=" * 50)
    
    # Test 4x4 with 1 mine
    print("\nTesting 4x4 with 1 mine:")
    
    # Test different mine positions
    test_positions = [
        (0, 0),  # Corner - should not be learnable (1-move win possible)
        (0, 1),  # Edge - should not be learnable (1-move win possible)  
        (1, 1),  # Center - should not be learnable (1-move win possible)
        (3, 3),  # Corner - should not be learnable (1-move win possible)
    ]
    
    for pos in test_positions:
        print(f"\n  Mine at {pos}:")
        
        env = MinesweeperEnv(
            max_board_size=(4, 4),
            initial_board_size=(4, 4),
            max_mines=1,
            initial_mines=1,
            learnable_only=False
        )
        
        # Place mine at specific position
        env._place_mines_at_positions([pos])
        
        # Test if this is learnable
        is_learnable = env._is_learnable_configuration([pos])
        print(f"    _is_learnable_configuration: {is_learnable}")
        
        # Test cascade simulation
        total_cells = 16
        mine_count = 1
        safe_cells = total_cells - mine_count
        
        # Simulate cascade from a non-mine position
        revealed = env._simulate_cascade(env.board, pos)
        print(f"    Cascade reveals: {revealed} cells")
        print(f"    Safe cells: {safe_cells}")
        print(f"    Is 1-move win: {revealed >= safe_cells}")
        
        # Test actual first move
        env.reset()
        # Find a non-mine position for first move
        for action in range(16):
            col = action % 4
            row = action // 4
            if not env.mines[row, col]:
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"    First move from ({row},{col}) - won: {info.get('won', False)}")
                break

def test_single_mine_audit():
    """Test single-mine boards with learnable_only=True."""
    print(f"\n{'='*50}")
    print("Testing Single-Mine Boards with learnable_only=True")
    print(f"{'='*50}")
    
    env = MinesweeperEnv(
        max_board_size=(4, 4),
        initial_board_size=(4, 4),
        max_mines=1,
        initial_mines=1,
        learnable_only=True,
        max_learnable_attempts=1000
    )
    
    learnable_count = 0
    total_tests = 100
    
    for i in range(total_tests):
        obs, info = env.reset()
        
        if info.get('learnable', False):
            learnable_count += 1
            print(f"  Board {i+1}: Learnable board found!")
            
            # Show mine position
            mine_positions = [(y, x) for y in range(4) for x in range(4) if env.mines[y, x]]
            print(f"    Mine position: {mine_positions[0]}")
            
            # Test if this actually requires 2+ moves
            has_instant_win = False
            for action in range(16):
                col = action % 4
                row = action // 4
                
                if not env.mines[row, col]:  # Only test non-mine positions
                    env.reset()
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    if info.get('won', False):
                        has_instant_win = True
                        print(f"    ❌ Action {action} (pos {row},{col}) caused instant win!")
                        break
            
            if not has_instant_win:
                print(f"    ✅ No instant wins found")
    
    print(f"\nResults: {learnable_count}/{total_tests} learnable boards ({learnable_count/total_tests*100:.1f}%)")

if __name__ == "__main__":
    test_single_mine_logic()
    test_single_mine_audit() 