#!/usr/bin/env python3
"""
Debug script for learnable filtering logic.
Tests each step of the learnable configuration verification.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from core.minesweeper_env import MinesweeperEnv

def test_learnable_configuration_logic():
    """Test the learnable configuration logic step by step."""
    print("🔍 Debugging Learnable Configuration Logic")
    print("=" * 60)
    
    # Test configurations that failed in the audit
    test_configs = [
        {"size": (4, 4), "mines": 2, "name": "4x4-2mines"},
        {"size": (5, 5), "mines": 2, "name": "5x5-2mines"},
        {"size": (6, 6), "mines": 3, "name": "6x6-3mines"},
    ]
    
    for config in test_configs:
        print(f"\n{'='*50}")
        print(f"Testing: {config['name']}")
        print(f"{'='*50}")
        
        # Create environment
        env = MinesweeperEnv(
            max_board_size=config['size'],
            initial_board_size=config['size'],
            max_mines=config['mines'],
            initial_mines=config['mines'],
            learnable_only=True,
            max_learnable_attempts=1000
        )
        
        # Test multiple boards
        for board_num in range(5):
            print(f"\n   Board {board_num + 1}:")
            
            # Reset to get a new board
            obs, info = env.reset()
            
            # Check learnable status
            is_learnable = info.get('learnable', False)
            print(f"      Learnable flag: {is_learnable}")
            
            if not is_learnable:
                print(f"      ⚠️  Board marked as non-learnable")
                continue
            
            # Test first move
            action = 0  # Corner (0,0)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if info.get('won', False):
                print(f"      ❌ First move caused instant win!")
            elif terminated and reward == -20:
                print(f"      ❌ First move hit mine!")
            else:
                print(f"      ✅ First move was safe")
            
            # Test if the board actually passes learnable criteria
            mine_positions = [(y, x) for y in range(env.current_board_height) 
                             for x in range(env.current_board_width) if env.mines[y, x]]
            
            print(f"      Mine positions: {mine_positions}")
            print(f"      Number of mines: {len(mine_positions)}")
            
            # Test the learnable configuration logic directly
            is_actually_learnable = env._is_learnable_configuration(mine_positions)
            print(f"      _is_learnable_configuration result: {is_actually_learnable}")
            
            # Test safe first move
            has_safe_first_move = env._has_safe_first_move()
            print(f"      _has_safe_first_move result: {has_safe_first_move}")
            
            # Check if the board should have been marked as learnable
            should_be_learnable = is_actually_learnable and has_safe_first_move
            print(f"      Should be learnable: {should_be_learnable}")
            
            if is_learnable != should_be_learnable:
                print(f"      ⚠️  MISMATCH: Board marked as {is_learnable} but should be {should_be_learnable}")

def test_multi_mine_learnable_logic():
    """Test the multi-mine learnable logic specifically."""
    print(f"\n{'='*60}")
    print("Testing Multi-Mine Learnable Logic")
    print(f"{'='*60}")
    
    # Create a simple test environment
    env = MinesweeperEnv(
        max_board_size=(4, 4),
        initial_board_size=(4, 4),
        max_mines=2,
        initial_mines=2,
        learnable_only=False  # Don't use learnable filtering for this test
    )
    
    print("Testing _is_multi_mine_learnable with different configurations:")
    
    # Test case 1: Two mines that create an instant win
    mine_positions_1 = [(0, 1), (1, 0)]  # Adjacent to corner (0,0)
    print(f"\n   Test 1 - Mines at {mine_positions_1}:")
    
    # Place mines manually
    env._place_mines_at_positions(mine_positions_1)
    
    # Test if this should be learnable
    is_learnable_1 = env._is_learnable_configuration(mine_positions_1)
    print(f"      _is_learnable_configuration: {is_learnable_1}")
    
    # Test first move
    env.reset()
    action = 0  # Corner (0,0)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if info.get('won', False):
        print(f"      First move result: Instant win")
    elif terminated and reward == -20:
        print(f"      First move result: Mine hit")
    else:
        print(f"      First move result: Safe")
    
    # Test case 2: Two mines that don't create instant win
    mine_positions_2 = [(2, 2), (3, 3)]  # Far from corner
    print(f"\n   Test 2 - Mines at {mine_positions_2}:")
    
    env._place_mines_at_positions(mine_positions_2)
    
    is_learnable_2 = env._is_learnable_configuration(mine_positions_2)
    print(f"      _is_learnable_configuration: {is_learnable_2}")
    
    # Test first move
    env.reset()
    action = 0  # Corner (0,0)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if info.get('won', False):
        print(f"      First move result: Instant win")
    elif terminated and reward == -20:
        print(f"      First move result: Mine hit")
    else:
        print(f"      First move result: Safe")

def test_single_mine_learnable_logic():
    """Test the single mine learnable logic."""
    print(f"\n{'='*60}")
    print("Testing Single Mine Learnable Logic")
    print(f"{'='*60}")
    
    env = MinesweeperEnv(
        max_board_size=(4, 4),
        initial_board_size=(4, 4),
        max_mines=1,
        initial_mines=1,
        learnable_only=False
    )
    
    print("Testing _is_single_mine_learnable with different positions:")
    
    # Test different mine positions
    test_positions = [(0, 0), (0, 1), (1, 1), (2, 2), (3, 3)]
    
    for pos in test_positions:
        print(f"\n   Mine at {pos}:")
        
        # Test the single mine learnable logic
        is_learnable = env._is_single_mine_learnable(pos)
        print(f"      _is_single_mine_learnable: {is_learnable}")
        
        # Place the mine and test first move
        env._place_mines_at_positions([pos])
        env.reset()
        
        action = 0  # Corner (0,0)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if info.get('won', False):
            print(f"      First move result: Instant win")
        elif terminated and reward == -20:
            print(f"      First move result: Mine hit")
        else:
            print(f"      First move result: Safe")

def main():
    """Run the debug tests."""
    print("🧪 Debug Learnable Filtering")
    print("=" * 60)
    
    # Test the overall learnable configuration logic
    test_learnable_configuration_logic()
    
    # Test multi-mine logic specifically
    test_multi_mine_learnable_logic()
    
    # Test single mine logic
    test_single_mine_learnable_logic()
    
    print(f"\n{'='*60}")
    print("DEBUG SUMMARY")
    print(f"{'='*60}")
    print("The issue is likely in _is_multi_mine_learnable() which always returns True.")
    print("This means any board with 2+ mines is marked as learnable without verification.")

if __name__ == "__main__":
    main() 