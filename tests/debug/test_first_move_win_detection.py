#!/usr/bin/env python3
"""
Test script to verify first-move win detection logic.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from core.minesweeper_env import MinesweeperEnv

def test_first_move_win_detection():
    """Test the first-move win detection logic."""
    print("🧪 Testing First-Move Win Detection")
    print("=" * 50)
    
    # Test case 1: Single mine in corner - should be learnable (no instant win)
    print("\nTest 1: Single mine in corner (0,0)")
    env = MinesweeperEnv(
        max_board_size=(4, 4),
        initial_board_size=(4, 4),
        max_mines=1,
        initial_mines=1,
        learnable_only=False
    )
    
    # Place mine at (0,0)
    env._place_mines_at_positions([(0, 0)])
    
    # Test if this is learnable
    is_learnable = env._is_learnable_configuration([(0, 0)])
    print(f"   _is_learnable_configuration: {is_learnable}")
    
    # Test first move from (0,1) - should not be instant win
    env.reset()
    obs, reward, terminated, truncated, info = env.step(1)  # action 1 = (0,1)
    print(f"   First move from (0,1) - won: {info.get('won', False)}")
    
    # Test case 2: Two mines that create instant win - should not be learnable
    print("\nTest 2: Two mines creating instant win")
    env = MinesweeperEnv(
        max_board_size=(4, 4),
        initial_board_size=(4, 4),
        max_mines=2,
        initial_mines=2,
        learnable_only=False
    )
    
    # Place mines at (1,1) and (2,2) - this should create instant win from (0,0)
    env._place_mines_at_positions([(1, 1), (2, 2)])
    
    # Debug: Show the board configuration
    print(f"   Board with adjacent counts:")
    print(env.board)
    
    # Test if this is learnable
    is_learnable = env._is_learnable_configuration([(1, 1), (2, 2)])
    print(f"   _is_learnable_configuration: {is_learnable}")
    
    # Test first move from (0,0) - should be instant win
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)  # action 0 = (0,0)
    print(f"   First move from (0,0) - won: {info.get('won', False)}")
    
    # Test manual cascade simulation from (0,0)
    print(f"   Manual cascade simulation from (0,0):")
    
    # Create a temporary board with the mines
    temp_board = np.zeros((4, 4), dtype=int)
    
    # Place mines
    for row, col in [(1, 1), (2, 2)]:
        temp_board[row, col] = 9
    
    # Fill in adjacent counts
    for row, col in [(1, 1), (2, 2)]:
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if (0 <= nr < 4 and 
                    0 <= nc < 4 and 
                    (nr, nc) not in [(1, 1), (2, 2)]):
                    temp_board[nr, nc] += 1
    
    print(f"   Board with adjacent counts:")
    print(temp_board)
    
    # Simulate clicking (0,0) and any cascading cells
    revealed = np.zeros_like(temp_board, dtype=bool)
    queue = [(0, 0)]
    
    while queue:
        r, c = queue.pop(0)
        if revealed[r, c]:
            continue
        
        revealed[r, c] = True
        
        # If this cell has no adjacent mines (value 0), cascade to neighbors
        if temp_board[r, c] == 0:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < 4 and 
                        0 <= nc < 4 and 
                        not revealed[nr, nc] and 
                        (nr, nc) not in [(1, 1), (2, 2)]):
                        queue.append((nr, nc))
    
    revealed_count = revealed.sum()
    total_cells = 16
    mine_count = 2
    safe_cells = total_cells - mine_count
    print(f"   Cascade from (0,0) reveals: {revealed_count} cells")
    print(f"   Safe cells: {safe_cells}")
    print(f"   Is instant win: {revealed_count >= safe_cells}")
    
    # Test case 3: Two mines that don't create instant win - should be learnable
    print("\nTest 3: Two mines not creating instant win")
    env = MinesweeperEnv(
        max_board_size=(4, 4),
        initial_board_size=(4, 4),
        max_mines=2,
        initial_mines=2,
        learnable_only=False
    )
    
    # Place mines at (0,0) and (3,3) - this should not create instant win
    env._place_mines_at_positions([(0, 0), (3, 3)])
    
    # Test if this is learnable
    is_learnable = env._is_learnable_configuration([(0, 0), (3, 3)])
    print(f"   _is_learnable_configuration: {is_learnable}")
    
    # Test first move from (0,1) - should not be instant win
    env.reset()
    obs, reward, terminated, truncated, info = env.step(1)  # action 1 = (0,1)
    print(f"   First move from (0,1) - won: {info.get('won', False)}")

def test_cascade_simulation():
    """Test the cascade simulation logic."""
    print(f"\n{'='*50}")
    print("Testing Cascade Simulation")
    print(f"{'='*50}")
    
    env = MinesweeperEnv(
        max_board_size=(4, 4),
        initial_board_size=(4, 4),
        max_mines=2,
        initial_mines=2,
        learnable_only=False
    )
    
    # Create a board where (0,0) should cascade to reveal most cells
    mine_positions = [(1, 1), (2, 2)]
    env._place_mines_at_positions(mine_positions)
    
    print(f"Mine positions: {mine_positions}")
    
    # Test cascade from (0,0)
    total_cells = env.current_board_height * env.current_board_width
    mine_count = len(mine_positions)
    safe_cells = total_cells - mine_count
    
    print(f"Total cells: {total_cells}")
    print(f"Mine count: {mine_count}")
    print(f"Safe cells: {safe_cells}")
    
    # Test the learnable configuration logic
    is_learnable = env._is_learnable_configuration(mine_positions)
    print(f"_is_learnable_configuration: {is_learnable}")
    
    # Test actual first move
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)  # action 0 = (0,0)
    print(f"Actual first move from (0,0) - won: {info.get('won', False)}")
    
    # Test manual cascade simulation from (0,0)
    print(f"\nManual cascade simulation from (0,0):")
    
    # Create a temporary board with the mines
    temp_board = np.zeros((4, 4), dtype=int)
    
    # Place mines
    for row, col in mine_positions:
        temp_board[row, col] = 9
    
    # Fill in adjacent counts
    for row, col in mine_positions:
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if (0 <= nr < 4 and 
                    0 <= nc < 4 and 
                    (nr, nc) not in mine_positions):
                    temp_board[nr, nc] += 1
    
    print(f"Board with adjacent counts:")
    print(temp_board)
    
    # Simulate clicking (0,0) and any cascading cells
    revealed = np.zeros_like(temp_board, dtype=bool)
    queue = [(0, 0)]
    
    while queue:
        r, c = queue.pop(0)
        if revealed[r, c]:
            continue
        
        revealed[r, c] = True
        
        # If this cell has no adjacent mines (value 0), cascade to neighbors
        if temp_board[r, c] == 0:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < 4 and 
                        0 <= nc < 4 and 
                        not revealed[nr, nc] and 
                        (nr, nc) not in mine_positions):
                        queue.append((nr, nc))
    
    revealed_count = revealed.sum()
    print(f"Cascade from (0,0) reveals: {revealed_count} cells")
    print(f"Is instant win: {revealed_count >= safe_cells}")
    print(f"Revealed array:")
    print(revealed)

if __name__ == "__main__":
    test_first_move_win_detection()
    test_cascade_simulation() 