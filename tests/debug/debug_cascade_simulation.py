#!/usr/bin/env python3
"""
Debug script to examine the cascade simulation and understand why it's not working correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from core.minesweeper_env import MinesweeperEnv

def debug_cascade_simulation():
    """Debug the cascade simulation logic."""
    print("🔍 Debugging Cascade Simulation")
    print("=" * 50)
    
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
    print(f"\nBoard state:")
    print(env.board)
    
    print(f"\nMines array:")
    print(env.mines)
    
    # Test cascade from (0,0) manually
    print(f"\nManual cascade simulation from (0,0):")
    
    # Create a copy of the board for simulation
    temp_board = env.board.copy()
    revealed = np.zeros_like(temp_board, dtype=bool)
    queue = [(0, 0)]
    
    step = 0
    while queue:
        r, c = queue.pop(0)
        if revealed[r, c]:
            continue
        
        revealed[r, c] = True
        step += 1
        print(f"Step {step}: Revealing ({r}, {c}) = {temp_board[r, c]}")
        
        # If this cell has no adjacent mines (value 0), cascade to neighbors
        if temp_board[r, c] == 0:
            print(f"  Cell ({r}, {c}) has value 0, cascading to neighbors...")
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
                        print(f"    Adding ({nr}, {nc}) to queue")
    
    print(f"\nFinal revealed cells: {revealed.sum()}")
    print(f"Revealed array:")
    print(revealed)
    
    # Test the multi-mine cascade simulation
    print(f"\nMulti-mine cascade simulation:")
    max_revealed = env._simulate_cascade_multi_mine(env.board, mine_positions)
    print(f"Maximum cascade reveals: {max_revealed} cells")

if __name__ == "__main__":
    debug_cascade_simulation() 