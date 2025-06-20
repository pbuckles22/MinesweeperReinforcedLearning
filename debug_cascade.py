#!/usr/bin/env python3
"""
Debug script to understand the cascade logic
"""

from src.core.minesweeper_env import MinesweeperEnv
from src.core.constants import REWARD_FIRST_CASCADE_SAFE, REWARD_WIN

def debug_cascade_logic():
    """Debug the cascade logic to understand why win is getting wrong reward."""
    print("=== Debugging 4x4 Cascade Logic ===\n")
    
    env = MinesweeperEnv(initial_board_size=(4, 4), initial_mines=2)
    env.reset()
    
    # Place mines at corners
    env.mines.fill(False)
    env.mines[0, 0] = True  # Mine at (0,0)
    env.mines[3, 3] = True  # Mine at (3,3)
    env._update_adjacent_counts()
    env.mines_placed = True
    
    print("Initial state:")
    print(f"  is_first_cascade: {env.is_first_cascade}")
    print(f"  first_cascade_done: {env.first_cascade_done}")
    print(f"  board values:")
    for i in range(4):
        for j in range(4):
            print(f"    ({i},{j}): {env.board[i,j]}")
    
    # Try different cells to find one that triggers a cascade
    test_cells = [(1,1), (1,2), (2,1), (2,2)]
    for row, col in test_cells:
        action = row * 4 + col
        print(f"\nTesting cell ({row},{col}) - action {action}:")
        print(f"  board value: {env.board[row,col]}")
        if env.board[row,col] == 0:
            print(f"  This should trigger a cascade!")
            break
    
    # First move at the cell that should trigger cascade (no special logic)
    if env.board[1,2] == 0:  # Let's try (1,2)
        action = 6  # (1,2)
        print(f"\nFirst move at (1,2):")
        state, reward, terminated, truncated, info = env.step(action)
        print(f"  reward: {reward}")
        print(f"  terminated: {terminated}")
        print(f"  won: {info.get('won', False)}")
        print(f"  is_first_cascade: {env.is_first_cascade}")
        print(f"  first_cascade_done: {env.first_cascade_done}")
        print(f"  revealed cells:")
        for i in range(4):
            for j in range(4):
                if env.revealed[i,j]:
                    print(f"    ({i},{j}) is revealed")

if __name__ == "__main__":
    debug_cascade_logic() 