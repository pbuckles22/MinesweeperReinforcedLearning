#!/usr/bin/env python3
"""
Debug Cascade Logic

This script debugs the cascade simulation to understand why certain positions
are not being detected as 1-move wins.
"""

import numpy as np

def make_board_4x4(mine_pos):
    """Create a 4×4 board with a mine at the specified position."""
    board = np.zeros((4, 4), dtype=int)
    row, col = mine_pos
    board[row, col] = 9  # Place mine
    
    # Fill in adjacent counts
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            nr, nc = row + dr, col + dc
            if (0 <= nr < 4 and 0 <= nc < 4 and (nr, nc) != mine_pos):
                board[nr, nc] += 1
    
    return board

def simulate_cascade_from_start(board, mine_pos, start_pos):
    """Simulate a cascade from a specific start position."""
    revealed = np.zeros((4, 4), dtype=bool)
    
    # BFS cascade simulation
    queue = [start_pos]
    while queue:
        r, c = queue.pop(0)
        if revealed[r, c]:
            continue
        
        revealed[r, c] = True
        
        # If this cell has no adjacent mines (value 0), cascade to neighbors
        if board[r, c] == 0:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < 4 and 
                        0 <= nc < 4 and 
                        not revealed[nr, nc] and 
                        (nr, nc) != mine_pos):
                        queue.append((nr, nc))
    
    return revealed.sum()

def find_best_cascade(board, mine_pos):
    """Try cascading from every non-mine cell and return the maximum."""
    max_revealed = 0
    best_start = None
    
    for r in range(4):
        for c in range(4):
            if (r, c) != mine_pos:
                revealed = simulate_cascade_from_start(board, mine_pos, (r, c))
                if revealed > max_revealed:
                    max_revealed = revealed
                    best_start = (r, c)
    
    return max_revealed, best_start

def debug_position(mine_pos):
    """Debug a specific mine position."""
    print(f"\n🔍 Debugging mine position {mine_pos}")
    print("=" * 40)
    
    board = make_board_4x4(mine_pos)
    
    print("Board with mine:")
    for r in range(4):
        row_str = "  "
        for c in range(4):
            if (r, c) == mine_pos:
                row_str += "M "
            else:
                row_str += f"{board[r, c]} "
        print(row_str)
    
    print(f"\nTrying cascade from each non-mine cell:")
    for r in range(4):
        for c in range(4):
            if (r, c) != mine_pos:
                revealed = simulate_cascade_from_start(board, mine_pos, (r, c))
                print(f"  Start at ({r}, {c}): reveals {revealed} cells")
    
    max_revealed, best_start = find_best_cascade(board, mine_pos)
    print(f"\nBest cascade: {max_revealed} cells from start position {best_start}")
    
    is_one_move_win = max_revealed == 15  # 16 total - 1 mine
    print(f"1-move win: {'✅ YES' if is_one_move_win else '❌ NO'}")
    
    return is_one_move_win

def main():
    print("🐛 Debugging Cascade Logic")
    print("=" * 50)
    
    # Test the problematic position (0,0)
    debug_position((0, 0))
    
    # Test a known 1-move win position (0,3)
    debug_position((0, 3))
    
    # Test all positions
    print(f"\n📊 Testing All Positions")
    print("=" * 50)
    
    one_move_wins = []
    for r in range(4):
        for c in range(4):
            mine_pos = (r, c)
            is_win = debug_position(mine_pos)
            if is_win:
                one_move_wins.append(mine_pos)
    
    print(f"\n📋 Final Results")
    print("=" * 50)
    print(f"1-move win positions: {one_move_wins}")
    print(f"Total 1-move wins: {len(one_move_wins)}")

if __name__ == "__main__":
    main() 