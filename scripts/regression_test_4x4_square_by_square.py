#!/usr/bin/env python3
"""
Regression Test: 4×4 Square by Square Analysis

This script tests each individual square of a 4×4 board to determine which mine positions
allow 1-move wins vs. require strategic play.
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

def simulate_cascade_4x4(board, mine_pos):
    """Simulate a cascade from a non-mine cell and return number of revealed cells."""
    revealed = np.zeros((4, 4), dtype=bool)
    
    # Find a non-mine cell to start the cascade
    start_pos = None
    for r in range(4):
        for c in range(4):
            if (r, c) != mine_pos:
                start_pos = (r, c)
                break
        if start_pos:
            break
    
    if not start_pos:
        return 0
    
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

def test_4x4_square_by_square():
    """Test each square of a 4×4 board individually."""
    
    print("🧪 4×4 Square by Square Regression Test")
    print("=" * 50)
    print("Testing each mine position to see if it allows 1-move win")
    print()
    
    # Create a visual representation of the board
    print("Board Layout (row, col):")
    print("  0  1  2  3")
    for r in range(4):
        print(f"{r} [ ] [ ] [ ] [ ]")
    print()
    
    one_move_wins = []
    learnable_positions = []
    
    # Test each position
    for row in range(4):
        for col in range(4):
            mine_pos = (row, col)
            board = make_board_4x4(mine_pos)
            revealed_cells = simulate_cascade_4x4(board, mine_pos)
            total_cells = 16
            
            is_one_move_win = revealed_cells == (total_cells - 1)
            
            if is_one_move_win:
                one_move_wins.append(mine_pos)
                status = "✅ 1-MOVE WIN"
            else:
                learnable_positions.append(mine_pos)
                status = "❌ LEARNABLE"
            
            print(f"Position ({row}, {col}): {status} (revealed {revealed_cells}/15 cells)")
            
            # Show the board for this position
            print("  Board:")
            for r in range(4):
                row_str = "  "
                for c in range(4):
                    if (r, c) == mine_pos:
                        row_str += "M "
                    else:
                        row_str += f"{board[r, c]} "
                print(row_str)
            print()
    
    # Summary
    print("📊 Summary")
    print("=" * 50)
    print(f"Total positions: 16")
    print(f"1-move wins: {len(one_move_wins)} ({len(one_move_wins)/16*100:.1f}%)")
    print(f"Learnable positions: {len(learnable_positions)} ({len(learnable_positions)/16*100:.1f}%)")
    print()
    
    print("1-Move Win Positions:")
    for pos in one_move_wins:
        print(f"  ({pos[0]}, {pos[1]})")
    print()
    
    print("Learnable Positions:")
    for pos in learnable_positions:
        print(f"  ({pos[0]}, {pos[1]})")
    print()
    
    # Visual summary
    print("Visual Summary (M = 1-move win, L = learnable):")
    print("  0  1  2  3")
    for r in range(4):
        row_str = f"{r} "
        for c in range(4):
            if (r, c) in one_move_wins:
                row_str += "[M]"
            else:
                row_str += "[L]"
        print(row_str)

if __name__ == "__main__":
    test_4x4_square_by_square() 