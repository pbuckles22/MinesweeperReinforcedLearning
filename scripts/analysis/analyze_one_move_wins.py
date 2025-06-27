#!/usr/bin/env python3
"""
Analyze One-Move Wins

Enumerate all possible single-mine placements on a board, simulate a cascade from a non-mine cell,
and count how many placements are 1-move wins (full cascade) vs. learnable (require 2+ moves).
"""

import numpy as np
from itertools import product

def simulate_cascade(board, mine_pos):
    """
    Simulate a cascade from the first non-mine cell and return the number of revealed cells.
    board: 2D numpy array (0 = empty, 9 = mine, >0 = adjacent mine count)
    mine_pos: (row, col) of the mine
    """
    h, w = board.shape
    revealed = np.zeros_like(board, dtype=bool)
    # Find a non-mine cell to start
    for r in range(h):
        for c in range(w):
            if (r, c) != mine_pos:
                start = (r, c)
                break
        else:
            continue
        break
    # BFS cascade
    queue = [start]
    while queue:
        r, c = queue.pop(0)
        if revealed[r, c]:
            continue
        revealed[r, c] = True
        if board[r, c] == 0:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not revealed[nr, nc] and (nr, nc) != mine_pos:
                        queue.append((nr, nc))
    # Count revealed
    return revealed.sum()

def make_board(h, w, mine_pos):
    board = np.zeros((h, w), dtype=int)
    r, c = mine_pos
    board[r, c] = 9
    # Fill in adjacent counts
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) != mine_pos:
                board[nr, nc] += 1
    return board

def analyze_one_move_wins(h, w):
    total = h * w
    one_move_win = 0
    learnable = 0
    for r in range(h):
        for c in range(w):
            mine_pos = (r, c)
            board = make_board(h, w, mine_pos)
            revealed = simulate_cascade(board, mine_pos)
            if revealed == total - 1:
                one_move_win += 1
            else:
                learnable += 1
    print(f"Board: {h}x{w}")
    print(f"  Total positions: {total}")
    print(f"  1-move wins: {one_move_win} ({one_move_win/total*100:.1f}%)")
    print(f"  Learnable: {learnable} ({learnable/total*100:.1f}%)")
    print()

def main():
    for size in [(4, 4), (5, 5), (6, 6), (8, 8)]:
        analyze_one_move_wins(*size)

if __name__ == "__main__":
    main() 