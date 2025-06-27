#!/usr/bin/env python3
"""
Test script to check ALL possible mine positions in a 4x4 board.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from core.minesweeper_env import MinesweeperEnv

def test_all_mine_positions():
    """Test all possible mine positions in a 4x4 board."""
    print("🧪 Testing ALL Mine Positions in 4x4 Board")
    print("=" * 60)
    
    # Test every possible mine position
    all_positions = [(row, col) for row in range(4) for col in range(4)]
    
    learnable_positions = []
    non_learnable_positions = []
    
    for pos in all_positions:
        print(f"\nMine at {pos}:")
        
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
        print(f"  _is_learnable_configuration: {is_learnable}")
        
        # Test cascade simulation
        total_cells = 16
        mine_count = 1
        safe_cells = total_cells - mine_count
        
        # Simulate cascade from a non-mine position
        revealed = env._simulate_cascade(env.board, pos)
        print(f"  Cascade reveals: {revealed} cells")
        print(f"  Safe cells: {safe_cells}")
        print(f"  Is 1-move win: {revealed >= safe_cells}")
        
        # Test actual first move from a non-mine position
        env.reset()
        for action in range(16):
            col = action % 4
            row = action // 4
            if not env.mines[row, col]:
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"  First move from ({row},{col}) - won: {info.get('won', False)}")
                break
        
        # Categorize the position
        if is_learnable:
            learnable_positions.append(pos)
            print(f"  ✅ LEARNABLE")
        else:
            non_learnable_positions.append(pos)
            print(f"  ❌ NOT LEARNABLE (1-move win possible)")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total positions: {len(all_positions)}")
    print(f"Learnable positions: {len(learnable_positions)}")
    print(f"Non-learnable positions: {len(non_learnable_positions)}")
    
    print(f"\nLearnable positions: {learnable_positions}")
    print(f"Non-learnable positions: {non_learnable_positions}")
    
    # Visual representation
    print(f"\nVisual representation (L=Learnable, N=Non-learnable):")
    board = [[' ' for _ in range(4)] for _ in range(4)]
    for pos in learnable_positions:
        board[pos[0]][pos[1]] = 'L'
    for pos in non_learnable_positions:
        board[pos[0]][pos[1]] = 'N'
    
    for row in range(4):
        print(f"  {' '.join(board[row])}")

if __name__ == "__main__":
    test_all_mine_positions() 