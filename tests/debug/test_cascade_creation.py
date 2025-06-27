#!/usr/bin/env python3
"""
Test to demonstrate cascade creation when moving mines.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from core.minesweeper_env import MinesweeperEnv

def test_cascade_creation():
    """Test if moving mines creates new cascades."""
    print("🔍 Testing Cascade Creation from Mine Movement")
    print("=" * 50)
    
    board_size = (4, 4)
    mines = 1
    
    print(f"\nTesting 4x4 board with 1 mine:")
    
    # Create environment
    env = MinesweeperEnv(
        max_board_size=board_size,
        initial_board_size=board_size,
        max_mines=mines,
        initial_mines=mines,
        learnable_only=True,
        max_learnable_attempts=1000
    )
    
    # Reset to get initial board
    obs, info = env.reset()
    
    if not info.get('learnable', False):
        print("   Board not learnable, skipping test")
        return
    
    print(f"   Initial board learnable: {info.get('learnable', False)}")
    
    # Check if first move is safe
    action = 0  # Corner (0,0)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated and reward == -20:
        print(f"   ❌ First move hit mine (environment should have prevented this)")
        return
    elif info.get('won', False):
        print(f"   ❌ First move caused instant win (environment should have prevented this)")
        return
    else:
        print(f"   ✅ First move was safe")
    
    # Now let's manually test what happens if we move a mine
    print(f"\n🔍 Testing manual mine movement:")
    
    # Get current mine position
    mine_pos = None
    for y in range(env.current_board_height):
        for x in range(env.current_board_width):
            if env.mines[y, x]:
                mine_pos = (y, x)
                break
        if mine_pos:
            break
    
    print(f"   Current mine position: {mine_pos}")
    
    # Test moving mine to different positions
    test_positions = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2), (3, 3)]
    
    for new_pos in test_positions:
        if new_pos == mine_pos:
            continue
            
        # Create a copy of the board
        temp_mines = env.mines.copy()
        temp_board = env.board.copy()
        
        # Move the mine
        temp_mines[mine_pos] = False
        temp_mines[new_pos] = True
        
        # Update adjacent counts
        temp_board.fill(0)
        for i in range(env.current_board_height):
            for j in range(env.current_board_width):
                if temp_mines[i, j]:
                    temp_board[i, j] = 9
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if (0 <= ni < env.current_board_height and 
                                0 <= nj < env.current_board_width):
                                temp_board[ni, nj] += 1
        
        # Test if this creates a cascade
        mine_positions = [(y, x) for y in range(env.current_board_height) 
                         for x in range(env.current_board_width) if temp_mines[y, x]]
        
        is_learnable = env._is_learnable_configuration(mine_positions)
        
        print(f"   Moving mine to {new_pos}: {'❌ Creates cascade' if not is_learnable else '✅ Still learnable'}")

def test_original_vs_adjusted():
    """Compare original board vs adjusted board."""
    print(f"\n🔍 Testing Original vs Adjusted Boards")
    print("=" * 50)
    
    board_size = (4, 4)
    mines = 1
    
    print(f"\nTesting multiple boards:")
    
    for test_num in range(5):
        print(f"\n   Test {test_num + 1}:")
        
        # Create environment
        env = MinesweeperEnv(
            max_board_size=board_size,
            initial_board_size=board_size,
            max_mines=mines,
            initial_mines=mines,
            learnable_only=True,
            max_learnable_attempts=1000
        )
        
        # Reset to get initial board
        obs, info = env.reset()
        
        if not info.get('learnable', False):
            print(f"      Board not learnable, skipping")
            continue
        
        # Check if first move is safe
        action = 0  # Corner (0,0)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated and reward == -20:
            print(f"      ❌ First move hit mine")
        elif info.get('won', False):
            print(f"      ❌ First move caused instant win")
        else:
            print(f"      ✅ First move was safe")
            
            # Check if this created a cascade
            revealed_count = np.sum(env.revealed)
            total_cells = env.current_board_height * env.current_board_width
            mine_count = env.current_mines
            
            if revealed_count >= (total_cells - mine_count - 1):
                print(f"      ⚠️  First move revealed most cells (cascade-like behavior)")
            else:
                print(f"      ✅ First move revealed {revealed_count} cells (normal)")

def main():
    """Run the cascade creation tests."""
    print("🧪 Cascade Creation Test")
    print("=" * 60)
    
    test_cascade_creation()
    test_original_vs_adjusted()
    
    print(f"\n{'='*60}")
    print("CONCLUSION")
    print(f"{'='*60}")
    print("The current environment moves mines to ensure safe first moves.")
    print("This could potentially create new cascades that make boards non-learnable.")
    print("Consider using pure filtering instead of mine adjustment.")

if __name__ == "__main__":
    main() 