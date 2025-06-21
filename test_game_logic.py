#!/usr/bin/env python3
"""
Simple test script to verify Minesweeper game logic.
This will help us identify any bugs in the core game mechanics.
"""

import numpy as np
from src.core.minesweeper_env import MinesweeperEnv

def test_simple_game():
    """Test a simple 3x3 game with 1 mine to verify basic logic."""
    print("=== Testing Simple 3x3 Game with 1 Mine ===")
    
    env = MinesweeperEnv(
        max_board_size=(3, 3),
        max_mines=1,
        initial_board_size=3,
        initial_mines=1,
        render_mode=None
    )
    
    state, info = env.reset(seed=42)  # Fixed seed for reproducibility
    
    print(f"Initial board size: {env.current_board_height}x{env.current_board_width}")
    print(f"Mines placed: {env.current_mines}")
    print(f"Initial state shape: {state.shape}")
    
    # Print the mine locations
    mine_locations = np.where(env.mines)
    print(f"Mine locations: {list(zip(mine_locations[0], mine_locations[1]))}")
    
    # Print the board values
    print("Board values (9=mine, 0-8=adjacent mine count):")
    print(env.board)
    
    # Try each cell and see what happens
    for action in range(9):
        row, col = action // 3, action % 3
        print(f"\n--- Testing action {action} (row={row}, col={col}) ---")
        
        # Reset for each test
        state, info = env.reset(seed=42)
        
        # Take the action
        next_state, reward, terminated, truncated, info = env.step(action)
        
        print(f"Action: {action} -> ({row}, {col})")
        print(f"Hit mine: {env.mines[row, col]}")
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")
        print(f"Won: {info['won']}")
        print(f"Revealed cells: {np.sum(env.revealed)}")
        
        if env.mines[row, col]:
            print("  -> Hit a mine!")
        else:
            print(f"  -> Safe cell with value: {env.board[row, col]}")
            print(f"  -> Revealed state: {env.revealed}")
            
            # If it's a safe cell, try to complete the game
            if not terminated:
                print("  -> Attempting to complete the game...")
                steps = 0
                while not terminated and steps < 10:
                    # Find next valid action
                    valid_actions = np.where(env.action_masks)[0]
                    if len(valid_actions) == 0:
                        print("  -> No valid actions left!")
                        break
                    
                    next_action = valid_actions[0]
                    next_row, next_col = next_action // 3, next_action % 3
                    print(f"  -> Next action: {next_action} ({next_row}, {next_col})")
                    
                    next_state, reward, terminated, truncated, info = env.step(next_action)
                    steps += 1
                    
                    print(f"  -> Step {steps}: reward={reward}, terminated={terminated}, won={info['won']}")
                    
                    if terminated:
                        if info['won']:
                            print("  -> GAME WON!")
                        else:
                            print("  -> Game lost (hit mine)")
                        break

def test_win_condition():
    """Test that win condition is correctly implemented."""
    print("\n=== Testing Win Condition ===")
    
    # Create a 2x2 board with 1 mine in a known position
    env = MinesweeperEnv(
        max_board_size=(2, 2),
        max_mines=1,
        initial_board_size=2,
        initial_mines=1,
        render_mode=None
    )
    
    # Force a specific mine placement by manipulating the board directly
    state, info = env.reset(seed=123)
    
    # Clear all mines and place one in position (0,0)
    env.mines.fill(False)
    env.mines[0, 0] = True
    env._update_adjacent_counts()
    env._update_enhanced_state()
    
    print("Board setup:")
    print(f"Mines: {env.mines}")
    print(f"Board values: {env.board}")
    
    # Now reveal the safe cells (1,0), (0,1), (1,1)
    safe_cells = [(1, 0), (0, 1), (1, 1)]
    
    for i, (row, col) in enumerate(safe_cells):
        action = row * 2 + col
        print(f"\nRevealing safe cell ({row}, {col}) - action {action}")
        
        next_state, reward, terminated, truncated, info = env.step(action)
        
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")
        print(f"Won: {info['won']}")
        print(f"Revealed: {env.revealed}")
        
        if terminated and info['won']:
            print("SUCCESS: Win condition correctly detected!")
            break
        elif terminated and not info['won']:
            print("ERROR: Game terminated but not won!")
            break

def test_cascade_logic():
    """Test that cascade logic works correctly."""
    print("\n=== Testing Cascade Logic ===")
    
    env = MinesweeperEnv(
        max_board_size=(3, 3),
        max_mines=1,
        initial_board_size=3,
        initial_mines=1,
        render_mode=None
    )
    
    state, info = env.reset(seed=456)
    
    # Place mine in corner and check cascade behavior
    env.mines.fill(False)
    env.mines[2, 2] = True  # Mine in bottom-right corner
    env._update_adjacent_counts()
    env._update_enhanced_state()
    
    print("Board with mine in corner (2,2):")
    print(f"Board values:\n{env.board}")
    
    # Reveal top-left corner (0,0) - should cascade
    action = 0  # (0,0)
    print(f"\nRevealing corner cell (0,0) - action {action}")
    
    next_state, reward, terminated, truncated, info = env.step(action)
    
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Won: {info['won']}")
    print(f"Revealed cells:\n{env.revealed}")
    print(f"Final state:\n{next_state[0]}")

if __name__ == "__main__":
    test_simple_game()
    test_win_condition()
    test_cascade_logic()
    print("\n=== Game Logic Tests Complete ===") 