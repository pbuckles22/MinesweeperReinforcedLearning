import pytest
import numpy as np
from src.core.minesweeper_env import MinesweeperEnv
from src.core.constants import (
    CELL_UNREVEALED,
    CELL_MINE,
    CELL_MINE_HIT,
    REWARD_FIRST_CASCADE_SAFE, REWARD_FIRST_CASCADE_HIT_MINE,
    REWARD_SAFE_REVEAL,
    REWARD_WIN,
    REWARD_HIT_MINE
)

def test_initialization(env):
    """Test that the environment initializes correctly."""
    # ... existing code ...

def test_reset(env):
    """Test that the environment resets correctly."""
    # ... existing code ...

def test_step(env):
    """Test that the environment steps correctly."""
    # ... existing code ...

def test_safe_cell_reveal(env):
    """Test that revealing a safe cell works correctly using only public API and explicit board setup."""
    env.reset()
    # Explicitly set up a board with a single mine at (0,0)
    env.mines.fill(False)
    env.mines[0, 0] = True
    env._update_adjacent_counts()
    env.mines_placed = True
    env.is_first_cascade = False
    env.first_cascade_done = True

    # Pick a safe cell to reveal (e.g., (1,1))
    action = 1 * env.current_board_width + 1
    state, reward, terminated, truncated, info = env.step(action)

    # The cell should be revealed (state has shape (2, height, width), channel 0 is game state)
    assert state[0, 1, 1] != CELL_UNREVEALED, "Safe cell should be revealed"
    # The game should not be terminated after revealing one safe cell
    assert not terminated, "Game should not be terminated after revealing one safe cell"
    # The reward should be non-negative (safe reveal or neutral)
    assert reward >= 0, "Reward should be non-negative for safe cell reveal"
    assert not truncated, "Game should not be truncated after safe cell reveal"

def test_safe_cell_cascade(env):
    """Test that revealing a safe cell with no adjacent mines reveals surrounding cells."""
    # Clear all mines first
    env.mines.fill(False)
    env.is_first_cascade = False  # Disable First cascade mine placement
    env.first_cascade_done = True  # Prevent mine placement in step()

    # Place mine at (0,0)
    env.mines[0, 0] = True
    env._update_adjacent_counts()
    env.mines_placed = True  # Prevent automatic mine placement in step()
    
    print("\nBoard state after mine placement:")
    print("Mines:")
    print(env.mines)
    print("Adjacent counts:")
    print(env.board)
    print("Mines placed flag:", env.mines_placed)
    
    # Pick (3,3) as the cascade cell (should be 0)
    assert not env.mines[3, 3], "Cell (3,3) should not be a mine"
    assert env.board[3, 3] == 0, f"Cell (3,3) should have no adjacent mines, got {env.board[3,3]}"
    
    # Reveal the safe cell at (3,3)
    action = 3 * env.current_board_width + 3
    print(f"\nRevealing cell (3,3) with action {action}")
    state, reward, terminated, truncated, info = env.step(action)
    
    print("\nState after revealing (3,3):")
    print(state)
    print("\nMine locations:")
    print(env.mines)
    print("\nRevealed cells:")
    print(env.revealed)
    
    # Check that all non-mine cells are revealed
    unrevealed_cells = []
    for i in range(env.current_board_height):
        for j in range(env.current_board_width):
            if not env.mines[i, j] and state[0, i, j] == CELL_UNREVEALED:
                unrevealed_cells.append((i, j))
    
    assert not unrevealed_cells, f"Non-mine cells still unrevealed: {unrevealed_cells}"
    # Since all non-mine cells are revealed, game should be terminated
    assert terminated, "Game should be terminated when all non-mine cells are revealed"
    assert not truncated, "Game should not be truncated"
    assert reward == REWARD_WIN, "Should get win reward"

def test_safe_cell_adjacent_mines(env):
    """Test that revealing a safe cell shows the correct number of adjacent mines."""
    env.mines.fill(False)
    env.mines[2, 2] = True
    env._update_adjacent_counts()
    print("\nBoard state for adjacent mines test:")
    print("Mines:")
    print(env.mines)
    print("Adjacent counts:")
    print(env.board)
    action = 1 * env.current_board_width + 1
    state, reward, terminated, truncated, info = env.step(action)
    print(f"\nState after revealing (1,1):\n{state}")
    print(f"Value at (1,1): {state[0,1,1]}")
    assert state[0, 1, 1] == env.board[1, 1], f"Cell (1,1) should show {env.board[1,1]} adjacent mines, got {state[0,1,1]}"

def test_win_condition(env):
    """Test that revealing all safe cells wins the game."""
    # Clear all mines first
    env.mines.fill(False)
    
    # Place mine at (1,1)
    env.mines[1, 1] = True
    env._update_adjacent_counts()
    env.mines_placed = True
    env.is_first_cascade = False
    env.first_cascade_done = True
    
    # Debug print initial board state
    print("\nInitial board state:")
    print("Mines:")
    print(env.mines)
    print("Adjacent counts:")
    print(env.board)
    
    # Reveal all safe cells
    safe_cells_revealed = 0
    for i in range(env.current_board_height):
        for j in range(env.current_board_width):
            if not env.mines[i, j] and env.revealed[i, j] == False:
                action = i * env.current_board_width + j
                print(f"\nRevealing cell ({i},{j})")
                state, reward, terminated, truncated, info = env.step(action)
                print(f"State after reveal:")
                print(state)
                print(f"Terminated: {terminated}, Truncated: {truncated}")
                print(f"Reward: {reward}")
                print(f"Info: {info}")
                safe_cells_revealed += 1
                if terminated:
                    break
        if terminated:
            break
    
    print(f"\nTotal safe cells revealed: {safe_cells_revealed}")
    print("Final state:")
    print(state)
    
    # Check that game is won
    assert terminated, "Game should be terminated"
    assert not truncated, "Game should not be truncated"
    assert reward == REWARD_WIN, "Should get win reward"
    assert info.get('won', False), "Game should be marked as won"
    
    # Verify all safe cells are revealed
    for i in range(env.current_board_height):
        for j in range(env.current_board_width):
            if not env.mines[i, j]:
                assert state[0, i, j] != CELL_UNREVEALED, f"Safe cell at ({i},{j}) should be revealed" 