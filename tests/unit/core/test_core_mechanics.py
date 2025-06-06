import pytest
import numpy as np
from src.core.minesweeper_env import MinesweeperEnv
from src.core.constants import (
    CELL_UNREVEALED,
    CELL_MINE,
    CELL_FLAGGED,
    CELL_MINE_HIT,
    REWARD_FIRST_MOVE_SAFE,
    REWARD_FIRST_MOVE_HIT_MINE,
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
    """Test that revealing a safe cell works correctly."""
    # Place mine at (1,1)
    env.mines[1, 1] = True
    env._update_adjacent_counts()
    
    # Reveal safe cell at (0,0)
    action = 0 * env.current_board_width + 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # Check that cell was revealed
    assert state[0, 0] != CELL_UNREVEALED
    assert not terminated
    assert not truncated
    assert reward == REWARD_SAFE_REVEAL

def test_safe_cell_cascade(env):
    """Test that revealing a safe cell with no adjacent mines reveals surrounding cells."""
    # Create a board with a safe area and one mine
    env.mines[0, 0] = True
    env._update_adjacent_counts()
    
    # Find a safe cell with no adjacent mines
    safe_cell = None
    for i in range(env.current_board_height):
        for j in range(env.current_board_width):
            if not env.mines[i, j] and env.board[i, j] == 0:
                safe_cell = (i, j)
                break
        if safe_cell:
            break
    
    assert safe_cell is not None, "No safe cell with zero adjacent mines found"
    
    # Reveal the safe cell
    action = safe_cell[0] * env.current_board_width + safe_cell[1]
    state, reward, terminated, truncated, info = env.step(action)
    
    # Check that surrounding cells were revealed
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            ni, nj = safe_cell[0] + di, safe_cell[1] + dj
            if 0 <= ni < env.current_board_height and 0 <= nj < env.current_board_width:
                assert state[ni, nj] != CELL_UNREVEALED

def test_safe_cell_adjacent_mines(env):
    """Test that revealing a safe cell shows the correct number of adjacent mines."""
    # Place mine at (1,1)
    env.mines[1, 1] = True
    env._update_adjacent_counts()
    
    # Reveal safe cell at (0,0)
    action = 0 * env.current_board_width + 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # Check that cell shows correct number of adjacent mines
    assert state[0, 0] == 1  # One adjacent mine
    assert not terminated
    assert not truncated
    assert reward == REWARD_SAFE_REVEAL

def test_win_condition(env):
    """Test that revealing all safe cells wins the game."""
    # Place mine at (1,1)
    env.mines[1, 1] = True
    env._update_adjacent_counts()
    
    # Flag the mine
    flag_action = (1 * env.current_board_width * env.current_board_height) + (1 * env.current_board_width + 1)
    state, reward, terminated, truncated, info = env.step(flag_action)
    
    # Reveal all safe cells
    for i in range(env.current_board_height):
        for j in range(env.current_board_width):
            if not env.mines[i, j]:
                action = i * env.current_board_width + j
                state, reward, terminated, truncated, info = env.step(action)
    
    # Check that game is won
    assert terminated
    assert not truncated
    assert reward == REWARD_WIN
    assert info.get('won', False) 