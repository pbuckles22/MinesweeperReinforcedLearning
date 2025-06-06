import pytest
import numpy as np
from src.core.minesweeper_env import MinesweeperEnv
from src.core.constants import CELL_UNREVEALED

@pytest.fixture
def env():
    """Create a test environment with known board state."""
    env = MinesweeperEnv(
        initial_board_size=3,
        initial_mines=1,
        early_learning_mode=True
    )
    # Force a specific board state for testing
    env.board = np.array([
        [0, 0, 0],
        [0, 9, 0],  # Mine at (1,1) represented by 9
        [0, 0, 0]
    ])
    env.mines = np.zeros((3, 3), dtype=bool)
    env.mines[1, 1] = True
    env._update_adjacent_counts()
    return env

def test_safe_cell_reveal(env):
    """Test that revealing a safe cell works correctly."""
    # Find a safe cell
    safe_cell = None
    for i in range(env.current_board_height):
        for j in range(env.current_board_width):
            if not env.mines[i, j]:
                safe_cell = (i, j)
                break
        if safe_cell:
            break
    
    # Reveal the safe cell
    action = safe_cell[0] * env.current_board_width + safe_cell[1]
    state, reward, terminated, truncated, info = env.step(action)
    
    assert not terminated
    assert reward >= 0
    assert state[safe_cell] != CELL_UNREVEALED

def test_safe_cell_cascade(env):
    """Test that revealing a safe cell with no adjacent mines reveals surrounding cells."""
    # Create a board with a safe area
    env.mines.fill(False)
    env.mines[0, 0] = True  # Place one mine in corner
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
    
    # Reveal the safe cell
    action = safe_cell[0] * env.current_board_width + safe_cell[1]
    state, reward, terminated, truncated, info = env.step(action)
    
    # Check that surrounding cells were revealed
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            ni, nj = safe_cell[0] + di, safe_cell[1] + dj
            if (0 <= ni < env.current_board_height and 
                0 <= nj < env.current_board_width and 
                (di != 0 or dj != 0)):
                assert state[ni, nj] != CELL_UNREVEALED

def test_safe_cell_adjacent_mines(env):
    """Test that adjacent mine counts are correct after placing a mine."""
    env.mines[1, 1] = True
    env._update_adjacent_counts()
    assert env.board[0, 0] == 1
    assert env.board[0, 1] == 1
    assert env.board[0, 2] == 1
    assert env.board[1, 0] == 1
    assert env.board[1, 1] == 0
    assert env.board[1, 2] == 1
    assert env.board[2, 0] == 1
    assert env.board[2, 1] == 1
    assert env.board[2, 2] == 1 