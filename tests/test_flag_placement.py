import pytest
import numpy as np
from src.core.minesweeper_env import MinesweeperEnv
from src.core.constants import CELL_FLAGGED, CELL_UNREVEALED

@pytest.fixture
def env():
    """Create a test environment."""
    return MinesweeperEnv(initial_board_size=3, initial_mines=1)

def test_flag_placement(env):
    """Test that placing a flag works correctly."""
    # Place flag at (1,1)
    flag_action = 1 * env.current_board_width + 1
    state, reward, terminated, truncated, info = env.step(flag_action)
    
    # Check that flag was placed
    assert state[1, 1] == CELL_FLAGGED
    assert not terminated
    assert not truncated
    assert reward == 0  # No reward for flag placement

def test_flag_removal(env):
    """Test that removing a flag works correctly."""
    # Place flag at (1,1)
    flag_action = 1 * env.current_board_width + 1
    state, reward, terminated, truncated, info = env.step(flag_action)
    
    # Remove flag at same position
    state, reward, terminated, truncated, info = env.step(flag_action)
    
    # Check that flag was removed
    assert state[1, 1] == CELL_UNREVEALED
    assert not terminated
    assert not truncated
    assert reward == 0  # No reward for flag removal

def test_flag_on_revealed_cell(env):
    """Test that flag cannot be placed on revealed cell."""
    # Reveal cell at (1,1)
    reveal_action = 1 * env.current_board_width + 1
    state, reward, terminated, truncated, info = env.step(reveal_action)
    
    # Try to place flag on revealed cell
    flag_action = 1 * env.current_board_width + 1
    state, reward, terminated, truncated, info = env.step(flag_action)
    
    # Check that flag was not placed
    assert state[1, 1] != CELL_FLAGGED
    assert not terminated
    assert not truncated
    assert reward < 0  # Negative reward for invalid action

def test_flag_count(env):
    """Test that flag count is tracked correctly."""
    # Place flag at (1,1)
    flag_action = 1 * env.current_board_width + 1
    state, reward, terminated, truncated, info = env.step(flag_action)
    
    # Check flag count
    assert info['flags_remaining'] == env.initial_mines - 1
    
    # Place flag at (0,0)
    flag_action = 0 * env.current_board_width + 0
    state, reward, terminated, truncated, info = env.step(flag_action)
    
    # Check flag count
    assert info['flags_remaining'] == env.initial_mines - 2
    
    # Remove flag at (1,1)
    flag_action = 1 * env.current_board_width + 1
    state, reward, terminated, truncated, info = env.step(flag_action)
    
    # Check flag count
    assert info['flags_remaining'] == env.initial_mines - 1

def test_flag_on_mine(env):
    """Test that flag can be placed on mine."""
    # Place mine at (1,1)
    env.mines[1, 1] = True
    env._update_adjacent_counts()
    
    # Place flag at mine location
    flag_action = 1 * env.current_board_width + 1
    state, reward, terminated, truncated, info = env.step(flag_action)
    
    # Check that flag was placed
    assert state[1, 1] == CELL_FLAGGED
    assert not terminated
    assert not truncated
    assert reward == 0  # No reward for flag placement 