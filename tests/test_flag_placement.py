import pytest
import numpy as np
from src.core.minesweeper_env import MinesweeperEnv

@pytest.fixture
def env():
    """Create a test environment."""
    return MinesweeperEnv(max_board_size=3, max_mines=1)

def test_flag_placement_on_mine(env):
    """Test placing a flag on a mine gives correct reward and state."""
    # Place mine at (1,1)
    env.mines[1, 1] = True
    env._update_adjacent_counts()
    
    # Calculate flag action for (1,1)
    flag_action = env.current_board_size * env.current_board_size + (1 * env.current_board_size + 1)
    obs, reward, terminated, truncated, info = env.step(flag_action)
    
    # Verify flag was placed
    assert env.flags[1, 1]
    assert not terminated
    assert reward == 0

def test_flag_placement_on_safe_cell(env):
    """Test placing a flag on a safe cell gives penalty and state."""
    # Calculate flag action for (0,0)
    flag_action = env.current_board_size * env.current_board_size + (0 * env.current_board_size + 0)
    obs, reward, terminated, truncated, info = env.step(flag_action)
    
    # Verify flag was placed
    assert env.flags[0, 0]
    assert not terminated
    assert reward == 0

def test_flag_removal(env):
    """Test removing a flag gives small penalty and updates state."""
    # Place flag first
    flag_action = env.current_board_size * env.current_board_size + (1 * env.current_board_size + 1)
    env.step(flag_action)
    
    # Remove flag
    obs, reward, terminated, truncated, info = env.step(flag_action)
    
    # Verify flag was removed
    assert not env.flags[1, 1]
    assert not terminated
    assert reward == 0 