import pytest
import numpy as np
from src.core.minesweeper_env import MinesweeperEnv
from src.core.constants import CELL_UNREVEALED, CELL_FLAGGED

@pytest.fixture
def env():
    return MinesweeperEnv(initial_board_size=3, initial_mines=1)

def test_reveal_already_revealed_cell(env):
    """Test that revealing an already revealed cell is invalid."""
    # Reveal a cell
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # Try to reveal the same cell again
    state, reward, terminated, truncated, info = env.step(action)
    
    assert reward < 0  # Should get negative reward for invalid action
    assert not terminated
    assert not truncated
    assert 'won' in info

def test_reveal_flagged_cell(env):
    """Test that revealing a flagged cell is invalid."""
    # Flag a cell
    action = env.current_board_width * env.current_board_height  # First flag action
    state, reward, terminated, truncated, info = env.step(action)
    
    # Try to reveal the flagged cell
    reveal_action = 0  # First reveal action
    state, reward, terminated, truncated, info = env.step(reveal_action)
    
    assert reward < 0  # Should get negative reward for invalid action
    assert not terminated
    assert not truncated
    assert 'won' in info

def test_flag_revealed_cell(env):
    """Test that flagging a revealed cell is invalid."""
    # Reveal a cell
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # Try to flag the revealed cell
    flag_action = env.current_board_width * env.current_board_height  # First flag action
    state, reward, terminated, truncated, info = env.step(flag_action)
    
    assert reward < 0  # Should get negative reward for invalid action
    assert not terminated
    assert not truncated
    assert 'won' in info

def test_flag_already_flagged_cell(env):
    """Test that flagging an already flagged cell is invalid."""
    # Flag a cell
    flag_action = env.current_board_width * env.current_board_height  # First flag action
    state, reward, terminated, truncated, info = env.step(flag_action)
    
    # Try to flag the same cell again
    state, reward, terminated, truncated, info = env.step(flag_action)
    
    assert reward < 0  # Should get negative reward for invalid action
    assert not terminated
    assert not truncated
    assert 'won' in info

def test_reveal_after_game_over(env):
    """Test that revealing after game over is invalid."""
    # Place mine at (0,0)
    env.mines[0, 0] = True
    env._update_adjacent_counts()
    
    # Hit the mine
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # Try to reveal another cell
    action = 1
    state, reward, terminated, truncated, info = env.step(action)
    
    assert reward < 0  # Should get negative reward for invalid action
    assert terminated  # Game should still be terminated
    assert not truncated
    assert 'won' in info 