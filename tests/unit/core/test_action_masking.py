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
    REWARD_HIT_MINE,
    REWARD_FLAG_PLACED,
    REWARD_FLAG_REMOVED,
    REWARD_INVALID_ACTION
)

@pytest.fixture
def env():
    return MinesweeperEnv(initial_board_size=3, initial_mines=1)

def test_reveal_already_revealed_cell():
    """Test that revealing an already revealed cell is invalid."""
    env = MinesweeperEnv(initial_board_size=5, initial_mines=1)
    env.reset()
    # Place mine far from (0,0) and (0,1)
    env.mines[:, :] = False
    env.mines[4, 4] = True
    env._update_adjacent_counts()
    # Reveal (0,0) using step
    action = 0  # (0,0)
    state, reward, terminated, truncated, info = env.step(action)
    # Reveal (0,1) to ensure there are other valid actions
    action_2 = 1  # (0,1)
    state, reward, terminated, truncated, info = env.step(action_2)
    # Now try to reveal (0,0) again (already revealed)
    state, reward, terminated, truncated, info = env.step(0)
    assert reward < 0  # Invalid action should give negative reward
    assert 'won' in info  # Info should contain expected keys

def test_reveal_flagged_cell(env):
    """Test that revealing a flagged cell is invalid."""
    env.reset()
    # Set up board with one mine in a corner to avoid immediate win
    env.mines[:, :] = False
    env.mines[2, 2] = True  # Place mine in bottom-right corner
    env._update_adjacent_counts()
    
    # Flag a cell
    action = env.current_board_width * env.current_board_height
    state, reward, terminated, truncated, info = env.step(action)
    # Try to reveal the flagged cell
    reveal_action = 0
    state, reward, terminated, truncated, info = env.step(reveal_action)
    assert reward < 0
    assert not terminated
    assert not truncated
    assert 'won' in info

def test_flag_revealed_cell():
    """Test that flagging a revealed cell is invalid."""
    env = MinesweeperEnv(initial_board_size=5, initial_mines=1)
    env.reset()
    # Place mine far from (0,0) and (0,1)
    env.mines[:, :] = False
    env.mines[4, 4] = True
    env._update_adjacent_counts()
    # Reveal (0,0) using step
    action = 0  # (0,0)
    state, reward, terminated, truncated, info = env.step(action)
    # Reveal (0,1) to ensure there are other valid actions
    action_2 = 1  # (0,1)
    state, reward, terminated, truncated, info = env.step(action_2)
    # Now try to flag (0,0) (already revealed)
    flag_action = env.current_board_width * env.current_board_height  # flag (0,0)
    state, reward, terminated, truncated, info = env.step(flag_action)
    assert reward < 0  # Invalid action should give negative reward
    assert 'won' in info  # Info should contain expected keys

def test_flag_already_flagged_cell(env):
    """Test that flagging an already flagged cell is invalid."""
    env.reset()
    # Set up board with one mine in a corner to avoid immediate win
    env.mines[:, :] = False
    env.mines[2, 2] = True  # Place mine in bottom-right corner
    env._update_adjacent_counts()
    
    # Flag a cell
    flag_action = env.current_board_width * env.current_board_height
    state, reward, terminated, truncated, info = env.step(flag_action)
    # Try to flag the same cell again
    state, reward, terminated, truncated, info = env.step(flag_action)
    assert reward < 0
    assert not terminated
    assert not truncated
    assert 'won' in info

def test_reveal_after_game_over(env):
    """Test that revealing after game over is invalid."""
    env.reset()
    # Set up a mine at (0,0) and make sure it's not the first move
    env.mines[:, :] = False
    env.mines[0, 0] = True
    env._update_adjacent_counts()
    env.is_first_move = False
    
    # Hit the mine (this should terminate the game)
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    assert terminated
    # Try to reveal another cell
    action = 1
    state, reward, terminated, truncated, info = env.step(action)
    assert reward < 0
    assert terminated
    assert not truncated
    assert 'won' in info

def test_action_masking_revealed_cells(env):
    """Test that revealed cells are masked."""
    env.reset()
    # Set up board with one mine in a corner to avoid immediate win
    env.mines[:, :] = False
    env.mines[2, 2] = True  # Place mine in bottom-right corner
    env._update_adjacent_counts()
    
    # Reveal a cell (should be safe)
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    # Check that the revealed cell is masked
    assert not env.action_masks[action]
    assert not env.action_masks[action + env.current_board_width * env.current_board_height]

def test_action_masking_flagged_cells(env):
    """Test that flagged cells are masked for reveal but not for unflag."""
    env.reset()
    # Set up board with one mine in a corner to avoid immediate win
    env.mines[:, :] = False
    env.mines[2, 2] = True  # Place mine in bottom-right corner
    env._update_adjacent_counts()
    
    # Flag a cell
    action = env.current_board_width * env.current_board_height
    state, reward, terminated, truncated, info = env.step(action)
    # Check that the flagged cell is masked for reveal but not for flag (unflag)
    assert env.action_masks[action]
    assert not env.action_masks[action - env.current_board_width * env.current_board_height]

def test_action_masking_game_over(env):
    """Test that all actions are masked when game is over."""
    env.reset()
    # Set up a mine at (0,0) and make sure it's not the first move
    env.mines[:, :] = False
    env.mines[0, 0] = True
    env._update_adjacent_counts()
    env.is_first_move = False
    
    # Hit mine (this should terminate the game)
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    assert terminated
    # Check that all actions are masked
    assert all(not mask for mask in env.action_masks) 