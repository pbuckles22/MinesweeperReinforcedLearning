import pytest
import numpy as np
from src.core.minesweeper_env import MinesweeperEnv
from src.core.constants import (
    CELL_FLAGGED, 
    CELL_UNREVEALED,
    CELL_MINE,
    CELL_MINE_HIT,
    REWARD_FLAG_MINE,
    REWARD_FLAG_SAFE,
    REWARD_UNFLAG,
    REWARD_FIRST_MOVE_SAFE,
    REWARD_FIRST_MOVE_HIT_MINE,
    REWARD_SAFE_REVEAL,
    REWARD_WIN,
    REWARD_HIT_MINE,
    REWARD_INVALID_ACTION
)

@pytest.fixture
def env():
    """Create a test environment."""
    return MinesweeperEnv(initial_board_size=3, initial_mines=1)

def test_flag_placement(env):
    """Test that placing a flag works correctly."""
    # Place flag at (1,1)
    flag_action = (1 * env.current_board_width * env.current_board_height) + (1 * env.current_board_width + 1)
    state, reward, terminated, truncated, info = env.step(flag_action)
    
    # Check that flag was placed
    assert state[1, 1] == CELL_FLAGGED
    assert not terminated
    assert not truncated
    # Reward should be REWARD_FLAG_MINE if cell is a mine, REWARD_FLAG_SAFE otherwise
    assert reward in [REWARD_FLAG_MINE, REWARD_FLAG_SAFE]

def test_flag_removal(env):
    """Test that removing a flag works correctly."""
    # Place flag at (1,1)
    flag_action = (1 * env.current_board_width * env.current_board_height) + (1 * env.current_board_width + 1)
    state, reward, terminated, truncated, info = env.step(flag_action)
    
    # Remove flag at same position
    state, reward, terminated, truncated, info = env.step(flag_action)
    
    # Check that flag was removed
    assert state[1, 1] == CELL_UNREVEALED
    assert not terminated
    assert not truncated
    assert reward == REWARD_UNFLAG  # Small penalty for removing a flag

def test_flag_on_revealed_cell(env):
    """Test that flag cannot be placed on revealed cell."""
    # Reveal cell at (1,1)
    reveal_action = 1 * env.current_board_width + 1
    state, reward, terminated, truncated, info = env.step(reveal_action)
    
    # Debug print statements
    print(f"State after reveal: {state[1, 1]}")
    print(f"Revealed status after reveal: {env.revealed[1, 1]}")
    
    # Try to place flag on revealed cell
    flag_action = (1 * env.current_board_width * env.current_board_height) + (1 * env.current_board_width + 1)
    state, reward, terminated, truncated, info = env.step(flag_action)
    
    # Debug print statements
    print(f"State after flag attempt: {state[1, 1]}")
    print(f"Revealed status after flag attempt: {env.revealed[1, 1]}")
    
    # Check that flag was not placed
    assert state[1, 1] != CELL_FLAGGED
    assert not terminated
    assert not truncated
    assert reward < 0  # Negative reward for invalid action

def test_flag_count(env):
    """Test that flag placement and removal works correctly."""
    # Place flag at (1,1)
    flag_action = (1 * env.current_board_width * env.current_board_height) + (1 * env.current_board_width + 1)
    state, reward, terminated, truncated, info = env.step(flag_action)
    
    # Verify flag was placed
    assert state[1, 1] == CELL_FLAGGED
    assert reward == REWARD_FLAG_SAFE  # Assuming (1,1) is not a mine
    
    # Place flag at (0,0)
    flag_action = (1 * env.current_board_width * env.current_board_height) + (0 * env.current_board_width + 0)
    state, reward, terminated, truncated, info = env.step(flag_action)
    
    # Verify second flag was placed
    assert state[0, 0] == CELL_FLAGGED
    assert reward == REWARD_FLAG_SAFE  # Assuming (0,0) is not a mine

def test_flag_on_mine(env):
    """Test that flag can be placed on mine."""
    # Place mine at (1,1)
    env.mines[1, 1] = True
    env._update_adjacent_counts()
    
    # Place flag at mine location
    flag_action = (1 * env.current_board_width * env.current_board_height) + (1 * env.current_board_width + 1)
    state, reward, terminated, truncated, info = env.step(flag_action)
    
    # Check that flag was placed
    assert state[1, 1] == CELL_FLAGGED
    assert not terminated
    assert not truncated
    assert reward == REWARD_FLAG_MINE  # Positive reward for correctly flagging a mine

def test_flag_mine_hit(env):
    """Test that flagged cells cannot be revealed and flagging/unflagging give no reward."""
    # Place mine at (1,1)
    env.mines[1, 1] = True
    env._update_adjacent_counts()

    # Place flag at mine location
    flag_action = (1 * env.current_board_width * env.current_board_height) + (1 * env.current_board_width + 1)
    state, reward, terminated, truncated, info = env.step(flag_action)

    # Check that flag was placed
    assert state[1, 1] == CELL_FLAGGED
    assert not terminated
    assert not truncated
    assert reward == 0  # No reward for flagging

    # Try to reveal flagged cell - should be invalid
    reveal_action = 1 * env.current_board_width + 1
    state, reward, terminated, truncated, info = env.step(reveal_action)
    assert state[1, 1] == CELL_FLAGGED  # Cell should still be flagged
    assert reward == REWARD_INVALID_ACTION  # Should get invalid action reward
    assert not terminated
    assert not truncated

    # Unflag the cell
    unflag_action = (1 * env.current_board_width * env.current_board_height) + (1 * env.current_board_width + 1)
    state, reward, terminated, truncated, info = env.step(unflag_action)
    assert state[1, 1] == CELL_UNREVEALED  # Cell should be unflag
    assert reward == 0  # No reward for unflagging
    assert not terminated
    assert not truncated 