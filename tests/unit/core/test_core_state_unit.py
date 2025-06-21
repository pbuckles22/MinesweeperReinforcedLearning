"""
Test suite for state management functionality.
Tests state reset, persistence, transitions, and counters.
"""

import pytest
import numpy as np
from src.core.minesweeper_env import MinesweeperEnv
from src.core.constants import (
    CELL_UNREVEALED,
    CELL_MINE_HIT,
    REWARD_FIRST_CASCADE_SAFE, REWARD_FIRST_CASCADE_HIT_MINE,
    REWARD_SAFE_REVEAL,
    REWARD_WIN,
    REWARD_HIT_MINE
)

@pytest.fixture
def env():
    """Create a test environment."""
    return MinesweeperEnv(initial_board_size=3, initial_mines=1)

def test_initial_state(env):
    """Test that initial state is correct."""
    env.reset()
    
    # Check that all cells are unrevealed initially
    assert np.all(env.state[0] == CELL_UNREVEALED)
    assert np.all(env.revealed == False)

def test_state_after_reveal(env):
    """Test that state is updated correctly after revealing a cell."""
    env.reset()
    
    # Reveal a cell
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # Check that the revealed cell is no longer unrevealed
    assert state[0, 0, 0] != CELL_UNREVEALED
    assert env.revealed[0, 0] == True

def test_state_persistence(env):
    """Test that state persists between actions."""
    env.reset()
    
    # Reveal a cell
    action = 0
    state1, reward1, terminated1, truncated1, info1 = env.step(action)
    
    # Reveal another cell
    action = 1
    state2, reward2, terminated2, truncated2, info2 = env.step(action)
    
    # Check that the first cell remains revealed
    assert state2[0, 0, 0] != CELL_UNREVEALED
    assert env.revealed[0, 0] == True

def test_state_reset(env):
    """Test that state is reset correctly."""
    env.reset()
    
    # Reveal some cells
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # Reset and check state is back to initial
    env.reset()
    assert np.all(env.state[0] == CELL_UNREVEALED)
    assert np.all(env.revealed == False)

def test_state_shape_consistency(env):
    """Test that state shape remains consistent across operations."""
    # Test initial state shape
    state, _ = env.reset()
    expected_shape = (4, env.current_board_height, env.current_board_width)
    assert state.shape == expected_shape, f"Initial state shape should be {expected_shape}, got {state.shape}"
    
    # Test state shape after reveal
    action = 0
    state, _, _, _, _ = env.step(action)
    assert state.shape == expected_shape, f"State shape after reveal should be {expected_shape}, got {state.shape}"
    
    # Test state shape after reset
    state, _ = env.reset()
    assert state.shape == expected_shape, f"State shape after reset should be {expected_shape}, got {state.shape}"

def test_state_bounds(env):
    """Test that state values are within expected bounds."""
    state, _ = env.reset()
    
    # Channel 0: Game state (-4 to 8)
    assert np.all(state[0] >= -4), "Channel 0 should be >= -4"
    assert np.all(state[0] <= 8), "Channel 0 should be <= 8"
    
    # Channel 1: Safety hints (-1 to 8)
    assert np.all(state[1] >= -1), "Channel 1 should be >= -1"
    assert np.all(state[1] <= 8), "Channel 1 should be <= 8"
    
    # Channel 2: Revealed cell count (0 to max_cells)
    max_cells = env.current_board_height * env.current_board_width
    assert np.all(state[2] >= 0), "Channel 2 should be >= 0"
    assert np.all(state[2] <= max_cells), f"Channel 2 should be <= {max_cells}"
    
    # Channel 3: Game progress indicators (0 to 1)
    assert np.all(state[3] >= 0), "Channel 3 should be >= 0"
    assert np.all(state[3] <= 1), "Channel 3 should be <= 1"

def test_state_transitions(env):
    """Test that state transitions are correct."""
    env.reset()
    
    # Test transition from unrevealed to revealed
    action = 0
    state_before = env.state.copy()
    state_after, reward, terminated, truncated, info = env.step(action)
    
    # The revealed cell should change
    assert state_before[0, 0, 0] == CELL_UNREVEALED
    assert state_after[0, 0, 0] != CELL_UNREVEALED

def test_mine_hit_state(env):
    """Test that hitting a mine updates state correctly."""
    env.reset()
    
    # Place mine at (0,0) and hit it
    env.mines[0, 0] = True
    env._update_adjacent_counts()
    env.mines_placed = True
    env.is_first_cascade = False
    env.first_cascade_done = True
    
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # Check that the hit cell shows mine hit
    assert state[0, 0, 0] == CELL_MINE_HIT
    assert env.revealed[0, 0] == True

def test_win_state(env):
    """Test that winning updates state correctly."""
    env.reset()
    
    # Set up simple win scenario: mine at corner, reveal all others
    env.mines.fill(False)
    env.mines[0, 0] = True  # Mine at corner
    env._update_adjacent_counts()
    env.mines_placed = True
    env.is_first_cascade = False
    env.first_cascade_done = True
    
    # Reveal all safe cells
    for i in range(1, env.current_board_width * env.current_board_height):
        state, reward, terminated, truncated, info = env.step(i)
        if terminated:
            # All safe cells should be revealed
            for row in range(env.current_board_height):
                for col in range(env.current_board_width):
                    if not env.mines[row, col]:
                        assert env.revealed[row, col] == True
            break

def test_safety_hints_channel(env):
    """Test that safety hints channel works correctly."""
    env.reset()
    
    # Check initial safety hints
    safety_hints = env.state[1]
    assert safety_hints.shape == (env.current_board_height, env.current_board_width)
    
    # Reveal a cell and check safety hints update
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    new_safety_hints = state[1]
    # Revealed cell should show UNKNOWN_SAFETY (-1) or 0.0 (if cell is a zero cell) in safety hints
    assert new_safety_hints[0, 0] in (-1, 0.0) 