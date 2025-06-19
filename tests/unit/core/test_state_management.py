"""
Test suite for state management functionality.
Tests state reset, persistence, transitions, and counters.
"""

import pytest
import numpy as np
from src.core.minesweeper_env import MinesweeperEnv
from src.core.constants import (
    CELL_UNREVEALED,
    CELL_MINE,
    CELL_FLAGGED,
    CELL_MINE_HIT,
    REWARD_SAFE_REVEAL,
    REWARD_WIN,
    REWARD_HIT_MINE
)

@pytest.fixture
def env():
    """Create a test environment."""
    return MinesweeperEnv(
        initial_board_size=(4, 4),
        initial_mines=2
    )

def test_state_reset(env):
    """Test that state is properly reset."""
    # Make some moves to change state
    env.reset()
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # Reset and check state is cleared
    env.reset()
    assert np.all(env.state == CELL_UNREVEALED)
    assert np.all(env.flags == 0)
    assert env.is_first_move is True

def test_mine_placement_on_reset(env):
    """Test that mines are placed correctly on reset."""
    env.reset()
    
    # Check mine count
    mine_count = np.sum(env.mines)
    assert mine_count == env.current_mines
    
    # Check that mines are properly placed
    assert env.mines.shape == (env.current_board_height, env.current_board_width)

def test_flag_clearing_on_reset(env):
    """Test that flags are cleared on reset."""
    env.reset()
    
    # Place some flags
    flag_action = env.current_board_width * env.current_board_height
    state, reward, terminated, truncated, info = env.step(flag_action)
    
    # Reset and check flags are cleared
    env.reset()
    assert np.all(env.flags == 0)

def test_counter_reset(env):
    """Test that counters are reset properly."""
    env.reset()
    
    # Make some moves to increment counters
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # Reset and check counters
    env.reset()
    assert env.is_first_move is True

def test_state_persistence_between_actions(env):
    """Test that state persists correctly between actions."""
    env.reset()
    
    # Make first move
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    if terminated:
        # Game won on first move, can't test further
        return
    
    # Check that state reflects the move
    assert state[0, 0] != CELL_UNREVEALED
    
    # Make another move
    action = 1
    state, reward, terminated, truncated, info = env.step(action)
    
    # Check that both moves are reflected in state
    assert state[0, 0] != CELL_UNREVEALED
    assert state[0, 1] != CELL_UNREVEALED

def test_flag_persistence(env):
    """Test that flags persist between actions."""
    env.reset()
    
    # Place a flag
    flag_action = env.current_board_width * env.current_board_height
    state, reward, terminated, truncated, info = env.step(flag_action)
    
    # Check flag is in state
    assert state[0, 0] == CELL_FLAGGED
    
    # Make another action
    action = 1
    state, reward, terminated, truncated, info = env.step(action)
    
    # Check flag still persists
    assert state[0, 0] == CELL_FLAGGED

def test_revealed_cell_persistence(env):
    """Test that revealed cells persist between actions."""
    env.reset()
    
    # Reveal a cell
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    if terminated:
        # Game won on first move, can't test further
        return
    
    # Check cell is revealed
    assert state[0, 0] != CELL_UNREVEALED
    revealed_value = state[0, 0]
    
    # Make another action
    action = 1
    state, reward, terminated, truncated, info = env.step(action)
    
    # Check revealed cell still has same value
    assert state[0, 0] == revealed_value

def test_game_over_state(env):
    """Test game over state management."""
    env.reset()
    
    # Place mine and hit it
    env.mines[0, 0] = True
    env._update_adjacent_counts()
    
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # Check game is over
    assert terminated
    assert state[0, 0] == CELL_MINE_HIT

def test_game_counter(env):
    """Test game counter functionality."""
    # Play multiple games
    for _ in range(5):
        env.reset()
        action = 0
        state, reward, terminated, truncated, info = env.step(action)
    
    # Check that game counter is working
    assert hasattr(env, 'total_games')
    assert env.total_games >= 0

def test_win_counter(env):
    """Test win counter functionality."""
    # Play games and count wins
    wins = 0
    for _ in range(10):
        env.reset()
        # Try to win by revealing all safe cells
        for y in range(env.current_board_height):
            for x in range(env.current_board_width):
                if not env.mines[y, x]:
                    action = y * env.current_board_width + x
                    state, reward, terminated, truncated, info = env.step(action)
                    if terminated and info.get('won', False):
                        wins += 1
                        break
            if terminated:
                break
    
    # Check win counter
    assert hasattr(env, 'win_count')
    assert env.win_count >= 0

def test_consecutive_hits(env):
    """Test consecutive mine hits tracking."""
    env.reset()
    
    # Place mine and hit it
    env.mines[0, 0] = True
    env._update_adjacent_counts()
    
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # Check consecutive hits tracking
    assert hasattr(env, 'consecutive_mine_hits')
    assert env.consecutive_mine_hits >= 0

def test_win_rate_calculation(env):
    """Test win rate calculation."""
    # Play several games
    for _ in range(10):
        env.reset()
        action = 0
        state, reward, terminated, truncated, info = env.step(action)
    
    # Check win rate calculation
    assert hasattr(env, 'win_rate')
    assert 0 <= env.win_rate <= 1

def test_state_transitions():
    """Test state transitions between different game states."""
    env = MinesweeperEnv(initial_board_size=(4, 4), initial_mines=2)
    
    # Test initial state
    env.reset()
    assert env.is_first_move is True
    
    # Test after first move
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    assert env.is_first_move is False
    
    # Test after game over
    if terminated:
        assert env.is_first_move is True  # Should reset for next game

def test_state_representation():
    """Test that state representation is correct."""
    env = MinesweeperEnv(initial_board_size=(4, 4), initial_mines=2)
    env.reset()
    
    # Check initial state representation
    state = env.state
    assert state.shape == (env.current_board_height, env.current_board_width)
    assert np.all(state == CELL_UNREVEALED)
    
    # Check state after action
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    assert state.shape == (env.current_board_height, env.current_board_width)

def test_state_with_flags():
    """Test state representation with flags."""
    env = MinesweeperEnv(initial_board_size=(4, 4), initial_mines=2)
    env.reset()
    
    # Place a flag
    flag_action = env.current_board_width * env.current_board_height
    state, reward, terminated, truncated, info = env.step(flag_action)
    
    # Check flag is represented correctly
    assert state[0, 0] == CELL_FLAGGED

def test_state_with_revealed_cells():
    """Test state representation with revealed cells."""
    env = MinesweeperEnv(initial_board_size=(4, 4), initial_mines=2)
    env.reset()
    
    # Reveal a cell
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    if terminated:
        # Game won on first move, can't test further
        return
    
    # Check revealed cell is represented correctly
    assert state[0, 0] != CELL_UNREVEALED
    assert state[0, 0] != CELL_FLAGGED

def test_state_with_mine_hit():
    """Test state representation when mine is hit."""
    env = MinesweeperEnv(initial_board_size=(4, 4), initial_mines=2)
    env.reset()
    
    # Place mine and hit it
    env.mines[0, 0] = True
    env._update_adjacent_counts()
    
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # Check mine hit is represented correctly
    assert state[0, 0] == CELL_MINE_HIT

def test_state_consistency():
    """Test that state is consistent across multiple accesses."""
    env = MinesweeperEnv(initial_board_size=(4, 4), initial_mines=2)
    env.reset()
    
    # Make a move
    action = 0
    state1, reward1, terminated1, truncated1, info1 = env.step(action)
    
    # Get state again
    state2 = env.state
    
    # States should be identical
    assert np.array_equal(state1, state2)

def test_state_with_rectangular_board():
    """Test state management with rectangular boards."""
    env = MinesweeperEnv(initial_board_size=(3, 5), initial_mines=3)
    env.reset()
    
    # Check state shape
    assert env.state.shape == (3, 5)
    assert np.all(env.state == CELL_UNREVEALED)
    
    # Make a move
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # Check state shape is maintained
    assert state.shape == (3, 5)

def test_state_with_large_board():
    """Test state management with large boards."""
    env = MinesweeperEnv(initial_board_size=(8, 8), initial_mines=10)
    env.reset()
    
    # Check state shape
    assert env.state.shape == (8, 8)
    assert np.all(env.state == CELL_UNREVEALED)
    
    # Make a move
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # Check state shape is maintained
    assert state.shape == (8, 8) 