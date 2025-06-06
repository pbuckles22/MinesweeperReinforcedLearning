import pytest
import numpy as np
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

def test_mine_hit_reward(env):
    """Test that hitting a mine gives the correct reward."""
    # Place mine at (1,1)
    env.mines[1, 1] = True
    env._update_adjacent_counts()
    
    # Reveal mine
    action = 1 * env.current_board_width + 1
    state, reward, terminated, truncated, info = env.step(action)
    
    # Check reward
    assert reward == REWARD_HIT_MINE
    assert terminated

def test_mine_hit_state(env):
    """Test that hitting a mine updates the state correctly."""
    # Place mine at (1,1)
    env.mines[1, 1] = True
    env._update_adjacent_counts()
    
    # Reveal mine
    action = 1 * env.current_board_width + 1
    state, reward, terminated, truncated, info = env.step(action)
    
    # Check that state is updated correctly
    assert state[1, 1] == CELL_MINE_HIT
    assert np.all(state[0, 0] == CELL_UNREVEALED)
    assert np.all(state[0, 1] == CELL_UNREVEALED)
    assert np.all(state[0, 2] == CELL_UNREVEALED)
    assert np.all(state[1, 0] == CELL_UNREVEALED)
    assert np.all(state[1, 2] == CELL_UNREVEALED)
    assert np.all(state[2, 0] == CELL_UNREVEALED)
    assert np.all(state[2, 1] == CELL_UNREVEALED)
    assert np.all(state[2, 2] == CELL_UNREVEALED)

def test_mine_hit_game_over(env):
    """Test that hitting a mine ends the game."""
    # Place mine at (1,1)
    env.mines[1, 1] = True
    env._update_adjacent_counts()
    
    # Reveal mine
    action = 1 * env.current_board_width + 1
    state, reward, terminated, truncated, info = env.step(action)
    
    # Check that game is terminated
    assert terminated
    assert not truncated
    assert state[1, 1] == CELL_MINE_HIT
    assert reward == REWARD_HIT_MINE

def test_first_move_mine_hit_reset(env):
    """Test that hitting a mine on first move resets the game."""
    # Place mine at (1,1)
    env.mines[1, 1] = True
    env._update_adjacent_counts()
    
    # Reveal mine on first move
    action = 1 * env.current_board_width + 1
    state, reward, terminated, truncated, info = env.step(action)
    
    # Check that game is reset
    assert reward == REWARD_FIRST_MOVE_HIT_MINE
    assert not terminated
    assert np.all(state == CELL_UNREVEALED)

def test_first_move_behavior(env):
    """Test that first move behavior is correct."""
    # Place mine at (1,1)
    env.mines[1, 1] = True
    env._update_adjacent_counts()
    
    # Reveal safe cell on first move
    action = 0 * env.current_board_width + 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # Check that game continues
    assert reward == REWARD_FIRST_MOVE_SAFE
    assert not terminated
    assert state[0, 0] != CELL_UNREVEALED 