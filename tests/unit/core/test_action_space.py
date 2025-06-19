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

def test_action_space_size(env):
    """Test that the action space size is correct."""
    expected_size = env.current_board_width * env.current_board_height * 2  # 2 for reveal and flag
    assert env.action_space.n == expected_size

def test_action_space_type(env):
    """Test that the action space type is correct."""
    assert env.action_space.__class__.__name__ == 'Discrete'

def test_action_space_bounds(env):
    """Test that the action space bounds are correct."""
    assert env.action_space.start == 0
    assert env.action_space.n == env.current_board_width * env.current_board_height * 2

def test_action_space_boundaries(env):
    """Test that action space boundaries are correct."""
    # Test reveal actions (0 to width*height-1)
    for i in range(env.current_board_height):
        for j in range(env.current_board_width):
            action = i * env.current_board_width + j
            assert 0 <= action < env.current_board_width * env.current_board_height
    
    # Test flag actions (width*height to 2*width*height-1)
    for i in range(env.current_board_height):
        for j in range(env.current_board_width):
            action = env.current_board_width * env.current_board_height + (i * env.current_board_width + j)
            assert env.current_board_width * env.current_board_height <= action < 2 * env.current_board_width * env.current_board_height

def test_action_space_mapping(env):
    """Test that action space maps correctly to board positions."""
    # Test reveal action mapping
    for i in range(env.current_board_height):
        for j in range(env.current_board_width):
            action = i * env.current_board_width + j
            row = action // env.current_board_width
            col = action % env.current_board_width
            assert row == i
            assert col == j
    
    # Test flag action mapping
    for i in range(env.current_board_height):
        for j in range(env.current_board_width):
            action = env.current_board_width * env.current_board_height + (i * env.current_board_width + j)
            row = (action - env.current_board_width * env.current_board_height) // env.current_board_width
            col = (action - env.current_board_width * env.current_board_height) % env.current_board_width
            assert row == i
            assert col == j

def test_action_space_consistency(env):
    """Test that action space remains consistent after board size changes."""
    # Get initial action space size
    initial_size = env.action_space.n
    
    # Change board size to something different than default (4x4)
    env.current_board_width = 5
    env.current_board_height = 5
    env.reset()
    
    # Action space should be updated
    new_size = env.action_space.n
    assert new_size == env.current_board_width * env.current_board_height * 2
    assert new_size != initial_size 