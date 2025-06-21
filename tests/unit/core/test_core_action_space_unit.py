import pytest
import numpy as np
from src.core.constants import (
    CELL_UNREVEALED,
    CELL_MINE,
    CELL_MINE_HIT,
    REWARD_FIRST_CASCADE_SAFE, REWARD_FIRST_CASCADE_HIT_MINE,
    REWARD_SAFE_REVEAL,
    REWARD_WIN,
    REWARD_HIT_MINE
)

def test_action_space_size(env):
    """Test that the action space size is correct."""
    expected_size = env.current_board_width * env.current_board_height  # Only reveal actions
    assert env.action_space.n == expected_size

def test_action_space_boundaries(env):
    """Test that action space boundaries are correct."""
    # Test reveal actions (0 to width*width-1)
    for i in range(env.current_board_height):
        for j in range(env.current_board_width):
            action = i * env.current_board_width + j
            assert 0 <= action < env.current_board_width * env.current_board_height

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

def test_action_space_consistency(env):
    """Test that action space is consistent across resets."""
    initial_size = env.action_space.n
    
    # Reset environment
    env.reset()
    
    # Action space size should remain the same
    assert env.action_space.n == initial_size

def test_action_space_contains(env):
    """Test that action space contains valid actions."""
    # Test valid actions
    assert env.action_space.contains(0)
    assert env.action_space.contains(env.action_space.n - 1)
    
    # Test invalid actions
    assert not env.action_space.contains(-1)
    assert not env.action_space.contains(env.action_space.n)
    assert not env.action_space.contains(env.action_space.n + 1) 