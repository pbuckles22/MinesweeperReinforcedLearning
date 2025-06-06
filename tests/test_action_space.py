import pytest
import numpy as np
from src.core.minesweeper_env import MinesweeperEnv

@pytest.fixture
def env():
    """Create a test environment with a known board state."""
    env = MinesweeperEnv(
        max_board_size=4,
        max_mines=2,
        initial_board_size=4,
        initial_mines=2,
        mine_spacing=1
    )
    env.reset()
    return env

def test_action_space_dimensions(env):
    """Test that action space has correct dimensions."""
    # Action space should be 2 * board area (reveal + flag actions)
    expected_size = env.current_board_width * env.current_board_height * 2
    assert env.action_space.n == expected_size

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
    
    # Change board size
    env.current_board_width = 4
    env.current_board_height = 4
    env.reset()
    
    # Action space should be updated
    new_size = env.action_space.n
    assert new_size == env.current_board_width * env.current_board_height * 2
    assert new_size != initial_size 