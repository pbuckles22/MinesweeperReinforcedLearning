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
    """Test that the action space dimensions match the board size."""
    # For a 4x4 board, we should have 32 actions (16 cells * 2 actions per cell)
    assert env.action_space.n == 32

def test_action_space_boundaries(env):
    """Test that invalid actions raise appropriate errors."""
    # Test negative action
    with pytest.raises(ValueError):
        env.step(-1)
    
    # Test action too large
    with pytest.raises(ValueError):
        env.step(env.action_space.n)
    
    # Test non-integer action
    with pytest.raises(ValueError):
        env.step(1.5)

def test_action_space_mapping(env):
    """Test that actions map correctly to board positions."""
    board_size = env.current_board_size

    # Test reveal action
    action = 5  # Should map to position (1,1) on a 4x4 board
    state, reward, terminated, truncated, info = env.step(action)
    
    # If this is the first move, check first move behavior
    if env.is_first_move:
        assert not terminated
        assert reward == 0
    else:
        # Check that the action was processed
        assert state[1, 1] != -1  # Cell should be revealed

def test_action_space_consistency(env):
    """Test that the action space remains consistent after actions and resets."""
    initial_action_space = env.action_space.n
    
    # Perform some actions
    for _ in range(5):
        env.step(0)
    
    # Check action space hasn't changed
    assert env.action_space.n == initial_action_space
    
    # Reset and check again
    env.reset()
    assert env.action_space.n == initial_action_space 