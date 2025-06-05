import pytest
import numpy as np
from src.core.minesweeper_env import MinesweeperEnv

@pytest.fixture
def env():
    """Create a test environment with known board state."""
    env = MinesweeperEnv(
        max_board_size=3,
        max_mines=1,
        early_learning_mode=True
    )
    env.current_board_size = 3
    env.current_mines = 1
    # Force a specific board state for testing
    env.board = np.array([
        [0, 0, 0],
        [0, 9, 0],  # Mine at (1,1) represented by 9
        [0, 0, 0]
    ])
    env.mines = [(1, 1)]
    env._update_adjacent_counts()
    env.state = np.full((3, 3), -1, dtype=np.int8)
    env.flags = np.zeros((3, 3), dtype=bool)
    env.revealed_count = 0
    return env

def test_mine_hit_termination(env):
    """Test that hitting a mine terminates the game."""
    # Calculate action index for the mine position (1,1)
    mine_action = 1 * env.current_board_size + 1
    
    # Reveal the mine
    obs, reward, terminated, truncated, info = env.step(mine_action)
    
    # Check game termination
    assert terminated
    assert not truncated
    assert reward == env.mine_penalty
    assert env.state[1, 1] == -2  # Mine should be marked as hit
    assert env.revealed_count == 1  # Only the mine should be revealed

def test_mine_hit_state_update(env):
    """Test that hitting a mine updates the state correctly."""
    # Calculate action index for the mine position (1,1)
    mine_action = 1 * env.current_board_size + 1
    
    # Reveal the mine
    obs, reward, terminated, truncated, info = env.step(mine_action)
    
    # Check state update
    assert env.state[1, 1] == -2  # Mine should be marked as hit
    # All other cells should remain hidden
    for x in range(3):
        for y in range(3):
            if (x, y) != (1, 1):
                assert env.state[x, y] == -1

def test_mine_hit_reward_breakdown(env):
    """Test that hitting a mine includes correct reward breakdown in info."""
    # Calculate action index for the mine position (1,1)
    mine_action = 1 * env.current_board_size + 1
    
    # Reveal the mine
    obs, reward, terminated, truncated, info = env.step(mine_action)
    
    # Check reward breakdown
    assert 'reward_breakdown' in info
    assert 'mine_hit' in info['reward_breakdown']
    assert info['reward_breakdown']['mine_hit'] == env.mine_penalty
    assert reward == env.mine_penalty 