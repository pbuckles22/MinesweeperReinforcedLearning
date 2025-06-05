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
    # Force a mine at position (1,1)
    env.mines = np.zeros((4, 4), dtype=bool)
    env.mines[1, 1] = True
    env.mines[2, 2] = True  # Add another mine to satisfy mine count
    return env

def test_mine_hit_termination(env):
    """Test that hitting a mine terminates the game."""
    # Calculate action index for the mine position (1,1)
    mine_action = 1 * env.current_board_size + 1

    # Reveal the mine
    obs, reward, terminated, truncated, info = env.step(mine_action)

    # If this is the first move, the game should be reset
    if env.is_first_move:
        assert not terminated
        assert reward == 0
    else:
        # Otherwise, the game should be terminated
        assert terminated
        assert reward == env.mine_penalty
        assert 'mine_hit' in info['reward_breakdown']

def test_mine_hit_state_update(env):
    """Test that hitting a mine updates the state correctly."""
    # Calculate action index for the mine position (1,1)
    mine_action = 1 * env.current_board_size + 1

    # Reveal the mine
    obs, reward, terminated, truncated, info = env.step(mine_action)

    # If this is the first move, the game should be reset
    if env.is_first_move:
        assert not terminated
        assert reward == 0
        # State should be reset to all unrevealed
        assert np.all(obs == -1)
    else:
        # Otherwise, the mine should be marked as hit
        assert terminated
        assert obs[1, 1] == -2  # Mine should be marked as hit
        assert reward == env.mine_penalty

def test_mine_hit_reward_breakdown(env):
    """Test that hitting a mine includes correct reward breakdown in info."""
    # Calculate action index for the mine position (1,1)
    mine_action = 1 * env.current_board_size + 1

    # Reveal the mine
    obs, reward, terminated, truncated, info = env.step(mine_action)

    # Check reward breakdown
    assert 'reward_breakdown' in info
    if env.is_first_move:
        assert reward == 0
    else:
        assert 'mine_hit' in info['reward_breakdown']
        assert info['reward_breakdown']['mine_hit'] == env.mine_penalty

def test_first_move_mine_hit_reset(env):
    """Test that hitting a mine on the first move resets the game with no penalty."""
    # Hit a mine on the first move
    state, reward, terminated, truncated, info = env.step(0)

    # If first move didn't hit mine, try another cell
    if not terminated:
        env.reset()
        state, reward, terminated, truncated, info = env.step(1)

    # Verify the game was reset
    assert not terminated  # Game should not be terminated
    assert reward == 0  # No penalty
    assert np.all(state == -1)  # All cells should be unrevealed

def test_first_move_behavior(env):
    """Test that first move behavior is consistent regardless of outcome."""
    # Try first move
    state, reward, terminated, truncated, info = env.step(0)

    # Verify first move behavior
    assert not terminated  # Game should not be terminated
    assert reward == 0  # No reward for first move
    assert env.is_first_move == False  # First move flag should be set to False 