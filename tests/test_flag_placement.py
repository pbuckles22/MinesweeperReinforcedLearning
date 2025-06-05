import pytest
import numpy as np
from src.core.minesweeper_env import MinesweeperEnv

@pytest.fixture
def env():
    env = MinesweeperEnv(
        max_board_size=4,
        max_mines=1,
        initial_mines=1,
        early_learning_mode=True
    )
    env.current_board_size = 3
    env.current_mines = 1
    env.board = np.array([
        [0, 0, 0],
        [0, 9, 0],  # Mine at (1,1)
        [0, 0, 0]
    ])
    env.mines = np.zeros((3, 3), dtype=bool)
    env.mines[1, 1] = True
    env._update_adjacent_counts()
    env.state = np.full((3, 3), -1, dtype=np.int8)
    env.flags = np.zeros((3, 3), dtype=bool)
    env.revealed_count = 0
    return env

def test_flag_placement_on_mine(env):
    """Test placing a flag on a mine gives correct reward and state."""
    flag_action = env.current_board_size * env.current_board_size + (1 * env.current_board_size + 1)
    obs, reward, terminated, truncated, info = env.step(flag_action)
    assert env.flags[1, 1] == True
    assert reward == 5.0
    assert info['reward_breakdown']['correct_flag'] == 5.0
    assert not terminated
    assert not truncated

def test_flag_placement_on_safe_cell(env):
    """Test placing a flag on a safe cell gives penalty and state."""
    flag_action = env.current_board_size * env.current_board_size + (0 * env.current_board_size + 0)
    obs, reward, terminated, truncated, info = env.step(flag_action)
    assert env.flags[0, 0] == True
    assert reward == -1.0
    assert info['reward_breakdown']['incorrect_flag'] == -1.0
    assert not terminated
    assert not truncated

def test_flag_removal(env):
    """Test removing a flag gives small penalty and updates state."""
    # Place flag first
    flag_action = env.current_board_size * env.current_board_size + (1 * env.current_board_size + 1)
    env.step(flag_action)
    # Remove flag
    obs, reward, terminated, truncated, info = env.step(flag_action)
    assert env.flags[1, 1] == False
    assert reward == -0.1
    assert info['reward_breakdown']['unflag'] == -0.1
    assert not terminated
    assert not truncated 