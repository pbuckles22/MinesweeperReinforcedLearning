import pytest
import numpy as np
from src.core.minesweeper_env import MinesweeperEnv
from src.core.constants import CELL_MINE_HIT, CELL_UNREVEALED

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
    # Place mine at (1,1)
    env.mines[1, 1] = True
    env._update_adjacent_counts()
    
    # Take action to hit mine
    mine_action = 1 * env.current_board_width + 1
    state, reward, terminated, truncated, info = env.step(mine_action)
    
    # If this is the first move, the game should be reset
    if env.is_first_move:
        assert not terminated
        assert reward == 0
    else:
        assert terminated
        assert not truncated
        assert state[1, 1] == CELL_MINE_HIT
        assert reward < 0  # Should be negative reward for hitting mine

def test_mine_hit_state_update(env):
    """Test that hitting a mine updates the state correctly."""
    # Place mine at (1,1)
    env.mines[1, 1] = True
    env._update_adjacent_counts()
    
    # Take action to hit mine
    mine_action = 1 * env.current_board_width + 1
    state, reward, terminated, truncated, info = env.step(mine_action)
    
    # If this is the first move, the game should be reset
    if env.is_first_move:
        assert not terminated
        assert reward == 0
        assert np.all(state == CELL_UNREVEALED)  # All cells should be unrevealed
    else:
        # Check that only the mine hit cell is revealed
        assert state[1, 1] == CELL_MINE_HIT
        for i in range(env.current_board_height):
            for j in range(env.current_board_width):
                if (i, j) != (1, 1):
                    assert state[i, j] == CELL_UNREVEALED

def test_mine_hit_reward_breakdown(env):
    """Test that hitting a mine provides correct reward breakdown."""
    # Place mine at (1,1)
    env.mines[1, 1] = True
    env._update_adjacent_counts()
    
    # Take action to hit mine
    mine_action = 1 * env.current_board_width + 1
    state, reward, terminated, truncated, info = env.step(mine_action)
    
    # If this is the first move, the game should be reset
    if env.is_first_move:
        assert not terminated
        assert reward == 0
    else:
        assert terminated
        assert reward < 0  # Should be negative reward for hitting mine
        assert 'won' in info

def test_first_move_mine_hit_reset(env):
    """Test that hitting a mine on first move resets the board."""
    # Place mine at (1,1)
    env.mines[1, 1] = True
    env._update_adjacent_counts()
    
    # Take action to hit mine on first move
    mine_action = 1 * env.current_board_width + 1
    state, reward, terminated, truncated, info = env.step(mine_action)
    
    # First move should be safe
    assert not terminated
    assert reward == 0
    assert not env.mines[1, 1]  # Mine should be moved
    assert np.all(state == CELL_UNREVEALED)  # All cells should be unrevealed

def test_first_move_behavior(env):
    """Test that first move is always safe."""
    # Try multiple first moves
    for _ in range(10):
        env.reset()
        # Take random first move
        first_action = np.random.randint(0, env.current_board_width * env.current_board_height)
        state, reward, terminated, truncated, info = env.step(first_action)
        
        # First move should never hit a mine
        assert not terminated
        assert reward == 0
        assert not env.mines[first_action // env.current_board_width, first_action % env.current_board_width] 