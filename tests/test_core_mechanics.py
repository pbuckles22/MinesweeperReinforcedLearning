import pytest
import numpy as np
from src.core.minesweeper_env import MinesweeperEnv

@pytest.fixture
def env():
    """Create a test environment with known board state."""
    env = MinesweeperEnv(
        max_board_size=4,
        max_mines=1,
        initial_mines=1,
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
    env.mines = np.zeros((3, 3), dtype=bool)
    env.mines[1, 1] = True
    env._update_adjacent_counts()
    env.state = np.full((3, 3), -1, dtype=np.int8)
    env.flags = np.zeros((3, 3), dtype=bool)
    env.revealed_count = 0
    return env

def test_safe_cell_reveal(env):
    """Test revealing a safe cell and its effects."""
    # Reveal a safe cell (0,0)
    obs, reward, terminated, truncated, info = env.step(0)
    
    # Check state update
    assert env.state[0, 0] == 1  # Should show 1 adjacent mine
    assert env.revealed_count == 1
    
    # Check reward
    assert reward > 0  # Should get positive reward for safe reveal
    
    # Check game not terminated
    assert not terminated
    assert not truncated
    
    # Check info dict
    assert 'revealed_cells' in info
    assert 'adjacent_mines' in info
    assert 'reward_breakdown' in info
    assert len(info['revealed_cells']) == 1
    assert (0, 0) in info['revealed_cells']
    
    # Check observation
    assert obs[0, 0] == 1  # Should show 1 adjacent mine
    # All other cells should be unrevealed
    unrevealed = np.delete(obs.flatten(), 0)
    assert np.all(unrevealed == -1)

def test_safe_cell_cascade(env):
    """Test the cascade effect when revealing a cell with no adjacent mines."""
    # Place the mine in the bottom-right corner
    env.board = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 9]  # Mine at (2,2)
    ])
    env.mines = np.zeros((3, 3), dtype=bool)
    env.mines[2, 2] = True
    env._update_adjacent_counts()  # Ensure adjacent mine counts are correct
    print('Board after _update_adjacent_counts:')
    print(env.board)
    env.state = np.full((3, 3), -1, dtype=np.int8)
    env.flags = np.zeros((3, 3), dtype=bool)
    env.revealed_count = 0

    # Reveal a cell with no adjacent mines (0,0)
    obs, reward, terminated, truncated, info = env.step(0)
    print('State after step:')
    print(env.state)

    # Check cascade effect
    assert env.revealed_count == 8  # All cells except the mine should be revealed
    assert env.state[2, 2] == -1  # Mine cell should remain hidden
    # All other cells should be revealed
    for x in range(3):
        for y in range(3):
            if (x, y) != (2, 2):
                assert env.state[x, y] == env.board[x, y]

def test_safe_cell_adjacent_mines(env):
    """Test revealing a cell with adjacent mines."""
    # Reveal a cell adjacent to mine (0,1)
    obs, reward, terminated, truncated, info = env.step(1)
    
    # Check state update
    assert env.state[0, 1] == 1  # Should show 1 adjacent mine
    assert env.revealed_count == 1
    
    # Check adjacent mines info
    assert 'adjacent_mines' in info
    assert len(info['adjacent_mines']) == 1  # Should have one adjacent mine
    assert (1, 1) in info['adjacent_mines']  # Mine should be at (1,1)
    
    # Check reward breakdown
    assert 'reward_breakdown' in info
    assert 'safe_reveal' in info['reward_breakdown']
    assert info['reward_breakdown']['safe_reveal'] > 0 