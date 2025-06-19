import pytest
import numpy as np
from src.core.minesweeper_env import MinesweeperEnv
from src.core.constants import (
    CELL_UNREVEALED,
    CELL_MINE_HIT,
    REWARD_FIRST_MOVE_HIT_MINE,
    REWARD_HIT_MINE
)

@pytest.fixture
def env():
    """Create a test environment."""
    return MinesweeperEnv(initial_board_size=3, initial_mines=1)

def test_first_move_mine_hit(env):
    """Test hitting a mine on first move."""
    env.reset()
    
    # Place mine at (0,0) and hit it
    env.mines[0, 0] = True
    env._update_adjacent_counts()
    env.mines_placed = True
    env.is_first_move = True
    env.first_move_done = False
    
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    assert reward == REWARD_FIRST_MOVE_HIT_MINE
    # After first move mine hit, the environment resets, so all cells should be unrevealed
    assert np.all(state[0] == CELL_UNREVEALED), "State should be reset to all unrevealed after first move mine hit"
    assert not terminated

def test_mine_hit_after_first_move(env):
    """Test hitting a mine after first move."""
    env.reset()
    
    # Make first move safe
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # Place mine at (1,0) and hit it (after reset and first move)
    env.mines.fill(False)
    env.mines[1, 0] = True
    env._update_adjacent_counts()
    env.mines_placed = True
    env.is_first_move = False
    env.first_move_done = True
    
    action = 3  # (1,0)
    state, reward, terminated, truncated, info = env.step(action)
    
    assert reward == REWARD_HIT_MINE
    assert state[0, 1, 0] == CELL_MINE_HIT
    assert env.revealed[1, 0] == True
    assert terminated

def test_multiple_mine_hits(env):
    """Test multiple mine hits in sequence."""
    env.reset()
    
    # Place multiple mines
    env.mines[0, 0] = True
    env.mines[1, 1] = True
    env._update_adjacent_counts()
    env.mines_placed = True
    env.is_first_move = False
    env.first_move_done = True
    
    # Hit first mine
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    assert reward == REWARD_HIT_MINE
    assert state[0, 0, 0] == CELL_MINE_HIT
    assert terminated

def test_mine_hit_state_consistency(env):
    """Test that mine hit state is consistent."""
    env.reset()
    
    # Place mine and hit it
    env.mines[0, 0] = True
    env._update_adjacent_counts()
    env.mines_placed = True
    env.is_first_move = False
    env.first_move_done = True
    
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # Check state consistency
    assert state[0, 0, 0] == CELL_MINE_HIT
    assert env.revealed[0, 0] == True
    assert env.mines[0, 0] == True  # Mine should still be there
    assert terminated
    assert not info.get('won', False)

def test_mine_hit_reward_consistency(env):
    """Test that mine hit rewards are consistent."""
    rewards = []
    for _ in range(3):
        env.reset()
        env.mines.fill(False)
        env.mines[0, 0] = True
        env._update_adjacent_counts()
        env.mines_placed = True
        env.is_first_move = False
        env.first_move_done = True
        action = 0
        state, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
    # All rewards should be the same
    assert all(r == REWARD_HIT_MINE for r in rewards) 