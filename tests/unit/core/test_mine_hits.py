import pytest
import numpy as np
from src.core.minesweeper_env import MinesweeperEnv
from src.core.constants import (
    CELL_UNREVEALED,
    CELL_MINE_HIT,
    REWARD_HIT_MINE,
    REWARD_FIRST_CASCADE_HIT_MINE,
    REWARD_FIRST_CASCADE_SAFE,
    REWARD_SAFE_REVEAL
)

@pytest.fixture
def env():
    """Create a test environment."""
    return MinesweeperEnv(initial_board_size=3, initial_mines=1)

def test_pre_cascade_mine_hit(env):
    """Test mine hit during pre-cascade period."""
    env.reset()
    
    # Set up a mine at the first action location
    env.mines.fill(False)
    env.mines[0, 0] = True  # Mine at first cell
    env._update_adjacent_counts()
    env.mines_placed = True
    
    # Hit mine on pre-cascade
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # Should get neutral reward since it's during pre-cascade period, but game should terminate
    assert reward == REWARD_FIRST_CASCADE_HIT_MINE, "Pre-cascade mine hit should give neutral reward"
    assert terminated, "Game should terminate on pre-cascade mine hit"
    assert state[0, 0, 0] == CELL_MINE_HIT, "Mine hit should be visible"

def test_mine_hit_after_pre_cascade(env):
    """Test mine hit after pre-cascade period."""
    env.reset()
    
    # Set up a mine at a specific location
    env.mines.fill(False)
    env.mines[1, 1] = True  # Mine at center
    env._update_adjacent_counts()
    env.mines_placed = True
    env.is_first_cascade = False  # Simulate post-cascade state
    env.first_cascade_done = True
    
    # Take a safe move first
    safe_action = 0
    state, reward, terminated, truncated, info = env.step(safe_action)
    
    # Now hit the mine (after pre-cascade period)
    mine_action = 1 * env.current_board_width + 1
    state, reward, terminated, truncated, info = env.step(mine_action)
    
    # Should get mine hit penalty since it's after pre-cascade period
    assert reward == REWARD_HIT_MINE, "Post-cascade mine hit should give mine hit penalty"
    assert terminated, "Game should terminate on mine hit"
    assert state[0, 1, 1] == CELL_MINE_HIT, "Mine hit should be visible"

def test_multiple_mine_hits(env):
    """Test multiple mine hits in a game."""
    env.reset()
    # Place multiple mines
    env.mines.fill(False)
    env.mines[0, 0] = True
    env.mines[1, 1] = True
    env.mines[2, 2] = True
    env._update_adjacent_counts()
    env.mines_placed = True
    env.is_first_cascade = False
    env.first_cascade_done = True

    # Make first move (can be a mine or safe, no special logic)
    action = 3
    state, reward, terminated, truncated, info = env.step(action)
    assert reward == REWARD_SAFE_REVEAL

    # Hit first mine
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    assert reward == REWARD_HIT_MINE
    assert terminated

def test_mine_hit_state_consistency(env):
    """Test that mine hit state is consistent."""
    env.reset()
    
    # Place mine and hit it
    env.mines[0, 0] = True
    env._update_adjacent_counts()
    env.mines_placed = True
    env.is_first_cascade = False
    env.first_cascade_done = True
    
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
        env.is_first_cascade = False
        env.first_cascade_done = True
        action = 0
        state, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
    # All rewards should be the same
    assert all(r == REWARD_HIT_MINE for r in rewards)

def test_mine_hit_with_cascade(env):
    """Test mine hit that triggers a cascade."""
    env.reset()
    # Place mine at edge
    env.mines.fill(False)
    env.mines[0, 1] = True
    env._update_adjacent_counts()
    env.mines_placed = True
    env.is_first_cascade = False
    env.first_cascade_done = True

    # Make first move (can be a mine or safe, no special logic)
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    assert reward == REWARD_SAFE_REVEAL

    # Hit mine
    action = 1
    state, reward, terminated, truncated, info = env.step(action)
    assert reward == REWARD_HIT_MINE
    assert terminated 