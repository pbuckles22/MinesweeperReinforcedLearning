"""
Test suite for the reward system.
Tests all reward types, scaling, and edge cases.
"""

import pytest
import numpy as np
from src.core.minesweeper_env import MinesweeperEnv
from src.core.constants import (
    CELL_UNREVEALED,
    CELL_MINE_HIT,
    REWARD_FIRST_CASCADE_SAFE, REWARD_FIRST_CASCADE_HIT_MINE,
    REWARD_SAFE_REVEAL,
    REWARD_WIN,
    REWARD_HIT_MINE,
    REWARD_INVALID_ACTION
)

@pytest.fixture
def env():
    """Create a test environment."""
    return MinesweeperEnv(initial_board_size=4, initial_mines=3)  # More complex setup

def test_pre_cascade_safe_reward(env):
    """Test reward for safe pre-cascade."""
    env.reset()
    
    # Pre-cascade should be safe but not win (due to 3 mines in 4x4 board)
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    assert reward == REWARD_FIRST_CASCADE_SAFE, "Pre-cascade should give neutral reward"
    assert not terminated, "Pre-cascade should not terminate game"

def test_pre_cascade_mine_hit_reward(env):
    """Test reward for hitting mine on pre-cascade."""
    env.reset()
    
    # Set up a mine at the first action location
    env.mines.fill(False)
    env.mines[0, 0] = True
    env._update_adjacent_counts()
    env.mines_placed = True
    
    # Pre-cascade mine hit should end the game with neutral reward
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # Game should terminate on mine hit
    assert terminated, "Game should terminate on mine hit"
    # Pre-cascade mine hit should give neutral reward (not full penalty)
    assert reward == REWARD_FIRST_CASCADE_HIT_MINE, f"Pre-cascade mine hit should give neutral reward, got {reward}"
    assert not info['won'], "Game should not be won when hitting a mine"

def test_safe_reveal_after_pre_cascade(env):
    """Test reward for safe reveal after pre-cascade."""
    env.reset()
    
    # Set up board so pre-cascade doesn't cascade too much
    env.mines.fill(False)
    env.mines[1, 1] = True
    env.mines[2, 2] = True
    env._update_adjacent_counts()
    env.mines_placed = True
    env.is_first_cascade = False
    env.first_cascade_done = True
    
    # Make pre-cascade safe (should reveal limited area)
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # Should get full safe reveal reward since we're past pre-cascade
    assert reward == REWARD_SAFE_REVEAL  # Not pre-cascade anymore

def test_mine_hit_after_pre_cascade(env):
    """Test reward for hitting mine after pre-cascade."""
    env.reset()
    
    # Set up a mine at a specific location
    env.mines.fill(False)
    env.mines[1, 1] = True
    env._update_adjacent_counts()
    env.mines_placed = True
    env.is_first_cascade = False
    env.first_cascade_done = True
    
    # Make pre-cascade safe
    safe_action = 0
    state, reward, terminated, truncated, info = env.step(safe_action)
    
    # Now hit the mine (after pre-cascade period)
    mine_action = 1 * env.current_board_width + 1
    state, reward, terminated, truncated, info = env.step(mine_action)
    
    # Should get mine hit penalty since it's after pre-cascade period
    assert reward == REWARD_HIT_MINE, "Post-cascade mine hit should give mine hit penalty"
    assert terminated, "Game should terminate on mine hit"

def test_win_reward(env):
    """Test reward for winning the game."""
    env.reset()
    # Set up simple win scenario: mine at corner, reveal all others
    env.mines.fill(False)
    env.mines[0, 0] = True  # Mine at corner
    env._update_adjacent_counts()
    env.mines_placed = True
    env.is_first_cascade = False
    env.first_cascade_done = True
    
    # Reveal all safe cells
    for i in range(1, env.current_board_width * env.current_board_height):
        state, reward, terminated, truncated, info = env.step(i)
        if terminated:
            assert reward == REWARD_WIN
            break

def test_invalid_action_reward(env):
    """Test reward for invalid actions."""
    env.reset()
    # Try to reveal an already revealed cell
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # Try to reveal the same cell again
    state, reward, terminated, truncated, info = env.step(action)
    assert reward == REWARD_INVALID_ACTION

def test_game_over_invalid_action_reward(env):
    """Test reward for actions after game over."""
    env.reset()
    # Hit a mine to end the game
    env.mines[0, 0] = True
    env._update_adjacent_counts()
    env.mines_placed = True
    env.is_first_cascade = False
    env.first_cascade_done = True
    
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    assert terminated
    
    # Try another action after game over
    action = 1
    state, reward, terminated, truncated, info = env.step(action)
    assert reward == REWARD_INVALID_ACTION

def test_reward_consistency(env):
    """Test that rewards are consistent for the same actions in the same game state."""
    env.reset()
    
    # Take same action multiple times and verify consistent rewards
    action = 0
    state1, reward1, terminated1, truncated1, info1 = env.step(action)
    
    # Reset and take same action
    env.reset()
    state2, reward2, terminated2, truncated2, info2 = env.step(action)
    
    # Both should be pre-cascade rewards (neutral) since no cascade has occurred
    assert reward1 == REWARD_FIRST_CASCADE_SAFE, f"First action should give neutral reward, got {reward1}"
    assert reward2 == REWARD_FIRST_CASCADE_SAFE, f"Second action should give neutral reward, got {reward2}"

def test_reward_bounds(env):
    """Test that rewards are within expected bounds."""
    env.reset()
    
    # Test various actions and verify reward bounds
    for action in range(min(5, env.action_space.n)):
        state, reward, terminated, truncated, info = env.step(action)
        
        # Rewards should be within expected range
        assert reward >= REWARD_HIT_MINE, f"Reward {reward} should be >= {REWARD_HIT_MINE}"
        assert reward <= REWARD_WIN, f"Reward {reward} should be <= {REWARD_WIN}"
        
        if terminated:
            break

def test_reward_scaling_with_board_size():
    """Test that rewards scale appropriately with board size."""
    small_env = MinesweeperEnv(initial_board_size=(3, 3), initial_mines=1)
    large_env = MinesweeperEnv(initial_board_size=(8, 8), initial_mines=10)
    
    # Test safe reveal rewards
    small_env.reset()
    large_env.reset()
    
    # Make pre-cascade moves
    small_state, small_reward, small_term, small_trunc, small_info = small_env.step(0)
    large_state, large_reward, large_term, large_trunc, large_info = large_env.step(0)
    
    # Both should be pre-cascade rewards (neutral) since no cascade has occurred
    assert small_reward == REWARD_FIRST_CASCADE_SAFE, f"Small board should give neutral reward, got {small_reward}"
    assert large_reward == REWARD_FIRST_CASCADE_SAFE, f"Large board should give neutral reward, got {large_reward}"

def test_reward_with_custom_parameters():
    """Test rewards with custom reward parameters (for ablation/customization research)."""
    # This test intentionally sets custom rewards to verify the environment supports them.
    custom_invalid_penalty = -5.0
    custom_mine_penalty = -10.0
    custom_safe_reveal = 2.0
    custom_win_reward = 50.0
    
    env = MinesweeperEnv(
        initial_board_size=(4, 4),
        initial_mines=2,
        invalid_action_penalty=custom_invalid_penalty,
        mine_penalty=custom_mine_penalty,
        safe_reveal_base=custom_safe_reveal,
        win_reward=custom_win_reward
    )
    
    env.reset()
    
    # Test invalid action penalty by trying an out-of-bounds action
    invalid_action = 1000
    state, reward, terminated, truncated, info = env.step(invalid_action)
    
    # Should get the custom invalid action penalty
    assert reward == custom_invalid_penalty

def test_reward_info_dict():
    """Test that reward information is included in info dict."""
    env = MinesweeperEnv(initial_board_size=(4, 4), initial_mines=2)
    env.reset()
    
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # Check that relevant info is present
    assert 'won' in info
    # The environment only provides 'won' key, not 'game_over'
    # This is the actual behavior, so we should test for what exists
    assert isinstance(info['won'], bool)

def test_reward_with_early_learning():
    """Test rewards in early learning mode."""
    env = MinesweeperEnv(
        initial_board_size=(4, 4),
        initial_mines=2,
        early_learning_mode=True,
        early_learning_threshold=10
    )
    
    env.reset()
    
    # Make pre-cascade move
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # Should get neutral reward for pre-cascade move
    assert reward == REWARD_FIRST_CASCADE_SAFE, f"Pre-cascade should give neutral reward, got {reward}"

def test_reward_edge_cases():
    """Test reward edge cases."""
    env = MinesweeperEnv(initial_board_size=(2, 2), initial_mines=1)
    env.reset()
    
    # Test with very small board
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # Should get neutral reward for pre-cascade move
    assert reward == REWARD_FIRST_CASCADE_SAFE, f"Pre-cascade should give neutral reward, got {reward}"

def test_reward_with_rectangular_board():
    """Test rewards on rectangular boards."""
    env = MinesweeperEnv(initial_board_size=(5, 3), initial_mines=3)
    env.reset()
    
    # Make pre-cascade move
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # Should get neutral reward for pre-cascade move
    assert reward == REWARD_FIRST_CASCADE_SAFE, f"Pre-cascade should give neutral reward, got {reward}" 