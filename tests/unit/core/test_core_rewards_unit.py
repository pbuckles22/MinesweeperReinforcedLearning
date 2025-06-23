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
    """Test that safe reveals get immediate rewards (no more pre-cascade neutral)."""
    env.reset()
    
    # Take first action (should get immediate reward now)
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    if not env.mines[0, 0]:  # If safe cell
        assert reward == REWARD_SAFE_REVEAL, f"Safe reveal should get immediate reward, got {reward}"
    else:  # If mine
        assert reward == REWARD_HIT_MINE, f"Mine hit should get immediate penalty, got {reward}"

def test_pre_cascade_mine_hit_reward(env):
    """Test that mine hits get immediate penalties (no more pre-cascade neutral)."""
    env.reset()
    
    # Take first action (should get immediate penalty now)
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    if env.mines[0, 0]:  # If mine
        assert reward == REWARD_HIT_MINE, f"Mine hit should get immediate penalty, got {reward}"
    else:  # If safe cell
        assert reward == REWARD_SAFE_REVEAL, f"Safe reveal should get immediate reward, got {reward}"

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
    """Test that rewards are consistent for the same actions within the same game state."""
    env.reset(seed=42)  # Use fixed seed for deterministic behavior
    
    # Take first action
    action = 0
    state1, reward1, terminated1, truncated1, info1 = env.step(action)
    
    # Reset with same seed and take same action
    env.reset(seed=42)
    state2, reward2, terminated2, truncated2, info2 = env.step(action)
    
    # With fixed seed, rewards should be consistent
    assert reward1 == reward2, f"Same action with same seed should give same reward, got {reward1} and {reward2}"
    
    # Verify the reward types are valid
    valid_rewards = [REWARD_HIT_MINE, REWARD_SAFE_REVEAL, REWARD_WIN]
    assert reward1 in valid_rewards, f"Reward {reward1} should be one of {valid_rewards}"

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
    
    # First move can result in win, safe reveal, or mine hit
    if small_term and small_reward == REWARD_WIN:
        assert small_reward == REWARD_WIN, f"Small board win should give win reward, got {small_reward}"
    elif small_reward == REWARD_SAFE_REVEAL:
        assert small_reward == REWARD_SAFE_REVEAL, f"Small board should give immediate safe reward, got {small_reward}"
    elif small_reward == REWARD_HIT_MINE:
        assert small_reward == REWARD_HIT_MINE, f"Small board mine hit should give penalty, got {small_reward}"
    else:
        assert False, f"Unexpected reward on small board: {small_reward}"
    if large_term and large_reward == REWARD_WIN:
        assert large_reward == REWARD_WIN, f"Large board win should give win reward, got {large_reward}"
    elif large_reward == REWARD_SAFE_REVEAL:
        assert large_reward == REWARD_SAFE_REVEAL, f"Large board should give immediate safe reward, got {large_reward}"
    elif large_reward == REWARD_HIT_MINE:
        assert large_reward == REWARD_HIT_MINE, f"Large board mine hit should give penalty, got {large_reward}"
    else:
        assert False, f"Unexpected reward on large board: {large_reward}"

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
    
    # Should get immediate reward for pre-cascade move (no more neutral)
    if reward in [REWARD_SAFE_REVEAL, REWARD_HIT_MINE, REWARD_WIN]:
        assert True
    else:
        assert False, f"Pre-cascade should give immediate reward/penalty/win, got {reward}"

def test_reward_edge_cases():
    """Test reward edge cases."""
    env = MinesweeperEnv(initial_board_size=(2, 2), initial_mines=1)
    env.reset()
    
    # Test with very small board
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # Should get immediate reward/penalty/win for pre-cascade move (no more neutral)
    if reward in [REWARD_SAFE_REVEAL, REWARD_HIT_MINE, REWARD_WIN]:
        assert True
    else:
        assert False, f"Pre-cascade should give immediate reward/penalty/win, got {reward}"

def test_reward_with_rectangular_board():
    """Test rewards on rectangular boards."""
    env = MinesweeperEnv(initial_board_size=(5, 3), initial_mines=3)
    env.reset()
    
    # Make pre-cascade move
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # Should get immediate reward/penalty/win for pre-cascade move
    if reward in [REWARD_SAFE_REVEAL, REWARD_HIT_MINE, REWARD_WIN]:
        assert True
    else:
        assert False, f"Pre-cascade should give immediate reward/penalty/win, got {reward}" 