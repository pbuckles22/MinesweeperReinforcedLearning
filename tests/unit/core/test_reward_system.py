"""
Test suite for the reward system.
Tests all reward types, scaling, and edge cases.
"""

import pytest
import numpy as np
from src.core.minesweeper_env import MinesweeperEnv
from src.core.constants import (
    CELL_UNREVEALED,
    CELL_MINE,
    CELL_FLAGGED,
    CELL_MINE_HIT,
    REWARD_FIRST_MOVE_SAFE,
    REWARD_FIRST_MOVE_HIT_MINE,
    REWARD_SAFE_REVEAL,
    REWARD_WIN,
    REWARD_HIT_MINE,
    REWARD_FLAG_PLACED,
    REWARD_FLAG_REMOVED,
    REWARD_INVALID_ACTION
)

@pytest.fixture
def env():
    """Create a test environment."""
    return MinesweeperEnv(
        initial_board_size=(4, 4),
        initial_mines=2,
        invalid_action_penalty=-1.0,
        mine_penalty=-2.0,
        flag_mine_reward=1.0,
        flag_safe_penalty=-0.5,
        unflag_penalty=-0.1,
        safe_reveal_base=0.1,
        win_reward=10.0
    )

def test_first_move_safe_reward(env):
    """Test reward for safe first move."""
    env.reset()
    
    # Find a safe cell for first move
    safe_found = False
    for y in range(env.current_board_height):
        for x in range(env.current_board_width):
            if not env.mines[y, x]:
                action = y * env.current_board_width + x
                state, reward, terminated, truncated, info = env.step(action)
                safe_found = True
                break
        if safe_found:
            break
    
    assert safe_found
    assert reward == REWARD_FIRST_MOVE_SAFE
    assert not terminated

def test_first_move_mine_hit_reward(env):
    """Test reward for hitting mine on first move."""
    env.reset()
    
    # Place mine at (0,0) and hit it
    env.mines[0, 0] = True
    env._update_adjacent_counts()
    
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    assert reward == REWARD_FIRST_MOVE_HIT_MINE
    assert terminated
    assert not info.get('won', False)

def test_safe_reveal_reward(env):
    """Test reward for revealing safe cells after first move."""
    env.reset()
    
    # Make first move to get past first move
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    if terminated:
        # Game won on first move, can't test further
        return
    
    # Find another safe cell to reveal
    for y in range(env.current_board_height):
        for x in range(env.current_board_width):
            if not env.mines[y, x] and state[y, x] == CELL_UNREVEALED:
                action = y * env.current_board_width + x
                state, reward, terminated, truncated, info = env.step(action)
                assert reward == REWARD_SAFE_REVEAL
                break

def test_mine_hit_reward(env):
    """Test reward for hitting mine after first move."""
    env.reset()
    
    # Make first move to get past first move
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    if terminated:
        # Game won on first move, reset and try different approach
        env.reset()
        # Place mine at (0,0) and make sure it's not the first move
        env.mines[0, 0] = True
        env._update_adjacent_counts()
        # Make a safe move first
        safe_action = 1
        state, reward, terminated, truncated, info = env.step(safe_action)
        if terminated:
            # Still won, skip this test
            return
    
    # Place mine and hit it
    env.mines[0, 0] = True
    env._update_adjacent_counts()
    
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    assert reward == REWARD_HIT_MINE
    assert terminated
    assert not info.get('won', False)

def test_flag_placement_reward(env):
    """Test reward for flagging a mine."""
    env.reset()
    
    # Find a mine to flag
    mine_found = False
    for y in range(env.current_board_height):
        for x in range(env.current_board_width):
            if env.mines[y, x]:
                action = y * env.current_board_width + x + env.current_board_width * env.current_board_height
                state, reward, terminated, truncated, info = env.step(action)
                mine_found = True
                assert reward == REWARD_FLAG_PLACED
                break
        if mine_found:
            break
    
    assert mine_found

def test_flag_safe_cell_penalty(env):
    """Test penalty for flagging a safe cell."""
    env.reset()
    
    # Find a safe cell to flag
    safe_found = False
    for y in range(env.current_board_height):
        for x in range(env.current_board_width):
            if not env.mines[y, x]:
                action = y * env.current_board_width + x + env.current_board_width * env.current_board_height
                state, reward, terminated, truncated, info = env.step(action)
                safe_found = True
                assert reward < 0  # Should be negative
                break
        if safe_found:
            break
    
    assert safe_found

def test_flag_removal_reward(env):
    """Test reward for removing a flag."""
    env.reset()
    
    # Flag a cell first
    action = env.current_board_width * env.current_board_height
    state, reward, terminated, truncated, info = env.step(action)
    
    # Remove the flag
    state, reward, terminated, truncated, info = env.step(action)
    
    assert reward == REWARD_FLAG_REMOVED

def test_win_reward(env):
    """Test reward for winning the game."""
    env.reset()
    
    # Win the game by revealing all safe cells
    for y in range(env.current_board_height):
        for x in range(env.current_board_width):
            if not env.mines[y, x]:
                action = y * env.current_board_width + x
                state, reward, terminated, truncated, info = env.step(action)
                if terminated and info.get('won', False):
                    assert reward == REWARD_WIN
                    break

def test_invalid_action_penalty(env):
    """Test penalty for invalid actions."""
    env.reset()
    
    # Reveal a cell
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    if terminated:
        # Game won on first move, can't test invalid action
        return
    
    # Try to reveal the same cell again (invalid)
    state, reward, terminated, truncated, info = env.step(action)
    
    assert reward < 0  # Should be negative penalty
    assert not terminated

def test_reward_scaling_with_board_size():
    """Test that rewards scale appropriately with board size."""
    small_env = MinesweeperEnv(initial_board_size=(3, 3), initial_mines=1)
    large_env = MinesweeperEnv(initial_board_size=(8, 8), initial_mines=10)
    
    # Test safe reveal rewards
    small_env.reset()
    large_env.reset()
    
    # Make first moves
    small_state, small_reward, small_term, small_trunc, small_info = small_env.step(0)
    large_state, large_reward, large_term, large_trunc, large_info = large_env.step(0)
    
    # Rewards should be the same for first move regardless of board size
    assert small_reward == large_reward

def test_reward_consistency():
    """Test that rewards are consistent across multiple games."""
    env = MinesweeperEnv(initial_board_size=(4, 4), initial_mines=2)
    
    rewards = []
    for _ in range(5):
        env.reset()
        action = 0
        state, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
    
    # All first move rewards should be the same
    assert len(set(rewards)) == 1

def test_reward_with_custom_parameters():
    """Test rewards with custom reward parameters."""
    env = MinesweeperEnv(
        initial_board_size=(4, 4),
        initial_mines=2,
        invalid_action_penalty=-5.0,
        mine_penalty=-10.0,
        flag_mine_reward=5.0,
        flag_safe_penalty=-2.0,
        unflag_penalty=-1.0,
        safe_reveal_base=2.0,
        win_reward=50.0
    )
    
    env.reset()
    
    # Test invalid action penalty
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    if not terminated:
        # Try invalid action
        state, reward, terminated, truncated, info = env.step(action)
        assert reward == -5.0

def test_reward_info_dict():
    """Test that reward information is included in info dict."""
    env = MinesweeperEnv(initial_board_size=(4, 4), initial_mines=2)
    env.reset()
    
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # Check that relevant info is present
    assert 'won' in info
    assert 'game_over' in info

def test_reward_with_early_learning():
    """Test rewards in early learning mode."""
    env = MinesweeperEnv(
        initial_board_size=(4, 4),
        initial_mines=2,
        early_learning_mode=True,
        early_learning_threshold=10
    )
    
    env.reset()
    
    # Make first move
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # Should get normal first move reward even in early learning mode
    assert reward == REWARD_FIRST_MOVE_SAFE or reward == REWARD_FIRST_MOVE_HIT_MINE

def test_reward_edge_cases():
    """Test reward edge cases."""
    env = MinesweeperEnv(initial_board_size=(2, 2), initial_mines=1)
    env.reset()
    
    # Test with very small board
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # Should get appropriate reward
    assert reward in [REWARD_FIRST_MOVE_SAFE, REWARD_FIRST_MOVE_HIT_MINE, REWARD_WIN]

def test_reward_with_rectangular_board():
    """Test rewards on rectangular boards."""
    env = MinesweeperEnv(initial_board_size=(3, 5), initial_mines=3)
    env.reset()
    
    # Make first move
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # Should get normal first move reward
    assert reward in [REWARD_FIRST_MOVE_SAFE, REWARD_FIRST_MOVE_HIT_MINE, REWARD_WIN] 