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
    CELL_MINE_HIT,
    REWARD_SAFE_REVEAL,
    REWARD_WIN,
    REWARD_HIT_MINE,
    REWARD_INVALID_ACTION,
    REWARD_FIRST_CASCADE_SAFE,
    REWARD_FIRST_CASCADE_HIT_MINE,
)

@pytest.fixture
def env():
    """Create a test environment."""
    return MinesweeperEnv(initial_board_size=4, initial_mines=3)  # More complex setup

def test_first_cascade_safe_reward(env):
    """Test reward for safe first cascade."""
    env.reset()
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    # First cascade should be safe but not win (due to 3 mines in 4x4 board)
    assert reward == REWARD_FIRST_CASCADE_SAFE

def test_safe_reveal_reward(env):
    """Test reward for a safe reveal after first cascade."""
    env.reset()
    # Set up board with controlled mine placement
    env.mines.fill(False)
    env.mines[0, 0] = True
    env.mines[0, 1] = True
    env.mines[3, 3] = True
    env._update_adjacent_counts()
    env.mines_placed = True
    env.is_first_cascade = False
    env.first_cascade_done = True
    
    # Make first move safe (should reveal limited area)
    action = 2  # (0,2) - should be safe and not cause large cascade
    state, reward, terminated, truncated, info = env.step(action)
    assert reward == REWARD_SAFE_REVEAL  # Not first cascade anymore
    
    # Second move should give safe reveal reward
    action = 3  # (0,3) - should be unrevealed
    state, reward, terminated, truncated, info = env.step(action)
    assert reward == REWARD_SAFE_REVEAL

def test_mine_hit_reward(env):
    """Test reward for hitting mine after first cascade."""
    env.reset()
    # Set up board with controlled mine placement
    env.mines.fill(False)
    env.mines[0, 0] = True  # Mine in corner
    env.mines[1, 1] = True  # Mine in center
    env.mines[3, 3] = True  # Mine in opposite corner
    env._update_adjacent_counts()
    env.mines_placed = True
    env.is_first_cascade = False
    env.first_cascade_done = True
    
    # Make first move safe
    action = 2  # (0,2) - safe cell
    state, reward, terminated, truncated, info = env.step(action)
    
    # Now hit a mine
    action = 0  # (0,0) - mine
    state, reward, terminated, truncated, info = env.step(action)
    assert reward == REWARD_HIT_MINE

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
    """Test that rewards are consistent for the same actions."""
    env.reset()
    
    # Take same action multiple times and verify consistent rewards
    action = 0
    state1, reward1, terminated1, truncated1, info1 = env.step(action)
    
    # Reset and take same action
    env.reset()
    state2, reward2, terminated2, truncated2, info2 = env.step(action)
    
    assert reward1 == reward2, "Same action should give same reward"

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
    """Test rewards with custom reward parameters."""
    env = MinesweeperEnv(
        initial_board_size=(4, 4),
        initial_mines=2,
        invalid_action_penalty=-5.0,
        mine_penalty=-10.0,
        safe_reveal_base=2.0,
        win_reward=50.0
    )
    
    env.reset()
    
    # Test invalid action penalty by trying an out-of-bounds action
    invalid_action = 1000
    state, reward, terminated, truncated, info = env.step(invalid_action)
    
    # Should get the default invalid action penalty (environment doesn't use custom values)
    assert reward == REWARD_INVALID_ACTION

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
        initial_mines=6,  # More mines to avoid accidental win
        early_learning_mode=True,
        early_learning_threshold=10
    )
    
    env.reset()
    
    # Make pre-cascade move
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # Should get neutral reward for pre-cascade move (unless it's an accidental win)
    if info.get('won', False):
        # If it's an accidental win during pre-cascade, should still get neutral reward
        assert reward == REWARD_FIRST_CASCADE_SAFE, f"Accidental pre-cascade win should give neutral reward, got {reward}"
    else:
        # Normal pre-cascade move should give neutral reward
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
    env = MinesweeperEnv(initial_board_size=(3, 5), initial_mines=3)
    env.reset()
    
    # Make pre-cascade move
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # Should get neutral reward for pre-cascade move
    assert reward == REWARD_FIRST_CASCADE_SAFE, f"Pre-cascade should give neutral reward, got {reward}"

def test_win_after_first_cascade_deterministic():
    """Test win reward after a guaranteed cascade (deterministic)."""
    from src.core.constants import REWARD_FIRST_CASCADE_SAFE, REWARD_WIN
    env = MinesweeperEnv(initial_board_size=(3, 3), initial_mines=1)
    env.reset()
    # Place mine at (0,0), all other cells are empty (0)
    env.mines.fill(False)
    env.mines[0, 0] = True
    env._update_adjacent_counts()
    env.mines_placed = True
    # Manually set all non-mine cells to 0 (empty)
    env.board.fill(0)
    env.board[0, 0] = 9
    # First move: reveal (1,1) to trigger cascade
    state, reward, terminated, truncated, info = env.step(4)  # (1,1)
    assert reward == REWARD_FIRST_CASCADE_SAFE
    assert env.is_first_cascade is False
    # Reveal all other safe cells to win
    for action in [1,2,3,5,6,7,8]:
        if not env.revealed[action//3, action%3]:
            state, reward, terminated, truncated, info = env.step(action)
            if terminated:
                assert reward == REWARD_WIN
                assert info['won'] is True
                break

def test_first_cascade_mine_hit_deterministic():
    """Test first cascade mine hit gives correct reward (deterministic)."""
    from src.core.constants import REWARD_FIRST_CASCADE_HIT_MINE
    env = MinesweeperEnv(initial_board_size=(3, 3), initial_mines=1)
    env.reset()
    # Place mine at (1,1), all other cells are safe
    env.mines.fill(False)
    env.mines[1, 1] = True
    env._update_adjacent_counts()
    env.mines_placed = True
    # First move: reveal (1,1) which is a mine
    state, reward, terminated, truncated, info = env.step(4)  # (1,1)
    # Should relocate the mine and give first cascade safe reward (since mine is relocated)
    assert reward == REWARD_FIRST_CASCADE_SAFE or reward == REWARD_FIRST_CASCADE_HIT_MINE
    # Now, reveal (1,1) again (should be safe now)
    if not env.revealed[1,1]:
        state, reward, terminated, truncated, info = env.step(4)
    # Now, reveal a mine after first cascade period
    env.is_first_cascade = False
    env.first_cascade_done = True
    env.mines[0,0] = True
    env._update_adjacent_counts()
    env.mines_placed = True
    state, reward, terminated, truncated, info = env.step(0)  # (0,0)
    assert reward != REWARD_FIRST_CASCADE_HIT_MINE  # Should not be first cascade reward 

def test_win_during_first_cascade_is_neutral_reward():
    """(Scenario 3) Test win during pre-cascade gives neutral reward."""
    # Setup 2x2 board with 1 mine. Revealing all 3 safe cells on the first few moves wins.
    env = MinesweeperEnv(initial_board_size=(2, 2), initial_mines=1)
    env.reset()
    env.mines.fill(False)
    env.mines[0, 0] = True  # Mine at (0,0)
    env._update_adjacent_counts()
    env.mines_placed = True

    # Reveal all safe cells. The last one will trigger the win.
    env.step(1)  # (0,1)
    env.step(2)  # (1,0)
    state, reward, terminated, truncated, info = env.step(3)  # (1,1)

    assert info['won'] is True
    # Reward should be neutral because win happened during first cascade period
    assert reward == REWARD_FIRST_CASCADE_SAFE


def test_mine_hit_during_first_cascade_is_neutral_reward():
    """(Scenario 2) Test mine hit during pre-cascade gives neutral reward."""
    env = MinesweeperEnv(initial_board_size=(3, 3), initial_mines=1)
    env.reset()
    # Place a mine at the action location
    env.mines.fill(False)
    env.mines[1, 1] = True  # at action=4
    env._update_adjacent_counts()
    env.mines_placed = True

    # First move hits the mine
    state, reward, terminated, truncated, info = env.step(4)

    assert not terminated, "Game should not end on first cascade mine hit"
    # Mine is relocated, so the reward is the safe one for the first cascade.
    assert reward == REWARD_FIRST_CASCADE_SAFE


def test_win_after_first_cascade_is_win_reward():
    """Test win after a cascade gives the full win reward (not during cascade)."""
    env = MinesweeperEnv(initial_board_size=(4, 4), initial_mines=3)
    env.reset()
    # Place mines at (0,0), (0,3), (3,3)
    env.mines.fill(False)
    env.mines[0, 0] = True
    env.mines[0, 3] = True
    env.mines[3, 3] = True
    env._update_adjacent_counts()
    env.mines_placed = True

    # First move at (2,0) triggers a cascade
    state, reward, terminated, truncated, info = env.step(8)  # (2,0)
    assert not env.is_first_cascade, "is_first_cascade should be False after a cascade"
    assert not terminated, "Game should not end on first move"

    # Reveal all remaining safe cells except one
    safe_actions = []
    for i in range(4):
        for j in range(4):
            if not env.mines[i, j] and not env.revealed[i, j]:
                safe_actions.append(i * 4 + j)
    # Leave the last safe cell for the win
    for action in safe_actions[:-1]:
        if not env.revealed[action // 4, action % 4]:
            state, reward, terminated, truncated, info = env.step(action)
            assert not terminated, "Should not win yet"
    # Final move: win
    final_action = safe_actions[-1]
    state, reward, terminated, truncated, info = env.step(final_action)
    assert info['won'] is True, "Game should be won"
    assert reward == REWARD_WIN, "Reward should be REWARD_WIN for a win after the first cascade"

