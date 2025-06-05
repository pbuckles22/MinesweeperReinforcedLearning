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
    return env

def test_reveal_already_revealed_cell(env):
    """Test that revealing an already revealed cell returns invalid action penalty."""
    # First reveal a cell
    state, reward, terminated, truncated, info = env.step(0)
    
    # If this is the first move, check first move behavior
    if env.is_first_move:
        assert not terminated
        assert reward == 0
    else:
        # Try to reveal the same cell again
        state, reward, terminated, truncated, info = env.step(0)
        assert reward == env.invalid_action_penalty
        assert not terminated
        assert 'invalid_action' in info['reward_breakdown']

def test_reveal_flagged_cell(env):
    """Test that revealing a flagged cell returns invalid action penalty."""
    # First flag a cell
    action = env.current_board_size * env.current_board_size  # First flag action
    state, reward, terminated, truncated, info = env.step(action)
    
    # Then try to reveal it
    state, reward, terminated, truncated, info = env.step(0)
    assert reward == env.invalid_action_penalty
    assert not terminated
    assert 'invalid_action' in info['reward_breakdown']

def test_flag_revealed_cell(env):
    """Test that flagging a revealed cell returns invalid action penalty."""
    # Reveal a cell
    state, reward, terminated, truncated, info = env.step(0)
    
    # If first move hit a mine and was a forced guess, try another cell
    if 'forced_guess_mine_hit_reset' in info['reward_breakdown']:
        state, reward, terminated, truncated, info = env.step(1)
    
    # Try to flag the revealed cell
    flag_action = env.current_board_size * env.current_board_size  # First flag action
    state, reward, terminated, truncated, info = env.step(flag_action)
    
    # Should get invalid action penalty
    assert reward == env.invalid_action_penalty
    assert not terminated
    assert 'invalid_action' in info['reward_breakdown']

def test_flag_already_flagged_cell(env):
    """Test that flagging an already flagged cell removes the flag."""
    # Flag a cell
    flag_action = env.current_board_size * env.current_board_size  # First flag action
    state, reward, terminated, truncated, info = env.step(flag_action)
    
    # Try to flag the same cell again
    state, reward, terminated, truncated, info = env.step(flag_action)
    
    # Should remove the flag
    assert reward == env.unflag_penalty
    assert not terminated
    assert 'unflag' in info['reward_breakdown']
    assert not env.flags[0, 0]  # Flag should be removed

def test_reveal_after_game_over(env):
    """Test that revealing a cell after game over returns invalid action penalty."""
    # Force game over by hitting a mine
    state, reward, terminated, truncated, info = env.step(0)

    # If first move didn't hit mine, try another cell
    if not terminated:
        state, reward, terminated, truncated, info = env.step(1)

    # If still not terminated, try one more cell
    if not terminated:
        state, reward, terminated, truncated, info = env.step(2)

    # If this is the first move, the game should be reset
    if env.is_first_move:
        assert not terminated
        assert reward == 0
    else:
        assert terminated  # Game should be over
        # Try to reveal another cell
        state, reward, terminated, truncated, info = env.step(3)
        assert reward == env.invalid_action_penalty
        assert terminated  # Game should still be over 