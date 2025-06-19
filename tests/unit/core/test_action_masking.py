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
    return MinesweeperEnv(initial_board_size=3, initial_mines=1)

def test_reveal_already_revealed_cell(env):
    """Test that revealing an already revealed cell is invalid."""
    # Reveal a cell
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # If the game ended on first move (won), we can't test revealing the same cell again
    if terminated:
        # Game won on first move, which is valid behavior
        assert info.get('won', False)
        return
    
    # Try to reveal the same cell again
    state, reward, terminated, truncated, info = env.step(action)
    
    assert reward < 0  # Should get negative reward for invalid action
    assert not terminated
    assert not truncated
    assert 'won' in info

def test_reveal_flagged_cell(env):
    """Test that revealing a flagged cell is invalid."""
    # Flag a cell
    action = env.current_board_width * env.current_board_height  # First flag action
    state, reward, terminated, truncated, info = env.step(action)
    
    # Try to reveal the flagged cell
    reveal_action = 0  # First reveal action
    state, reward, terminated, truncated, info = env.step(reveal_action)
    
    assert reward < 0  # Should get negative reward for invalid action
    assert not terminated
    assert not truncated
    assert 'won' in info

def test_flag_revealed_cell(env):
    """Test that flagging a revealed cell is invalid."""
    # Reveal a cell
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # If the game ended on first move (won), we can't test flagging the revealed cell
    if terminated:
        # Game won on first move, which is valid behavior
        assert info.get('won', False)
        return
    
    # Try to flag the revealed cell
    flag_action = env.current_board_width * env.current_board_height  # First flag action
    state, reward, terminated, truncated, info = env.step(flag_action)
    
    assert reward < 0  # Should get negative reward for invalid action
    assert not terminated
    assert not truncated
    assert 'won' in info

def test_flag_already_flagged_cell(env):
    """Test that flagging an already flagged cell is invalid."""
    # Flag a cell
    flag_action = env.current_board_width * env.current_board_height  # First flag action
    state, reward, terminated, truncated, info = env.step(flag_action)
    
    # Try to flag the same cell again
    state, reward, terminated, truncated, info = env.step(flag_action)
    
    assert reward < 0  # Should get negative reward for invalid action
    assert not terminated
    assert not truncated
    assert 'won' in info

def test_reveal_after_game_over(env):
    """Test that revealing after game over is invalid."""
    # First, make a safe move to get past first move
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # If the game ended on first move (won), we need to reset and try a different approach
    if terminated:
        # Reset and try to create a scenario where we can hit a mine
        env.reset()
        # Place mine at (0,0) and make sure it's not the first move
        env.mines[0, 0] = True
        env._update_adjacent_counts()
        # Make a safe move first
        safe_action = 1  # Try a different cell
        state, reward, terminated, truncated, info = env.step(safe_action)
        if terminated:
            # Still won, skip this test
            return
    
    # Place mine at (0,0) and hit it
    env.mines[0, 0] = True
    env._update_adjacent_counts()
    
    # Hit the mine (this should terminate the game)
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # Try to reveal another cell
    action = 1
    state, reward, terminated, truncated, info = env.step(action)
    
    assert reward < 0  # Should get negative reward for invalid action
    assert terminated  # Game should still be terminated
    assert not truncated
    assert 'won' in info

def test_action_masking_revealed_cells(env):
    """Test that revealed cells are masked."""
    # Reveal a cell
    action = 0  # Reveal top-left cell
    state, reward, terminated, truncated, info = env.step(action)
    
    # Check that the revealed cell is masked
    assert not env.action_masks[action]
    assert not env.action_masks[action + env.current_board_width * env.current_board_height]  # Flag action

def test_action_masking_flagged_cells(env):
    """Test that flagged cells are masked."""
    # Flag a cell
    action = env.current_board_width * env.current_board_height  # Flag top-left cell
    state, reward, terminated, truncated, info = env.step(action)
    
    # Check that the flagged cell is masked
    assert not env.action_masks[action]
    assert not env.action_masks[action - env.current_board_width * env.current_board_height]  # Reveal action

def test_action_masking_game_over(env):
    """Test that all actions are masked when game is over."""
    # First, make a safe move to get past first move
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # If the game ended on first move (won), we need to reset and try a different approach
    if terminated:
        # Reset and try to create a scenario where we can hit a mine
        env.reset()
        # Place mine at (0,0) and make sure it's not the first move
        env.mines[0, 0] = True
        env._update_adjacent_counts()
        # Make a safe move first
        safe_action = 1  # Try a different cell
        state, reward, terminated, truncated, info = env.step(safe_action)
        if terminated:
            # Still won, skip this test
            return
    
    # Now we need to create a scenario where hitting a mine will terminate the game (not first move)
    # First, make sure we're past the first move by making a safe move
    if env.is_first_move:
        # Find a safe cell to reveal first
        for i in range(env.current_board_height):
            for j in range(env.current_board_width):
                if not env.mines[i, j]:
                    safe_action = i * env.current_board_width + j
                    state, reward, terminated, truncated, info = env.step(safe_action)
                    if terminated:
                        # Game won, skip this test
                        return
                    break
            if not env.is_first_move:
                break
    
    # Now place a mine and hit it (this should terminate since it's not the first move)
    env.mines[0, 0] = True
    env._update_adjacent_counts()
    
    # Hit mine (this should terminate the game)
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # Check that all actions are masked
    assert all(not mask for mask in env.action_masks) 