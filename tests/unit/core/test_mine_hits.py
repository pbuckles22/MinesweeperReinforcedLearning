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
    REWARD_HIT_MINE
)

def test_mine_hit_reward():
    env = MinesweeperEnv()
    env.reset()
    env.mines[0, 0] = True
    env.is_first_move = False
    state, reward, terminated, truncated, info = env.step(0)
    assert reward == REWARD_HIT_MINE

def test_mine_hit_state():
    env = MinesweeperEnv()
    env.reset()
    env.mines[0, 0] = True
    env.is_first_move = False
    state, reward, terminated, truncated, info = env.step(0)
    assert state[0, 0] == CELL_MINE_HIT

def test_mine_hit_game_over(env):
    """Test that hitting a mine on a subsequent move ends the game."""
    env.reset()
    
    # Place mine at (1,1)
    env.mines[1, 1] = True
    env._update_adjacent_counts()
    
    # First make a safe move to get past first move
    safe_action = 0 * env.current_board_width + 0  # Move at (0,0)
    state, reward, terminated, truncated, info = env.step(safe_action)
    assert not terminated, "Safe move should not terminate the game"
    
    # Now hit the mine
    mine_action = 1 * env.current_board_width + 1  # Move at (1,1)
    state, reward, terminated, truncated, info = env.step(mine_action)
    
    # Check that game is terminated
    assert terminated, "Game should terminate after hitting a mine"
    assert not truncated
    assert state[1, 1] == CELL_MINE_HIT
    assert reward == REWARD_HIT_MINE

def test_first_move_mine_hit_reset(env):
    """Test that hitting a mine on first move resets the game."""
    # Place mine at (1,1)
    env.mines[1, 1] = True
    env._update_adjacent_counts()
    
    # Reveal mine on first move
    action = 1 * env.current_board_width + 1
    state, reward, terminated, truncated, info = env.step(action)
    
    # Check that game is reset
    assert reward == REWARD_FIRST_MOVE_HIT_MINE
    assert not terminated
    assert np.all(state == CELL_UNREVEALED)

def test_first_move_behavior(env):
    """Test that first move behavior is correct."""
    # Place mine at (1,1)
    env.mines[1, 1] = True
    env._update_adjacent_counts()
    
    # Reveal safe cell on first move
    action = 0 * env.current_board_width + 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # Check that game continues
    assert reward == REWARD_FIRST_MOVE_SAFE
    assert not terminated
    assert state[0, 0] != CELL_UNREVEALED 