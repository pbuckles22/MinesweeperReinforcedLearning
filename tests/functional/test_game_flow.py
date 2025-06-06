import pytest
import numpy as np
from src.core.minesweeper_env import MinesweeperEnv
from src.core.constants import (
    CELL_UNREVEALED,
    CELL_MINE,
    CELL_FLAGGED,
    CELL_MINE_HIT,
    REWARD_WIN,
    REWARD_HIT_MINE
)

@pytest.fixture
def env():
    """Create a test environment."""
    return MinesweeperEnv(
        max_board_size=(10, 10),
        max_mines=10,
        initial_board_size=(4, 4),
        initial_mines=2
    )

def test_complete_game_win(env):
    """Test a complete game scenario ending in a win."""
    env.reset()
    
    # First, reveal all safe cells
    for y in range(env.current_board_height):
        for x in range(env.current_board_width):
            if not env.mines[y, x]:
                action = y * env.current_board_width + x
                state, reward, terminated, truncated, info = env.step(action)
                if terminated:  # If we hit a mine, reset and try again
                    env.reset()
                    break
    
    # Then flag all mines
    for y in range(env.current_board_height):
        for x in range(env.current_board_width):
            if env.mines[y, x]:
                action = y * env.current_board_width + x + env.current_board_width * env.current_board_height
                state, reward, terminated, truncated, info = env.step(action)
                if terminated:  # If we won, verify win condition
                    assert info['won']
                    assert reward == REWARD_WIN
                    return
    
    # If we get here, we should have won
    assert terminated
    assert info['won']
    assert reward == REWARD_WIN

def test_complete_game_loss(env):
    """Test a complete game scenario ending in a loss."""
    env.reset()
    
    # Find and hit a mine
    for y in range(env.current_board_height):
        for x in range(env.current_board_width):
            if env.mines[y, x]:
                action = y * env.current_board_width + x
                state, reward, terminated, truncated, info = env.step(action)
                assert terminated
                assert not info['won']
                assert reward == REWARD_HIT_MINE
                return

def test_game_with_flags(env):
    """Test a game scenario using flags."""
    env.reset()
    
    # First, flag all mines
    for y in range(env.current_board_height):
        for x in range(env.current_board_width):
            if env.mines[y, x]:
                action = y * env.current_board_width + x + env.current_board_width * env.current_board_height
                state, reward, terminated, truncated, info = env.step(action)
                assert not terminated
                assert state[y, x] == CELL_FLAGGED
    
    # Then reveal all safe cells
    for y in range(env.current_board_height):
        for x in range(env.current_board_width):
            if not env.mines[y, x]:
                action = y * env.current_board_width + x
                state, reward, terminated, truncated, info = env.step(action)
                if terminated:  # If we won, verify win condition
                    assert info['won']
                    assert reward == REWARD_WIN
                    return

def test_game_with_wrong_flags(env):
    """Test a game scenario with incorrect flag placements."""
    env.reset()
    
    # Flag some safe cells
    safe_cells = [(y, x) for y in range(env.current_board_height) 
                 for x in range(env.current_board_width) 
                 if not env.mines[y, x]][:2]  # Flag 2 safe cells
    
    for y, x in safe_cells:
        action = y * env.current_board_width + x + env.current_board_width * env.current_board_height
        state, reward, terminated, truncated, info = env.step(action)
        assert not terminated
        assert state[y, x] == CELL_FLAGGED
    
    # Try to win the game
    for y in range(env.current_board_height):
        for x in range(env.current_board_width):
            if not env.mines[y, x] and (y, x) not in safe_cells:
                action = y * env.current_board_width + x
                state, reward, terminated, truncated, info = env.step(action)
                if terminated:  # If we won, verify win condition
                    assert info['won']
                    assert reward == REWARD_WIN
                    return 