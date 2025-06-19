"""
Test suite for early learning mode functionality.
Tests corner safety, edge safety, threshold behavior, and transitions.
"""

import pytest
import numpy as np
from src.core.minesweeper_env import MinesweeperEnv
from src.core.constants import (
    CELL_UNREVEALED,
    CELL_MINE,
    CELL_FLAGGED,
    CELL_MINE_HIT,
    REWARD_SAFE_REVEAL,
    REWARD_WIN,
    REWARD_HIT_MINE
)

@pytest.fixture
def early_learning_env():
    """Create a test environment with early learning mode enabled."""
    return MinesweeperEnv(
        initial_board_size=(4, 4),
        initial_mines=2,
        early_learning_mode=True,
        early_learning_threshold=200,
        early_learning_corner_safe=True,
        early_learning_edge_safe=True
    )

def test_early_learning_initialization(early_learning_env):
    """Test that early learning mode is properly initialized."""
    assert early_learning_env.early_learning_mode is True
    assert early_learning_env.early_learning_threshold == 200
    assert early_learning_env.early_learning_corner_safe is True
    assert early_learning_env.early_learning_edge_safe is True
    assert early_learning_env.current_board_width == 4
    assert early_learning_env.current_board_height == 4
    assert early_learning_env.current_mines == 2

def test_corner_safety(early_learning_env):
    """Test that corners are safe when corner_safe is enabled."""
    early_learning_env.reset()
    
    # Check that corners don't contain mines
    corners = [(0, 0), (0, 3), (3, 0), (3, 3)]
    for row, col in corners:
        assert not early_learning_env.mines[row, col], f"Corner ({row}, {col}) contains a mine"

def test_edge_safety(early_learning_env):
    """Test that edges are safe when edge_safe is enabled."""
    early_learning_env.reset()
    
    # Check that edges don't contain mines
    edges = []
    # Top and bottom edges
    for col in range(4):
        edges.extend([(0, col), (3, col)])
    # Left and right edges (excluding corners already checked)
    for row in range(1, 3):
        edges.extend([(row, 0), (row, 3)])
    
    for row, col in edges:
        assert not early_learning_env.mines[row, col], f"Edge ({row}, {col}) contains a mine"

def test_early_learning_disabled():
    """Test that early learning mode can be disabled."""
    env = MinesweeperEnv(
        initial_board_size=(4, 4),
        initial_mines=2,
        early_learning_mode=False,
        early_learning_corner_safe=False,
        early_learning_edge_safe=False
    )
    
    assert env.early_learning_mode is False
    assert env.early_learning_corner_safe is False
    assert env.early_learning_edge_safe is False

def test_threshold_behavior(early_learning_env):
    """Test that early learning mode respects the threshold."""
    # Simulate games up to threshold
    for game in range(200):
        early_learning_env.reset()
        
        # Play a quick game (just make one move)
        action = 0
        state, reward, terminated, truncated, info = early_learning_env.step(action)
        
        # Check if we're still in early learning mode
        if game < 200:
            # Should still be in early learning mode
            assert early_learning_env.early_learning_mode is True
        else:
            # Should transition out of early learning mode
            assert early_learning_env.early_learning_mode is False

def test_parameter_updates(early_learning_env):
    """Test that parameters update correctly during early learning."""
    initial_width = early_learning_env.current_board_width
    initial_height = early_learning_env.current_board_height
    initial_mines = early_learning_env.current_mines
    
    # Simulate successful games to trigger progression
    for _ in range(50):
        early_learning_env.reset()
        # Win the game by revealing all safe cells
        for y in range(early_learning_env.current_board_height):
            for x in range(early_learning_env.current_board_width):
                if not early_learning_env.mines[y, x]:
                    action = y * early_learning_env.current_board_width + x
                    state, reward, terminated, truncated, info = early_learning_env.step(action)
                    if terminated and info.get('won', False):
                        break
            if terminated:
                break
    
    # Check if parameters have been updated
    assert (early_learning_env.current_board_width > initial_width or
            early_learning_env.current_board_height > initial_height or
            early_learning_env.current_mines > initial_mines)

def test_state_preservation(early_learning_env):
    """Test that state is preserved correctly during early learning."""
    early_learning_env.reset()
    
    # Make some moves
    action = 0
    state, reward, terminated, truncated, info = early_learning_env.step(action)
    
    # Check that state is properly updated
    assert state[0, 0] != CELL_UNREVEALED
    assert not terminated or info.get('won', False)
    
    # Reset and check state is cleared
    early_learning_env.reset()
    assert np.all(early_learning_env.state == CELL_UNREVEALED)
    assert np.all(early_learning_env.flags == 0)

def test_transition_out_of_early_learning(early_learning_env):
    """Test transition out of early learning mode."""
    # Set threshold to a low value for testing
    early_learning_env.early_learning_threshold = 5
    
    # Play games until threshold is reached
    for _ in range(6):
        early_learning_env.reset()
        # Make one move to simulate a game
        action = 0
        state, reward, terminated, truncated, info = early_learning_env.step(action)
    
    # Should have transitioned out of early learning mode
    assert early_learning_env.early_learning_mode is False

def test_early_learning_with_large_board():
    """Test early learning mode with larger initial board."""
    env = MinesweeperEnv(
        initial_board_size=(6, 6),
        initial_mines=4,
        early_learning_mode=True,
        early_learning_threshold=100,
        early_learning_corner_safe=True,
        early_learning_edge_safe=True
    )
    
    assert env.current_board_width == 6
    assert env.current_board_height == 6
    assert env.current_mines == 4
    assert env.early_learning_mode is True
    
    # Test corner safety on larger board
    env.reset()
    corners = [(0, 0), (0, 5), (5, 0), (5, 5)]
    for row, col in corners:
        assert not env.mines[row, col]

def test_early_learning_mine_spacing():
    """Test that mine spacing works correctly in early learning mode."""
    env = MinesweeperEnv(
        initial_board_size=(4, 4),
        initial_mines=2,
        early_learning_mode=True,
        mine_spacing=2
    )
    
    env.reset()
    
    # Check mine spacing
    mine_positions = np.where(env.mines)
    for i in range(len(mine_positions[0])):
        row, col = mine_positions[0][i], mine_positions[1][i]
        
        # Check that no other mines are within spacing distance
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if dr == 0 and dc == 0:
                    continue
                new_row, new_col = row + dr, col + dc
                if (0 <= new_row < env.current_board_height and 
                    0 <= new_col < env.current_board_width):
                    assert not env.mines[new_row, new_col]

def test_early_learning_win_rate_tracking(early_learning_env):
    """Test that win rate is tracked during early learning."""
    # Play several games
    for _ in range(10):
        early_learning_env.reset()
        # Simulate a game (win or lose randomly)
        action = 0
        state, reward, terminated, truncated, info = early_learning_env.step(action)
    
    # Check that win rate tracking is working
    assert hasattr(early_learning_env, 'win_rate')
    assert 0 <= early_learning_env.win_rate <= 1
    assert hasattr(early_learning_env, 'total_games')
    assert early_learning_env.total_games >= 0 