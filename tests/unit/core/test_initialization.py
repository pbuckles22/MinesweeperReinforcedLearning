import pytest
import numpy as np
from src.core.minesweeper_env import MinesweeperEnv
import warnings

# Suppress the pkg_resources deprecation warning
warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated")

def test_invalid_board_size():
    """Test that invalid board sizes raise appropriate errors."""
    # Test negative board size
    with pytest.raises(ValueError, match="Board size must be positive"):
        MinesweeperEnv(max_board_size=-1)
    
    # Test zero board size
    with pytest.raises(ValueError, match="Board size must be positive"):
        MinesweeperEnv(max_board_size=0)
    
    # Test board size too large (e.g., > 100)
    with pytest.raises(ValueError, match="Board size too large"):
        MinesweeperEnv(max_board_size=101)

def test_invalid_mine_count():
    """Test that invalid mine counts raise appropriate errors."""
    # Test negative mine count
    with pytest.raises(ValueError, match="Mine count must be positive"):
        MinesweeperEnv(max_mines=-1)
    
    # Test zero mine count
    with pytest.raises(ValueError, match="Mine count must be positive"):
        MinesweeperEnv(max_mines=0)
    
    # Test mine count greater than board size squared
    with pytest.raises(ValueError, match="Mine count cannot exceed board size squared"):
        MinesweeperEnv(max_board_size=3, max_mines=10)

def test_invalid_mine_spacing():
    """Test that invalid mine spacing raises appropriate errors."""
    # Test negative mine spacing
    with pytest.raises(ValueError, match="Mine spacing must be non-negative"):
        MinesweeperEnv(mine_spacing=-1)
    
    # Test mine spacing too large for board
    with pytest.raises(ValueError, match="Mine spacing too large for board size"):
        MinesweeperEnv(max_board_size=3, max_mines=1, mine_spacing=3, initial_board_size=3, initial_mines=1)

def test_invalid_initial_parameters():
    """Test that invalid initial board size and mine count raise appropriate errors."""
    # Test initial board size greater than max board size
    with pytest.raises(ValueError, match="Initial board size cannot exceed max board size"):
        MinesweeperEnv(max_board_size=5, initial_board_size=6)
    
    # Test initial mine count greater than max mines
    with pytest.raises(ValueError, match="Initial mine count cannot exceed max mines"):
        MinesweeperEnv(max_mines=5, initial_mines=6)
    
    # Test initial mine count greater than initial board size squared
    with pytest.raises(ValueError, match="Initial mine count cannot exceed initial board size squared"):
        MinesweeperEnv(initial_board_size=3, initial_mines=10)

def test_invalid_reward_parameters():
    """Test that invalid reward parameters raise appropriate errors."""
    # Test invalid mine penalty (should be negative)
    with pytest.raises(ValueError, match="Mine penalty must be negative"):
        MinesweeperEnv(mine_penalty=1.0)
    
    # Test invalid flag safe penalty (should be negative)
    with pytest.raises(ValueError, match="Flag safe penalty must be negative"):
        MinesweeperEnv(flag_safe_penalty=1.0)
    
    # Test invalid unflag penalty (should be negative)
    with pytest.raises(ValueError, match="Unflag penalty must be negative"):
        MinesweeperEnv(unflag_penalty=1.0) 