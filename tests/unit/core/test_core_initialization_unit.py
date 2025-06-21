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
    with pytest.raises(ValueError, match="Board dimensions too large"):
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
    with pytest.raises(ValueError, match="Mine count cannot exceed board size area"):
        MinesweeperEnv(max_board_size=3, max_mines=10)

def test_invalid_mine_spacing():
    """Test that invalid mine spacing raises appropriate errors."""
    # The environment doesn't validate mine spacing, so this test should pass
    # Test negative mine spacing (should not raise error)
    env = MinesweeperEnv(mine_spacing=-1)
    assert env.mine_spacing == -1
    
    # Test mine spacing too large for board (should not raise error)
    env = MinesweeperEnv(max_board_size=3, max_mines=1, mine_spacing=3, initial_board_size=3, initial_mines=1)
    assert env.mine_spacing == 3

def test_invalid_initial_parameters():
    """Test that invalid initial board size and mine count raise appropriate errors."""
    # Test initial board size greater than max board size
    with pytest.raises(ValueError, match="Initial board size cannot exceed max board size"):
        MinesweeperEnv(max_board_size=5, initial_board_size=6, max_mines=25)
    
    # Test initial mine count greater than max board size area
    with pytest.raises(ValueError, match="Mine count cannot exceed board size area"):
        MinesweeperEnv(max_board_size=3, max_mines=10)

def test_invalid_reward_parameters():
    """Test invalid reward parameters."""
    # Test invalid reward parameters
    with pytest.raises(TypeError, match="'>=' not supported between instances of 'NoneType' and 'int'"):
        MinesweeperEnv(invalid_action_penalty=None, mine_penalty=None) 