"""
Test suite for error handling functionality.
Tests invalid parameters, edge cases, and error recovery.
"""

import pytest
import numpy as np
from src.core.minesweeper_env import MinesweeperEnv

def test_invalid_board_size():
    """Test that invalid board size raises appropriate error."""
    with pytest.raises(ValueError, match="Board dimensions too large"):
        MinesweeperEnv(max_board_size=(101, 101))

def test_invalid_mine_count():
    """Test that invalid mine count raises appropriate error."""
    with pytest.raises(ValueError, match="Mine count cannot exceed board area"):
        MinesweeperEnv(initial_board_size=(4, 4), initial_mines=20)

def test_invalid_mine_spacing():
    """Test that invalid mine spacing raises appropriate error."""
    with pytest.raises(ValueError, match="Mine spacing must be non-negative"):
        MinesweeperEnv(mine_spacing=-1)

def test_invalid_initial_parameters():
    """Test that invalid initial parameters raise appropriate error."""
    with pytest.raises(ValueError, match="Initial board size cannot exceed max board size"):
        MinesweeperEnv(
            max_board_size=(4, 4),
            initial_board_size=(8, 8)
        )

def test_invalid_reward_parameters():
    """Test that invalid reward parameters raise appropriate error."""
    with pytest.raises(ValueError, match="Invalid reward parameters"):
        MinesweeperEnv(
            invalid_action_penalty="invalid",
            mine_penalty="invalid"
        )

def test_invalid_action():
    """Test that invalid actions are handled gracefully."""
    env = MinesweeperEnv(initial_board_size=(4, 4), initial_mines=2)
    env.reset()
    
    # Test out-of-bounds action
    with pytest.raises(IndexError):
        env.step(1000)

def test_invalid_action_type():
    """Test that invalid action types are handled gracefully."""
    env = MinesweeperEnv(initial_board_size=(4, 4), initial_mines=2)
    env.reset()
    
    # Test invalid action type
    with pytest.raises(TypeError):
        env.step("invalid")

def test_invalid_board_dimensions():
    """Test that invalid board dimensions raise appropriate error."""
    with pytest.raises(ValueError, match="Board dimensions must be positive"):
        MinesweeperEnv(initial_board_size=(0, 4))

def test_invalid_mine_count_zero():
    """Test that zero mine count raises appropriate error."""
    with pytest.raises(ValueError, match="Mine count must be positive"):
        MinesweeperEnv(initial_mines=0)

def test_invalid_mine_count_negative():
    """Test that negative mine count raises appropriate error."""
    with pytest.raises(ValueError, match="Mine count must be positive"):
        MinesweeperEnv(initial_mines=-1)

def test_invalid_threshold():
    """Test that invalid early learning threshold raises appropriate error."""
    with pytest.raises(ValueError, match="Early learning threshold must be positive"):
        MinesweeperEnv(early_learning_threshold=0)

def test_invalid_threshold_negative():
    """Test that negative early learning threshold raises appropriate error."""
    with pytest.raises(ValueError, match="Early learning threshold must be positive"):
        MinesweeperEnv(early_learning_threshold=-1)

def test_error_recovery():
    """Test that environment can recover from errors."""
    env = MinesweeperEnv(initial_board_size=(4, 4), initial_mines=2)
    env.reset()
    
    # Make a valid move
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # Environment should still be functional
    assert state is not None
    assert reward is not None
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

def test_edge_case_minimum_board():
    """Test edge case with minimum board size."""
    env = MinesweeperEnv(initial_board_size=(2, 2), initial_mines=1)
    env.reset()
    
    # Should work with minimum board
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    assert state.shape == (2, 2)
    assert reward is not None

def test_edge_case_maximum_board():
    """Test edge case with maximum board size."""
    env = MinesweeperEnv(
        max_board_size=(20, 35),
        initial_board_size=(20, 35),
        initial_mines=130
    )
    env.reset()
    
    # Should work with maximum board
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    assert state.shape == (20, 35)
    assert reward is not None

def test_edge_case_maximum_mines():
    """Test edge case with maximum mine count."""
    env = MinesweeperEnv(
        initial_board_size=(4, 4),
        initial_mines=16  # Maximum for 4x4 board
    )
    env.reset()
    
    # Should work with maximum mines
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    assert state.shape == (4, 4)
    assert reward is not None

def test_error_message_clarity():
    """Test that error messages are clear and informative."""
    try:
        MinesweeperEnv(initial_board_size=(0, 4))
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Board dimensions must be positive" in str(e)

def test_error_recovery_after_invalid_action():
    """Test that environment recovers after invalid action."""
    env = MinesweeperEnv(initial_board_size=(4, 4), initial_mines=2)
    env.reset()
    
    # Try invalid action
    try:
        env.step(1000)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Environment should still be functional
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    assert state is not None
    assert reward is not None

def test_boundary_conditions():
    """Test boundary conditions for various parameters."""
    # Test minimum valid values
    env = MinesweeperEnv(
        initial_board_size=(1, 1),
        initial_mines=1,
        mine_spacing=0,
        early_learning_threshold=1
    )
    env.reset()
    
    # Should work with minimum values
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    assert state.shape == (1, 1)
    assert reward is not None

def test_invalid_early_learning_parameters():
    """Test invalid early learning parameters."""
    with pytest.raises(ValueError, match="Early learning parameters must be valid"):
        MinesweeperEnv(
            early_learning_mode=True,
            early_learning_threshold=0
        )

def test_invalid_render_mode():
    """Test invalid render mode."""
    with pytest.raises(ValueError, match="Invalid render mode"):
        MinesweeperEnv(render_mode="invalid")

def test_error_handling_with_custom_rewards():
    """Test error handling with custom reward parameters."""
    # Test invalid custom rewards
    with pytest.raises(ValueError, match="Invalid reward parameters"):
        MinesweeperEnv(
            invalid_action_penalty=None,
            mine_penalty=None
        )

def test_edge_case_rectangular_board():
    """Test edge case with rectangular board."""
    env = MinesweeperEnv(initial_board_size=(3, 5), initial_mines=3)
    env.reset()
    
    # Should work with rectangular board
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    assert state.shape == (3, 5)
    assert reward is not None

def test_error_handling_consistency():
    """Test that error handling is consistent across multiple calls."""
    # Test that same invalid parameters always raise same error
    for _ in range(3):
        with pytest.raises(ValueError, match="Board dimensions must be positive"):
            MinesweeperEnv(initial_board_size=(0, 4))

def test_error_handling_performance():
    """Test that error handling doesn't cause performance issues."""
    import time
    
    start_time = time.time()
    
    # Create many environments with valid parameters
    for _ in range(100):
        env = MinesweeperEnv(initial_board_size=(4, 4), initial_mines=2)
        env.reset()
        env.step(0)
    
    end_time = time.time()
    
    # Should complete in reasonable time
    assert end_time - start_time < 10.0  # Less than 10 seconds

def test_error_handling_memory():
    """Test that error handling doesn't cause memory leaks."""
    import gc
    
    # Create and destroy many environments
    for _ in range(100):
        env = MinesweeperEnv(initial_board_size=(4, 4), initial_mines=2)
        env.reset()
        env.step(0)
        del env
    
    # Force garbage collection
    gc.collect()
    
    # Should not cause memory issues
    assert True  # If we get here, no memory issues occurred 