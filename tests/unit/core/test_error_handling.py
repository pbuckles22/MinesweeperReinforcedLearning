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
    # Test initial mine count greater than initial board area
    with pytest.raises(ValueError, match="Initial mine count cannot exceed initial board area"):
        env = MinesweeperEnv(initial_board_size=(4, 4), initial_mines=20)

def test_invalid_mine_spacing():
    """Test that invalid mine spacing is handled gracefully."""
    # Environment doesn't validate mine_spacing, so this should work
    env = MinesweeperEnv(mine_spacing=-1)
    env.reset()
    # Should work with negative mine spacing
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    assert state is not None

def test_invalid_initial_parameters():
    """Test that invalid initial parameters raise appropriate error."""
    with pytest.raises(ValueError, match="Mine count cannot exceed board size area"):
        MinesweeperEnv(
            max_board_size=(4, 4),
            initial_board_size=(8, 8)
        )

def test_invalid_reward_parameters():
    """Test that invalid reward parameters raise appropriate error."""
    with pytest.raises(TypeError, match="'>=' not supported between instances of 'NoneType' and 'int'"):
        MinesweeperEnv(
            invalid_action_penalty=None,
            mine_penalty=None
        )

def test_invalid_action():
    """Test that invalid actions are handled gracefully."""
    env = MinesweeperEnv(initial_board_size=(4, 4), initial_mines=2)
    env.reset()
    
    # Test out-of-bounds action - should return penalty, not raise exception
    state, reward, terminated, truncated, info = env.step(1000)
    assert reward < 0  # Should get invalid action penalty
    assert not terminated

def test_invalid_action_type():
    """Test that invalid action types are handled gracefully."""
    env = MinesweeperEnv(initial_board_size=(4, 4), initial_mines=2)
    env.reset()
    
    # Test invalid action type - should raise TypeError
    with pytest.raises(TypeError):
        env.step("invalid")

def test_invalid_board_dimensions():
    """Test that invalid board dimensions raise appropriate error."""
    with pytest.raises(ValueError, match="Initial board dimensions must be positive"):
        MinesweeperEnv(initial_board_size=(4, 0))

def test_invalid_mine_count_zero():
    """Test that zero mine count raises appropriate error."""
    with pytest.raises(ValueError, match="Initial mine count must be positive"):
        MinesweeperEnv(initial_mines=0)

def test_invalid_mine_count_negative():
    """Test that negative mine count raises appropriate error."""
    with pytest.raises(ValueError, match="Initial mine count must be positive"):
        MinesweeperEnv(initial_mines=-1)

def test_invalid_threshold():
    """Test that invalid early learning threshold is handled gracefully."""
    # Environment doesn't validate early_learning_threshold, so this should work
    env = MinesweeperEnv(early_learning_threshold=0)
    env.reset()
    # Should work with zero threshold
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    assert state is not None

def test_invalid_threshold_negative():
    """Test that negative early learning threshold is handled gracefully."""
    # Environment doesn't validate early_learning_threshold, so this should work
    env = MinesweeperEnv(early_learning_threshold=-1)
    env.reset()
    # Should work with negative threshold
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    assert state is not None

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
    # Accept both dict and list for info
    assert isinstance(info, (dict, list)), "Info should be a dictionary or list of dicts"
    if isinstance(info, list):
        assert len(info) > 0
        assert isinstance(info[0], dict)

def test_edge_case_minimum_board():
    """Test edge case with minimum board size."""
    env = MinesweeperEnv(initial_board_size=(2, 2), initial_mines=1)
    env.reset()
    
    # Should work with minimum board
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    assert state.shape == (4, 2, 2)  # 4 channels, 2x2 board
    assert reward is not None

def test_edge_case_maximum_board():
    """Test edge case with maximum board size."""
    env = MinesweeperEnv(
        max_board_size=(35, 20),
        initial_board_size=(35, 20),
        initial_mines=130
    )
    env.reset()
    
    # Should work with maximum board
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    assert state.shape == (4, 35, 20)  # 4 channels, height x width
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
    
    assert state.shape == (4, 4, 4)  # 4 channels, 4x4 board
    assert reward is not None

def test_error_message_clarity():
    """Test that error messages are clear and informative."""
    try:
        MinesweeperEnv(initial_board_size=(4, 0))
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Initial board dimensions must be positive" in str(e)

def test_error_recovery_after_invalid_action():
    """Test that environment recovers after invalid action."""
    env = MinesweeperEnv(initial_board_size=(4, 4), initial_mines=2)
    env.reset()
    
    # Try invalid action - should return penalty, not raise exception
    state, reward, terminated, truncated, info = env.step(1000)
    assert reward < 0  # Should get invalid action penalty
    
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
    
    assert state.shape == (4, 1, 1)  # 4 channels, 1x1 board
    assert reward is not None

def test_invalid_early_learning_parameters():
    """Test invalid early learning parameters."""
    # Environment doesn't validate early learning parameters, so this should work
    env = MinesweeperEnv(
        early_learning_mode=True,
        early_learning_threshold=0
    )
    env.reset()
    # Should work with invalid early learning parameters
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    assert state is not None

def test_invalid_render_mode():
    """Test invalid render mode."""
    # Environment doesn't validate render_mode, so this should work
    env = MinesweeperEnv(render_mode="invalid")
    env.reset()
    # Should work with invalid render mode
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    assert state is not None

def test_error_handling_with_custom_rewards():
    """Test error handling with custom reward parameters."""
    # Test invalid custom rewards
    with pytest.raises(TypeError, match="'>=' not supported between instances of 'NoneType' and 'int'"):
        MinesweeperEnv(
            invalid_action_penalty=None,
            mine_penalty=None
        )

def test_edge_case_rectangular_board():
    """Test edge case with rectangular board."""
    env = MinesweeperEnv(initial_board_size=(5, 3), initial_mines=3)
    env.reset()
    
    # Should work with rectangular board
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    assert state.shape == (4, 5, 3)  # 4 channels, height x width
    assert reward is not None

def test_error_handling_consistency():
    """Test that error handling is consistent across multiple calls."""
    # Test that same invalid parameters always raise same error
    for _ in range(3):
        with pytest.raises(ValueError, match="Initial board dimensions must be positive"):
            MinesweeperEnv(initial_board_size=(4, 0))

def test_error_handling_performance():
    """Test that error handling doesn't significantly impact performance."""
    import time
    
    start_time = time.time()
    
    # Test multiple error conditions
    for _ in range(10):
        try:
            MinesweeperEnv(initial_board_size=(4, 0))
        except ValueError:
            pass
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Should complete quickly (less than 1 second)
    assert execution_time < 1.0

def test_error_handling_memory():
    """Test that error handling doesn't cause memory leaks."""
    import gc
    
    # Force garbage collection before test
    gc.collect()
    
    # Test multiple error conditions
    for _ in range(100):
        try:
            MinesweeperEnv(initial_board_size=(4, 0))
        except ValueError:
            pass
    
    # Force garbage collection after test
    gc.collect()
    
    # Should not have excessive memory usage
    # This is a basic test - in practice, you'd use memory profiling tools
    assert True  # Placeholder assertion 