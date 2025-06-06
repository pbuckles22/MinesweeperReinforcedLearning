import pytest
import numpy as np
import time
import psutil
import os
from src.core.minesweeper_env import MinesweeperEnv

@pytest.fixture
def env():
    """Create a test environment."""
    return MinesweeperEnv(
        max_board_size=(20, 35),
        max_mines=130,
        initial_board_size=(4, 4),
        initial_mines=2
    )

def test_large_board_performance(env):
    """Test performance with maximum board size."""
    # Set to maximum size
    env.current_board_width = env.max_board_width
    env.current_board_height = env.max_board_height
    env.current_mines = env.max_mines
    env.reset()
    
    # Measure time for 100 steps
    start_time = time.time()
    for _ in range(100):
        # Take random actions
        action = np.random.randint(0, env.action_space.n)
        env.step(action)
    end_time = time.time()
    
    # Calculate average time per step
    avg_time = (end_time - start_time) / 100
    assert avg_time < 0.1  # Should take less than 100ms per step

def test_many_mines_performance(env):
    """Test performance with maximum mine count."""
    # Set to maximum mine count
    env.current_mines = env.max_mines
    env.reset()
    
    # Measure time for 100 steps
    start_time = time.time()
    for _ in range(100):
        # Take random actions
        action = np.random.randint(0, env.action_space.n)
        env.step(action)
    end_time = time.time()
    
    # Calculate average time per step
    avg_time = (end_time - start_time) / 100
    assert avg_time < 0.1  # Should take less than 100ms per step

def test_memory_usage(env):
    """Test memory usage during gameplay."""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # Play 100 games
    for _ in range(100):
        env.reset()
        # Play until game over
        while True:
            action = np.random.randint(0, env.action_space.n)
            _, _, terminated, _, _ = env.step(action)
            if terminated:
                break
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Memory increase should be reasonable (less than 100MB)
    assert memory_increase < 100 * 1024 * 1024

def test_reset_performance(env):
    """Test performance of reset operation."""
    # Measure time for 100 resets
    start_time = time.time()
    for _ in range(100):
        env.reset()
    end_time = time.time()
    
    # Calculate average time per reset
    avg_time = (end_time - start_time) / 100
    assert avg_time < 0.1  # Should take less than 100ms per reset

def test_state_update_performance(env):
    """Test performance of state updates."""
    env.reset()
    
    # Measure time for 100 state updates
    start_time = time.time()
    for _ in range(100):
        # Take random actions
        action = np.random.randint(0, env.action_space.n)
        env.step(action)
    end_time = time.time()
    
    # Calculate average time per state update
    avg_time = (end_time - start_time) / 100
    assert avg_time < 0.1  # Should take less than 100ms per update

def test_rapid_actions(env):
    """Test performance under rapid action sequences."""
    env.reset()
    
    # Take 1000 actions as fast as possible
    start_time = time.time()
    for _ in range(1000):
        action = np.random.randint(0, env.action_space.n)
        env.step(action)
    end_time = time.time()
    
    # Calculate average time per action
    avg_time = (end_time - start_time) / 1000
    assert avg_time < 0.1  # Should take less than 100ms per action 