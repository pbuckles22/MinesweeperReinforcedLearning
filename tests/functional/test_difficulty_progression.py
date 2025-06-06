import pytest
import numpy as np
from src.core.minesweeper_env import MinesweeperEnv
from src.core.constants import REWARD_WIN

@pytest.fixture
def env():
    """Create a test environment with curriculum learning enabled."""
    return MinesweeperEnv(
        max_board_size=(20, 35),
        max_mines=130,
        initial_board_size=(4, 4),
        initial_mines=2,
        early_learning_mode=True,
        early_learning_threshold=200
    )

def test_early_learning_progression(env):
    """Test progression through early learning stages."""
    # Start with initial size
    assert env.current_board_width == 4
    assert env.current_board_height == 4
    assert env.current_mines == 2
    
    # Simulate successful games to trigger progression
    for _ in range(10):
        env.reset()
        # Win the game
        for y in range(env.current_board_height):
            for x in range(env.current_board_width):
                if not env.mines[y, x]:
                    action = y * env.current_board_width + x
                    state, reward, terminated, truncated, info = env.step(action)
                    if terminated and info['won']:
                        break
    
    # Check if board size or mine count increased
    assert (env.current_board_width > 4 or 
            env.current_board_height > 4 or 
            env.current_mines > 2)

def test_difficulty_levels(env):
    """Test progression through different difficulty levels."""
    # Test Easy (9x9, 10 mines)
    env.current_board_width = 9
    env.current_board_height = 9
    env.current_mines = 10
    env.reset()
    assert env.current_board_width == 9
    assert env.current_board_height == 9
    assert env.current_mines == 10
    
    # Test Normal (16x16, 40 mines)
    env.current_board_width = 16
    env.current_board_height = 16
    env.current_mines = 40
    env.reset()
    assert env.current_board_width == 16
    assert env.current_board_height == 16
    assert env.current_mines == 40
    
    # Test Hard (16x30, 99 mines)
    env.current_board_width = 16
    env.current_board_height = 30
    env.current_mines = 99
    env.reset()
    assert env.current_board_width == 16
    assert env.current_board_height == 30
    assert env.current_mines == 99

def test_curriculum_limits(env):
    """Test that curriculum learning respects maximum limits."""
    # Set to maximum values
    env.current_board_width = env.max_board_width
    env.current_board_height = env.max_board_height
    env.current_mines = env.max_mines
    env.reset()
    
    # Simulate successful games
    for _ in range(10):
        env.reset()
        # Win the game
        for y in range(env.current_board_height):
            for x in range(env.current_board_width):
                if not env.mines[y, x]:
                    action = y * env.current_board_width + x
                    state, reward, terminated, truncated, info = env.step(action)
                    if terminated and info['won']:
                        break
    
    # Verify we haven't exceeded maximums
    assert env.current_board_width <= env.max_board_width
    assert env.current_board_height <= env.max_board_height
    assert env.current_mines <= env.max_mines

def test_win_rate_tracking(env):
    """Test win rate tracking and its effect on progression."""
    # Simulate games with varying success rates
    for _ in range(20):
        env.reset()
        # Randomly win or lose
        if np.random.random() < 0.7:  # 70% win rate
            # Win the game
            for y in range(env.current_board_height):
                for x in range(env.current_board_width):
                    if not env.mines[y, x]:
                        action = y * env.current_board_width + x
                        state, reward, terminated, truncated, info = env.step(action)
                        if terminated and info['won']:
                            break
        else:
            # Lose the game
            for y in range(env.current_board_height):
                for x in range(env.current_board_width):
                    if env.mines[y, x]:
                        action = y * env.current_board_width + x
                        state, reward, terminated, truncated, info = env.step(action)
                        if terminated:
                            break
    
    # Check if win rate is being tracked
    assert hasattr(env, 'win_rate')
    assert 0 <= env.win_rate <= 1 