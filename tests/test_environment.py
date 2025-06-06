import pytest
import os
import sys
import numpy as np
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.core.minesweeper_env import MinesweeperEnv
import unittest
import gymnasium as gym
from src.core.constants import CELL_UNREVEALED

def test_imports():
    """Test that all required imports are available"""
    print("Testing imports...")
    assert True  # If we got here, imports worked
    print("✓ All imports successful")

def test_environment_creation():
    """Test that the environment can be created and reset"""
    print("\nTesting environment creation...")
    env = MinesweeperEnv(max_board_size=4, max_mines=2, mine_spacing=2)
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4, 4)
    assert isinstance(info, dict)
    print("✓ Environment created and reset successfully")
    print(f"✓ State shape: {obs.shape}")
    print(f"✓ Info: {info}")

def test_basic_actions():
    """Test that basic actions work"""
    print("\nTesting basic actions...")
    env = MinesweeperEnv(max_board_size=4, max_mines=2, mine_spacing=2)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)  # Reveal first cell
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, (float, np.floating, int))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    print("✓ Basic action successful")
    print(f"✓ Reward: {reward}")
    print(f"✓ Terminated: {terminated}")
    print(f"✓ Truncated: {truncated}")
    print(f"✓ Info: {info}")

def test_pygame():
    """Test that pygame can be initialized"""
    print("\nTesting pygame...")
    pygame.init()
    assert pygame.get_init()
    print("✓ Pygame initialized successfully")
    pygame.quit()

def main():
    """Run all environment tests"""
    print("Starting environment tests...\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Environment Creation Test", test_environment_creation),
        ("Basic Actions Test", test_basic_actions),
        ("Pygame Test", test_pygame)
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        if not test_func():
            all_passed = False
            print(f"✗ {test_name} failed")
        else:
            print(f"✓ {test_name} passed")
    
    print("\nTest Summary:")
    if all_passed:
        print("✓ All environment tests passed!")
        return 0
    else:
        print("✗ Some tests failed. Please check the output above.")
        return 1

class TestMinesweeperEnv:
    @pytest.fixture
    def env(self):
        """Create a test environment"""
        return MinesweeperEnv(max_board_size=4, max_mines=2, mine_spacing=2)

    def test_initialization(self, env):
        """Test that the environment initializes correctly"""
        assert env.max_board_size == 4
        assert env.max_mines == 2
        assert env.mine_spacing == 2

    def test_invalid_action(self, env):
        """Test that invalid actions are handled (should not crash)"""
        env.reset()
        with pytest.raises((ValueError, IndexError)):
            env.step(100)  # Out of bounds

    def test_mine_reveal(self, env):
        """Test revealing a mine."""
        # Find a mine on the board
        mine_found = False
        for i in range(env.current_board_width * env.current_board_height):
            row = i // env.current_board_width
            col = i % env.current_board_width
            if env.mines[row, col]:
                mine_found = True
                action = i
                break
        
        assert mine_found, "No mine found on board"
        
        # Reveal the mine
        state, reward, terminated, truncated, info = env.step(action)
        
        # If this is the first move, the game should be reset
        if env.is_first_move:
            assert not terminated
            assert reward == 0
        else:
            assert reward == env.mine_penalty
            assert terminated
            assert 'mine_hit' in info['reward_breakdown']

    def test_reset(self, env):
        """Test that reset returns the correct observation and info"""
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (4, 4)
        assert isinstance(info, dict)

    def test_step(self, env):
        """Test that step returns the correct observation, reward, terminated, truncated, and info"""
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)  # Reveal first cell
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (4, 4)
        assert isinstance(reward, (float, np.floating, int))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

@pytest.fixture
def env():
    return MinesweeperEnv(initial_board_size=3, initial_mines=1)

def test_initialization(env):
    """Test that environment initializes correctly."""
    assert env.current_board_width == 3
    assert env.current_board_height == 3
    assert env.initial_mines == 1
    assert env.is_first_move
    assert env.flags_remaining == env.initial_mines
    assert env.mines.shape == (3, 3)
    assert env.board.shape == (3, 3)
    assert np.all(env.board == CELL_UNREVEALED)

def test_reset(env):
    """Test that environment resets correctly."""
    # Make some moves
    env.step(0)  # Reveal first cell
    env.step(4)  # Place flag in middle
    
    # Reset environment
    state = env.reset()
    
    # Check that everything is reset
    assert env.current_board_width == 3
    assert env.current_board_height == 3
    assert env.initial_mines == 1
    assert env.is_first_move
    assert env.flags_remaining == env.initial_mines
    assert env.mines.shape == (3, 3)
    assert env.board.shape == (3, 3)
    assert np.all(state == CELL_UNREVEALED)
    assert np.all(env.board == CELL_UNREVEALED)

def test_board_size_initialization():
    """Test that different board sizes initialize correctly."""
    # Test 5x5 board
    env = MinesweeperEnv(initial_board_size=5, initial_mines=3)
    assert env.current_board_width == 5
    assert env.current_board_height == 5
    assert env.initial_mines == 3
    assert env.mines.shape == (5, 5)
    assert env.board.shape == (5, 5)
    
    # Test 10x10 board
    env = MinesweeperEnv(initial_board_size=10, initial_mines=10)
    assert env.current_board_width == 10
    assert env.current_board_height == 10
    assert env.initial_mines == 10
    assert env.mines.shape == (10, 10)
    assert env.board.shape == (10, 10)

def test_mine_count_initialization():
    """Test that different mine counts initialize correctly."""
    # Test with 1 mine
    env = MinesweeperEnv(initial_board_size=3, initial_mines=1)
    assert env.initial_mines == 1
    assert np.sum(env.mines) == 1
    
    # Test with 2 mines
    env = MinesweeperEnv(initial_board_size=3, initial_mines=2)
    assert env.initial_mines == 2
    assert np.sum(env.mines) == 2
    
    # Test with maximum mines (board_size - 1)
    env = MinesweeperEnv(initial_board_size=3, initial_mines=8)
    assert env.initial_mines == 8
    assert np.sum(env.mines) == 8

def test_adjacent_mines_initialization(env):
    """Test that adjacent mine counts are calculated correctly."""
    # Place mine at (1,1)
    env.mines[1, 1] = True
    env._update_adjacent_counts()
    
    # Check adjacent counts
    assert env.board[0, 0] == 1
    assert env.board[0, 1] == 1
    assert env.board[0, 2] == 1
    assert env.board[1, 0] == 1
    assert env.board[1, 1] == 0  # Mine location
    assert env.board[1, 2] == 1
    assert env.board[2, 0] == 1
    assert env.board[2, 1] == 1
    assert env.board[2, 2] == 1

def test_environment_initialization():
    """Test environment initialization with different parameters."""
    # Test with default parameters
    env = MinesweeperEnv()
    assert env.current_board_width == env.initial_board_width
    assert env.current_board_height == env.initial_board_height
    assert env.current_mines == env.initial_mines
    
    # Test with custom parameters
    env = MinesweeperEnv(initial_board_size=(4, 5), initial_mines=3)
    assert env.current_board_width == 4
    assert env.current_board_height == 5
    assert env.current_mines == 3

def test_board_creation(env):
    """Test that board is created with correct dimensions and mine count."""
    assert env.board.shape == (env.current_board_height, env.current_board_width)
    assert np.sum(env.mines) == env.current_mines

def test_mine_placement(env):
    """Test that mines are placed correctly."""
    # Count mines
    mine_count = np.sum(env.mines)
    assert mine_count == env.current_mines
    
    # Check mine spacing
    for i in range(env.current_board_height):
        for j in range(env.current_board_width):
            if env.mines[i, j]:
                # Check surrounding cells
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if (0 <= ni < env.current_board_height and 
                            0 <= nj < env.current_board_width and 
                            (di != 0 or dj != 0)):
                            assert not env.mines[ni, nj]

def test_safe_cell_reveal(env):
    """Test that revealing a safe cell works correctly."""
    # Find a safe cell
    safe_cell = None
    for i in range(env.current_board_height):
        for j in range(env.current_board_width):
            if not env.mines[i, j]:
                safe_cell = (i, j)
                break
        if safe_cell:
            break
    
    # Reveal the safe cell
    action = safe_cell[0] * env.current_board_width + safe_cell[1]
    state, reward, terminated, truncated, info = env.step(action)
    
    assert not terminated
    assert reward >= 0
    assert state[safe_cell] != CELL_UNREVEALED

def test_difficulty_levels():
    """Test different difficulty levels."""
    # Test easy difficulty
    env = MinesweeperEnv(initial_board_size=9, initial_mines=10)
    assert env.current_board_width == 9
    assert env.current_board_height == 9
    assert env.current_mines == 10
    
    # Test normal difficulty
    env = MinesweeperEnv(initial_board_size=16, initial_mines=40)
    assert env.current_board_width == 16
    assert env.current_board_height == 16
    assert env.current_mines == 40
    
    # Test hard difficulty
    env = MinesweeperEnv(initial_board_size=(16, 30), initial_mines=99)
    assert env.current_board_width == 16
    assert env.current_board_height == 30
    assert env.current_mines == 99

def test_rectangular_board_actions(env):
    """Test actions on rectangular board."""
    # Create rectangular board
    env = MinesweeperEnv(initial_board_size=(4, 5), initial_mines=3)
    
    # Test actions in different positions
    for i in range(env.current_board_height):
        for j in range(env.current_board_width):
            action = i * env.current_board_width + j
            state, reward, terminated, truncated, info = env.step(action)
            assert not terminated  # First move should be safe
            assert state.shape == (env.current_board_height, env.current_board_width)

def test_curriculum_progression(env):
    """Test that curriculum learning progresses correctly."""
    # Start with small board
    assert env.current_board_width == env.initial_board_width
    assert env.current_board_height == env.initial_board_height
    assert env.current_mines == env.initial_mines
    
    # Win multiple games to trigger progression
    for _ in range(5):
        # Find and reveal all safe cells
        for i in range(env.current_board_height):
            for j in range(env.current_board_width):
                if not env.mines[i, j]:
                    action = i * env.current_board_width + j
                    env.step(action)
        
        # Reset for next game
        env.reset()
    
    # Board size should have increased
    assert (env.current_board_width > env.initial_board_width or 
            env.current_board_height > env.initial_board_height or
            env.current_mines > env.initial_mines)

def test_win_condition(env):
    """Test win condition detection."""
    # Reveal all safe cells
    for i in range(env.current_board_height):
        for j in range(env.current_board_width):
            if not env.mines[i, j]:
                action = i * env.current_board_width + j
                state, reward, terminated, truncated, info = env.step(action)
    
    # Game should be won
    assert terminated
    assert reward > 0  # Should get positive reward for winning
    assert not truncated

if __name__ == "__main__":
    sys.exit(main()) 