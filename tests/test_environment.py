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
        """Test that revealing a mine ends the game"""
        env.reset()
        # Find a mine by checking the mines array
        for y in range(env.current_board_size):
            for x in range(env.current_board_size):
                if env.mines[y, x]:
                    action = y * env.current_board_size + x
                    obs, reward, terminated, truncated, info = env.step(action)
                    assert reward == env.mine_penalty  # Should be -10.0
                    assert terminated  # Game should end
                    assert not truncated
                    assert 'mine_hit' in info['reward_breakdown']
                    assert info['reward_breakdown']['mine_hit'] == env.mine_penalty
                    return
        pytest.fail("No mine found on the board")

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

if __name__ == "__main__":
    sys.exit(main()) 