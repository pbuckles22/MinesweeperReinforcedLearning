import os
import sys
import numpy as np
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from core.minesweeper_env import MinesweeperEnv
import unittest
import gymnasium as gym

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    try:
        import numpy as np
        import pygame
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from core.minesweeper_env import MinesweeperEnv
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_environment_creation():
    """Test that the environment can be created and reset"""
    print("\nTesting environment creation...")
    try:
        env = MinesweeperEnv(board_size=3, num_mines=1)
        state, _ = env.reset()
        print("✓ Environment created and reset successfully")
        print(f"✓ State shape: {state.shape}")
        return True
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        return False

def test_basic_actions():
    """Test that basic actions work in the environment"""
    print("\nTesting basic actions...")
    try:
        env = MinesweeperEnv(board_size=3, num_mines=1)
        state, _ = env.reset()
        
        # Test a reveal action (first cell)
        action = 0  # Reveal first cell (0,0)
        state, reward, terminated, truncated, _ = env.step(action)
        print("✓ Basic action successful")
        print(f"✓ Reward: {reward}")
        print(f"✓ Terminated: {terminated}")
        return True
    except Exception as e:
        print(f"✗ Basic action test failed: {e}")
        return False

def test_pygame():
    """Test that pygame can be initialized"""
    print("\nTesting pygame...")
    try:
        pygame.init()
        pygame.quit()
        print("✓ Pygame initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Pygame initialization failed: {e}")
        return False

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

class TestMinesweeperEnv(unittest.TestCase):
    def setUp(self):
        self.env = MinesweeperEnv(board_size=10, num_mines=10)

    def test_initialization(self):
        """Test that the environment initializes correctly"""
        self.assertEqual(self.env.board_size, 10)
        self.assertEqual(self.env.num_mines, 10)
        self.assertIsNotNone(self.env.board)
        self.assertIsNotNone(self.env.mines)
        self.assertIsNotNone(self.env.revealed)

    def test_reset(self):
        """Test that reset returns the correct observation and info"""
        obs, info = self.env.reset()
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(info, dict)
        self.assertEqual(obs.shape, (10, 10))
        self.assertTrue(np.all(obs == -1))  # All cells should be hidden initially

    def test_step(self):
        """Test that step returns the correct observation, reward, terminated, truncated, and info"""
        self.env.reset()
        obs, reward, terminated, truncated, info = self.env.step(0)  # Reveal first cell
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)
        self.assertEqual(obs.shape, (10, 10))

    def test_invalid_action(self):
        """Test that invalid actions raise an error"""
        self.env.reset()
        with self.assertRaises(ValueError):
            self.env.step(100)  # Out of bounds

    def test_mine_reveal(self):
        """Test that revealing a mine ends the episode"""
        self.env.reset()
        # Find a mine position
        mine_pos = np.where(self.env.mines == 1)
        if len(mine_pos[0]) > 0:
            x, y = mine_pos[0][0], mine_pos[1][0]
            action = x * self.env.current_board_size + y  # Convert (x,y) to single integer
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.assertTrue(terminated)
            self.assertLess(reward, 0)  # Negative reward for hitting a mine

if __name__ == "__main__":
    sys.exit(main()) 