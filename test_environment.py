import unittest
import gymnasium as gym
import numpy as np
from minesweeper_env import MinesweeperEnv

class TestMinesweeperEnvironment(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test."""
        self.env = MinesweeperEnv()

    def test_environment_creation(self):
        """Test that the environment can be created and initialized."""
        self.assertIsInstance(self.env, MinesweeperEnv)
        self.assertEqual(self.env.board_size, 5)
        self.assertEqual(self.env.num_mines, 4)

    def test_environment_reset(self):
        """Test that the environment resets properly."""
        # Take some actions
        self.env.step(0)
        self.env.step(1)
        
        # Reset the environment
        obs, info = self.env.reset()
        
        # Check that the board is reset
        self.assertEqual(obs.shape, (5, 5))
        self.assertTrue(np.all(obs == -1))  # All cells should be hidden
        self.assertEqual(info['mines_remaining'], 4)
        self.assertFalse(info['done'])

    def test_environment_step(self):
        """Test basic environment step functionality."""
        obs, reward, terminated, truncated, info = self.env.step(0)
        
        # Check observation shape
        self.assertEqual(obs.shape, (5, 5))
        
        # Check reward is a number
        self.assertIsInstance(reward, (int, float))
        
        # Check done flags
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        
        # Check info dict
        self.assertIsInstance(info, dict)
        self.assertIn('mines_remaining', info)
        self.assertIn('done', info)

    def test_invalid_action(self):
        """Test that invalid actions are handled correctly."""
        # Try an invalid action (out of bounds)
        obs, reward, terminated, truncated, info = self.env.step(25)  # 5x5 board has indices 0-24
        
        # Should get negative reward for invalid action
        self.assertLess(reward, 0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)

    def test_environment_consistency(self):
        """Test that the environment maintains consistency between steps."""
        # Take a step
        obs1, reward1, terminated1, truncated1, info1 = self.env.step(0)
        
        # Take another step
        obs2, reward2, terminated2, truncated2, info2 = self.env.step(1)
        
        # Check that revealed cells stay revealed
        revealed_cells1 = (obs1 != -1)
        revealed_cells2 = (obs2 != -1)
        self.assertTrue(np.all(revealed_cells1 <= revealed_cells2))

    def test_environment_completion(self):
        """Test that the environment properly handles game completion."""
        # Play until game is done
        done = False
        while not done:
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
        
        # Check that the game is marked as done
        self.assertTrue(terminated or truncated)
        self.assertTrue(info['done'])

if __name__ == '__main__':
    unittest.main() 