import unittest
import os
import shutil
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from minesweeper_env import MinesweeperEnv

class TestTrainAgent(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        self.test_logs_dir = "test_logs/"
        os.makedirs(self.test_logs_dir, exist_ok=True)
        
        # Create a small test environment
        self.env = DummyVecEnv([lambda: MinesweeperEnv(board_size=3, num_mines=2)])
        
    def tearDown(self):
        """Clean up after each test"""
        try:
            if os.path.exists(self.test_logs_dir):
                shutil.rmtree(self.test_logs_dir)
        except Exception as e:
            print(f"Warning: Could not clean up test_logs directory: {e}")

    def test_environment_creation(self):
        """Test that the environment is created correctly"""
        self.assertEqual(self.env.observation_space.shape, (3, 3, 2))
        self.assertEqual(self.env.action_space.n, 9)  # 3x3 board = 9 possible actions

    def test_environment_reset(self):
        """Test that the environment resets correctly"""
        obs = self.env.reset()
        self.assertEqual(obs.shape, (1, 3, 3, 2))
        # Check that no cells are revealed initially
        self.assertTrue(np.all(obs[0, :, :, 0] == 0))

    def test_environment_step(self):
        """Test that the environment responds correctly to actions"""
        obs = self.env.reset()
        action = 0  # Click top-left corner
        obs, reward, done, info = self.env.step([action])
        
        self.assertEqual(obs.shape, (1, 3, 3, 2))
        self.assertIsInstance(reward[0], (float, np.floating))
        self.assertIsInstance(done[0], (bool, np.bool_))

    def test_invalid_action(self):
        """Test that the environment handles invalid actions"""
        obs = self.env.reset()
        # Try an action outside the valid range
        action = 9  # Invalid for 3x3 board
        with self.assertRaises(Exception):
            self.env.step([action])

    def test_environment_consistency(self):
        """Test that the environment maintains consistent state"""
        obs = self.env.reset()
        # Make a move
        action = 0
        obs1, _, _, _ = self.env.step([action])
        # Reset and make the same move
        obs = self.env.reset()
        obs2, _, _, _ = self.env.step([action])
        # States should be identical
        np.testing.assert_array_equal(obs1, obs2)

    def test_environment_completion(self):
        """Test that the environment properly detects game completion"""
        obs = self.env.reset()
        done = False
        steps = 0
        while not done and steps < 100:  # Prevent infinite loops
            action = np.random.randint(0, 9)
            obs, reward, done, _ = self.env.step([action])
            steps += 1
        self.assertTrue(done[0])  # Game should end at some point

if __name__ == '__main__':
    unittest.main() 