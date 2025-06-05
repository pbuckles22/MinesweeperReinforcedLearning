import pytest
import os
import shutil
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.core.minesweeper_env import MinesweeperEnv

class TestTrainAgent:
    @pytest.fixture
    def env(self):
        """Create a test environment"""
        env = MinesweeperEnv(max_board_size=4, max_mines=2, mine_spacing=2)
        return DummyVecEnv([lambda: env])

    def setUp(self):
        """Set up test environment before each test"""
        self.test_logs_dir = "test_logs/"
        os.makedirs(self.test_logs_dir, exist_ok=True)

    def tearDown(self):
        """Clean up after each test"""
        try:
            if os.path.exists(self.test_logs_dir):
                shutil.rmtree(self.test_logs_dir)
        except Exception as e:
            print(f"Warning: Could not clean up test_logs directory: {e}")

    def test_environment_creation(self, env):
        """Test that the environment is created correctly"""
        assert env.observation_space.shape == (4, 4)
        assert env.action_space.n == 16  # 4x4 board = 16 possible actions

    def test_environment_reset(self, env):
        """Test that the environment resets correctly"""
        obs = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (1, 4, 4)  # DummyVecEnv adds a batch dimension

    def test_environment_step(self, env):
        """Test that the environment responds correctly to actions"""
        env.reset()
        obs, reward, terminated, truncated, info = env.step([0])  # Reveal first cell
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (1, 4, 4)  # DummyVecEnv adds a batch dimension
        assert isinstance(reward, np.ndarray)
        assert isinstance(terminated, np.ndarray)
        assert isinstance(truncated, np.ndarray)
        assert isinstance(info, list)

    def test_environment_consistency(self, env):
        """Test that the environment maintains consistent state"""
        obs = env.reset()
        initial_state = obs.copy()
        
        # Take a step
        obs, _, _, _, _ = env.step([0])
        
        # Reset and verify state is different
        obs = env.reset()
        assert not np.array_equal(obs, initial_state)

    def test_environment_completion(self, env):
        """Test that the environment properly detects game completion"""
        obs = env.reset()
        terminated = False
        truncated = False
        
        # Try to complete the game
        for _ in range(100):  # Limit steps to prevent infinite loops
            obs, _, terminated, truncated, _ = env.step([0])
            if terminated or truncated:
                break
        
        assert terminated or truncated

    def test_invalid_action(self, env):
        """Test that the environment handles invalid actions"""
        env.reset()
        with pytest.raises((ValueError, IndexError)):
            env.step([100])  # Out of bounds

if __name__ == '__main__':
    pytest.main() 