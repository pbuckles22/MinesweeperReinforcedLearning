"""
RL Agent/Environment Integration Tests

These tests verify that the Minesweeper environment works with RL agents and vectorized environments.
Non-determinism is expected: tests only check for valid behaviors, not specific outcomes.
"""
import pytest
import os
import shutil
import numpy as np
from stable_baselines3 import PPO
from src.core.minesweeper_env import MinesweeperEnv
from src.core.vec_env import DummyVecEnv
from src.core.constants import (
    CELL_UNREVEALED,
    CELL_MINE,
    CELL_FLAGGED,
    CELL_MINE_HIT,
    REWARD_FIRST_MOVE_SAFE,
    REWARD_FIRST_MOVE_HIT_MINE,
    REWARD_SAFE_REVEAL,
    REWARD_WIN,
    REWARD_HIT_MINE,
    REWARD_INVALID_ACTION
)

class TestTrainAgent:
    @pytest.fixture
    def env(self):
        """Create a test environment."""
        env_fn = lambda: MinesweeperEnv(
            max_board_size=4,
            max_mines=2,
            early_learning_mode=True
        )
        return DummyVecEnv([env_fn])

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
        assert env.action_space.shape == (1,)  # Single discrete action
        assert env.observation_space.shape == (1, 4, 4)  # (num_envs, height, width)

    def test_environment_reset(self, env):
        """Test that the environment resets correctly"""
        obs, info = env.reset()
        assert obs.shape == (1, 4, 4)  # (num_envs, height, width)
        assert np.all(obs == CELL_UNREVEALED)  # All cells should be hidden

    def test_environment_step(self, env):
        """Test that the environment responds correctly to actions"""
        env.reset()
        obs, reward, terminated, truncated, info = env.step([0])  # Reveal first cell
        assert obs.shape == (1, 4, 4)
        assert isinstance(reward, np.ndarray)
        assert isinstance(terminated, np.ndarray)
        assert isinstance(truncated, np.ndarray)
        assert isinstance(info, dict)

    def test_environment_consistency(self, env):
        """Test that the environment maintains consistent state"""
        obs, _ = env.reset()
        initial_state = obs.copy()

        # Take a step
        obs, reward, terminated, truncated, info = env.step([0])
        assert not np.array_equal(obs, initial_state)  # State should change
        
        # If we hit a mine on the first move, reset and try again
        if terminated[0]:
            obs, _ = env.reset()
            initial_state = obs.copy()
            obs, reward, terminated, truncated, info = env.step([1])
            assert not np.array_equal(obs, initial_state)  # State should change
        
        assert not terminated[0]  # Game should not end on first move

    def test_environment_completion(self, env):
        """Test that the environment properly detects game completion"""
        obs, _ = env.reset()
        terminated = np.array([False])
        truncated = np.array([False])
        board_size = 4  # Since we're using a 4x4 board in the test

        # Try to complete the game by revealing cells systematically
        for i in range(board_size * board_size):
            # Convert linear index to 2D coordinates
            row = i // board_size
            col = i % board_size
            # Convert to action space index (multiply by 2 since each cell has reveal/flag actions)
            action = (row * board_size + col) * 2
            obs, _, terminated, truncated, info = env.step([action])
            if terminated[0] or truncated[0]:
                break

        assert terminated[0] or truncated[0]  # Game should end

    def test_invalid_action(self, env):
        """Test that the environment handles invalid actions"""
        env.reset()
        obs, reward, terminated, truncated, info = env.step([100])  # Action out of bounds
        assert reward[0] == REWARD_INVALID_ACTION  # Should return invalid action penalty
        assert not terminated[0]  # Should not terminate the game
        assert not truncated[0]   # Should not truncate the game

if __name__ == '__main__':
    pytest.main() 