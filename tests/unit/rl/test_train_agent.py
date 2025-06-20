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
from stable_baselines3.common.vec_env import DummyVecEnv
from src.core.minesweeper_env import MinesweeperEnv
from src.core.constants import (
    CELL_UNREVEALED,
    CELL_MINE,
    CELL_MINE_HIT,
    REWARD_SAFE_REVEAL,
    REWARD_WIN,
    REWARD_HIT_MINE,
    REWARD_INVALID_ACTION,
    REWARD_FIRST_CASCADE_SAFE,
    REWARD_FIRST_CASCADE_HIT_MINE,
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
        # Vector environments have different action space structure
        assert hasattr(env.action_space, 'n') or env.action_space.shape == ()  # Discrete or empty shape
        # Vector environments have shape (channels, height, width) not (num_envs, channels, height, width)
        assert env.observation_space.shape == (2, 4, 4)  # (channels, height, width)

    def test_environment_reset(self, env):
        """Test that the environment resets correctly"""
        # Handle both gym and gymnasium APIs
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs = reset_result
            info = {}
            
        # Vector environments return shape (num_envs, channels, height, width)
        assert obs.shape == (1, 2, 4, 4)  # (num_envs, channels, height, width)
        assert np.all(obs[0, 0] == CELL_UNREVEALED)  # All cells should be hidden in channel 0

    def test_environment_step(self, env):
        """Test that the environment responds correctly to actions"""
        env.reset()
        # Handle both gym and gymnasium APIs
        step_result = env.step([0])  # Reveal first cell
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        else:
            obs, reward, done, info = step_result
            terminated = done
            truncated = np.array([False])
            
        assert obs.shape == (1, 2, 4, 4)
        assert isinstance(reward, np.ndarray)
        assert isinstance(terminated, np.ndarray)
        assert isinstance(truncated, np.ndarray)
        # Accept both dict and list for info
        assert isinstance(info, (list, dict))
        if isinstance(info, list):
            assert len(info) > 0
            assert isinstance(info[0], dict)
        # Accept both array and list for truncated
        assert isinstance(truncated, (np.ndarray, list))

    def test_environment_consistency(self, env):
        """Test that the environment maintains consistent state"""
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result
            
        initial_state = obs.copy()

        # Take a step
        step_result = env.step([0])
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        else:
            obs, reward, done, info = step_result
            terminated = done
            truncated = np.array([False])
            
        assert not np.array_equal(obs, initial_state)  # State should change
        
        # If we hit a mine on the first move, reset and try again
        if terminated[0]:
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs, _ = reset_result
            else:
                obs = reset_result
            initial_state = obs.copy()
            step_result = env.step([1])
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
            else:
                obs, reward, done, info = step_result
                terminated = done
            assert not np.array_equal(obs, initial_state)  # State should change
        
        assert not terminated[0]  # Game should not end on first move

    def test_environment_completion(self, env):
        """Test that the environment properly handles multiple steps without errors"""
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result
            
        terminated = np.array([False])
        truncated = np.array([False])
        
        # Get action space size from the underlying environment
        if hasattr(env.action_space, 'n'):
            action_space_size = env.action_space.n
        else:
            # For vectorized environments, get size from shape
            action_space_size = env.action_space.shape[0] if env.action_space.shape else 16

        # Test that we can take multiple steps without errors
        for i in range(min(10, action_space_size)):  # Test up to 10 steps or action space size
            action = i % action_space_size  # Use modulo to stay within bounds
            step_result = env.step([action])
            
            if len(step_result) == 5:
                obs, _, terminated, truncated, info = step_result
            else:
                obs, _, done, info = step_result
                terminated = done
                truncated = np.array([False])
            
            # Check that the environment returns valid data
            assert obs.shape == (1, 2, 4, 4)
            assert isinstance(terminated, np.ndarray)
            assert isinstance(truncated, np.ndarray)
            
            # If game ends, that's fine
            if terminated[0] or truncated[0]:
                break

        # Test passes if we can step through without errors
        assert True

    def test_invalid_action(self, env):
        """Test that the environment handles invalid actions"""
        env.reset()
        step_result = env.step([100])  # Action out of bounds
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        else:
            obs, reward, done, info = step_result
            terminated = done
            truncated = np.array([False])
            
        assert reward[0] == REWARD_INVALID_ACTION  # Should return invalid action penalty
        assert not terminated[0]  # Should not terminate the game
        assert not truncated[0]   # Should not truncate the game

if __name__ == '__main__':
    pytest.main()
