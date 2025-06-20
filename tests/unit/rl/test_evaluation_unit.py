import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from stable_baselines3.common.vec_env import DummyVecEnv
from src.core.train_agent import evaluate_model
from src.core.minesweeper_env import MinesweeperEnv
from src.core.constants import REWARD_WIN, REWARD_HIT_MINE


class TestEvaluationFunction:
    """Test the evaluate_model function for proper win detection and vectorized environments."""
    
    @pytest.mark.timeout(30)
    def test_evaluate_model_with_vectorized_env(self):
        """Test that evaluate_model works with vectorized environments."""
        # Create a mock model
        mock_model = Mock()
        mock_model.predict.return_value = (np.array([0]), None)  # Always choose action 0
        
        # Create a vectorized environment
        def make_env():
            env = MinesweeperEnv(
                initial_board_size=(4, 4),
                initial_mines=2,
                early_learning_mode=True
            )
            return env
        
        vec_env = DummyVecEnv([make_env])
        
        # Run evaluation
        results = evaluate_model(mock_model, vec_env, n_episodes=5)
        
        # Check that results are returned
        assert "win_rate" in results
        assert "avg_reward" in results
        assert "avg_length" in results
        assert "reward_ci" in results
        assert "length_ci" in results
        assert "n_episodes" in results
        
        # Check that n_episodes is correct
        assert results["n_episodes"] == 5
        
        # Check that win_rate is a percentage
        assert 0 <= results["win_rate"] <= 100
        
        # Check that avg_reward is a float
        assert isinstance(results["avg_reward"], float)
        
        # Check that avg_length is a float
        assert isinstance(results["avg_length"], float)
    
    @pytest.mark.timeout(30)
    def test_evaluate_model_win_detection(self):
        """Test that evaluate_model correctly detects wins from info dictionary."""
        # Create a mock model
        mock_model = Mock()
        mock_model.predict.return_value = (np.array([0]), None)
        
        # Create a mock environment that simulates wins
        class MockEnv:
            def __init__(self, should_win=True):
                self.should_win = should_win
                self.step_count = 0
            
            def reset(self):
                self.step_count = 0
                return np.zeros((4, 4, 2)), {}
            
            def step(self, action):
                self.step_count += 1
                # Simulate a win after 3 steps if should_win is True
                if self.should_win and self.step_count >= 3:
                    return (np.zeros((4, 4, 2)), REWARD_WIN, True, False, {"won": True})
                elif not self.should_win and self.step_count >= 3:
                    return (np.zeros((4, 4, 2)), REWARD_HIT_MINE, True, False, {"won": False})
                else:
                    return (np.zeros((4, 4, 2)), 1.0, False, False, {"won": False})
        
        # Test with winning episodes
        winning_env = MockEnv(should_win=True)
        results = evaluate_model(mock_model, winning_env, n_episodes=3)
        assert results["win_rate"] == 100.0  # All episodes should be wins
        
        # Test with losing episodes
        losing_env = MockEnv(should_win=False)
        results = evaluate_model(mock_model, losing_env, n_episodes=3)
        assert results["win_rate"] == 0.0  # No episodes should be wins
    
    @pytest.mark.timeout(30)
    def test_evaluate_model_with_real_env(self):
        """Test evaluate_model with a real MinesweeperEnv."""
        # Create a mock model that always chooses the first available valid action
        class ValidActionModel:
            def predict(self, obs):
                # Find the first valid action from the action mask
                env = self.env
                valid_actions = np.where(env.action_masks)[0]
                if len(valid_actions) > 0:
                    return np.array([valid_actions[0]]), None
                else:
                    return np.array([0]), None  # fallback
        
        # Create a real environment
        env = MinesweeperEnv(
            initial_board_size=(4, 4),
            initial_mines=2,
            early_learning_mode=True
        )
        model = ValidActionModel()
        model.env = env
        
        # Run evaluation
        results = evaluate_model(model, env, n_episodes=3)
        
        # Check that results are valid
        assert "win_rate" in results
        assert "avg_reward" in results
        assert "avg_length" in results
        assert results["n_episodes"] == 3
        
        # Check that win_rate is reasonable (0-100%)
        assert 0 <= results["win_rate"] <= 100
        
        # Check that avg_length is positive
        assert results["avg_length"] > 0
    
    @pytest.mark.timeout(30)
    def test_evaluate_model_statistics(self):
        """Test that evaluate_model calculates statistics correctly."""
        # Create a mock model
        mock_model = Mock()
        mock_model.predict.return_value = (np.array([0]), None)
        
        # Create a mock environment with known rewards
        class MockEnv:
            def __init__(self, rewards=[10, 20, 30]):
                self.rewards = rewards
                self.current_step = 0
            
            def reset(self):
                self.current_step = 0
                return np.zeros((4, 4, 2)), {}
            
            def step(self, action):
                if self.current_step < len(self.rewards):
                    reward = self.rewards[self.current_step]
                    self.current_step += 1
                    done = self.current_step >= len(self.rewards)
                    return (np.zeros((4, 4, 2)), reward, done, False, {"won": reward > 0})
                else:
                    return (np.zeros((4, 4, 2)), 0, True, False, {"won": False})
        
        env = MockEnv()
        results = evaluate_model(mock_model, env, n_episodes=1)
        
        # Check that average reward is calculated correctly
        # The episode should have 3 steps with rewards [10, 20, 30], total = 60
        expected_avg_reward = 60.0  # Sum of all rewards in the episode
        assert abs(results["avg_reward"] - expected_avg_reward) < 0.01
        
        # Check that average length is correct
        assert results["avg_length"] == 3.0
    
    @pytest.mark.timeout(30)
    def test_evaluate_model_confidence_intervals(self):
        """Test that confidence intervals are calculated correctly."""
        # Create a mock model
        mock_model = Mock()
        mock_model.predict.return_value = (np.array([0]), None)
        
        # Create a mock environment
        class MockEnv:
            def reset(self):
                return np.zeros((4, 4, 2)), {}
            
            def step(self, action):
                return (np.zeros((4, 4, 2)), 1.0, True, False, {"won": True})
        
        env = MockEnv()
        results = evaluate_model(mock_model, env, n_episodes=5)
        
        # Check that confidence intervals are floats
        assert isinstance(results["reward_ci"], float)
        assert isinstance(results["length_ci"], float)
        
        # Check that confidence intervals are non-negative
        assert results["reward_ci"] >= 0
        assert results["length_ci"] >= 0 