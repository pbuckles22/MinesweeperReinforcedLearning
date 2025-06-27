"""
Unit Tests for train_agent.py

These tests focus on unit testing individual components of the training agent:
- ExperimentTracker class
- IterationCallback class
- make_env function
- evaluate_model function
- Curriculum stages and progression
"""

import pytest
import os
import tempfile
import shutil
import json
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.core.train_agent import (
    ExperimentTracker,
    IterationCallback,
    make_env,
    evaluate_model
)
from src.core.minesweeper_env import MinesweeperEnv
from src.core.constants import (
    REWARD_INVALID_ACTION,
    REWARD_HIT_MINE,
    REWARD_SAFE_REVEAL,
    REWARD_WIN
)
from stable_baselines3.common.vec_env import DummyVecEnv

class TestExperimentTracker:
    """Unit tests for ExperimentTracker class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_init(self, temp_dir):
        """Test ExperimentTracker initialization."""
        tracker = ExperimentTracker(experiment_dir=temp_dir)
        assert tracker.experiment_dir == temp_dir
        assert tracker.current_run is None
        assert "training" in tracker.metrics
        assert "validation" in tracker.metrics
        assert "hyperparameters" in tracker.metrics
        assert "metadata" in tracker.metrics
        assert os.path.exists(temp_dir)
    
    def test_start_new_run(self, temp_dir):
        """Test starting a new experiment run."""
        tracker = ExperimentTracker(experiment_dir=temp_dir)
        hyperparams = {"learning_rate": 0.001, "batch_size": 64}
        tracker.start_new_run(hyperparams)
        
        assert tracker.current_run is not None
        assert os.path.exists(tracker.current_run)
        assert tracker.metrics["hyperparameters"] == hyperparams
        assert "start_time" in tracker.metrics["metadata"]
        
        # Verify metrics file was created
        metrics_file = os.path.join(tracker.current_run, "metrics.json")
        assert os.path.exists(metrics_file)
    
    def test_add_training_metric(self, temp_dir):
        """Test adding training metrics."""
        tracker = ExperimentTracker(experiment_dir=temp_dir)
        tracker.start_new_run({})
        
        tracker.add_training_metric("win_rate", 0.75, 100)
        assert len(tracker.metrics["training"]) == 1
        metric = tracker.metrics["training"][0]
        assert metric["metric"] == "win_rate"
        assert metric["value"] == 0.75
        assert metric["step"] == 100
    
    def test_add_validation_metric(self, temp_dir):
        """Test adding validation metrics."""
        tracker = ExperimentTracker(experiment_dir=temp_dir)
        tracker.start_new_run({})
        
        tracker.add_validation_metric("test_win_rate", 0.8, confidence_interval=(0.75, 0.85))
        assert len(tracker.metrics["validation"]) == 1
        metric = tracker.metrics["validation"][0]
        assert metric["metric"] == "test_win_rate"
        assert metric["value"] == 0.8
        # Handle both tuple and list formats for confidence interval
        ci = metric["confidence_interval"]
        assert ci == (0.75, 0.85) or ci == [0.75, 0.85]

class TestIterationCallback:
    """Unit tests for IterationCallback class."""
    
    @pytest.fixture
    def callback(self):
        """Create a test callback instance."""
        return IterationCallback(verbose=0, debug_level=2)
    
    def test_init(self, callback):
        """Test callback initialization."""
        assert callback.iterations == 0
        assert callback.debug_level == 2
        assert callback.learning_phase == "Initial Random"
        assert callback.curriculum_stage == 1
        assert callback.best_reward == float('-inf')
        assert callback.best_win_rate == 0
    
    def test_log(self, callback):
        """Test logging functionality."""
        with patch('builtins.print') as mock_print:
            callback.log("Test message", level=2)
            mock_print.assert_called_once()
            
            # Test log level filtering
            callback.debug_level = 1
            callback.log("Debug message", level=2)
            assert mock_print.call_count == 1  # Should not increment
    
    def test_update_learning_phase(self, callback):
        """Test learning phase updates."""
        # Test initial phase - should stay as "Initial Random" for low performance
        callback._update_learning_phase(0.0, 5.0)
        assert callback.learning_phase == "Initial Random"
        
        # Test phase progression with better performance
        # Need to set iterations > 5 to get past "Initial Random" phase
        callback.iterations = 6
        callback._update_learning_phase(10.0, 25.0)
        assert callback.learning_phase == "Basic Strategy"  # win_rate >= 30
        
        callback._update_learning_phase(20.0, 45.0)
        assert callback.learning_phase == "Intermediate"  # win_rate >= 50
        
        callback._update_learning_phase(30.0, 65.0)
        assert callback.learning_phase == "Advanced"  # win_rate >= 70
        
        callback._update_learning_phase(40.0, 75.0)
        assert callback.learning_phase == "Expert"  # win_rate >= 70
    
    def test_on_step(self, callback):
        """Test step callback functionality."""
        # Mock model and episode info
        model = Mock()
        model.ep_info_buffer = [
            {"r": 10.0, "l": 5, "won": True},
            {"r": -5.0, "l": 3, "won": False}
        ]
        callback.model = model
        callback.num_timesteps = 100
        
        # Test step callback
        with patch('builtins.print'):  # Suppress prints
            result = callback._on_step()
        
        assert result is True  # Should continue training
        # best_reward should be the average of the episode rewards, not the max
        assert callback.best_reward == 2.5  # (10.0 + -5.0) / 2
        assert callback.best_win_rate == 50.0  # 1/2 episodes won

class TestMakeEnv:
    """Unit tests for make_env function."""
    
    def test_make_env_basic(self):
        """Test basic environment creation."""
        env_fn = make_env(max_board_size=4, max_mines=2)
        env = env_fn()
        
        # Environment is wrapped in Monitor, so check the underlying env
        from stable_baselines3.common.monitor import Monitor
        assert isinstance(env, Monitor)
        # Get the underlying environment (may be wrapped in FirstMoveDiscardWrapper)
        underlying_env = env.env
        # If it's wrapped in FirstMoveDiscardWrapper, get the actual environment
        if hasattr(underlying_env, 'env'):
            underlying_env = underlying_env.env
        assert isinstance(underlying_env, MinesweeperEnv)
        assert underlying_env.max_board_size_int == 4
        assert underlying_env.max_mines == 2
        assert underlying_env.early_learning_mode is False
        assert underlying_env.early_learning_threshold == 200
        assert underlying_env.reward_invalid_action == REWARD_INVALID_ACTION
        assert underlying_env.mine_penalty == REWARD_HIT_MINE
        assert underlying_env.safe_reveal_base == REWARD_SAFE_REVEAL
        assert underlying_env.win_reward == REWARD_WIN
    
    def test_make_env_parameters(self):
        """Test environment creation with different parameters."""
        env_fn = make_env(max_board_size=8, max_mines=10)
        env = env_fn()
        
        # Environment is wrapped in Monitor, so check the underlying env
        from stable_baselines3.common.monitor import Monitor
        assert isinstance(env, Monitor)
        # Get the underlying environment (may be wrapped in FirstMoveDiscardWrapper)
        underlying_env = env.env
        # If it's wrapped in FirstMoveDiscardWrapper, get the actual environment
        if hasattr(underlying_env, 'env'):
            underlying_env = underlying_env.env
        assert underlying_env.max_board_size_int == 8
        assert underlying_env.max_mines == 10
        assert underlying_env.initial_board_width == 8  # Should match max_board_size
        assert underlying_env.initial_mines == 10  # Should match max_mines
        assert underlying_env.reward_invalid_action == REWARD_INVALID_ACTION
        assert underlying_env.mine_penalty == REWARD_HIT_MINE
        assert underlying_env.safe_reveal_base == REWARD_SAFE_REVEAL
        assert underlying_env.win_reward == REWARD_WIN

class TestEvaluateModel:
    """Unit tests for evaluate_model function."""
    
    @pytest.fixture
    def mock_env(self):
        """Create a mock environment."""
        env = Mock()
        env.reset.return_value = (np.zeros((1, 2, 4, 4)), {})
        env.step.return_value = (np.zeros((1, 2, 4, 4)), 1.0, False, False, {})
        return env
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = Mock()
        model.predict.return_value = (0, None)
        return model
    
    @pytest.mark.timeout(30)
    def test_evaluate_model_basic(self, mock_model, mock_env):
        """Test basic model evaluation."""
        # Mock environment to simulate a simple game
        mock_env.reset.return_value = (np.zeros((1, 2, 4, 4)), {})
        mock_env.step.side_effect = [
            (np.zeros((1, 2, 4, 4)), 1.0, False, False, {}),
            (np.zeros((1, 2, 4, 4)), 2.0, True, False, {"won": True})
        ]
        
        result = evaluate_model(mock_model, mock_env, n_episodes=1)
        
        assert isinstance(result, dict)
        assert "win_rate" in result
        assert "avg_reward" in result
        assert "avg_length" in result
        assert "reward_ci" in result
        assert "length_ci" in result
        assert "n_episodes" in result
        assert result["avg_reward"] == 3.0  # Sum of rewards (1.0 + 2.0)
        assert result["n_episodes"] == 1
    
    @pytest.mark.timeout(30)
    def test_evaluate_model_multiple_episodes(self, mock_model, mock_env):
        """Test evaluation with multiple episodes."""
        # Mock environment to simulate multiple games
        mock_env.reset.return_value = (np.zeros((1, 2, 4, 4)), {})
        mock_env.step.side_effect = [
            (np.zeros((1, 2, 4, 4)), 1.0, False, False, {}),
            (np.zeros((1, 2, 4, 4)), 2.0, True, False, {"won": True}),
            (np.zeros((1, 2, 4, 4)), 0.5, False, False, {}),
            (np.zeros((1, 2, 4, 4)), -1.0, True, False, {"won": False})
        ]
        
        result = evaluate_model(mock_model, mock_env, n_episodes=2)
        
        assert result["avg_reward"] == 1.25  # (3.0 + -0.5) / 2
        assert result["n_episodes"] == 2
        assert result["reward_ci"] > 0  # Should have non-zero standard deviation

if __name__ == '__main__':
    pytest.main([__file__]) 