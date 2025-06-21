"""
Comprehensive Tests for train_agent.py

These tests cover all major components of the training agent:
- ExperimentTracker class
- IterationCallback class  
- make_env function
- parse_args function
- main function
- evaluate_model function
"""

import pytest
import os
import tempfile
import shutil
import json
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# TEMPORARY QUICK FIX: Add project root to sys.path for import resolution. Revert when test infra is unified.
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from src.core.train_agent import (
    ExperimentTracker,
    IterationCallback,
    make_env,
    parse_args,
    main,
    evaluate_model
)
from src.core.minesweeper_env import MinesweeperEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor


class TestExperimentTracker:
    """Test the ExperimentTracker class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def tracker(self, temp_dir):
        """Create an ExperimentTracker instance."""
        return ExperimentTracker(experiment_dir=temp_dir)
    
    def test_init(self, temp_dir):
        """Test ExperimentTracker initialization."""
        tracker = ExperimentTracker(experiment_dir=temp_dir)
        
        assert tracker.experiment_dir == temp_dir
        assert tracker.current_run is None
        assert tracker.metrics["training"] == []
        assert tracker.metrics["validation"] == []
        assert tracker.metrics["hyperparameters"] == {}
        assert tracker.metrics["metadata"] == {}
        
        # Check that directory was created
        assert os.path.exists(temp_dir)
    
    def test_start_new_run(self, tracker):
        """Test starting a new experiment run."""
        hyperparameters = {
            "learning_rate": 0.001,
            "batch_size": 64,
            "random_seed": 42
        }
        
        tracker.start_new_run(hyperparameters)
        
        assert tracker.current_run is not None
        assert "run_" in tracker.current_run
        assert tracker.metrics["hyperparameters"] == hyperparameters
        assert "start_time" in tracker.metrics["metadata"]
        assert tracker.metrics["metadata"]["random_seed"] == 42
        
        # Check that run directory was created
        assert os.path.exists(tracker.current_run)
        
        # Check that metrics file was created
        metrics_file = os.path.join(tracker.current_run, "metrics.json")
        assert os.path.exists(metrics_file)
    
    def test_add_training_metric(self, tracker):
        """Test adding training metrics."""
        tracker.start_new_run({"test": "params"})
        
        tracker.add_training_metric("loss", 0.5, 100)
        tracker.add_training_metric("accuracy", 0.8, 200)
        
        assert len(tracker.metrics["training"]) == 2
        assert tracker.metrics["training"][0]["metric"] == "loss"
        assert tracker.metrics["training"][0]["value"] == 0.5
        assert tracker.metrics["training"][0]["step"] == 100
        assert tracker.metrics["training"][1]["metric"] == "accuracy"
        assert tracker.metrics["training"][1]["value"] == 0.8
        assert tracker.metrics["training"][1]["step"] == 200
    
    def test_add_validation_metric(self, tracker):
        """Test adding validation metrics."""
        tracker.start_new_run({"test": "params"})
        
        tracker.add_validation_metric("test_accuracy", 0.85)
        tracker.add_validation_metric("test_loss", 0.3, confidence_interval=0.05)
        
        assert len(tracker.metrics["validation"]) == 2
        assert tracker.metrics["validation"][0]["metric"] == "test_accuracy"
        assert tracker.metrics["validation"][0]["value"] == 0.85
        assert tracker.metrics["validation"][0]["confidence_interval"] is None
        assert tracker.metrics["validation"][1]["metric"] == "test_loss"
        assert tracker.metrics["validation"][1]["value"] == 0.3
        assert tracker.metrics["validation"][1]["confidence_interval"] == 0.05
    
    def test_save_metrics(self, tracker):
        """Test that metrics are saved to file."""
        tracker.start_new_run({"test": "params"})
        tracker.add_training_metric("test", 1.0, 1)
        
        metrics_file = os.path.join(tracker.current_run, "metrics.json")
        assert os.path.exists(metrics_file)
        
        with open(metrics_file, 'r') as f:
            saved_metrics = json.load(f)
        
        assert saved_metrics["training"][0]["metric"] == "test"
        assert saved_metrics["training"][0]["value"] == 1.0


class TestIterationCallback:
    """Test the IterationCallback class."""
    
    @pytest.fixture
    def callback(self):
        """Create an IterationCallback instance."""
        return IterationCallback(verbose=1, debug_level=2)
    
    def test_init(self, callback):
        """Test IterationCallback initialization."""
        assert callback.start_time > 0
        assert callback.iterations == 0
        assert callback.debug_level == 2
        assert callback.learning_phase == "Initial Random"
        assert callback.curriculum_stage == 1
        assert callback.experiment_tracker is None
    
    def test_log(self, callback, capsys):
        """Test logging functionality."""
        # Test default level logging
        callback.log("Test message")
        captured = capsys.readouterr()
        assert "Test message" in captured.out
        
        # Test high level logging (should not appear)
        callback.debug_level = 1
        callback.log("Debug message", level=3)
        captured = capsys.readouterr()
        assert "Debug message" not in captured.out
        
        # Test forced logging
        callback.log("Forced message", level=3, force=True)
        captured = capsys.readouterr()
        assert "Forced message" in captured.out
    
    def test_update_learning_phase(self, callback):
        """Test learning phase updates."""
        # Set up the callback to simulate some iterations
        callback.iterations = 5  # Skip the early return condition
        
        # Test initial phase
        callback._update_learning_phase(0.0, 5.0)
        assert callback.learning_phase == "Early Learning"
        
        # Test phase progression
        callback._update_learning_phase(10.0, 25.0)
        assert callback.learning_phase == "Basic Strategy"
        
        callback._update_learning_phase(20.0, 45.0)
        assert callback.learning_phase == "Intermediate"
        
        callback._update_learning_phase(30.0, 65.0)
        assert callback.learning_phase == "Advanced"
        
        callback._update_learning_phase(40.0, 75.0)
        assert callback.learning_phase == "Expert"
    
    def test_get_env_attr(self, callback):
        """Test getting environment attributes."""
        # Create simple environment classes instead of Mock objects
        class DummyEnv:
            pass
        
        inner_env = DummyEnv()
        inner_env.test_attr = "test_value"
        
        middle_env = DummyEnv()
        middle_env.env = inner_env
        
        outer_env = DummyEnv()
        outer_env.env = middle_env
        
        # Test attribute retrieval
        result = callback.get_env_attr(outer_env, "test_attr")
        assert result == "test_value"
        
        # Test non-existent attribute
        result = callback.get_env_attr(outer_env, "non_existent")
        assert result is None
        
        # Test with environment that has no wrappers
        simple_env = DummyEnv()
        simple_env.test_attr = "simple_value"
        result = callback.get_env_attr(simple_env, "test_attr")
        assert result == "simple_value"


class TestMakeEnv:
    """Test the make_env function."""
    
    def test_make_env(self):
        """Test environment creation."""
        env_fn = make_env(max_board_size=4, max_mines=2)
        assert callable(env_fn)
        
        # Test that the function creates a valid environment
        env = env_fn()
        # The environment is wrapped in Monitor, so check the underlying env
        assert isinstance(env, Monitor)
        # Check the underlying environment
        assert isinstance(env.env, MinesweeperEnv)
        assert env.env.max_board_size_int == 4
        assert env.env.max_mines == 2
        assert env.env.early_learning_mode is True
        assert env.env.early_learning_threshold == 200
        assert env.env.early_learning_corner_safe is True
        assert env.env.early_learning_edge_safe is True
        assert env.env.mine_spacing == 1
        assert env.env.initial_board_width == 4
        assert env.env.initial_mines == 2


class TestParseArgs:
    """Test the parse_args function."""
    
    @patch('src.core.train_agent.argparse.ArgumentParser.parse_args')
    def test_parse_args_defaults(self, mock_parse_args):
        """Test argument parsing with defaults."""
        mock_args = Mock()
        mock_args.total_timesteps = 1000000
        mock_args.eval_freq = 10000
        mock_args.n_eval_episodes = 100
        mock_args.save_freq = 50000
        mock_args.learning_rate = 0.0003
        mock_args.n_steps = 2048
        mock_args.batch_size = 64
        mock_args.n_epochs = 10
        mock_args.gamma = 0.99
        mock_args.gae_lambda = 0.95
        mock_args.clip_range = 0.2
        mock_args.clip_range_vf = None
        mock_args.ent_coef = 0.01
        mock_args.vf_coef = 0.5
        mock_args.max_grad_norm = 0.5
        mock_args.use_sde = False
        mock_args.sde_sample_freq = -1
        mock_args.target_kl = None
        mock_args.policy = "MlpPolicy"
        mock_args.verbose = 1
        mock_args.seed = None
        mock_args.device = "auto"
        mock_args._init_setup_model = True
        
        mock_parse_args.return_value = mock_args
        
        args = parse_args()
        
        assert args.total_timesteps == 1000000
        assert args.eval_freq == 10000
        assert args.n_eval_episodes == 100
        assert args.learning_rate == 0.0003
        assert args.policy == "MlpPolicy"
        assert args.verbose == 1


class TestEvaluateModel:
    """Test the evaluate_model function."""
    
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
        
        assert "win_rate" in result
        assert "avg_reward" in result
        assert "avg_length" in result
        assert "reward_ci" in result
        assert "length_ci" in result
        assert "n_episodes" in result
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
        
        assert result["n_episodes"] == 2
        assert result["avg_reward"] == 1.25  # (3.0 + -0.5) / 2
        assert result["avg_length"] == 2.0  # Both episodes took 2 steps


class TestMainFunction:
    """Test the main function."""
    
    @patch('src.core.train_agent.parse_args')
    @patch('src.core.train_agent.ExperimentTracker')
    @patch('src.core.train_agent.DummyVecEnv')
    @patch('src.core.train_agent.PPO')
    @patch('src.core.train_agent.CustomEvalCallback')
    @patch('src.core.train_agent.IterationCallback')
    @patch('src.core.train_agent.evaluate_model')
    def test_main_function_structure(self, mock_evaluate_model, mock_iteration_callback, mock_custom_eval_callback, 
                                   mock_ppo, mock_dummy_vec_env, mock_experiment_tracker, 
                                   mock_parse_args):
        """Test the main function structure and component creation."""
        # Mock arguments
        mock_args = Mock()
        mock_args.total_timesteps = 1000
        mock_args.eval_freq = 100
        mock_args.n_eval_episodes = 10
        mock_args.learning_rate = 0.001
        mock_args.n_steps = 128
        mock_args.batch_size = 32
        mock_args.n_epochs = 5
        mock_args.gamma = 0.99
        mock_args.gae_lambda = 0.95
        mock_args.clip_range = 0.2
        mock_args.clip_range_vf = None
        mock_args.ent_coef = 0.01
        mock_args.vf_coef = 0.5
        mock_args.max_grad_norm = 0.5
        mock_args.use_sde = False
        mock_args.sde_sample_freq = -1
        mock_args.target_kl = None
        mock_args.policy = "MlpPolicy"
        mock_args.verbose = 1
        mock_args.seed = None
        mock_args.device = "auto"
        mock_args._init_setup_model = True
        
        mock_parse_args.return_value = mock_args
        
        # Mock experiment tracker with proper dict support
        mock_tracker_instance = Mock()
        mock_tracker_instance.metrics = {
            "stage_completion": {
                "stage_1": {
                    "name": "Beginner",
                    "win_rate": 0.5,
                    "mean_reward": 10.0,
                    "std_reward": 2.0,
                    "completed_at": "2024-12-19 12:00:00"
                },
                "stage_2": {
                    "name": "Intermediate",
                    "win_rate": 0.6,
                    "mean_reward": 15.0,
                    "std_reward": 3.0,
                    "completed_at": "2024-12-19 12:01:00"
                },
                "stage_3": {
                    "name": "Easy",
                    "win_rate": 0.5,
                    "mean_reward": 20.0,
                    "std_reward": 4.0,
                    "completed_at": "2024-12-19 12:02:00"
                },
                "stage_4": {
                    "name": "Normal",
                    "win_rate": 0.4,
                    "mean_reward": 25.0,
                    "std_reward": 5.0,
                    "completed_at": "2024-12-19 12:03:00"
                },
                "stage_5": {
                    "name": "Hard",
                    "win_rate": 0.3,
                    "mean_reward": 30.0,
                    "std_reward": 6.0,
                    "completed_at": "2024-12-19 12:04:00"
                },
                "stage_6": {
                    "name": "Expert",
                    "win_rate": 0.2,
                    "mean_reward": 35.0,
                    "std_reward": 7.0,
                    "completed_at": "2024-12-19 12:05:00"
                },
                "stage_7": {
                    "name": "Chaotic",
                    "win_rate": 0.1,
                    "mean_reward": 40.0,
                    "std_reward": 8.0,
                    "completed_at": "2024-12-19 12:06:00"
                }
            }
        }
        mock_tracker_instance._save_metrics = Mock()
        mock_experiment_tracker.return_value = mock_tracker_instance
        
        # Mock environment
        mock_env_instance = Mock()
        mock_env_instance.reset.return_value = np.zeros((1, 2, 4, 4))
        mock_env_instance.step.return_value = (np.zeros((1, 2, 4, 4)), np.array([1.0]), np.array([False]), np.array([False]), [{}])
        mock_dummy_vec_env.return_value = mock_env_instance
        
        # Mock model
        mock_model_instance = Mock()
        mock_model_instance.set_env = Mock()
        mock_model_instance.learn = Mock()
        mock_model_instance.save = Mock()
        mock_model_instance.predict.return_value = (0, None)  # Return proper tuple
        mock_ppo.return_value = mock_model_instance
        
        # Mock callbacks
        mock_custom_eval_callback_instance = Mock()
        mock_custom_eval_callback.return_value = mock_custom_eval_callback_instance
        mock_iteration_callback_instance = Mock()
        mock_iteration_callback.return_value = mock_iteration_callback_instance
        
        # Mock evaluate_model to return simple results
        mock_evaluate_model.return_value = {
            "win_rate": 50.0,
            "avg_reward": 10.0,
            "avg_length": 5.0,
            "reward_ci": 2.0,
            "length_ci": 1.0,
            "n_episodes": 10
        }
        
        # Call main function
        with patch('builtins.print'):  # Suppress print statements
            main()
        
        # Verify components were created
        mock_experiment_tracker.assert_called_once()
        mock_dummy_vec_env.assert_called()
        mock_ppo.assert_called_once()
        mock_custom_eval_callback.assert_called_once()
        mock_iteration_callback.assert_called_once()
        
        # Verify model training was called
        mock_model_instance.learn.assert_called()


class TestIntegration:
    """Integration tests for the training agent."""
    
    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create a temporary directory for testing."""
        return str(tmp_path)
    
    def test_experiment_tracker_with_real_env(self, temp_dir):
        """Test experiment tracker with real environment."""
        tracker = ExperimentTracker(experiment_dir=temp_dir)
        tracker.start_new_run({"test": "params"})
        
        # Create a real environment
        env_fn = make_env(max_board_size=4, max_mines=2)
        env = env_fn()
        
        # Add some metrics
        tracker.add_training_metric("episode_reward", 10.5, 100)
        tracker.add_validation_metric("test_win_rate", 0.6)
        
        # Verify metrics were saved
        metrics_file = os.path.join(tracker.current_run, "metrics.json")
        assert os.path.exists(metrics_file)
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        assert len(metrics["training"]) == 1
        assert len(metrics["validation"]) == 1
        assert metrics["training"][0]["value"] == 10.5
        assert metrics["validation"][0]["value"] == 0.6
    
    def test_callback_with_real_model(self):
        """Test callback with real model (minimal training)."""
        # Create a real environment
        env_fn = make_env(max_board_size=4, max_mines=2)
        env = DummyVecEnv([env_fn])
        
        # Create a real model
        model = PPO("MlpPolicy", env, verbose=0)
        
        # Create callback
        callback = IterationCallback(verbose=0, debug_level=0)
        
        # Train for a few steps
        with patch('builtins.print'):  # Suppress print statements
            model.learn(total_timesteps=100, callback=callback, progress_bar=False)
        
        # Verify callback was used
        assert callback.iterations > 0 or callback.num_timesteps > 0


if __name__ == "__main__":
    pytest.main([__file__]) 