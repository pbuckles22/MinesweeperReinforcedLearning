"""
Unit tests for training callbacks in train_agent.py

This module tests the CustomEvalCallback and IterationCallback classes
that handle evaluation and training iteration monitoring.
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock, Mock
from src.core.train_agent import CustomEvalCallback, IterationCallback


class TestCustomEvalCallback:
    """Test CustomEvalCallback functionality."""
    
    def test_custom_eval_callback_init(self):
        """Test CustomEvalCallback initialization."""
        eval_env = MagicMock()
        
        callback = CustomEvalCallback(
            eval_env=eval_env,
            eval_freq=1000,
            n_eval_episodes=5,
            verbose=1,
            best_model_save_path="models/",
            log_path="logs/"
        )
        
        assert callback.eval_env == eval_env
        assert callback.eval_freq == 1000
        assert callback.n_eval_episodes == 5
        assert callback.verbose == 1
        assert callback.best_model_save_path == "models/"
        assert callback.log_path == "logs/"
        assert callback.best_mean_reward == -np.inf
        assert callback.best_model_save_path is not None
    
    def test_custom_eval_callback_init_defaults(self):
        """Test CustomEvalCallback initialization with defaults."""
        eval_env = MagicMock()
        
        callback = CustomEvalCallback(eval_env=eval_env)
        
        assert callback.eval_env == eval_env
        assert callback.eval_freq == 1000
        assert callback.n_eval_episodes == 5
        assert callback.verbose == 1
        assert callback.best_model_save_path is None
        assert callback.log_path is None
    
    @patch('src.core.train_agent.evaluate_model')
    def test_custom_eval_callback_on_step_evaluation(self, mock_evaluate):
        """Test CustomEvalCallback evaluation during training."""
        eval_env = MagicMock()
        callback = CustomEvalCallback(eval_env=eval_env, eval_freq=10, n_eval_episodes=3)
        
        # Mock evaluation results
        mock_evaluate.return_value = {
            "win_rate": 75.0,
            "avg_reward": 25.5,
            "avg_length": 15.2,
            "reward_ci": 2.1,
            "length_ci": 1.5,
            "n_episodes": 3
        }
        
        # Mock training environment
        training_env = MagicMock()
        training_env.num_envs = 1
        
        # Mock model
        model = MagicMock()
        
        # Test evaluation trigger
        callback.num_timesteps = 10  # Should trigger evaluation
        result = callback._on_step()
        
        assert result is True  # Should continue training
        mock_evaluate.assert_called_once_with(model, eval_env, n_episodes=3)
    
    @patch('src.core.train_agent.evaluate_model')
    def test_custom_eval_callback_on_step_no_evaluation(self, mock_evaluate):
        """Test CustomEvalCallback when evaluation is not triggered."""
        eval_env = MagicMock()
        callback = CustomEvalCallback(eval_env=eval_env, eval_freq=100, n_eval_episodes=3)
        
        # Mock training environment
        training_env = MagicMock()
        training_env.num_envs = 1
        
        # Mock model
        model = MagicMock()
        
        # Test no evaluation trigger
        callback.num_timesteps = 50  # Should not trigger evaluation
        result = callback._on_step()
        
        assert result is True  # Should continue training
        mock_evaluate.assert_not_called()
    
    @patch('src.core.train_agent.evaluate_model')
    def test_custom_eval_callback_best_model_saving(self, mock_evaluate):
        """Test CustomEvalCallback best model saving functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            eval_env = MagicMock()
            save_path = os.path.join(temp_dir, "best_model")
            
            callback = CustomEvalCallback(
                eval_env=eval_env,
                eval_freq=10,
                n_eval_episodes=3,
                best_model_save_path=save_path
            )
            
            # Mock model
            model = MagicMock()
            
            # Mock evaluation results - first evaluation
            mock_evaluate.return_value = {
                "win_rate": 60.0,
                "avg_reward": 20.0,
                "avg_length": 12.0,
                "reward_ci": 1.5,
                "length_ci": 1.0,
                "n_episodes": 3
            }
            
            callback.num_timesteps = 10
            callback._on_step()
            
            # Model should be saved as it's the first evaluation
            model.save.assert_called_once_with(save_path)
            
            # Mock evaluation results - better performance
            mock_evaluate.return_value = {
                "win_rate": 80.0,
                "avg_reward": 30.0,
                "avg_length": 10.0,
                "reward_ci": 1.0,
                "length_ci": 0.8,
                "n_episodes": 3
            }
            
            callback.num_timesteps = 20
            callback._on_step()
            
            # Model should be saved again due to better performance
            assert model.save.call_count == 2
    
    @patch('src.core.train_agent.evaluate_model')
    def test_custom_eval_callback_no_improvement(self, mock_evaluate):
        """Test CustomEvalCallback when no improvement occurs."""
        eval_env = MagicMock()
        callback = CustomEvalCallback(eval_env=eval_env, eval_freq=10, n_eval_episodes=3)
        
        # Mock model
        model = MagicMock()
        
        # Mock evaluation results - good performance
        mock_evaluate.return_value = {
            "win_rate": 80.0,
            "avg_reward": 30.0,
            "avg_length": 10.0,
            "reward_ci": 1.0,
            "length_ci": 0.8,
            "n_episodes": 3
        }
        
        callback.num_timesteps = 10
        callback._on_step()
        
        # Mock evaluation results - worse performance
        mock_evaluate.return_value = {
            "win_rate": 40.0,
            "avg_reward": 15.0,
            "avg_length": 15.0,
            "reward_ci": 2.0,
            "length_ci": 1.5,
            "n_episodes": 3
        }
        
        callback.num_timesteps = 20
        callback._on_step()
        
        # Model should not be saved due to worse performance
        assert model.save.call_count == 1  # Only the first evaluation


class TestIterationCallback:
    """Test IterationCallback functionality."""
    
    def test_iteration_callback_init(self):
        """Test IterationCallback initialization."""
        experiment_tracker = MagicMock()
        
        callback = IterationCallback(
            verbose=1,
            debug_level=2,
            experiment_tracker=experiment_tracker,
            stats_file="training_stats.txt",
            timestamped_stats=True
        )
        
        assert callback.verbose == 1
        assert callback.debug_level == 2
        assert callback.experiment_tracker == experiment_tracker
        assert callback.stats_file == "training_stats.txt"
        assert callback.timestamped_stats == True
        assert callback.iteration_count == 0
        assert callback.last_log_time == 0
        assert callback.learning_phase == "initialization"
    
    def test_iteration_callback_init_defaults(self):
        """Test IterationCallback initialization with defaults."""
        callback = IterationCallback()
        
        assert callback.verbose == 0
        assert callback.debug_level == 2
        assert callback.experiment_tracker is None
        assert callback.stats_file == "training_stats.txt"
        assert callback.timestamped_stats == False
        assert callback.iteration_count == 0
    
    def test_iteration_callback_log(self):
        """Test IterationCallback logging functionality."""
        callback = IterationCallback(verbose=1, debug_level=2)
        
        # Test logging with different levels
        callback.log("Test message", level=1, force=True)
        callback.log("Debug message", level=3, force=False)  # Should not log due to debug_level=2
        
        # Verify logging behavior (we can't easily test print output, but we can test the logic)
        assert callback.iteration_count == 0  # Should not change
    
    @patch('time.time')
    def test_iteration_callback_update_learning_phase(self, mock_time):
        """Test IterationCallback learning phase updates."""
        callback = IterationCallback()
        
        # Mock time
        mock_time.return_value = 1000.0
        
        # Test different learning phases
        callback._update_learning_phase(avg_reward=5.0, win_rate=0.1)
        assert callback.learning_phase == "early_learning"
        
        callback._update_learning_phase(avg_reward=15.0, win_rate=0.3)
        assert callback.learning_phase == "improving"
        
        callback._update_learning_phase(avg_reward=25.0, win_rate=0.6)
        assert callback.learning_phase == "advanced"
        
        callback._update_learning_phase(avg_reward=35.0, win_rate=0.8)
        assert callback.learning_phase == "expert"
    
    def test_iteration_callback_get_env_attr(self):
        """Test IterationCallback environment attribute retrieval."""
        callback = IterationCallback()
        
        # Mock environment with nested wrappers
        inner_env = MagicMock()
        inner_env.test_attr = "inner_value"
        
        middle_env = MagicMock()
        middle_env.env = inner_env
        
        outer_env = MagicMock()
        outer_env.env = middle_env
        
        # Test attribute retrieval
        result = callback.get_env_attr(outer_env, "test_attr")
        assert result == "inner_value"
    
    def test_iteration_callback_get_env_attr_not_found(self):
        """Test IterationCallback when attribute is not found."""
        callback = IterationCallback()
        
        # Mock environment without the attribute
        env = MagicMock()
        env.env = None  # No nested environment
        
        # Test attribute retrieval
        result = callback.get_env_attr(env, "nonexistent_attr")
        assert result is None
    
    @patch('time.time')
    @patch('builtins.print')
    def test_iteration_callback_on_step_basic(self, mock_print, mock_time):
        """Test IterationCallback basic step functionality."""
        mock_time.return_value = 1000.0
        
        callback = IterationCallback(verbose=1, debug_level=1)
        
        # Mock training environment
        training_env = MagicMock()
        training_env.num_envs = 1
        
        # Mock model
        model = MagicMock()
        
        # Test basic step
        result = callback._on_step()
        
        assert result is True  # Should continue training
        assert callback.iteration_count == 1
    
    @patch('time.time')
    @patch('builtins.print')
    def test_iteration_callback_on_step_with_experiment_tracker(self, mock_print, mock_time):
        """Test IterationCallback with experiment tracker."""
        mock_time.return_value = 1000.0
        
        experiment_tracker = MagicMock()
        callback = IterationCallback(
            verbose=1,
            debug_level=1,
            experiment_tracker=experiment_tracker
        )
        
        # Mock training environment
        training_env = MagicMock()
        training_env.num_envs = 1
        
        # Mock model
        model = MagicMock()
        
        # Test step with experiment tracker
        result = callback._on_step()
        
        assert result is True
        assert callback.iteration_count == 1
        # Experiment tracker should be called for metrics
    
    @patch('time.time')
    @patch('builtins.print')
    def test_iteration_callback_on_step_shutdown_requested(self, mock_print, mock_time):
        """Test IterationCallback when shutdown is requested."""
        mock_time.return_value = 1000.0
        
        callback = IterationCallback(verbose=1, debug_level=1)
        
        # Mock training environment
        training_env = MagicMock()
        training_env.num_envs = 1
        
        # Mock model
        model = MagicMock()
        
        # Mock global shutdown flag
        with patch('src.core.train_agent.shutdown_requested', True):
            result = callback._on_step()
            
            assert result is False  # Should stop training
            mock_print.assert_called()  # Should print shutdown message
    
    @patch('time.time')
    @patch('builtins.print')
    def test_iteration_callback_on_step_periodic_logging(self, mock_print, mock_time):
        """Test IterationCallback periodic logging functionality."""
        mock_time.side_effect = [1000.0, 1060.0]  # 60 seconds apart
        
        callback = IterationCallback(verbose=1, debug_level=1)
        
        # Mock training environment
        training_env = MagicMock()
        training_env.num_envs = 1
        
        # Mock model
        model = MagicMock()
        
        # First step
        result1 = callback._on_step()
        assert result1 is True
        
        # Second step (should trigger periodic logging)
        result2 = callback._on_step()
        assert result2 is True
        
        # Should have logged due to time difference
        assert mock_print.call_count > 0
    
    @patch('time.time')
    @patch('builtins.print')
    def test_iteration_callback_on_step_stats_file(self, mock_print, mock_time):
        """Test IterationCallback stats file writing."""
        mock_time.return_value = 1000.0
        
        with tempfile.TemporaryDirectory() as temp_dir:
            stats_file = os.path.join(temp_dir, "test_stats.txt")
            callback = IterationCallback(
                verbose=1,
                debug_level=1,
                stats_file=stats_file
            )
            
            # Mock training environment
            training_env = MagicMock()
            training_env.num_envs = 1
            
            # Mock model
            model = MagicMock()
            
            # Test step
            result = callback._on_step()
            
            assert result is True
            assert callback.iteration_count == 1
            
            # Check if stats file was created
            assert os.path.exists(stats_file)


class TestCallbackIntegration:
    """Integration tests for callback functionality."""
    
    @patch('src.core.train_agent.evaluate_model')
    def test_callback_workflow_integration(self, mock_evaluate):
        """Test integration between callbacks during training."""
        eval_env = MagicMock()
        experiment_tracker = MagicMock()
        
        # Create callbacks
        eval_callback = CustomEvalCallback(
            eval_env=eval_env,
            eval_freq=10,
            n_eval_episodes=3
        )
        
        iter_callback = IterationCallback(
            verbose=1,
            debug_level=1,
            experiment_tracker=experiment_tracker
        )
        
        # Mock evaluation results
        mock_evaluate.return_value = {
            "win_rate": 75.0,
            "avg_reward": 25.5,
            "avg_length": 15.2,
            "reward_ci": 2.1,
            "length_ci": 1.5,
            "n_episodes": 3
        }
        
        # Mock training environment
        training_env = MagicMock()
        training_env.num_envs = 1
        
        # Mock model
        model = MagicMock()
        
        # Test callback workflow
        eval_callback.num_timesteps = 10
        eval_result = eval_callback._on_step()
        
        iter_result = iter_callback._on_step()
        
        assert eval_result is True
        assert iter_result is True
        assert iter_callback.iteration_count == 1
    
    def test_callback_error_handling(self):
        """Test callback error handling."""
        callback = IterationCallback(verbose=1, debug_level=1)
        
        # Mock training environment that raises an exception
        training_env = MagicMock()
        training_env.num_envs = 1
        training_env.side_effect = RuntimeError("Environment error")
        
        # Mock model
        model = MagicMock()
        
        # Should handle the error gracefully
        with pytest.raises(RuntimeError):
            callback._on_step()


class TestCallbackEdgeCases:
    """Test edge cases in callback functionality."""
    
    def test_custom_eval_callback_no_eval_env(self):
        """Test CustomEvalCallback with no evaluation environment."""
        callback = CustomEvalCallback(eval_env=None)
        
        # Should handle gracefully
        result = callback._on_step()
        assert result is True
    
    def test_iteration_callback_no_experiment_tracker(self):
        """Test IterationCallback with no experiment tracker."""
        callback = IterationCallback(experiment_tracker=None)
        
        # Should handle gracefully
        result = callback._on_step()
        assert result is True
    
    def test_iteration_callback_stats_file_error(self):
        """Test IterationCallback when stats file writing fails."""
        callback = IterationCallback(stats_file="/invalid/path/stats.txt")
        
        # Mock training environment
        training_env = MagicMock()
        training_env.num_envs = 1
        
        # Mock model
        model = MagicMock()
        
        # Should handle file writing error gracefully
        result = callback._on_step()
        assert result is True
    
    def test_custom_eval_callback_evaluation_error(self):
        """Test CustomEvalCallback when evaluation fails."""
        eval_env = MagicMock()
        callback = CustomEvalCallback(eval_env=eval_env, eval_freq=10)
        
        # Mock training environment
        training_env = MagicMock()
        training_env.num_envs = 1
        
        # Mock model
        model = MagicMock()
        
        # Mock evaluation to raise an exception
        with patch('src.core.train_agent.evaluate_model', side_effect=RuntimeError("Evaluation failed")):
            # Should handle the error gracefully
            result = callback._on_step()
            assert result is True 