"""
Unit tests for error handling in train_agent.py main function

This module tests critical error handling scenarios:
- Shutdown handling (graceful shutdown)
- MLflow integration failures
- Model saving errors
- Training interruption (KeyboardInterrupt)
- Resource cleanup
- Error recovery mechanisms
"""

import pytest
import os
import tempfile
import shutil
import signal
import sys
from unittest.mock import patch, MagicMock, mock_open
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.core.train_agent import (
    main,
    signal_handler,
    shutdown_requested,
    make_env,
    ExperimentTracker
)


class TestShutdownHandling:
    """Test graceful shutdown handling during training."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_signal_handler_sets_shutdown_flag(self):
        """Test that signal handler correctly sets shutdown flag."""
        # Import the global variable from the module
        import src.core.train_agent as train_agent_module
        
        # Reset global flag
        train_agent_module.shutdown_requested = False
        
        # Simulate SIGINT signal
        signal_handler(signal.SIGINT, None)
        
        assert train_agent_module.shutdown_requested is True
    
    @patch('src.core.train_agent.shutdown_requested', True)
    def test_main_function_checks_shutdown_before_training(self):
        """Test that main function checks shutdown flag before starting training."""
        with patch('src.core.train_agent.parse_args') as mock_parse:
            # Mock args to avoid actual training
            mock_args = MagicMock()
            mock_args.total_timesteps = 1000
            mock_args.verbose = 0
            mock_args.eval_freq = 100
            mock_args.n_eval_episodes = 5
            mock_args.policy = "MlpPolicy"
            mock_args.learning_rate = 0.0003
            mock_args.n_steps = 2048
            mock_args.batch_size = 64
            mock_args.n_epochs = 10
            mock_args.gamma = 0.99
            mock_args.gae_lambda = 0.95
            mock_args.clip_range = 0.2
            mock_args.clip_range_vf = None
            mock_args.ent_coef = 0.0
            mock_args.vf_coef = 0.5
            mock_args.max_grad_norm = 0.5
            mock_args.use_sde = False
            mock_args.sde_sample_freq = -1
            mock_args.target_kl = None
            mock_args.seed = None
            mock_args.device = "auto"
            mock_args._init_setup_model = True
            mock_args.strict_progression = False
            mock_args.timestamped_stats = False
            mock_parse.return_value = mock_args
            
            # Mock MLflow to avoid actual logging
            with patch('src.core.train_agent.mlflow') as mock_mlflow:
                mock_mlflow.start_run.return_value.__enter__ = MagicMock()
                mock_mlflow.start_run.return_value.__exit__ = MagicMock()
                
                # Mock PPO to avoid actual model creation
                with patch('src.core.train_agent.PPO') as mock_ppo:
                    mock_model = MagicMock()
                    mock_ppo.return_value = mock_model
                    
                    # Mock environment creation
                    with patch('src.core.train_agent.make_env') as mock_make_env:
                        mock_env = MagicMock()
                        mock_make_env.return_value = mock_env
                        
                        # Mock DummyVecEnv
                        with patch('src.core.train_agent.DummyVecEnv') as mock_dummy_vec:
                            mock_vec_env = MagicMock()
                            mock_dummy_vec.return_value = mock_vec_env
                            
                            # Call main function
                            with patch('builtins.print'):  # Suppress prints
                                main()
                            
                            # Verify that training was not started due to shutdown
                            mock_model.learn.assert_not_called()


class TestMLflowIntegrationFailures:
    """Test handling of MLflow integration failures."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @patch('src.core.train_agent.shutdown_requested', False)
    def test_main_function_handles_mlflow_start_failure(self):
        """Test that main function handles MLflow start_run failure gracefully."""
        with patch('src.core.train_agent.parse_args') as mock_parse:
            # Mock args
            mock_args = MagicMock()
            mock_args.total_timesteps = 100
            mock_args.verbose = 0
            mock_args.eval_freq = 50
            mock_args.n_eval_episodes = 2
            mock_args.policy = "MlpPolicy"
            mock_args.learning_rate = 0.0003
            mock_args.n_steps = 64
            mock_args.batch_size = 32
            mock_args.n_epochs = 4
            mock_args.gamma = 0.99
            mock_args.gae_lambda = 0.95
            mock_args.clip_range = 0.2
            mock_args.clip_range_vf = None
            mock_args.ent_coef = 0.0
            mock_args.vf_coef = 0.5
            mock_args.max_grad_norm = 0.5
            mock_args.use_sde = False
            mock_args.sde_sample_freq = -1
            mock_args.target_kl = None
            mock_args.seed = None
            mock_args.device = "auto"
            mock_args._init_setup_model = True
            mock_args.strict_progression = False
            mock_args.timestamped_stats = False
            mock_parse.return_value = mock_args
            
            # Mock MLflow to raise exception
            with patch('src.core.train_agent.mlflow') as mock_mlflow:
                mock_mlflow.start_run.side_effect = Exception("MLflow not available")
                
                # Mock PPO to avoid actual model creation
                with patch('src.core.train_agent.PPO') as mock_ppo:
                    mock_model = MagicMock()
                    mock_ppo.return_value = mock_model
                    
                    # Mock environment creation
                    with patch('src.core.train_agent.make_env') as mock_make_env:
                        mock_env = MagicMock()
                        mock_make_env.return_value = mock_env
                        
                        # Mock DummyVecEnv
                        with patch('src.core.train_agent.DummyVecEnv') as mock_dummy_vec:
                            mock_vec_env = MagicMock()
                            mock_dummy_vec.return_value = mock_vec_env
                            
                            # Call main function - should not crash
                            with patch('builtins.print'):  # Suppress prints
                                main()
                            
                            # Verify that training still proceeded despite MLflow failure
                            mock_model.learn.assert_called()


class TestModelSavingErrors:
    """Test handling of model saving errors."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @patch('src.core.train_agent.shutdown_requested', False)
    def test_main_function_handles_model_save_failure(self):
        """Test that main function handles model saving failures gracefully."""
        with patch('src.core.train_agent.parse_args') as mock_parse:
            # Mock args
            mock_args = MagicMock()
            mock_args.total_timesteps = 100
            mock_args.verbose = 0
            mock_args.eval_freq = 50
            mock_args.n_eval_episodes = 2
            mock_args.policy = "MlpPolicy"
            mock_args.learning_rate = 0.0003
            mock_args.n_steps = 64
            mock_args.batch_size = 32
            mock_args.n_epochs = 4
            mock_args.gamma = 0.99
            mock_args.gae_lambda = 0.95
            mock_args.clip_range = 0.2
            mock_args.clip_range_vf = None
            mock_args.ent_coef = 0.0
            mock_args.vf_coef = 0.5
            mock_args.max_grad_norm = 0.5
            mock_args.use_sde = False
            mock_args.sde_sample_freq = -1
            mock_args.target_kl = None
            mock_args.seed = None
            mock_args.device = "auto"
            mock_args._init_setup_model = True
            mock_args.strict_progression = False
            mock_args.timestamped_stats = False
            mock_parse.return_value = mock_args
            
            # Mock MLflow
            with patch('src.core.train_agent.mlflow') as mock_mlflow:
                mock_mlflow.start_run.return_value.__enter__ = MagicMock()
                mock_mlflow.start_run.return_value.__exit__ = MagicMock()
                
                # Mock PPO with save failure
                with patch('src.core.train_agent.PPO') as mock_ppo:
                    mock_model = MagicMock()
                    mock_model.save.side_effect = PermissionError("Permission denied")
                    mock_ppo.return_value = mock_model
                    
                    # Mock environment creation
                    with patch('src.core.train_agent.make_env') as mock_make_env:
                        mock_env = MagicMock()
                        mock_make_env.return_value = mock_env
                        
                        # Mock DummyVecEnv
                        with patch('src.core.train_agent.DummyVecEnv') as mock_dummy_vec:
                            mock_vec_env = MagicMock()
                            mock_dummy_vec.return_value = mock_vec_env
                            
                            # Call main function - should not crash
                            with patch('builtins.print'):  # Suppress prints
                                main()
                            
                            # Verify that training completed despite save failure
                            mock_model.learn.assert_called()


class TestTrainingInterruption:
    """Test handling of training interruption scenarios."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @patch('src.core.train_agent.shutdown_requested', False)
    def test_main_function_handles_keyboard_interrupt(self):
        """Test that main function handles KeyboardInterrupt gracefully."""
        with patch('src.core.train_agent.parse_args') as mock_parse:
            # Mock args
            mock_args = MagicMock()
            mock_args.total_timesteps = 1000
            mock_args.verbose = 0
            mock_args.eval_freq = 100
            mock_args.n_eval_episodes = 5
            mock_args.policy = "MlpPolicy"
            mock_args.learning_rate = 0.0003
            mock_args.n_steps = 64
            mock_args.batch_size = 32
            mock_args.n_epochs = 4
            mock_args.gamma = 0.99
            mock_args.gae_lambda = 0.95
            mock_args.clip_range = 0.2
            mock_args.clip_range_vf = None
            mock_args.ent_coef = 0.0
            mock_args.vf_coef = 0.5
            mock_args.max_grad_norm = 0.5
            mock_args.use_sde = False
            mock_args.sde_sample_freq = -1
            mock_args.target_kl = None
            mock_args.seed = None
            mock_args.device = "auto"
            mock_args._init_setup_model = True
            mock_args.strict_progression = False
            mock_args.timestamped_stats = False
            mock_parse.return_value = mock_args
            
            # Mock MLflow
            with patch('src.core.train_agent.mlflow') as mock_mlflow:
                mock_mlflow.start_run.return_value.__enter__ = MagicMock()
                mock_mlflow.start_run.return_value.__exit__ = MagicMock()
                
                # Mock PPO with KeyboardInterrupt during training
                with patch('src.core.train_agent.PPO') as mock_ppo:
                    mock_model = MagicMock()
                    mock_model.learn.side_effect = KeyboardInterrupt("Training interrupted")
                    mock_ppo.return_value = mock_model
                    
                    # Mock environment creation
                    with patch('src.core.train_agent.make_env') as mock_make_env:
                        mock_env = MagicMock()
                        mock_make_env.return_value = mock_env
                        
                        # Mock DummyVecEnv
                        with patch('src.core.train_agent.DummyVecEnv') as mock_dummy_vec:
                            mock_vec_env = MagicMock()
                            mock_dummy_vec.return_value = mock_vec_env
                            
                            # Call main function - should handle KeyboardInterrupt
                            with patch('builtins.print'):  # Suppress prints
                                main()
                            
                            # Verify that checkpoint was saved
                            mock_model.save.assert_called_with("models/checkpoint_interrupted")
    
    @patch('src.core.train_agent.shutdown_requested', False)
    def test_main_function_handles_general_exception(self):
        """Test that main function handles general exceptions gracefully."""
        with patch('src.core.train_agent.parse_args') as mock_parse:
            # Mock args
            mock_args = MagicMock()
            mock_args.total_timesteps = 1000
            mock_args.verbose = 0
            mock_args.eval_freq = 100
            mock_args.n_eval_episodes = 5
            mock_args.policy = "MlpPolicy"
            mock_args.learning_rate = 0.0003
            mock_args.n_steps = 64
            mock_args.batch_size = 32
            mock_args.n_epochs = 4
            mock_args.gamma = 0.99
            mock_args.gae_lambda = 0.95
            mock_args.clip_range = 0.2
            mock_args.clip_range_vf = None
            mock_args.ent_coef = 0.0
            mock_args.vf_coef = 0.5
            mock_args.max_grad_norm = 0.5
            mock_args.use_sde = False
            mock_args.sde_sample_freq = -1
            mock_args.target_kl = None
            mock_args.seed = None
            mock_args.device = "auto"
            mock_args._init_setup_model = True
            mock_args.strict_progression = False
            mock_args.timestamped_stats = False
            mock_parse.return_value = mock_args
            
            # Mock MLflow
            with patch('src.core.train_agent.mlflow') as mock_mlflow:
                mock_mlflow.start_run.return_value.__enter__ = MagicMock()
                mock_mlflow.start_run.return_value.__exit__ = MagicMock()
                
                # Mock PPO with general exception during training
                with patch('src.core.train_agent.PPO') as mock_ppo:
                    mock_model = MagicMock()
                    mock_model.learn.side_effect = RuntimeError("Training failed")
                    mock_ppo.return_value = mock_model
                    
                    # Mock environment creation
                    with patch('src.core.train_agent.make_env') as mock_make_env:
                        mock_env = MagicMock()
                        mock_make_env.return_value = mock_env
                        
                        # Mock DummyVecEnv
                        with patch('src.core.train_agent.DummyVecEnv') as mock_dummy_vec:
                            mock_vec_env = MagicMock()
                            mock_dummy_vec.return_value = mock_vec_env
                            
                            # Call main function - should handle exception and save checkpoint, then re-raise
                            with patch('builtins.print'):  # Suppress prints
                                with pytest.raises(RuntimeError, match="Training failed"):
                                    main()
                            
                            # Verify that checkpoint was saved
                            mock_model.save.assert_called_with("models/checkpoint_error")


class TestResourceCleanup:
    """Test proper resource cleanup in error scenarios."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @patch('src.core.train_agent.shutdown_requested', False)
    def test_main_function_cleans_up_resources_on_success(self):
        """Test that main function cleans up resources on successful completion."""
        with patch('src.core.train_agent.parse_args') as mock_parse:
            # Mock args
            mock_args = MagicMock()
            mock_args.total_timesteps = 100
            mock_args.verbose = 0
            mock_args.eval_freq = 50
            mock_args.n_eval_episodes = 2
            mock_args.policy = "MlpPolicy"
            mock_args.learning_rate = 0.0003
            mock_args.n_steps = 64
            mock_args.batch_size = 32
            mock_args.n_epochs = 4
            mock_args.gamma = 0.99
            mock_args.gae_lambda = 0.95
            mock_args.clip_range = 0.2
            mock_args.clip_range_vf = None
            mock_args.ent_coef = 0.0
            mock_args.vf_coef = 0.5
            mock_args.max_grad_norm = 0.5
            mock_args.use_sde = False
            mock_args.sde_sample_freq = -1
            mock_args.target_kl = None
            mock_args.seed = None
            mock_args.device = "auto"
            mock_args._init_setup_model = True
            mock_args.strict_progression = False
            mock_args.timestamped_stats = False
            mock_parse.return_value = mock_args
            
            # Mock MLflow
            with patch('src.core.train_agent.mlflow') as mock_mlflow:
                mock_mlflow.start_run.return_value.__enter__ = MagicMock()
                mock_mlflow.start_run.return_value.__exit__ = MagicMock()
                
                # Mock PPO
                with patch('src.core.train_agent.PPO') as mock_ppo:
                    mock_model = MagicMock()
                    mock_ppo.return_value = mock_model
                    
                    # Mock environment creation
                    with patch('src.core.train_agent.make_env') as mock_make_env:
                        mock_env = MagicMock()
                        mock_make_env.return_value = mock_env
                        
                        # Mock DummyVecEnv
                        with patch('src.core.train_agent.DummyVecEnv') as mock_dummy_vec:
                            mock_vec_env = MagicMock()
                            mock_dummy_vec.return_value = mock_vec_env
                            
                            # Call main function
                            with patch('builtins.print'):  # Suppress prints
                                main()
                            
                            # Verify that environments were closed
                            mock_vec_env.close.assert_called()
    
    @patch('src.core.train_agent.shutdown_requested', False)
    def test_main_function_cleans_up_resources_on_exception(self):
        """Test that main function cleans up resources even when exceptions occur."""
        with patch('src.core.train_agent.parse_args') as mock_parse:
            # Mock args
            mock_args = MagicMock()
            mock_args.total_timesteps = 1000
            mock_args.verbose = 0
            mock_args.eval_freq = 100
            mock_args.n_eval_episodes = 5
            mock_args.policy = "MlpPolicy"
            mock_args.learning_rate = 0.0003
            mock_args.n_steps = 64
            mock_args.batch_size = 32
            mock_args.n_epochs = 4
            mock_args.gamma = 0.99
            mock_args.gae_lambda = 0.95
            mock_args.clip_range = 0.2
            mock_args.clip_range_vf = None
            mock_args.ent_coef = 0.0
            mock_args.vf_coef = 0.5
            mock_args.max_grad_norm = 0.5
            mock_args.use_sde = False
            mock_args.sde_sample_freq = -1
            mock_args.target_kl = None
            mock_args.seed = None
            mock_args.device = "auto"
            mock_args._init_setup_model = True
            mock_args.strict_progression = False
            mock_args.timestamped_stats = False
            mock_parse.return_value = mock_args
            
            # Mock MLflow
            with patch('src.core.train_agent.mlflow') as mock_mlflow:
                mock_mlflow.start_run.return_value.__enter__ = MagicMock()
                mock_mlflow.start_run.return_value.__exit__ = MagicMock()
                
                # Mock PPO with exception
                with patch('src.core.train_agent.PPO') as mock_ppo:
                    mock_model = MagicMock()
                    mock_model.learn.side_effect = RuntimeError("Training failed")
                    mock_ppo.return_value = mock_model
                    
                    # Mock environment creation
                    with patch('src.core.train_agent.make_env') as mock_make_env:
                        mock_env = MagicMock()
                        mock_make_env.return_value = mock_env
                        
                        # Mock DummyVecEnv
                        with patch('src.core.train_agent.DummyVecEnv') as mock_dummy_vec:
                            mock_vec_env = MagicMock()
                            mock_dummy_vec.return_value = mock_vec_env
                            
                            # Call main function - should clean up despite exception, then re-raise
                            with patch('builtins.print'):  # Suppress prints
                                with pytest.raises(RuntimeError, match="Training failed"):
                                    main()
                            
                            # Verify that environments were closed even after exception
                            mock_vec_env.close.assert_called()
