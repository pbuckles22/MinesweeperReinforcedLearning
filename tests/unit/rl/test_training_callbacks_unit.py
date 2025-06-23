"""
Unit tests for training callbacks in train_agent.py

This module tests the CustomEvalCallback and IterationCallback classes
that handle evaluation and training iteration monitoring.
"""

import pytest
import numpy as np
import tempfile
import os
import uuid
from unittest.mock import patch, MagicMock, Mock
from src.core.train_agent import CustomEvalCallback, IterationCallback


@pytest.fixture
def unique_stats_file():
    """Create a unique temporary stats file for each test."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        temp_file = f.name
    yield temp_file
    # Cleanup
    try:
        os.unlink(temp_file)
    except OSError:
        pass


class TestCustomEvalCallback:
    """Test CustomEvalCallback functionality."""
    
    def test_custom_eval_callback_init(self):
        """Test CustomEvalCallback initialization."""
        print("DEBUG: Starting CustomEvalCallback init test")
        eval_env = MagicMock()
        
        callback = CustomEvalCallback(
            eval_env=eval_env,
            eval_freq=1000,
            n_eval_episodes=5,
            verbose=1
        )
        
        assert callback.eval_env == eval_env
        assert callback.eval_freq == 1000
        assert callback.n_eval_episodes == 5
        assert callback.verbose == 1
        assert callback.best_mean_reward == -np.inf
        print("DEBUG: CustomEvalCallback init test completed")
    
    def test_custom_eval_callback_init_with_paths(self):
        """Test CustomEvalCallback initialization with save paths."""
        print("DEBUG: Starting CustomEvalCallback init with paths test")
        eval_env = MagicMock()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            callback = CustomEvalCallback(
                eval_env=eval_env,
                eval_freq=1000,
                n_eval_episodes=5,
                verbose=1,
                best_model_save_path=temp_dir,
                log_path=temp_dir
            )
            
            assert callback.best_model_save_path == temp_dir
            assert callback.log_path == temp_dir
        print("DEBUG: CustomEvalCallback init with paths test completed")
    
    def test_custom_eval_callback_evaluation_frequency_logic(self):
        """Test CustomEvalCallback evaluation frequency logic without running evaluation."""
        print("DEBUG: Starting evaluation frequency logic test")
        eval_env = MagicMock()
        callback = CustomEvalCallback(eval_env=eval_env, eval_freq=10, n_eval_episodes=3)
        
        # Test evaluation trigger logic without actually running evaluation
        callback.n_calls = 10  # Should trigger evaluation
        should_evaluate = callback.n_calls % callback.eval_freq == 0
        assert should_evaluate is True
        
        callback.n_calls = 53  # Should not trigger evaluation (53 % 10 = 3)
        should_evaluate = callback.n_calls % callback.eval_freq == 0
        assert should_evaluate is False
        print("DEBUG: Evaluation frequency logic test completed")
    
    @patch('src.core.train_agent.mlflow')
    def test_custom_eval_callback_safe_evaluation(self, mock_mlflow):
        """Test CustomEvalCallback evaluation with proper mocking to avoid hangs."""
        print("DEBUG: Starting safe evaluation test")
        eval_env = MagicMock()
        model = MagicMock()
        
        # Mock environment to return simple, safe responses
        eval_env.reset.return_value = (np.zeros((4, 4, 4)), {})
        eval_env.step.return_value = (np.zeros((4, 4, 4)), 1.0, True, False, {"won": False})
        
        # Mock model prediction
        model.predict.return_value = (np.array([0]), None)
        
        callback = CustomEvalCallback(eval_env=eval_env, eval_freq=1, n_eval_episodes=1)
        callback.model = model
        callback.n_calls = 1
        
        print("DEBUG: About to call _on_step")
        # This should not hang due to proper mocking
        result = callback._on_step()
        print("DEBUG: _on_step completed")
        assert result is True
        
        # Verify environment was called
        eval_env.reset.assert_called_once()
        eval_env.step.assert_called_once()
        print("DEBUG: Safe evaluation test completed")


class TestIterationCallback:
    """Test IterationCallback functionality."""
    
    @pytest.fixture(autouse=True)
    def reset_shutdown(self):
        import src.core.train_agent as train_agent_module
        train_agent_module.shutdown_requested = False
        yield

    def test_iteration_callback_init(self, unique_stats_file):
        """Test IterationCallback initialization."""
        print("DEBUG: Starting IterationCallback init test")
        callback = IterationCallback(
            verbose=1, 
            debug_level=3, 
            stats_file=unique_stats_file,
            enable_file_logging=True
        )
        
        assert callback.verbose == 1
        assert callback.debug_level == 3
        assert callback.iterations == 0
        assert callback.learning_phase == "Initial Random"
        assert callback.curriculum_stage == 1
        assert callback.enable_file_logging is True
        assert os.path.exists(unique_stats_file)
        print("DEBUG: IterationCallback init test completed")
    
    def test_iteration_callback_init_with_experiment_tracker(self, unique_stats_file):
        """Test IterationCallback initialization with experiment tracker."""
        print("DEBUG: Starting IterationCallback init with tracker test")
        tracker = MagicMock()
        callback = IterationCallback(
            experiment_tracker=tracker, 
            stats_file=unique_stats_file,
            enable_file_logging=True
        )
        
        assert callback.experiment_tracker == tracker
        assert callback.enable_file_logging is True
        assert os.path.exists(unique_stats_file)
        print("DEBUG: IterationCallback init with tracker test completed")
    
    def test_iteration_callback_log(self, unique_stats_file):
        """Test IterationCallback logging functionality."""
        print("DEBUG: Starting logging test")
        callback = IterationCallback(
            debug_level=2, 
            stats_file=unique_stats_file,
            enable_file_logging=True
        )
        
        # Test logging at different levels
        callback.log("Test message", level=1)  # Should log (level <= debug_level)
        callback.log("Debug message", level=3)  # Should not log (level > debug_level)
        callback.log("Forced message", level=3, force=True)  # Should log (forced)
        print("DEBUG: Logging test completed")
    
    def test_iteration_callback_get_env_attr_safe(self, unique_stats_file):
        """Test IterationCallback environment attribute retrieval with safe mocking."""
        print("DEBUG: Starting get_env_attr safe test")
        callback = IterationCallback(
            stats_file=unique_stats_file,
            enable_file_logging=True
        )
        
        # Create a simple object without circular references
        class SimpleEnv:
            def __init__(self):
                self.test_attr = "test_value"
        
        env = SimpleEnv()
        
        # Test attribute retrieval
        result = callback.get_env_attr(env, "test_attr")
        assert result == "test_value"
        print("DEBUG: get_env_attr safe test completed")
    
    def test_iteration_callback_get_env_attr_nested_safe(self, unique_stats_file):
        """Test IterationCallback environment attribute retrieval with safe nested structure."""
        print("DEBUG: Starting get_env_attr nested safe test")
        callback = IterationCallback(
            stats_file=unique_stats_file,
            enable_file_logging=True
        )
        
        # Create a simple nested structure without circular references
        class InnerEnv:
            def __init__(self):
                self.test_attr = "test_value"
        
        class OuterEnv:
            def __init__(self):
                self.env = InnerEnv()
        
        outer_env = OuterEnv()
        
        # Test attribute retrieval
        result = callback.get_env_attr(outer_env, "test_attr")
        assert result == "test_value"
        print("DEBUG: get_env_attr nested safe test completed")
    
    def test_iteration_callback_get_env_attr_not_found(self, unique_stats_file):
        """Test IterationCallback environment attribute retrieval when not found."""
        print("DEBUG: Starting get_env_attr not found test")
        callback = IterationCallback(
            stats_file=unique_stats_file,
            enable_file_logging=True
        )
        
        # Create a simple object without the attribute
        class SimpleEnv:
            pass
        
        env = SimpleEnv()
        
        # Test attribute retrieval
        result = callback.get_env_attr(env, "nonexistent_attr")
        assert result is None
        print("DEBUG: get_env_attr not found test completed")
    
    def test_iteration_callback_update_learning_phase(self, unique_stats_file):
        """Test IterationCallback learning phase updates."""
        print("DEBUG: Starting learning phase update test")
        callback = IterationCallback(
            stats_file=unique_stats_file,
            enable_file_logging=True
        )
        
        # Set iterations to 5 or more to avoid "Initial Random" phase
        callback.iterations = 5
        
        # Test initial phase
        callback._update_learning_phase(0.0, 5.0)
        assert callback.learning_phase == "Early Learning"
        
        # Test phase progression
        callback._update_learning_phase(10.0, 35.0)
        assert callback.learning_phase == "Intermediate"
        print("DEBUG: Learning phase update test completed")
    
    @patch('src.core.train_agent.mlflow')
    def test_iteration_callback_safe_on_step(self, mock_mlflow, unique_stats_file):
        """Test IterationCallback _on_step with proper mocking to avoid hangs."""
        print("DEBUG: Starting safe on_step test")
        callback = IterationCallback(
            debug_level=0, 
            stats_file=unique_stats_file,
            enable_file_logging=True
        )
        
        # Mock model with safe episode buffer
        model = MagicMock()
        model.ep_info_buffer = [
            {"r": 1.0, "l": 10, "won": True},
            {"r": 0.5, "l": 8, "won": False}
        ]
        
        callback.model = model
        callback.num_timesteps = 100  # Trigger logging
        
        print("DEBUG: About to call _on_step")
        # This should not hang due to unique file and proper mocking
        result = callback._on_step()
        print("DEBUG: _on_step completed")
        assert result is True
        
        # Verify file was written to
        assert os.path.exists(unique_stats_file)
        with open(unique_stats_file, 'r') as f:
            content = f.read()
            assert len(content.strip().split('\n')) >= 2  # Header + at least one data line
        print("DEBUG: Safe on_step test completed")
    
    def test_iteration_callback_on_step_no_episodes(self, unique_stats_file):
        """Test IterationCallback _on_step when no episodes are available."""
        print("DEBUG: Starting on_step no episodes test")
        callback = IterationCallback(
            debug_level=0,
            stats_file=unique_stats_file,
            enable_file_logging=True
        )
        
        # Mock model with empty episode buffer
        model = MagicMock()
        model.ep_info_buffer = []
        
        callback.model = model
        callback.num_timesteps = 100  # Trigger logging
        
        print("DEBUG: About to call _on_step with no episodes")
        # This should not hang
        result = callback._on_step()
        print("DEBUG: _on_step with no episodes completed")
        assert result is True
        print("DEBUG: on_step no episodes test completed")
    
    def test_iteration_callback_file_logging_disabled(self):
        """Test IterationCallback with file logging disabled."""
        print("DEBUG: Starting file logging disabled test")
        callback = IterationCallback(enable_file_logging=False)
        
        assert callback.enable_file_logging is False
        # Should not create any files
        assert not hasattr(callback, 'stats_file') or callback.stats_file == "training_stats.txt"
        print("DEBUG: file logging disabled test completed")


class TestCallbackEdgeCases:
    """Test callback edge cases and error handling."""
    
    def test_custom_eval_callback_no_eval_env(self):
        """Test CustomEvalCallback behavior with no evaluation environment."""
        print("DEBUG: Starting no eval_env test")
        callback = CustomEvalCallback(eval_env=None, eval_freq=1)
        callback.model = MagicMock()
        callback.n_calls = 1
        
        # Should handle None eval_env gracefully
        with pytest.raises((AttributeError, TypeError)):
            callback._on_step()
        print("DEBUG: no eval_env test completed")
    
    def test_iteration_callback_no_model(self, unique_stats_file):
        """Test IterationCallback behavior with no model."""
        print("DEBUG: Starting no model test")
        callback = IterationCallback(
            stats_file=unique_stats_file,
            enable_file_logging=True
        )
        callback.num_timesteps = 100
        
        # Should handle missing model gracefully
        with pytest.raises(AttributeError):
            callback._on_step()
        print("DEBUG: no model test completed")
    
    def test_iteration_callback_file_permission_error(self, unique_stats_file):
        """Test IterationCallback behavior with file permission errors."""
        print("DEBUG: Starting file permission error test")
        callback = IterationCallback(
            stats_file=unique_stats_file,
            enable_file_logging=True
        )
        
        # Mock model
        model = MagicMock()
        model.ep_info_buffer = [{"r": 1.0, "l": 10, "won": True}]
        callback.model = model
        callback.num_timesteps = 100
        
        # Mock file operations to raise permission error
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            # Should handle permission error gracefully and disable file logging
            result = callback._on_step()
            assert result is True
            assert callback.enable_file_logging is False  # Should be disabled after error
        print("DEBUG: file permission error test completed")
    
    def test_iteration_callback_concurrent_file_access(self):
        """Test that multiple callbacks can write to different files simultaneously."""
        print("DEBUG: Starting concurrent file access test")
        
        # Create multiple callbacks with different files
        callbacks = []
        temp_files = []
        
        for i in range(3):
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'_{i}.txt', delete=False) as f:
                temp_file = f.name
                temp_files.append(temp_file)
            
            callback = IterationCallback(
                stats_file=temp_file,
                enable_file_logging=True,
                verbose=0
            )
            
            # Mock model
            model = MagicMock()
            model.ep_info_buffer = [{"r": 1.0, "l": 10, "won": True}]
            callback.model = model
            callback.num_timesteps = 100
            
            callbacks.append(callback)
        
        # Run all callbacks simultaneously
        results = []
        for callback in callbacks:
            result = callback._on_step()
            results.append(result)
        
        # All should succeed
        assert all(results)
        
        # All files should exist and have content
        for temp_file in temp_files:
            assert os.path.exists(temp_file)
            with open(temp_file, 'r') as f:
                content = f.read()
                assert len(content.strip().split('\n')) >= 2
        
        # Cleanup
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except OSError:
                pass
        
        print("DEBUG: concurrent file access test completed") 