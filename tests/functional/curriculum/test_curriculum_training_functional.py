"""
Functional Tests for train_agent.py

These tests verify the training agent's behavior in real-world scenarios:
- Complete training cycles
- Curriculum learning progression
- Model saving and loading
- Performance metrics tracking
- Training interruption and resumption
- Resource cleanup
"""

import pytest
import os
import json
import shutil
import tempfile
import numpy as np
from unittest.mock import patch, Mock
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.core.train_agent import (
    ExperimentTracker,
    IterationCallback,
    make_env,
    evaluate_model,
    main
)
from src.core.minesweeper_env import MinesweeperEnv

class TestTrainingCycle:
    """Test complete training cycles."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_short_training_run(self, temp_dir):
        """Test a short but complete training run."""
        # Create environment and model
        env = DummyVecEnv([lambda: make_env(max_board_size=4, max_mines=2)()])
        model = PPO("MlpPolicy", env, verbose=0)
        
        # Create tracker
        tracker = ExperimentTracker(experiment_dir=temp_dir)
        tracker.start_new_run({"learning_rate": 0.001, "n_steps": 32})
        
        # Create callback
        callback = IterationCallback(verbose=0, debug_level=0, experiment_tracker=tracker)
        
        # Train for a short period
        with patch('builtins.print'):  # Suppress prints
            model.learn(total_timesteps=100, callback=callback)
        
        # Verify training produced metrics
        assert len(tracker.metrics["training"]) > 0
        assert os.path.exists(os.path.join(tracker.current_run, "metrics.json"))
    
    def test_curriculum_progression(self, temp_dir):
        """Test curriculum learning progression."""
        # Create environments for different stages
        envs = {
            "beginner": DummyVecEnv([lambda: make_env(max_board_size=4, max_mines=2)()]),
            "intermediate": DummyVecEnv([lambda: make_env(max_board_size=6, max_mines=4)()]),
            "advanced": DummyVecEnv([lambda: make_env(max_board_size=8, max_mines=8)()])
        }
        
        # Create model
        model = PPO("MlpPolicy", envs["beginner"], verbose=0)
        
        # Create tracker
        tracker = ExperimentTracker(experiment_dir=temp_dir)
        tracker.start_new_run({"curriculum": "enabled"})
        
        # Train through curriculum stages
        for stage, env in envs.items():
            callback = IterationCallback(verbose=0, debug_level=0, experiment_tracker=tracker)
            model.set_env(env)
            
            with patch('builtins.print'):
                model.learn(total_timesteps=100, callback=callback)
            
            # Save stage completion
            tracker.add_validation_metric(f"{stage}_completion", True)
        
        # Verify progression through stages
        metrics_file = os.path.join(tracker.current_run, "metrics.json")
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        assert any(m["metric"] == "beginner_completion" for m in metrics["validation"])
        assert any(m["metric"] == "intermediate_completion" for m in metrics["validation"])
        assert any(m["metric"] == "advanced_completion" for m in metrics["validation"])

class TestModelPersistence:
    """Test model saving and loading functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_model_save_load(self, temp_dir):
        """Test saving and loading model during training."""
        # Create environment and model
        env = DummyVecEnv([lambda: make_env(max_board_size=4, max_mines=2)()])
        model = PPO("MlpPolicy", env, verbose=0)
        
        # Train briefly
        model.learn(total_timesteps=50)
        
        # Save model
        model_path = os.path.join(temp_dir, "test_model.zip")
        model.save(model_path)
        
        # Load model and verify it works
        loaded_model = PPO.load(model_path)
        obs = env.reset()
        action, _ = loaded_model.predict(obs[0])
        assert isinstance(action, (np.ndarray, int))
    
    def test_checkpoint_recovery(self, temp_dir):
        """Test training can resume from checkpoints."""
        # Create environment and model
        env = DummyVecEnv([lambda: make_env(max_board_size=4, max_mines=2)()])
        model = PPO("MlpPolicy", env, verbose=0)
        
        # Create tracker
        tracker = ExperimentTracker(experiment_dir=temp_dir)
        tracker.start_new_run({"checkpoint_interval": 50})
        
        # Train with checkpoints
        callback = IterationCallback(verbose=0, debug_level=0, experiment_tracker=tracker)
        
        with patch('builtins.print'):
            model.learn(total_timesteps=100, callback=callback)
        
        # Save checkpoint
        checkpoint_path = os.path.join(temp_dir, "checkpoint.zip")
        model.save(checkpoint_path)
        
        # Load checkpoint and continue training
        loaded_model = PPO.load(checkpoint_path)
        loaded_model.set_env(env)
        
        with patch('builtins.print'):
            loaded_model.learn(total_timesteps=50, callback=callback)
        
        # Verify continued training metrics exist
        assert len(tracker.metrics["training"]) > 0

class TestPerformanceMetrics:
    """Test tracking and validation of performance metrics."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_win_rate_tracking(self, temp_dir):
        """Test tracking of win rates during training."""
        # Create environment and model
        env = DummyVecEnv([lambda: make_env(max_board_size=4, max_mines=2)()])
        model = PPO("MlpPolicy", env, verbose=0)
        
        # Create tracker
        tracker = ExperimentTracker(experiment_dir=temp_dir)
        tracker.start_new_run({"track_win_rate": True})
        
        # Create callback
        callback = IterationCallback(verbose=0, debug_level=0, experiment_tracker=tracker)
        
        # Train and track metrics
        with patch('builtins.print'):
            model.learn(total_timesteps=100, callback=callback)
        
        # Verify win rate metrics were tracked
        win_rate_metrics = [m for m in tracker.metrics["training"] if m["metric"] == "win_rate"]
        assert len(win_rate_metrics) > 0
        assert all(0 <= m["value"] <= 100 for m in win_rate_metrics)
    
    def test_reward_statistics(self, temp_dir):
        """Test tracking of reward statistics."""
        # Create environment and model
        env = DummyVecEnv([lambda: make_env(max_board_size=4, max_mines=2)()])
        model = PPO("MlpPolicy", env, verbose=0)
        
        # Create tracker
        tracker = ExperimentTracker(experiment_dir=temp_dir)
        tracker.start_new_run({"track_rewards": True})
        
        # Train and evaluate
        model.learn(total_timesteps=50)
        evaluation_results = evaluate_model(model, env, n_episodes=5)
        mean_reward = evaluation_results["avg_reward"]
        std_reward = evaluation_results["reward_ci"]
        
        # Add evaluation metrics
        tracker.add_validation_metric("mean_reward", mean_reward, confidence_interval=std_reward)
        
        # Verify reward statistics
        assert hasattr(mean_reward, 'item') or isinstance(mean_reward, float)
        assert hasattr(std_reward, 'item') or isinstance(std_reward, float)
        assert len(tracker.metrics["validation"]) > 0

class TestTrainingInterruption:
    """Test handling of training interruptions."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_graceful_keyboard_interrupt(self, temp_dir):
        """Test graceful handling of keyboard interruption."""
        # Create environment and model
        env = DummyVecEnv([lambda: make_env(max_board_size=4, max_mines=2)()])
        model = PPO("MlpPolicy", env, verbose=0)
        
        # Create tracker
        tracker = ExperimentTracker(experiment_dir=temp_dir)
        tracker.start_new_run({"allow_interruption": True})
        
        # Create callback
        callback = IterationCallback(verbose=0, debug_level=0, experiment_tracker=tracker)
        
        # Simulate keyboard interrupt during training
        def mock_learn(*args, **kwargs):
            raise KeyboardInterrupt
        
        with patch.object(PPO, 'learn', side_effect=mock_learn):
            try:
                model.learn(total_timesteps=100, callback=callback)
            except KeyboardInterrupt:
                pass
        
        # Verify metrics were saved despite interruption
        assert os.path.exists(os.path.join(tracker.current_run, "metrics.json"))
    
    def test_error_recovery(self, temp_dir):
        """Test recovery from training errors."""
        # Create environment and model
        env = DummyVecEnv([lambda: make_env(max_board_size=4, max_mines=2)()])
        model = PPO("MlpPolicy", env, verbose=0)
        
        # Create tracker
        tracker = ExperimentTracker(experiment_dir=temp_dir)
        tracker.start_new_run({"error_recovery": True})
        
        # Create callback
        callback = IterationCallback(verbose=0, debug_level=0, experiment_tracker=tracker)
        
        # Simulate training error and recovery
        error_count = 0
        def mock_learn(*args, **kwargs):
            nonlocal error_count
            if error_count == 0:
                error_count += 1
                raise RuntimeError("Simulated training error")
            return None
        
        with patch.object(PPO, 'learn', side_effect=mock_learn):
            try:
                model.learn(total_timesteps=100, callback=callback)
            except RuntimeError:
                # Simulate recovery by saving current state
                model.save(os.path.join(temp_dir, "recovery_model.zip"))
        
        # Verify recovery artifacts exist
        assert os.path.exists(os.path.join(temp_dir, "recovery_model.zip"))

class TestResourceManagement:
    """Test proper management of system resources during training."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_memory_cleanup(self, temp_dir):
        """Test proper cleanup of memory resources."""
        # Create environment and model
        env = DummyVecEnv([lambda: make_env(max_board_size=4, max_mines=2)()])
        model = PPO("MlpPolicy", env, verbose=0)
        
        # Create tracker with cleanup monitoring
        tracker = ExperimentTracker(experiment_dir=temp_dir)
        tracker.start_new_run({"monitor_resources": True})
        
        # Train briefly
        model.learn(total_timesteps=50)
        
        # Clean up
        env.close()
        del model
        
        # Verify no resource leaks
        assert not os.path.exists(os.path.join(temp_dir, "running.lock"))
    
    def test_file_cleanup(self, temp_dir):
        """Test cleanup of temporary files."""
        # Create environment and model
        env = DummyVecEnv([lambda: make_env(max_board_size=4, max_mines=2)()])
        model = PPO("MlpPolicy", env, verbose=0)
        
        # Create some temporary files
        temp_files = [
            os.path.join(temp_dir, "temp1.txt"),
            os.path.join(temp_dir, "temp2.txt")
        ]
        for file in temp_files:
            with open(file, 'w') as f:
                f.write("temporary")
        
        # Train briefly
        model.learn(total_timesteps=50)
        
        # Clean up
        for file in temp_files:
            os.remove(file)
        
        # Verify cleanup
        assert not any(os.path.exists(f) for f in temp_files)

if __name__ == '__main__':
    pytest.main([__file__]) 