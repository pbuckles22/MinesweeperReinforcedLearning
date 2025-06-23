"""
Unit tests for ExperimentTracker class in train_agent.py

This module tests the experiment tracking functionality that manages
training metrics, validation data, and experiment persistence.
"""

import pytest
import json
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime
from src.core.train_agent import ExperimentTracker


class TestExperimentTrackerInitialization:
    """Test ExperimentTracker initialization and basic functionality."""
    
    def test_experiment_tracker_init_default(self):
        """Test ExperimentTracker initialization with default parameters."""
        tracker = ExperimentTracker()
        
        assert tracker.experiment_dir == "experiments"
        assert isinstance(tracker.metrics, dict)
        assert tracker.metrics == {
            "training": [],
            "validation": [],
            "hyperparameters": {},
            "metadata": {}
        }
        assert tracker.current_run is None
    
    def test_experiment_tracker_init_custom_dir(self):
        """Test ExperimentTracker initialization with custom directory."""
        tracker = ExperimentTracker("custom_experiments")
        
        assert tracker.experiment_dir == "custom_experiments"
        assert isinstance(tracker.metrics, dict)
    
    def test_experiment_tracker_init_creates_directory(self):
        """Test that ExperimentTracker creates the experiment directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = os.path.join(temp_dir, "test_experiments")
            
            tracker = ExperimentTracker(experiment_dir)
            
            assert os.path.exists(experiment_dir)
            assert tracker.experiment_dir == experiment_dir


class TestExperimentTrackerRunManagement:
    """Test experiment run management functionality."""
    
    def test_start_new_run(self):
        """Test starting a new experiment run."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = os.path.join(temp_dir, "test_experiments")
            tracker = ExperimentTracker(experiment_dir)
            
            hyperparameters = {
                "learning_rate": 0.001,
                "batch_size": 64,
                "total_timesteps": 1000000
            }
            
            tracker.start_new_run(hyperparameters)
            
            assert tracker.current_run is not None
            assert tracker.metrics["hyperparameters"] == hyperparameters
            assert "start_time" in tracker.metrics["metadata"]
            assert "random_seed" in tracker.metrics["metadata"]
    
    @patch('src.core.train_agent.datetime')
    def test_start_new_run_generates_unique_ids(self, mock_datetime):
        """Test that multiple runs generate unique IDs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = os.path.join(temp_dir, "test_experiments")
            tracker = ExperimentTracker(experiment_dir)
            
            hyperparameters = {"test": "value"}
            
            # Mock different timestamps
            mock_datetime.now.return_value.strftime.side_effect = ["20240101_120000", "20240101_120001"]
            
            # Start first run
            tracker.start_new_run(hyperparameters)
            first_run = tracker.current_run
            
            # Start second run
            tracker.start_new_run(hyperparameters)
            second_run = tracker.current_run
            
            assert first_run != second_run
            assert first_run is not None
            assert second_run is not None
    
    @patch('src.core.train_agent.datetime')
    def test_start_new_run_preserves_previous_metrics(self, mock_datetime):
        """Test that starting a new run preserves previous metrics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = os.path.join(temp_dir, "test_experiments")
            tracker = ExperimentTracker(experiment_dir)
            
            # Mock timestamp
            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"
            
            # Add some metrics to first run
            tracker.metrics["test_metric"] = 42
            tracker.start_new_run({"test": "value"})
            
            # Start second run
            tracker.start_new_run({"test2": "value2"})
            
            # Previous metrics should be preserved in the metrics dict
            assert "test_metric" in tracker.metrics
            assert tracker.metrics["test_metric"] == 42


class TestExperimentTrackerMetrics:
    """Test metric tracking functionality."""
    
    def test_add_training_metric(self):
        """Test adding training metrics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = os.path.join(temp_dir, "test_experiments")
            tracker = ExperimentTracker(experiment_dir)
            
            tracker.add_training_metric("loss", 0.5, 100)
            tracker.add_training_metric("reward", 15.2, 200)
            
            assert "training" in tracker.metrics
            assert len(tracker.metrics["training"]) == 2
            
            # Check first metric
            first_metric = tracker.metrics["training"][0]
            assert first_metric["metric"] == "loss"
            assert first_metric["value"] == 0.5
            assert first_metric["step"] == 100
            
            # Check second metric
            second_metric = tracker.metrics["training"][1]
            assert second_metric["metric"] == "reward"
            assert second_metric["value"] == 15.2
            assert second_metric["step"] == 200
    
    def test_add_validation_metric(self):
        """Test adding validation metrics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = os.path.join(temp_dir, "test_experiments")
            tracker = ExperimentTracker(experiment_dir)
            
            tracker.add_validation_metric("win_rate", 0.75)
            tracker.add_validation_metric("mean_reward", 12.5, confidence_interval=2.1)
            
            assert "validation" in tracker.metrics
            assert len(tracker.metrics["validation"]) == 2
            
            # Check first metric
            first_metric = tracker.metrics["validation"][0]
            assert first_metric["metric"] == "win_rate"
            assert first_metric["value"] == 0.75
            assert "confidence_interval" not in first_metric
            
            # Check second metric
            second_metric = tracker.metrics["validation"][1]
            assert second_metric["metric"] == "mean_reward"
            assert second_metric["value"] == 12.5
            assert second_metric["confidence_interval"] == 2.1
    
    def test_add_validation_metric_with_timestamp(self):
        """Test adding validation metrics with automatic timestamping."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = os.path.join(temp_dir, "test_experiments")
            tracker = ExperimentTracker(experiment_dir)
            
            tracker.add_validation_metric("test_metric", 42.0)
            
            validation_metrics = tracker.metrics["validation"]
            assert len(validation_metrics) == 1
            
            metric = validation_metrics[0]
            assert "timestamp" in metric
            assert isinstance(metric["timestamp"], str)
            
            # Verify timestamp is a valid ISO format
            datetime.fromisoformat(metric["timestamp"])


class TestExperimentTrackerPersistence:
    """Test experiment data persistence functionality."""
    
    def test_save_metrics_creates_file(self):
        """Test that _save_metrics creates a metrics file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = os.path.join(temp_dir, "test_experiments")
            tracker = ExperimentTracker(experiment_dir)
            
            # Add some metrics
            tracker.metrics["test_metric"] = 42
            tracker.metrics["nested_data"] = {"key": "value"}
            
            # Save metrics
            tracker._save_metrics()
            
            # Check that metrics file was created
            metrics_file = os.path.join(experiment_dir, "metrics.json")
            assert os.path.exists(metrics_file)
            
            # Verify content
            with open(metrics_file, 'r') as f:
                saved_metrics = json.load(f)
            
            assert saved_metrics["test_metric"] == 42
            assert saved_metrics["nested_data"]["key"] == "value"
    
    def test_save_metrics_handles_complex_data(self):
        """Test that _save_metrics handles complex data structures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = os.path.join(temp_dir, "test_experiments")
            tracker = ExperimentTracker(experiment_dir)
            
            # Add complex metrics
            tracker.metrics = {
                "simple_value": 42,
                "list_data": [1, 2, 3, 4, 5],
                "nested_dict": {
                    "level1": {
                        "level2": {
                            "level3": "deep_value"
                        }
                    }
                },
                "training_metrics": [
                    {"metric_name": "loss", "value": 0.5, "step": 100},
                    {"metric_name": "reward", "value": 15.2, "step": 200}
                ]
            }
            
            # Save metrics
            tracker._save_metrics()
            
            # Verify content
            metrics_file = os.path.join(experiment_dir, "metrics.json")
            with open(metrics_file, 'r') as f:
                saved_metrics = json.load(f)
            
            assert saved_metrics["simple_value"] == 42
            assert saved_metrics["list_data"] == [1, 2, 3, 4, 5]
            assert saved_metrics["nested_dict"]["level1"]["level2"]["level3"] == "deep_value"
            assert len(saved_metrics["training_metrics"]) == 2
    
    def test_save_metrics_handles_file_errors(self):
        """Test that _save_metrics handles file system errors gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = os.path.join(temp_dir, "test_experiments")
            tracker = ExperimentTracker(experiment_dir)
            
            # Add some metrics
            tracker.metrics["test_metric"] = 42
            
            # Mock open to raise an exception
            with pytest.raises(PermissionError):
                with patch('builtins.open', side_effect=PermissionError("Permission denied")):
                    tracker._save_metrics()
    
    def test_save_metrics_creates_backup(self):
        """Test that _save_metrics creates backup files for previous runs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = os.path.join(temp_dir, "test_experiments")
            tracker = ExperimentTracker(experiment_dir)
            
            # First save
            tracker.metrics["run1"] = "data1"
            tracker._save_metrics()
            
            # Second save
            tracker.metrics["run2"] = "data2"
            tracker._save_metrics()
            
            # Check that both files exist
            metrics_file = os.path.join(experiment_dir, "metrics.json")
            backup_file = os.path.join(experiment_dir, "metrics_backup.json")
            
            assert os.path.exists(metrics_file)
            assert os.path.exists(backup_file)
            
            # Verify current file has latest data
            with open(metrics_file, 'r') as f:
                current_metrics = json.load(f)
            assert "run2" in current_metrics
            
            # Verify backup file has previous data
            with open(backup_file, 'r') as f:
                backup_metrics = json.load(f)
            assert "run1" in backup_metrics


class TestExperimentTrackerIntegration:
    """Integration tests for ExperimentTracker workflow."""
    
    def test_complete_experiment_workflow(self):
        """Test complete experiment tracking workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = os.path.join(temp_dir, "test_experiments")
            tracker = ExperimentTracker(experiment_dir)
            
            # Start experiment
            hyperparameters = {
                "learning_rate": 0.001,
                "batch_size": 64,
                "total_timesteps": 1000000
            }
            tracker.start_new_run(hyperparameters)
            
            # Add training metrics
            for step in range(0, 1000, 100):
                tracker.add_training_metric("loss", 1.0 - step/1000, step)
                tracker.add_training_metric("reward", step/10, step)
            
            # Add validation metrics
            tracker.add_validation_metric("win_rate", 0.75)
            tracker.add_validation_metric("mean_reward", 12.5, confidence_interval=2.1)
            
            # Save metrics
            tracker._save_metrics()
            
            # Verify final state
            assert tracker.current_run is not None
            assert "run_id" in tracker.metrics
            assert "start_time" in tracker.metrics
            assert "hyperparameters" in tracker.metrics
            assert "training_metrics" in tracker.metrics
            assert "validation_metrics" in tracker.metrics
            
            # Verify training metrics
            training_metrics = tracker.metrics["training_metrics"]
            assert len(training_metrics) == 20  # 10 steps * 2 metrics per step
            
            # Verify validation metrics
            validation_metrics = tracker.metrics["validation_metrics"]
            assert len(validation_metrics) == 2
            
            # Verify file was created
            metrics_file = os.path.join(experiment_dir, "metrics.json")
            assert os.path.exists(metrics_file)
    
    def test_multiple_runs_workflow(self):
        """Test workflow with multiple experiment runs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = os.path.join(temp_dir, "test_experiments")
            tracker = ExperimentTracker(experiment_dir)
            
            # First run
            tracker.start_new_run({"lr": 0.001})
            tracker.add_training_metric("loss", 0.5, 100)
            tracker.add_validation_metric("win_rate", 0.6)
            tracker._save_metrics()
            
            # Second run
            tracker.start_new_run({"lr": 0.0001})
            tracker.add_training_metric("loss", 0.3, 100)
            tracker.add_validation_metric("win_rate", 0.8)
            tracker._save_metrics()
            
            # Verify both runs are tracked
            assert "run_id" in tracker.metrics
            assert len(tracker.metrics["training_metrics"]) == 2
            assert len(tracker.metrics["validation_metrics"]) == 2


class TestExperimentTrackerEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_experiment_tracker_with_empty_metrics(self):
        """Test ExperimentTracker with no metrics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = os.path.join(temp_dir, "test_experiments")
            tracker = ExperimentTracker(experiment_dir)
            
            # Save empty metrics
            tracker._save_metrics()
            
            metrics_file = os.path.join(experiment_dir, "metrics.json")
            assert os.path.exists(metrics_file)
            
            with open(metrics_file, 'r') as f:
                saved_metrics = json.load(f)
            
            assert saved_metrics == {}
    
    def test_experiment_tracker_with_none_values(self):
        """Test ExperimentTracker with None values in metrics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = os.path.join(temp_dir, "test_experiments")
            tracker = ExperimentTracker(experiment_dir)
            
            tracker.metrics = {
                "none_value": None,
                "valid_value": 42
            }
            
            tracker._save_metrics()
            
            metrics_file = os.path.join(experiment_dir, "metrics.json")
            with open(metrics_file, 'r') as f:
                saved_metrics = json.load(f)
            
            assert saved_metrics["none_value"] is None
            assert saved_metrics["valid_value"] == 42
    
    def test_experiment_tracker_directory_creation_failure(self):
        """Test ExperimentTracker when directory creation fails."""
        # Use a path that should be read-only or cause permission issues
        read_only_dir = "/tmp/readonly_test_dir"
        
        # Create a read-only directory
        try:
            os.makedirs(read_only_dir, exist_ok=True)
            os.chmod(read_only_dir, 0o444)  # Read-only
            
            # Should raise a PermissionError
            with pytest.raises(PermissionError):
                tracker = ExperimentTracker(os.path.join(read_only_dir, "experiments"))
        finally:
            # Clean up
            try:
                os.chmod(read_only_dir, 0o755)
                shutil.rmtree(read_only_dir)
            except:
                pass 