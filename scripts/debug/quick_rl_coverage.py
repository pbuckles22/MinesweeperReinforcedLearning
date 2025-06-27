#!/usr/bin/env python3
"""
Quick RL Coverage Analysis - Minimal memory footprint
Only runs essential RL tests without heavy training simulations.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path

def create_minimal_rl_test():
    """Create a minimal RL test that only tests essential functionality."""
    
    minimal_test = '''
import sys
import os
import tempfile
import shutil
import json
import numpy as np
import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.train_agent import (
    ExperimentTracker,
    IterationCallback,
    make_env,
    evaluate_model
)
from src.core.minesweeper_env import MinesweeperEnv

class TestMinimalRL(unittest.TestCase):
    """Minimal RL tests for coverage analysis."""
    
    def test_experiment_tracker_init(self):
        """Test ExperimentTracker initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = ExperimentTracker(experiment_dir=temp_dir)
            assert tracker.experiment_dir == temp_dir
            assert tracker.current_run is None
            assert "training" in tracker.metrics
            assert "validation" in tracker.metrics
    
    def test_experiment_tracker_start_run(self):
        """Test starting a new experiment run."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = ExperimentTracker(experiment_dir=temp_dir)
            hyperparams = {"learning_rate": 0.001}
            tracker.start_new_run(hyperparams)
            assert tracker.current_run is not None
            assert os.path.exists(tracker.current_run)
    
    def test_iteration_callback_init(self):
        """Test IterationCallback initialization."""
        callback = IterationCallback(verbose=0, debug_level=1)
        assert callback.iterations == 0
        assert callback.debug_level == 1
        assert callback.learning_phase == "Initial Random"
    
    def test_make_env_basic(self):
        """Test basic environment creation."""
        env_fn = make_env(max_board_size=4, max_mines=2)
        env = env_fn()
        # Environment is wrapped in Monitor, so check the underlying env
        from stable_baselines3.common.monitor import Monitor
        assert isinstance(env, Monitor)
        underlying_env = env.env
        assert isinstance(underlying_env, MinesweeperEnv)
        assert underlying_env.max_board_size_int == 4
        assert underlying_env.max_mines == 2
    
    @patch('src.core.train_agent.PPO')
    def test_evaluate_model_mock(self, mock_ppo):
        """Test evaluate_model with mocked components."""
        # Mock the model
        mock_model = Mock()
        mock_model.predict.return_value = ([0], None)
        
        # Mock the environment
        mock_env = Mock()
        mock_env.reset.return_value = (np.zeros((4, 4, 4)), {})
        mock_env.step.return_value = (np.zeros((4, 4, 4)), 10.0, True, False, {})
        
        # Test evaluation
        result = evaluate_model(mock_model, mock_env, 2)
        assert isinstance(result, dict)
        assert "mean_reward" in result or "avg_reward" in result
        assert "win_rate" in result

if __name__ == "__main__":
    # Run the tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMinimalRL)
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
'''
    
    return minimal_test

def run_minimal_rl_coverage():
    """Run minimal RL coverage analysis."""
    
    print("Creating minimal RL test for coverage analysis...")
    
    # Create temporary test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(create_minimal_rl_test())
        temp_test_file = f.name
    
    try:
        print("Running minimal RL coverage...")
        
        # Run coverage with the minimal test
        cmd = [
            sys.executable, "-m", "coverage", "run",
            "--source=src/core",
            "--omit=*/tests/*,*/venv/*,*/__pycache__/*",
            temp_test_file
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        
        if result.returncode == 0:
            print("✅ Minimal RL coverage completed successfully")
            
            # Generate reports
            print("Generating coverage reports...")
            
            # Terminal report
            subprocess.run([
                sys.executable, "-m", "coverage", "report",
                "--show-missing"
            ], check=True)
            
            # HTML report
            subprocess.run([
                sys.executable, "-m", "coverage", "html",
                "--directory=htmlcov/unit_rl_minimal"
            ], check=True)
            
            # JSON report
            subprocess.run([
                sys.executable, "-m", "coverage", "json",
                "-o=htmlcov/unit_rl_minimal/coverage.json"
            ], check=True)
            
            print("📊 Coverage reports generated in htmlcov/unit_rl_minimal/")
            return True
            
        else:
            print(f"❌ Minimal RL coverage failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Minimal RL coverage timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running minimal RL coverage: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists(temp_test_file):
            os.unlink(temp_test_file)

if __name__ == "__main__":
    success = run_minimal_rl_coverage()
    sys.exit(0 if success else 1) 