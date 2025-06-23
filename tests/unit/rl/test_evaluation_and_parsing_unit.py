"""
Unit tests for evaluation and argument parsing in train_agent.py

This module tests the evaluate_model function and argument parsing functionality
that are essential for model evaluation and command-line interface.
"""

import pytest
import numpy as np
import argparse
from unittest.mock import patch, MagicMock, Mock
from src.core.train_agent import evaluate_model, parse_args


class TestEvaluateModel:
    """Test the evaluate_model function with different environment types."""
    
    def test_evaluate_model_non_vectorized_env(self):
        """Test evaluation with non-vectorized environment."""
        # Mock model
        model = MagicMock()
        model.predict.return_value = (np.array([0]), None)
        
        # Mock environment
        env = MagicMock()
        env.reset.return_value = (np.zeros((4, 4, 4)), {})
        
        # Mock step to return gymnasium-style results (5 values)
        step_results = [
            (np.zeros((4, 4, 4)), 15.0, False, False, {"won": False}),  # Continue
            (np.zeros((4, 4, 4)), 15.0, False, False, {"won": False}),  # Continue
            (np.zeros((4, 4, 4)), 500.0, True, False, {"won": True}),   # Win
        ]
        env.step.side_effect = step_results
        
        # Evaluate
        results = evaluate_model(model, env, n_episodes=1)
        
        # Verify results
        assert results["win_rate"] == 100.0  # 1 win out of 1 episode
        assert results["avg_reward"] == 530.0  # 15 + 15 + 500
        assert results["avg_length"] == 3.0
        assert results["n_episodes"] == 1
        assert "reward_ci" in results
        assert "length_ci" in results
    
    def test_evaluate_model_vectorized_env(self):
        """Test evaluation with vectorized environment."""
        # Mock model
        model = MagicMock()
        model.predict.return_value = (np.array([[0]]), None)
        
        # Mock vectorized environment
        env = MagicMock()
        env.num_envs = 1
        env.reset.return_value = np.zeros((1, 4, 4, 4))
        
        # Mock step to return vectorized results
        step_results = [
            (np.zeros((1, 4, 4, 4)), np.array([15.0]), np.array([False]), np.array([False]), [{"won": False}]),
            (np.zeros((1, 4, 4, 4)), np.array([15.0]), np.array([False]), np.array([False]), [{"won": False}]),
            (np.zeros((1, 4, 4, 4)), np.array([500.0]), np.array([True]), np.array([False]), [{"won": True}]),
        ]
        env.step.side_effect = step_results
        
        # Evaluate
        results = evaluate_model(model, env, n_episodes=1)
        
        # Verify results
        assert results["win_rate"] == 100.0
        assert results["avg_reward"] == 530.0
        assert results["avg_length"] == 3.0
    
    def test_evaluate_model_old_gym_api(self):
        """Test evaluation with old gym API (4 values)."""
        # Mock model
        model = MagicMock()
        model.predict.return_value = (np.array([0]), None)
        
        # Mock environment
        env = MagicMock()
        env.reset.return_value = (np.zeros((4, 4, 4)), {})
        
        # Mock step to return old gym-style results (4 values)
        step_results = [
            (np.zeros((4, 4, 4)), 15.0, False, False),  # Continue
            (np.zeros((4, 4, 4)), 15.0, False, False),  # Continue
            (np.zeros((4, 4, 4)), 500.0, True, False),  # Win
        ]
        env.step.side_effect = step_results
        
        # Evaluate
        results = evaluate_model(model, env, n_episodes=1)
        
        # Verify results
        assert results["win_rate"] == 100.0
        assert results["avg_reward"] == 530.0
        assert results["avg_length"] == 3.0
    
    def test_evaluate_model_multiple_episodes(self):
        """Test evaluation with multiple episodes."""
        # Mock model
        model = MagicMock()
        model.predict.return_value = (np.array([0]), None)
        
        # Mock environment
        env = MagicMock()
        env.reset.return_value = (np.zeros((4, 4, 4)), {})
        
        # Mock step to return mixed results
        step_results = [
            # Episode 1: Win
            (np.zeros((4, 4, 4)), 500.0, True, False, {"won": True}),
            # Episode 2: Lose
            (np.zeros((4, 4, 4)), -20.0, True, False, {"won": False}),
            # Episode 3: Win
            (np.zeros((4, 4, 4)), 500.0, True, False, {"won": True}),
        ]
        env.step.side_effect = step_results
        
        # Evaluate
        results = evaluate_model(model, env, n_episodes=3)
        
        # Verify results
        assert results["win_rate"] == pytest.approx(66.67, abs=0.01)  # 2 wins out of 3
        assert results["avg_reward"] == pytest.approx(326.67, abs=0.01)  # (500 + (-20) + 500) / 3
        assert results["avg_length"] == 1.0  # All episodes end in 1 step
        assert results["n_episodes"] == 3
    
    def test_evaluate_model_confidence_intervals(self):
        """Test that confidence intervals are calculated correctly."""
        # Mock model
        model = MagicMock()
        model.predict.return_value = (np.array([0]), None)
        
        # Mock environment
        env = MagicMock()
        env.reset.return_value = (np.zeros((4, 4, 4)), {})
        
        # Mock step to return varied rewards
        step_results = [
            (np.zeros((4, 4, 4)), 10.0, True, False, {"won": False}),
            (np.zeros((4, 4, 4)), 20.0, True, False, {"won": False}),
            (np.zeros((4, 4, 4)), 30.0, True, False, {"won": False}),
        ]
        env.step.side_effect = step_results
        
        # Evaluate
        results = evaluate_model(model, env, n_episodes=3)
        
        # Verify confidence intervals
        assert results["reward_ci"] > 0  # Should have some standard error
        assert results["length_ci"] == 0  # All episodes same length
        assert results["avg_reward"] == 20.0
    
    def test_evaluate_model_single_episode_confidence(self):
        """Test confidence interval calculation with single episode."""
        # Mock model
        model = MagicMock()
        model.predict.return_value = (np.array([0]), None)
        
        # Mock environment
        env = MagicMock()
        env.reset.return_value = (np.zeros((4, 4, 4)), {})
        env.step.return_value = (np.zeros((4, 4, 4)), 100.0, True, False, {"won": True})
        
        # Evaluate
        results = evaluate_model(model, env, n_episodes=1)
        
        # With single episode, confidence interval should be 0
        assert results["reward_ci"] == 0.0
        assert results["length_ci"] == 0.0
    
    def test_evaluate_model_vectorized_env_complex(self):
        """Test evaluation with complex vectorized environment behavior."""
        # Mock model
        model = MagicMock()
        model.predict.return_value = (np.array([[0, 1]]), None)
        
        # Mock vectorized environment with multiple environments
        env = MagicMock()
        env.num_envs = 2
        env.reset.return_value = np.zeros((2, 4, 4, 4))
        
        # Mock step to return complex vectorized results
        step_results = [
            (
                np.zeros((2, 4, 4, 4)), 
                np.array([15.0, 20.0]), 
                np.array([False, True]), 
                np.array([False, False]), 
                [{"won": False}, {"won": True}]
            ),
        ]
        env.step.side_effect = step_results
        
        # Evaluate
        results = evaluate_model(model, env, n_episodes=1)
        
        # Verify results
        assert results["win_rate"] == 100.0  # One win in the episode
        assert results["avg_reward"] == 17.5  # (15 + 20) / 2
        assert results["avg_length"] == 1.0


class TestArgumentParsing:
    """Test argument parsing functionality."""
    
    def test_parse_args_default_values(self):
        """Test argument parsing with default values."""
        with patch('sys.argv', ['train_agent.py']):
            args = parse_args()
            
            assert args.total_timesteps == 1000000
            assert args.eval_freq == 10000
            assert args.n_eval_episodes == 100
            assert args.save_freq == 50000
            assert args.learning_rate == 0.0003
            assert args.n_steps == 2048
            assert args.batch_size == 64
            assert args.n_epochs == 10
            assert args.gamma == 0.99
            assert args.gae_lambda == 0.95
            assert args.clip_range == 0.2
            assert args.ent_coef == 0.01
            assert args.vf_coef == 0.5
            assert args.max_grad_norm == 0.5
            assert args.use_sde == False
            assert args.sde_sample_freq == -1
            assert args.target_kl is None
            assert args.policy == "MlpPolicy"
            assert args.verbose == 1
            assert args.seed is None
            assert args.device == "auto"
            assert args._init_setup_model == True
            assert args.strict_progression == False
            assert args.timestamped_stats == False
    
    def test_parse_args_custom_values(self):
        """Test argument parsing with custom values."""
        custom_args = [
            'train_agent.py',
            '--total_timesteps', '500000',
            '--learning_rate', '0.001',
            '--batch_size', '128',
            '--device', 'cuda',
            '--verbose', '2',
            '--seed', '42',
            '--strict_progression', 'True',
            '--timestamped_stats', 'True'
        ]
        
        with patch('sys.argv', custom_args):
            args = parse_args()
            
            assert args.total_timesteps == 500000
            assert args.learning_rate == 0.001
            assert args.batch_size == 128
            assert args.device == 'cuda'
            assert args.verbose == 2
            assert args.seed == 42
            assert args.strict_progression == True
            assert args.timestamped_stats == True
    
    def test_parse_args_float_values(self):
        """Test argument parsing with float values."""
        custom_args = [
            'train_agent.py',
            '--learning_rate', '0.0001',
            '--gamma', '0.95',
            '--gae_lambda', '0.9',
            '--clip_range', '0.1',
            '--ent_coef', '0.005',
            '--vf_coef', '0.25',
            '--max_grad_norm', '0.3',
            '--target_kl', '0.01'
        ]
        
        with patch('sys.argv', custom_args):
            args = parse_args()
            
            assert args.learning_rate == 0.0001
            assert args.gamma == 0.95
            assert args.gae_lambda == 0.9
            assert args.clip_range == 0.1
            assert args.ent_coef == 0.005
            assert args.vf_coef == 0.25
            assert args.max_grad_norm == 0.3
            assert args.target_kl == 0.01
    
    def test_parse_args_boolean_flags(self):
        """Test argument parsing with boolean flags."""
        custom_args = [
            'train_agent.py',
            '--use_sde', 'True',
            '--strict_progression', 'True',
            '--timestamped_stats', 'True'
        ]
        
        with patch('sys.argv', custom_args):
            args = parse_args()
            
            assert args.use_sde == True
            assert args.strict_progression == True
            assert args.timestamped_stats == True
    
    def test_parse_args_optional_values(self):
        """Test argument parsing with optional values."""
        custom_args = [
            'train_agent.py',
            '--clip_range_vf', '0.1',
            '--sde_sample_freq', '8',
            '--target_kl', '0.01'
        ]
        
        with patch('sys.argv', custom_args):
            args = parse_args()
            
            assert args.clip_range_vf == 0.1
            assert args.sde_sample_freq == 8
            assert args.target_kl == 0.01
    
    def test_parse_args_policy_selection(self):
        """Test argument parsing with different policy types."""
        custom_args = [
            'train_agent.py',
            '--policy', 'CnnPolicy'
        ]
        
        with patch('sys.argv', custom_args):
            args = parse_args()
            
            assert args.policy == 'CnnPolicy'


class TestEvaluateModelEdgeCases:
    """Test edge cases in evaluate_model function."""
    
    def test_evaluate_model_zero_episodes(self):
        """Test evaluation with zero episodes."""
        model = MagicMock()
        env = MagicMock()
        
        results = evaluate_model(model, env, n_episodes=0)
        
        assert results["win_rate"] == 0.0
        assert results["avg_reward"] == 0.0
        assert results["avg_length"] == 0.0
        assert results["n_episodes"] == 0
        assert results["reward_ci"] == 0.0
        assert results["length_ci"] == 0.0
    
    def test_evaluate_model_environment_reset_failure(self):
        """Test evaluation when environment reset fails."""
        model = MagicMock()
        env = MagicMock()
        env.reset.side_effect = RuntimeError("Environment reset failed")
        
        with pytest.raises(RuntimeError):
            evaluate_model(model, env, n_episodes=1)
    
    def test_evaluate_model_model_predict_failure(self):
        """Test evaluation when model prediction fails."""
        model = MagicMock()
        model.predict.side_effect = RuntimeError("Model prediction failed")
        
        env = MagicMock()
        env.reset.return_value = (np.zeros((4, 4, 4)), {})
        
        with pytest.raises(RuntimeError):
            evaluate_model(model, env, n_episodes=1)
    
    def test_evaluate_model_step_failure(self):
        """Test evaluation when environment step fails."""
        model = MagicMock()
        model.predict.return_value = (np.array([0]), None)
        
        env = MagicMock()
        env.reset.return_value = (np.zeros((4, 4, 4)), {})
        env.step.side_effect = RuntimeError("Environment step failed")
        
        with pytest.raises(RuntimeError):
            evaluate_model(model, env, n_episodes=1)
    
    def test_evaluate_model_vectorized_env_detection_edge_case(self):
        """Test vectorized environment detection edge case."""
        model = MagicMock()
        model.predict.return_value = (np.array([0]), None)
        
        # Mock environment that has num_envs but no step method
        env = MagicMock()
        env.num_envs = 1
        # Don't set step method
        
        with pytest.raises(AttributeError):
            evaluate_model(model, env, n_episodes=1)


class TestArgumentParsingEdgeCases:
    """Test edge cases in argument parsing."""
    
    def test_parse_args_invalid_integer(self):
        """Test argument parsing with invalid integer values."""
        custom_args = [
            'train_agent.py',
            '--total_timesteps', 'invalid'
        ]
        
        with patch('sys.argv', custom_args):
            with pytest.raises(SystemExit):
                parse_args()
    
    def test_parse_args_invalid_float(self):
        """Test argument parsing with invalid float values."""
        custom_args = [
            'train_agent.py',
            '--learning_rate', 'invalid'
        ]
        
        with patch('sys.argv', custom_args):
            with pytest.raises(SystemExit):
                parse_args()
    
    def test_parse_args_missing_value(self):
        """Test argument parsing with missing values."""
        custom_args = [
            'train_agent.py',
            '--learning_rate'  # Missing value
        ]
        
        with patch('sys.argv', custom_args):
            with pytest.raises(SystemExit):
                parse_args()
    
    def test_parse_args_unknown_argument(self):
        """Test argument parsing with unknown arguments."""
        custom_args = [
            'train_agent.py',
            '--unknown_arg', 'value'
        ]
        
        with patch('sys.argv', custom_args):
            with pytest.raises(SystemExit):
                parse_args() 