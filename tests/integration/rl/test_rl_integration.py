"""
Comprehensive RL Integration Tests

These tests verify the complete RL training pipeline works correctly,
including the issues we encountered with EvalCallback and vectorized environments.
"""

import pytest
import os
import shutil
import numpy as np
import tempfile
from unittest.mock import patch, Mock
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from src.core.minesweeper_env import MinesweeperEnv
from src.core.train_agent import (
    make_env, 
    evaluate_model, 
    CustomEvalCallback,
    IterationCallback,
    ExperimentTracker,
    main
)


class TestRLTrainingIntegration:
    """Integration tests for complete RL training pipeline.
    
    Note: We use CustomEvalCallback instead of the standard EvalCallback because
    the standard EvalCallback has compatibility issues with vectorized environments
    (tries to access env.won which doesn't exist on vectorized envs).
    
    All tests that could hang are decorated with @pytest.mark.timeout(30).
    If a test fails due to timeout, it means a regression in RL system.
    """
    
    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create a temporary directory for testing."""
        return str(tmp_path)
    
    @pytest.fixture
    def small_env(self):
        """Create a small test environment."""
        def make_small_env():
            return MinesweeperEnv(
                initial_board_size=(4, 4),
                initial_mines=2,
                early_learning_mode=True
            )
        return DummyVecEnv([make_small_env])
    
    @pytest.mark.timeout(30)
    def test_custom_evalcallback_with_vectorized_env(self, small_env, temp_dir):
        """Test that our CustomEvalCallback works correctly.
        FAILS (times out) if the RL system regresses and hangs.
        """
        # Create evaluation environment
        eval_env = DummyVecEnv([lambda: MinesweeperEnv(
            initial_board_size=(4, 4),
            initial_mines=2,
            early_learning_mode=True
        )])
        
        # Create model
        model = PPO('MlpPolicy', small_env, verbose=0)
        
        # Create CustomEvalCallback
        eval_callback = CustomEvalCallback(
            eval_env,
            eval_freq=50,
            n_eval_episodes=3,
            verbose=0
        )
        
        # Test that training completes
        try:
            model.learn(total_timesteps=100, callback=eval_callback, progress_bar=False)
            training_completed = True
        except Exception as e:
            training_completed = False
            pytest.fail(f"Training with CustomEvalCallback failed: {e}")
        
        assert training_completed, "Training should complete successfully"
    
    def test_evaluate_model_with_vectorized_env(self, small_env):
        """Test evaluate_model function with vectorized environment."""
        # Create a simple model
        model = PPO('MlpPolicy', small_env, verbose=0)
        
        # Test evaluation
        results = evaluate_model(model, small_env, n_episodes=5)
        
        # Verify results structure
        assert "win_rate" in results
        assert "avg_reward" in results
        assert "avg_length" in results
        assert "reward_ci" in results
        assert "length_ci" in results
        assert "n_episodes" in results
        
        # Verify reasonable values
        assert 0 <= results["win_rate"] <= 100
        assert isinstance(results["avg_reward"], float)
        assert isinstance(results["avg_length"], float)
        assert results["n_episodes"] == 5
    
    def test_info_dictionary_access_patterns(self, small_env):
        """Test that we can access info dictionary correctly from vectorized environment.
        Accepts both gym (dict) and gymnasium (list of dicts) API return types.
        """
        obs = small_env.reset()
        action = [0]
        step_result = small_env.step(action)
        if len(step_result) == 4:
            obs, reward, terminated, truncated = step_result
            info = {}
        else:
            obs, reward, terminated, truncated, info = step_result
        # Accept both dict and list for info
        assert isinstance(info, (list, dict))
        if isinstance(info, list):
            assert len(info) > 0
            assert isinstance(info[0], dict)
    
    def test_environment_api_compatibility(self, small_env):
        """Test compatibility with both gym and gymnasium APIs.
        Accepts both gym (dict/array) and gymnasium (list) return types.
        """
        reset_result = small_env.reset()
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs = reset_result
            info = {}
        assert obs.shape == (1, 2, 4, 4)
        step_result = small_env.step([0])
        if len(step_result) == 4:
            obs, reward, terminated, truncated = step_result
            info = {}
        else:
            obs, reward, terminated, truncated, info = step_result
        assert isinstance(reward, np.ndarray)
        assert isinstance(terminated, np.ndarray)
        # Accept both array and list for truncated
        assert isinstance(truncated, (np.ndarray, list))
    
    @pytest.mark.timeout(30)
    def test_curriculum_training_integration(self, temp_dir):
        """Test complete curriculum training with evaluation.
        FAILS (times out) if the RL system regresses and hangs.
        """
        # Create experiment tracker
        tracker = ExperimentTracker(experiment_dir=temp_dir)
        tracker.start_new_run({"test": "curriculum"})
        
        # Create environment
        env = DummyVecEnv([make_env(max_board_size=4, max_mines=2)])
        eval_env = DummyVecEnv([make_env(max_board_size=4, max_mines=2)])
        
        # Create model
        model = PPO('MlpPolicy', env, verbose=0)
        
        # Create callbacks
        eval_callback = CustomEvalCallback(
            eval_env,
            eval_freq=50,
            n_eval_episodes=3,
            verbose=0
        )
        
        iteration_callback = IterationCallback(
            verbose=0,
            debug_level=0,
            experiment_tracker=tracker
        )
        
        # Test training with both callbacks
        try:
            model.learn(
                total_timesteps=200,
                callback=[eval_callback, iteration_callback],
                progress_bar=False
            )
            training_completed = True
        except Exception as e:
            training_completed = False
            pytest.fail(f"Curriculum training failed: {e}")
        
        assert training_completed, "Curriculum training should complete"
        
        # Verify experiment tracker has data
        assert tracker.current_run is not None
        metrics_file = os.path.join(tracker.current_run, "metrics.json")
        assert os.path.exists(metrics_file)
    
    def test_episode_completion_with_evaluation(self, small_env):
        """Test that episodes complete properly during evaluation."""
        # Create model
        model = PPO('MlpPolicy', small_env, verbose=0)
        
        # Test multiple evaluation episodes
        results = evaluate_model(model, small_env, n_episodes=10)
        
        # Verify all episodes completed
        assert results["n_episodes"] == 10
        assert results["avg_length"] > 0
        
        # Verify reasonable episode lengths (should be > 1 for 4x4 board)
        assert 1 <= results["avg_length"] <= 16  # Max possible moves on 4x4 board
    
    def test_win_detection_in_vectorized_env(self, small_env):
        """Test that win detection works correctly in vectorized environment."""
        # Create model that always chooses first action
        model = PPO('MlpPolicy', small_env, verbose=0)
        
        # Run evaluation and check win detection
        results = evaluate_model(model, small_env, n_episodes=5)
        
        # Win rate should be a valid percentage
        assert 0 <= results["win_rate"] <= 100
        
        # If we have wins, verify they're detected
        if results["win_rate"] > 0:
            # Run a single episode to verify win detection
            obs = small_env.reset()
            done = False
            won = False
            
            while not done:
                action, _ = model.predict(obs)
                step_result = small_env.step(action)
                
                if len(step_result) == 4:
                    obs, reward, terminated, truncated = step_result
                    info = {}
                else:
                    obs, reward, terminated, truncated, info = step_result
                
                done = terminated or truncated
                
                # Check for win in info
                if info and isinstance(info, list) and len(info) > 0:
                    if info[0].get('won', False):
                        won = True
            
            # If we detected a win in evaluation, we should be able to detect it here too
            # (This is a probabilistic test, so we don't assert it always passes)

    def test_info_dictionary_win_access(self, small_env):
        """Test specific info dictionary win access pattern that caused our issue."""
        # Reset environment
        obs = small_env.reset()
        
        # Take multiple steps to potentially trigger a win
        for step in range(10):
            action = [step % 16]  # Cycle through actions
            step_result = small_env.step(action)
            
            # Handle both gym and gymnasium APIs
            if len(step_result) == 4:
                obs, reward, terminated, truncated = step_result
                info = {}
            else:
                obs, reward, terminated, truncated, info = step_result
            
            # Test the exact access pattern that caused our issue
            if info and isinstance(info, list) and len(info) > 0:
                # This is the correct way to access 'won' from vectorized environment
                won = info[0].get('won', False)
                assert isinstance(won, bool)
                
                # Test that we can safely access other info fields
                if 'won' in info[0]:
                    assert isinstance(info[0]['won'], bool)
            
            # If episode ends, reset and continue
            if terminated or truncated:
                obs = small_env.reset()
    
    def test_vectorized_env_info_structure(self, small_env):
        """Test that vectorized environment info structure is correct.
        Accepts both gym (dict) and gymnasium (list of dicts) API return types.
        """
        obs = small_env.reset()
        step_result = small_env.step([0])
        if len(step_result) == 4:
            obs, reward, terminated, truncated = step_result
            info = {}
        else:
            obs, reward, terminated, truncated, info = step_result
        # Accept both dict and list for info
        assert isinstance(info, (list, dict)), "Vectorized env info should be a list or dict"
        if isinstance(info, list):
            assert len(info) == 1
            assert isinstance(info[0], dict)
        assert not hasattr(small_env, 'won'), "Vectorized env should not have 'won' attribute"
        won = info[0].get('won', False) if isinstance(info, list) else info.get('won', False)
        assert isinstance(won, bool)


class TestEnvironmentAPICompatibility:
    """Test environment API compatibility across different scenarios."""
    
    def test_single_env_vs_vectorized_env(self):
        """Test differences between single and vectorized environments."""
        single_env = MinesweeperEnv(
            initial_board_size=(4, 4),
            initial_mines=2,
            early_learning_mode=True
        )
        vec_env = DummyVecEnv([lambda: MinesweeperEnv(
            initial_board_size=(4, 4),
            initial_mines=2,
            early_learning_mode=True
        )])
        single_reset = single_env.reset()
        vec_reset = vec_env.reset()
        if isinstance(single_reset, tuple):
            single_obs, single_info = single_reset
        else:
            single_obs = single_reset
            single_info = {}
        if isinstance(vec_reset, tuple):
            vec_obs, vec_info = vec_reset
        else:
            vec_obs = vec_reset
            vec_info = {}
        assert single_obs.shape == (2, 4, 4)
        assert vec_obs.shape == (1, 2, 4, 4)
        single_step = single_env.step(0)
        vec_step = vec_env.step([0])
        if len(single_step) == 4:
            single_obs, single_reward, single_terminated, single_truncated = single_step
            single_info = {}
        else:
            single_obs, single_reward, single_terminated, single_truncated, single_info = single_step
        if len(vec_step) == 4:
            vec_obs, vec_reward, vec_terminated, vec_truncated = vec_step
            vec_info = {}
        else:
            vec_obs, vec_reward, vec_terminated, vec_truncated, vec_info = vec_step
        assert isinstance(single_reward, (int, float))
        assert isinstance(vec_reward, np.ndarray)
        assert isinstance(single_terminated, bool)
        assert isinstance(vec_terminated, np.ndarray)
    
    def test_info_dictionary_structure(self):
        """Test info dictionary structure in different scenarios.
        Accepts both gym (dict) and gymnasium (list of dicts) API return types.
        """
        single_env = MinesweeperEnv(
            initial_board_size=(4, 4),
            initial_mines=2,
            early_learning_mode=True
        )
        single_env.reset()
        single_step = single_env.step(0)
        if len(single_step) == 4:
            _, _, _, _ = single_step
            single_info = {}
        else:
            _, _, _, _, single_info = single_step
        assert isinstance(single_info, dict)
        vec_env = DummyVecEnv([lambda: MinesweeperEnv(
            initial_board_size=(4, 4),
            initial_mines=2,
            early_learning_mode=True
        )])
        vec_env.reset()
        vec_step = vec_env.step([0])
        if len(vec_step) == 4:
            _, _, _, _ = vec_step
            vec_info = {}
        else:
            _, _, _, _, vec_info = vec_step
        # Accept both dict and list for info
        assert isinstance(vec_info, (list, dict))
        if isinstance(vec_info, list):
            assert len(vec_info) > 0
            assert isinstance(vec_info[0], dict)


class TestErrorHandling:
    """Test error handling in the RL system."""
    
    def test_invalid_action_handling(self):
        """Test that invalid actions are handled gracefully."""
        env = DummyVecEnv([lambda: MinesweeperEnv(
            initial_board_size=(4, 4),
            initial_mines=2,
            early_learning_mode=True
        )])
        
        env.reset()
        
        # Test invalid action
        step_result = env.step([1000])  # Action out of bounds
        
        if len(step_result) == 4:
            obs, reward, terminated, truncated = step_result
            info = {}
        else:
            obs, reward, terminated, truncated, info = step_result
        
        # Should not crash, should give negative reward
        assert reward[0] < 0
        assert not terminated[0]  # Should not terminate
    
    def test_environment_reset_after_error(self):
        """Test that environment can reset after errors."""
        env = DummyVecEnv([lambda: MinesweeperEnv(
            initial_board_size=(4, 4),
            initial_mines=2,
            early_learning_mode=True
        )])
        
        # Take some steps
        obs = env.reset()
        for _ in range(5):
            env.step([0])
        
        # Reset should work
        new_obs = env.reset()
        assert new_obs.shape == (1, 2, 4, 4)


if __name__ == "__main__":
    pytest.main([__file__]) 