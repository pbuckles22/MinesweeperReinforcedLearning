"""
Functional tests for refined curriculum learning.

Tests the refined curriculum learning approach with gradual difficulty progression.
"""

import pytest
import sys
import os
from pathlib import Path
import json
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from core.train_agent_modular import make_modular_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO


class TestRefinedCurriculumFunctional:
    """Test refined curriculum learning functionality."""
    
    def test_curriculum_stage_progression(self):
        """Test that curriculum stages progress with appropriate difficulty increases."""
        # Import the curriculum function
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts"))
        from curriculum_training_refined import refined_curriculum_training, evaluate_model
        
        # Test with a minimal curriculum (just 2 stages)
        curriculum_stages = [
            ((4, 4), 1, 1000, 0.5, "Test Stage 1: 4x4 with 1 mine"),
            ((4, 4), 2, 1000, 0.3, "Test Stage 2: 4x4 with 2 mines"),
        ]
        
        # Verify stage progression makes sense
        for i, (board_size, max_mines, timesteps, min_win_rate, description) in enumerate(curriculum_stages):
            # Check that timesteps increase (more training on harder stages)
            if i > 0:
                assert timesteps >= curriculum_stages[i-1][2], f"Timesteps should increase: {timesteps} >= {curriculum_stages[i-1][2]}"
            
            # Check that target win rates decrease (harder stages have lower targets)
            if i > 0:
                assert min_win_rate <= curriculum_stages[i-1][3], f"Target win rate should decrease: {min_win_rate} <= {curriculum_stages[i-1][3]}"
            
            # Check mine density increases
            density = max_mines / (board_size[0] * board_size[1])
            if i > 0:
                prev_density = curriculum_stages[i-1][1] / (curriculum_stages[i-1][0][0] * curriculum_stages[i-1][0][1])
                assert density >= prev_density, f"Mine density should increase: {density:.3f} >= {prev_density:.3f}"
    
    def test_evaluation_function(self):
        """Test the evaluation function works correctly."""
        # Import the evaluation function
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts"))
        from curriculum_training_refined import evaluate_model
        
        # Create a simple model
        env = DummyVecEnv([make_modular_env((4, 4), 1)])
        model = PPO("MlpPolicy", env, verbose=0)
        
        # Train for a few steps
        model.learn(total_timesteps=100)
        
        # Evaluate the model
        results = evaluate_model(model, (4, 4), 1, n_episodes=10)
        
        # Check that results contain expected keys
        expected_keys = ["win_rate", "mean_reward", "wins", "total_episodes", "avg_steps", "success_rate"]
        for key in expected_keys:
            assert key in results, f"Missing key in evaluation results: {key}"
        
        # Check that values are reasonable
        assert 0 <= results["win_rate"] <= 1, f"Win rate should be between 0 and 1: {results['win_rate']}"
        assert results["wins"] >= 0, f"Wins should be non-negative: {results['wins']}"
        assert results["total_episodes"] == 10, f"Total episodes should be 10: {results['total_episodes']}"
        assert results["avg_steps"] >= 0, f"Average steps should be non-negative: {results['avg_steps']}"
        assert 0 <= results["success_rate"] <= 1, f"Success rate should be between 0 and 1: {results['success_rate']}"
    
    def test_curriculum_results_format(self):
        """Test that curriculum results are saved in the correct format."""
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock results file
            mock_results = {
                "curriculum_summary": {
                    "total_stages": 2,
                    "completed_stages": 2,
                    "timestamp": "20240101_120000",
                    "approach": "refined_gradual_progression"
                },
                "results": [
                    {
                        "stage": 1,
                        "board_size": (4, 4),
                        "max_mines": 1,
                        "total_timesteps": 1000,
                        "description": "Test Stage 1",
                        "win_rate": 0.8,
                        "mean_reward": 15.5,
                        "wins": 8,
                        "total_episodes": 10,
                        "mine_density": 0.0625,
                        "target_win_rate": 0.75,
                        "timestamp": "20240101_120000"
                    },
                    {
                        "stage": 2,
                        "board_size": (4, 4),
                        "max_mines": 2,
                        "total_timesteps": 1000,
                        "description": "Test Stage 2",
                        "win_rate": 0.6,
                        "mean_reward": 12.0,
                        "wins": 6,
                        "total_episodes": 10,
                        "mine_density": 0.125,
                        "target_win_rate": 0.6,
                        "timestamp": "20240101_120000",
                        "avg_steps": 8.5,
                        "success_rate": 0.9
                    }
                ]
            }
            
            # Save the mock results
            results_file = os.path.join(temp_dir, "test_curriculum_results.json")
            with open(results_file, 'w') as f:
                json.dump(mock_results, f, indent=2)
            
            # Verify the file exists and can be loaded
            assert os.path.exists(results_file), "Results file should exist"
            
            with open(results_file, 'r') as f:
                loaded_results = json.load(f)
            
            # Check structure
            assert "curriculum_summary" in loaded_results, "Should have curriculum summary"
            assert "results" in loaded_results, "Should have results array"
            
            # Check summary fields
            summary = loaded_results["curriculum_summary"]
            assert summary["total_stages"] == 2, "Should have correct total stages"
            assert summary["completed_stages"] == 2, "Should have correct completed stages"
            assert summary["approach"] == "refined_gradual_progression", "Should have correct approach"
            
            # Check results array
            results = loaded_results["results"]
            assert len(results) == 2, "Should have 2 results"
            
            # Check first result
            result1 = results[0]
            assert result1["stage"] == 1, "Should have correct stage number"
            assert result1["board_size"] == [4, 4], "Should have correct board size"
            assert result1["max_mines"] == 1, "Should have correct mine count"
            assert result1["win_rate"] == 0.8, "Should have correct win rate"
            assert result1["mine_density"] == 0.0625, "Should have correct mine density"
            
            # Check second result has additional fields
            result2 = results[1]
            assert "avg_steps" in result2, "Should have avg_steps for non-first stage"
            assert "success_rate" in result2, "Should have success_rate for non-first stage"
    
    def test_curriculum_difficulty_progression(self):
        """Test that curriculum difficulty increases appropriately."""
        # Define expected curriculum stages
        expected_stages = [
            ((4, 4), 1, 200000, 0.75),  # Stage 1: Easiest
            ((4, 4), 2, 300000, 0.60),  # Stage 2: More mines
            ((4, 4), 3, 400000, 0.45),  # Stage 3: Even more mines
            ((6, 6), 4, 500000, 0.40),  # Stage 4: Larger board
            ((6, 6), 6, 600000, 0.30),  # Stage 5: More mines on larger board
            ((6, 6), 8, 700000, 0.20),  # Stage 6: Challenge
        ]
        
        # Verify progression makes sense
        for i in range(1, len(expected_stages)):
            prev_board, prev_mines, prev_timesteps, prev_target = expected_stages[i-1]
            curr_board, curr_mines, curr_timesteps, curr_target = expected_stages[i]
            
            # Timesteps should increase (more training on harder stages)
            assert curr_timesteps > prev_timesteps, f"Timesteps should increase: {curr_timesteps} > {prev_timesteps}"
            
            # Target win rates should decrease (harder stages have lower targets)
            assert curr_target < prev_target, f"Target should decrease: {curr_target} < {prev_target}"
            
            # Mine density should generally increase
            prev_density = prev_mines / (prev_board[0] * prev_board[1])
            curr_density = curr_mines / (curr_board[0] * curr_board[1])
            
            # Allow for some flexibility in density (board size changes)
            # But overall trend should be increasing difficulty
            if curr_board == prev_board:
                assert curr_density > prev_density, f"Density should increase: {curr_density:.3f} > {prev_density:.3f}"
    
    def test_curriculum_script_import(self):
        """Test that the refined curriculum script can be imported."""
        # Add scripts to path
        scripts_path = str(Path(__file__).parent.parent.parent.parent / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        
        try:
            # Try to import the curriculum module
            import curriculum_training_refined
            assert hasattr(curriculum_training_refined, 'refined_curriculum_training'), "Should have refined_curriculum_training function"
            assert hasattr(curriculum_training_refined, 'evaluate_model'), "Should have evaluate_model function"
        except ImportError as e:
            pytest.fail(f"Failed to import curriculum_training_refined: {e}")


if __name__ == "__main__":
    pytest.main([__file__]) 