"""
Performance Tests for Minesweeper RL Environment

These tests verify that the environment performs well under various conditions,
including large boards, high mine densities, and rapid state transitions.
"""

import pytest
import numpy as np
import time
from src.core.minesweeper_env import MinesweeperEnv
from src.core.constants import (
    CELL_UNREVEALED, CELL_MINE_HIT,
    REWARD_WIN, REWARD_HIT_MINE, REWARD_SAFE_REVEAL
)

class TestPerformance:
    """Test environment performance under various conditions."""
    
    def test_large_board_performance(self):
        """Test performance with large boards."""
        env = MinesweeperEnv(initial_board_size=(16, 16), initial_mines=40)
        
        # Measure reset time
        start_time = time.time()
        env.reset(seed=42)
        reset_time = time.time() - start_time
        
        # Reset should be reasonably fast
        assert reset_time < 1.0, f"Reset should be fast, took {reset_time:.3f}s"
        
        # Measure step time
        start_time = time.time()
        for action in range(10):  # Make 10 moves
            state, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break
        step_time = time.time() - start_time
        
        # Steps should be very fast
        assert step_time < 0.1, f"Steps should be very fast, took {step_time:.3f}s for 10 steps"
    
    def test_high_mine_density_performance(self):
        """Test performance with high mine density."""
        env = MinesweeperEnv(initial_board_size=(10, 10), initial_mines=80)
        
        # Measure reset time
        start_time = time.time()
        env.reset(seed=42)
        reset_time = time.time() - start_time
        
        # Reset should be fast even with high density
        assert reset_time < 0.5, f"Reset should be fast with high density, took {reset_time:.3f}s"
        
        # Measure step time
        start_time = time.time()
        moves_made = 0
        for action in range(20):  # Try up to 20 moves
            state, reward, terminated, truncated, info = env.step(action)
            moves_made += 1
            if terminated:
                break
        step_time = time.time() - start_time
        
        # Steps should be fast
        assert step_time < 0.1, f"Steps should be fast with high density, took {step_time:.3f}s for {moves_made} steps"
    
    def test_cascade_performance(self):
        """Test performance during cascade revelations."""
        env = MinesweeperEnv(initial_board_size=(8, 8), initial_mines=5)
        env.reset(seed=42)
        
        # Set up a large cascade scenario
        env.mines.fill(False)
        # Place mines only in bottom-right corner to create large cascade
        env.mines[6:, 6:] = True
        env._update_adjacent_counts()
        env.mines_placed = True
        env.is_first_cascade = False
        env.first_cascade_done = True
        
        # Measure cascade performance
        start_time = time.time()
        action = 0  # (0,0) - should trigger large cascade
        state, reward, terminated, truncated, info = env.step(action)
        cascade_time = time.time() - start_time
        
        # Cascade should be fast
        assert cascade_time < 0.1, f"Cascade should be fast, took {cascade_time:.3f}s"
        
        # Should reveal many cells
        revealed_cells = np.sum(state[0] != CELL_UNREVEALED)
        assert revealed_cells > 20, f"Should reveal many cells in cascade, revealed {revealed_cells}"
    
    def test_rapid_state_transitions(self):
        """Test performance during rapid state transitions."""
        env = MinesweeperEnv(initial_board_size=(6, 6), initial_mines=8)
        env.reset(seed=42)
        
        # Measure rapid transitions
        start_time = time.time()
        for action in range(20):  # Make 20 rapid moves
            state, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break
        total_time = time.time() - start_time
        
        # Should handle rapid transitions well
        assert total_time < 0.1, f"Rapid transitions should be fast, took {total_time:.3f}s for 20 moves"
    
    def test_memory_usage_consistency(self):
        """Test that memory usage remains consistent."""
        env = MinesweeperEnv(initial_board_size=(8, 8), initial_mines=10)
        
        # Track state sizes
        state_sizes = []
        
        for _ in range(5):  # Play 5 games
            env.reset(seed=42)
            initial_state_size = env.state.nbytes
            
            # Make several moves
            for action in range(10):
                state, reward, terminated, truncated, info = env.step(action)
                state_sizes.append(state.nbytes)
                if terminated:
                    break
        
        # State sizes should be consistent
        expected_size = env.state.shape[0] * env.state.shape[1] * env.state.shape[2] * env.state.dtype.itemsize
        for size in state_sizes:
            assert size == expected_size, f"State size should be consistent, got {size}, expected {expected_size}"
    
    def test_action_space_performance(self):
        """Test performance of action space operations."""
        env = MinesweeperEnv(initial_board_size=(12, 12), initial_mines=20)
        env.reset(seed=42)
        
        # Measure action space operations
        start_time = time.time()
        for _ in range(100):  # Test 100 operations
            action_count = env.action_space.n
            masks = env.action_masks
            valid_actions = np.sum(masks)
        action_time = time.time() - start_time
        
        # Action space operations should be very fast
        assert action_time < 0.1, f"Action space operations should be very fast, took {action_time:.3f}s for 100 ops"
        
        # Verify action space consistency
        assert action_count == 144, "Action space should be 12x12=144"
        assert valid_actions <= action_count, "Valid actions should not exceed total actions"
    
    def test_observation_space_performance(self):
        """Test performance of observation space operations."""
        env = MinesweeperEnv(initial_board_size=(10, 10), initial_mines=15)
        env.reset(seed=42)
        
        # Measure observation space operations
        start_time = time.time()
        for _ in range(100):  # Test 100 operations
            obs_shape = env.observation_space.shape
            obs_bounds = env.observation_space.low, env.observation_space.high
            obs_contains = env.observation_space.contains(env.state)
        obs_time = time.time() - start_time
        
        # Observation space operations should be very fast
        assert obs_time < 0.1, f"Observation space operations should be very fast, took {obs_time:.3f}s for 100 ops"
        
        # Verify observation space consistency
        assert obs_shape == (4, 10, 10), "Observation shape should be (4, 10, 10)"
        assert obs_contains, "State should be within observation space bounds"
    
    def test_concurrent_environment_creation(self):
        """Test performance of creating multiple environments."""
        start_time = time.time()
        
        # Create multiple environments
        envs = []
        for i in range(10):
            env = MinesweeperEnv(
                initial_board_size=(6, 6),
                initial_mines=8,
                max_board_size=(12, 12),
                max_mines=20
            )
            env.reset(seed=42 + i)
            envs.append(env)
        
        creation_time = time.time() - start_time
        
        # Environment creation should be fast
        assert creation_time < 1.0, f"Environment creation should be fast, took {creation_time:.3f}s for 10 envs"
        
        # Test that all environments work
        for i, env in enumerate(envs):
            action = 0
            state, reward, terminated, truncated, info = env.step(action)
            assert state.shape == (4, 6, 6), f"Environment {i} should have correct state shape"
    
    def test_large_scale_simulation(self):
        """Test performance in large-scale simulation scenarios."""
        env = MinesweeperEnv(initial_board_size=(20, 20), initial_mines=80)
        
        # Simulate many games
        start_time = time.time()
        total_moves = 0
        games_played = 0
        
        for game in range(10):  # Play 10 games
            env.reset(seed=42 + game)
            moves_in_game = 0
            
            # Play until game ends
            for action in range(100):  # Max 100 moves per game
                state, reward, terminated, truncated, info = env.step(action)
                moves_in_game += 1
                total_moves += 1
                
                if terminated:
                    break
            
            games_played += 1
        
        simulation_time = time.time() - start_time
        
        # Large-scale simulation should be reasonably fast
        assert simulation_time < 5.0, f"Large-scale simulation should be reasonably fast, took {simulation_time:.3f}s"
        assert total_moves > 0, "Should have made some moves"
        assert games_played == 10, "Should have played 10 games"
    
    def test_rectangular_board_performance(self):
        """Test performance with rectangular boards."""
        env = MinesweeperEnv(initial_board_size=(8, 16), initial_mines=20)
        
        # Measure reset time
        start_time = time.time()
        env.reset(seed=42)
        reset_time = time.time() - start_time
        
        # Reset should be fast for rectangular boards
        assert reset_time < 0.5, f"Reset should be fast for rectangular boards, took {reset_time:.3f}s"
        
        # Measure step time
        start_time = time.time()
        for action in range(10):  # Make 10 moves
            state, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break
        step_time = time.time() - start_time
        
        # Steps should be fast
        assert step_time < 0.1, f"Steps should be fast for rectangular boards, took {step_time:.3f}s"
        
        # Verify rectangular dimensions
        assert env.state.shape == (4, 8, 16), "State should match rectangular dimensions"
        assert env.action_space.n == 128, "Action space should match rectangular dimensions"
    
    def test_early_learning_performance(self):
        """Test performance with early learning mode enabled."""
        env = MinesweeperEnv(
            initial_board_size=(8, 8),
            initial_mines=10,
            early_learning_mode=True,
            early_learning_corner_safe=True,
            early_learning_edge_safe=True
        )
        
        # Measure reset time with early learning
        start_time = time.time()
        env.reset(seed=42)
        reset_time = time.time() - start_time
        
        # Reset should be fast with early learning
        assert reset_time < 0.5, f"Reset should be fast with early learning, took {reset_time:.3f}s"
        
        # Measure step time with early learning
        start_time = time.time()
        for action in range(10):  # Make 10 moves
            state, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break
        step_time = time.time() - start_time
        
        # Steps should be fast
        assert step_time < 0.1, f"Steps should be fast with early learning, took {step_time:.3f}s"
    
    def test_difficulty_progression_performance(self):
        """Test performance during difficulty progression."""
        env = MinesweeperEnv(
            initial_board_size=(4, 4),
            initial_mines=2,
            max_board_size=(12, 12),
            max_mines=30
        )
        
        # Test progression performance
        start_time = time.time()
        
        for step in range(5):  # Test 5 progression steps
            # Increase difficulty
            width = 4 + step * 2
            height = 4 + step * 2
            mines = 2 + step * 3
            
            env.current_board_width = width
            env.current_board_height = height
            env.current_mines = mines
            env.reset(seed=42)
            
            # Make a few moves
            for action in range(3):
                state, reward, terminated, truncated, info = env.step(action)
                if terminated:
                    break
        
        progression_time = time.time() - start_time
        
        # Progression should be reasonably fast
        assert progression_time < 2.0, f"Difficulty progression should be reasonably fast, took {progression_time:.3f}s"

if __name__ == "__main__":
    pytest.main([__file__]) 