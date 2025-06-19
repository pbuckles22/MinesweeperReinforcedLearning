"""
Functional Requirements Tests for Minesweeper RL Environment

These tests focus on functional requirements rather than implementation details.
They ensure the environment behaves correctly according to Minesweeper rules
and RL environment requirements.
"""

import pytest
import numpy as np
from src.core.minesweeper_env import MinesweeperEnv
from src.core.constants import (
    CELL_UNREVEALED, CELL_MINE, CELL_FLAGGED, CELL_MINE_HIT,
    REWARD_WIN, REWARD_HIT_MINE, REWARD_SAFE_REVEAL
)

class TestCoreGameMechanics:
    """Test core Minesweeper game mechanics."""
    
    def test_mine_placement_avoids_first_cell(self):
        """REQUIREMENT: First revealed cell should never be a mine."""
        env = MinesweeperEnv(initial_board_size=4, initial_mines=15)
        env.reset(seed=42)
        
        # First move should always be safe
        action = 0  # Reveal top-left cell
        state, reward, terminated, truncated, info = env.step(action)
        
        # The revealed cell should not be a mine
        assert state[0, 0] != CELL_MINE_HIT, "First revealed cell should not be a mine"
        assert not terminated, "First move should not end the game"
    
    def test_cascade_revelation(self):
        """REQUIREMENT: Revealing a cell with 0 adjacent mines should cascade to neighbors."""
        env = MinesweeperEnv(initial_board_size=4, initial_mines=1)
        env.reset(seed=42)
        
        # Place mine at (3,3) and reveal (0,0) which should cascade
        env.mines.fill(False)
        env.mines[3, 3] = True
        env._update_adjacent_counts()
        env.mines_placed = True
        env.is_first_move = False
        env.first_move_done = True
        
        # Reveal cell that should trigger cascade
        action = 0  # (0,0)
        state, reward, terminated, truncated, info = env.step(action)
        
        # Should reveal multiple cells due to cascade
        revealed_cells = np.sum(state != CELL_UNREVEALED)
        assert revealed_cells > 1, "Cascade should reveal multiple cells"
    
    def test_win_condition_all_safe_cells_revealed(self):
        """REQUIREMENT: Game should be won when all non-mine cells are revealed."""
        env = MinesweeperEnv(initial_board_size=3, initial_mines=1)
        env.reset(seed=42)
        
        # Place mine at (2,2) and reveal all other cells
        env.mines.fill(False)
        env.mines[2, 2] = True
        env._update_adjacent_counts()
        env.mines_placed = True
        env.is_first_move = False
        env.first_move_done = True
        
        # Reveal all safe cells
        safe_cells = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1)]
        for row, col in safe_cells:
            action = row * env.current_board_width + col
            state, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break
        
        assert terminated, "Game should be terminated when all safe cells are revealed"
        assert info.get('won', False), "Game should be marked as won"
        assert reward == REWARD_WIN, "Should receive win reward"
    
    def test_loss_condition_mine_hit(self):
        """REQUIREMENT: Game should end when a mine is revealed."""
        env = MinesweeperEnv(initial_board_size=3, initial_mines=1)
        env.reset(seed=42)
        
        # Place mine at (1,1) and reveal it
        env.mines.fill(False)
        env.mines[1, 1] = True
        env._update_adjacent_counts()
        env.mines_placed = True
        env.is_first_move = False
        env.first_move_done = True
        
        # Reveal the mine
        action = 1 * env.current_board_width + 1
        state, reward, terminated, truncated, info = env.step(action)
        
        assert terminated, "Game should be terminated when mine is hit"
        assert not info.get('won', False), "Game should not be marked as won"
        assert reward == REWARD_HIT_MINE, "Should receive mine hit penalty"
        assert state[1, 1] == CELL_MINE_HIT, "Hit cell should show mine hit"

class TestFlaggingSystem:
    """Test flagging system functionality."""
    
    def test_flag_placement_on_unrevealed_cell(self):
        """REQUIREMENT: Should be able to flag unrevealed cells."""
        env = MinesweeperEnv(initial_board_size=4, initial_mines=2)
        env.reset(seed=42)
        
        # Flag action (action space is 2x board size)
        flag_action = env.current_board_width * env.current_board_height + 0
        state, reward, terminated, truncated, info = env.step(flag_action)
        
        assert state[0, 0] == CELL_FLAGGED, "Cell should be flagged"
        assert not terminated, "Flagging should not end the game"
    
    def test_cannot_flag_revealed_cell(self):
        """REQUIREMENT: Should not be able to flag revealed cells."""
        env = MinesweeperEnv(initial_board_size=4, initial_mines=2)
        env.reset(seed=42)
        
        # First reveal a cell
        reveal_action = 0
        state, reward, terminated, truncated, info = env.step(reveal_action)
        
        # Try to flag the revealed cell
        flag_action = env.current_board_width * env.current_board_height + 0
        state, reward, terminated, truncated, info = env.step(flag_action)
        
        # Should get invalid action penalty
        assert reward < 0, "Should get penalty for invalid flag action"
    
    def test_flag_removal(self):
        """REQUIREMENT: Should be able to remove flags."""
        env = MinesweeperEnv(initial_board_size=4, initial_mines=2)
        env.reset(seed=42)
        
        # Place a flag
        flag_action = env.current_board_width * env.current_board_height + 0
        state, reward, terminated, truncated, info = env.step(flag_action)
        assert state[0, 0] == CELL_FLAGGED, "Cell should be flagged"
        
        # Remove the flag (same action)
        state, reward, terminated, truncated, info = env.step(flag_action)
        assert state[0, 0] == CELL_UNREVEALED, "Cell should be unflagged"

class TestRLEnvironmentRequirements:
    """Test RL-specific environment requirements."""
    
    def test_action_space_consistency(self):
        """REQUIREMENT: Action space should be consistent and valid."""
        env = MinesweeperEnv(initial_board_size=4, initial_mines=2)
        env.reset(seed=42)
        
        # Action space should be 2x board size (reveal + flag actions)
        expected_size = env.current_board_width * env.current_board_height * 2
        assert env.action_space.n == expected_size, "Action space size should match board size * 2"
        
        # All actions should be valid indices
        assert env.action_space.contains(0), "Action 0 should be valid"
        assert env.action_space.contains(expected_size - 1), "Last action should be valid"
        assert not env.action_space.contains(expected_size), "Action beyond range should be invalid"
    
    def test_observation_space_consistency(self):
        """REQUIREMENT: Observation space should be consistent."""
        env = MinesweeperEnv(initial_board_size=4, initial_mines=2)
        env.reset(seed=42)
        
        # Observation should match expected shape
        expected_shape = (env.current_board_height, env.current_board_width)
        assert env.observation_space.shape == expected_shape, "Observation shape should match board dimensions"
        
        # Observation should be within bounds
        assert env.observation_space.contains(env.state), "State should be within observation space bounds"
    
    def test_deterministic_reset(self):
        """REQUIREMENT: Same seed should produce same board."""
        env = MinesweeperEnv(initial_board_size=4, initial_mines=2)
        
        # Reset with same seed
        state1, info1 = env.reset(seed=42)
        state2, info2 = env.reset(seed=42)
        
        # States should be identical
        np.testing.assert_array_equal(state1, state2, "Same seed should produce same state")
    
    def test_state_consistency_between_steps(self):
        """REQUIREMENT: State should be consistent between steps."""
        env = MinesweeperEnv(initial_board_size=4, initial_mines=2)
        env.reset(seed=42)
        
        # Make a move
        action = 0
        state1, reward1, terminated1, truncated1, info1 = env.step(action)
        
        # Make another move
        action = 1
        state2, reward2, terminated2, truncated2, info2 = env.step(action)
        
        # Previously revealed cells should remain revealed
        if state1[0, 0] != CELL_UNREVEALED:
            assert state2[0, 0] == state1[0, 0], "Previously revealed cells should remain revealed"
    
    def test_info_dictionary_consistency(self):
        """REQUIREMENT: Info dictionary should provide consistent game state."""
        env = MinesweeperEnv(initial_board_size=4, initial_mines=2)
        env.reset(seed=42)
        
        # Info should contain required keys
        assert 'won' in env.info, "Info should contain 'won' key"
        
        # Make a move
        action = 0
        state, reward, terminated, truncated, info = env.step(action)
        
        # Info should be updated
        assert 'won' in info, "Info should contain 'won' key after step"
        if terminated:
            assert isinstance(info['won'], bool), "'won' should be boolean"

class TestCurriculumLearning:
    """Test curriculum learning functionality."""
    
    def test_early_learning_mode_safety(self):
        """REQUIREMENT: Early learning mode should provide safety features."""
        env = MinesweeperEnv(
            initial_board_size=4, 
            initial_mines=2,
            early_learning_mode=True,
            early_learning_corner_safe=True,
            early_learning_edge_safe=True
        )
        env.reset(seed=42)
        
        # In early learning mode, corners and edges should be safer
        # (This is a probabilistic test - may need adjustment)
        corner_actions = [0, 3, 12, 15]  # Corners of 4x4 board
        safe_corners = 0
        
        for action in corner_actions:
            env.reset(seed=42)
            state, reward, terminated, truncated, info = env.step(action)
            if not terminated:
                safe_corners += 1
        
        # At least some corners should be safe in early learning mode
        assert safe_corners > 0, "Early learning mode should provide some safe corners"
    
    def test_difficulty_progression(self):
        """REQUIREMENT: Environment should support difficulty progression."""
        env = MinesweeperEnv(
            initial_board_size=3,
            initial_mines=1,
            early_learning_mode=True,
            early_learning_threshold=5
        )
        
        # Play several games to trigger progression
        for _ in range(6):
            env.reset(seed=42)
            action = 0
            state, reward, terminated, truncated, info = env.step(action)
        
        # Environment should have progressed (board size or mine count increased)
        assert (env.current_board_width > 3 or 
                env.current_board_height > 3 or 
                env.current_mines > 1), "Difficulty should have progressed" 