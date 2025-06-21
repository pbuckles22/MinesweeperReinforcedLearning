"""
Test suite for the Minesweeper environment.
See TEST_CHECKLIST.md for comprehensive test coverage plan.
"""

import pytest
import numpy as np
from src.core.minesweeper_env import MinesweeperEnv
from src.core.constants import (
    CELL_UNREVEALED, CELL_MINE, CELL_MINE_HIT,
    REWARD_FIRST_CASCADE_SAFE, REWARD_FIRST_CASCADE_HIT_MINE,
    REWARD_SAFE_REVEAL, REWARD_WIN, REWARD_HIT_MINE
)

class TestMinesweeperEnv:
    """Test cases for the Minesweeper environment."""

    def setup_method(self):
        """Set up test environment before each test."""
        self.env = MinesweeperEnv(
            max_board_size=(10, 10),
            max_mines=10,
            initial_board_size=(4, 4),
            initial_mines=2
        )

    def test_initialization(self):
        """Test environment initialization with default parameters."""
        env = MinesweeperEnv()
        assert env.max_board_size == (35, 20)  # (height, width) format
        assert env.max_board_height == 35
        assert env.max_board_width == 20
        assert env.max_mines == 130
        assert env.initial_board_size == (4, 4)
        assert env.initial_mines == 2
        assert env.mine_spacing == 1
        assert env.early_learning_mode == False
        assert env.early_learning_threshold == 200
        assert env.early_learning_corner_safe == True
        assert env.early_learning_edge_safe == True

        # Check parameters are set correctly
        assert self.env.max_board_width == 10
        assert self.env.max_board_height == 10
        assert self.env.max_mines == 10
        assert self.env.early_learning_mode is False
        assert self.env.early_learning_threshold == 200
        assert self.env.early_learning_corner_safe is True
        assert self.env.early_learning_edge_safe is True
        assert self.env.mine_spacing == 1
        assert self.env.initial_board_width == 4
        assert self.env.initial_board_height == 4

    def test_board_creation(self):
        """Verify board is created with correct dimensions and initialization."""
        # Check board dimensions
        assert self.env.board.shape == (4, 4)
        assert self.env.state.shape == (4, 4, 4)  # 4 channels
        assert self.env.mines.shape == (4, 4)

        # Verify board is square
        assert self.env.board.shape[0] == self.env.board.shape[1]
        assert self.env.state.shape[1] == self.env.state.shape[2]  # height x width
        assert self.env.mines.shape[0] == self.env.mines.shape[1]

        # Check board is properly initialized with hidden cells
        assert np.all(self.env.state[0] == CELL_UNREVEALED)  # All cells should be hidden initially

        # Verify board size matches current dimensions
        assert self.env.board.shape[0] == self.env.current_board_height
        assert self.env.board.shape[1] == self.env.current_board_width

    def test_mine_placement(self):
        """Verify mines are placed correctly using only public API and deterministic setup."""
        # Use a fixed seed for deterministic mine placement
        env = MinesweeperEnv(initial_board_size=(4, 4), initial_mines=2)
        env.reset(seed=42)
        
        # Check mine count by examining the mines array directly
        # This is the most reliable way to count mines
        actual_mines = np.sum(env.mines)
        expected_mines = env.current_mines
        
        assert actual_mines == expected_mines, f"Expected {expected_mines} mines, but found {actual_mines} mines"

    def test_difficulty_levels(self):
        """Test environment with different difficulty levels using public API only."""
        difficulty_configs = [
            ('easy', 9, 9, 10),
            ('normal', 16, 16, 40),
            ('hard', 16, 30, 99),
            ('expert', 18, 24, 115),
            ('chaotic', 20, 35, 130)
        ]
        for name, height, width, mines in difficulty_configs:
            env = MinesweeperEnv(
                max_board_size=(height, width),  # (height, width) format
                max_mines=mines,
                initial_board_size=(height, width),  # (height, width) format
                initial_mines=mines,
                mine_spacing=0  # Disable mine spacing for testing
            )
            env.reset(seed=42)
            # Test board and state shapes
            assert env.state.shape == (4, height, width)  # 4 channels
            # Test action space
            expected_actions = width * height  # Only reveal actions
            assert env.action_space.n == expected_actions
            # Test observation space
            assert env.observation_space.shape == (4, height, width)  # 4 channels
            
            # Test mine count by examining the mines array directly
            actual_mines = np.sum(env.mines)
            assert actual_mines == mines, f"Difficulty {name}: Expected {mines} mines, but found {actual_mines} mines"

    def test_rectangular_board_actions(self):
        """Test actions on rectangular boards using public API only."""
        height, width, mines = 16, 30, 99  # (height, width) format
        env = MinesweeperEnv(
            max_board_size=(height, width),  # (height, width) format
            max_mines=mines,
            initial_board_size=(height, width),  # (height, width) format
            initial_mines=mines,
            mine_spacing=0
        )
        env.reset(seed=42)
        # Test reveal actions (first 5 cells)
        for i in range(5):
            state, reward, terminated, truncated, info = env.step(i)
            # Allow any mine hit to terminate
            if terminated:
                env.reset(seed=42)
        # Check board and state shapes
        assert env.state.shape == (4, height, width)  # 4 channels
        assert env.observation_space.shape == (4, height, width)  # 4 channels
        assert env.action_space.n == width * height  # Only reveal actions

    def test_safe_cell_reveal(self):
        """Test safe cell reveal."""
        env = MinesweeperEnv(initial_board_size=4, initial_mines=2)
        env.reset()
        
        # Take an action
        action = 0
        state, reward, terminated, truncated, info = env.step(action)
        
        # Check that the revealed cell is no longer unrevealed
        assert state[0, 0, 0] != CELL_UNREVEALED

    def test_curriculum_progression(self):
        """Test curriculum learning progression through difficulty levels."""
        # Start with beginner level
        env = MinesweeperEnv(
            max_board_size=(35, 20),
            max_mines=130,
            initial_board_size=(4, 4),
            initial_mines=2
        )
        
        # Test progression through stages
        stages = [
            (4, 4, 2, 1),    # Beginner
            (6, 6, 4, 1),    # Intermediate
            (9, 9, 10, 1),   # Easy
            (16, 16, 40, 1), # Normal
            (16, 30, 99, 0), # Hard (set mine_spacing=0)
            (18, 24, 115, 0),# Expert (set mine_spacing=0)
            (20, 35, 130, 0) # Chaotic (set mine_spacing=0)
        ]
        
        for width, height, mines, spacing in stages:
            # Update board size and mines
            env.current_board_width = width
            env.current_board_height = height
            env.current_mines = mines
            env.mine_spacing = spacing
            env.reset()
            
            # Verify dimensions and mine count
            assert env.current_board_width == width
            assert env.current_board_height == height
            assert env.current_mines == mines
            
            # Verify board shapes
            assert env.board.shape == (height, width)
            assert env.state.shape == (4, height, width)  # 4 channels
            assert env.mines.shape == (height, width)
            
            # Verify action space
            expected_actions = width * height  # Only reveal actions
            assert env.action_space.n == expected_actions
            
            # Verify observation space
            assert env.observation_space.shape == (4, height, width)  # 4 channels
            
            # Test mine placement
            mine_count = np.sum(env.mines)
            assert mine_count == mines

    def test_win_condition_rectangular(self):
        """Test win condition on rectangular boards."""
        env = MinesweeperEnv(initial_board_size=(5, 3), initial_mines=3)
        env.reset(seed=42)
        
        # Try to reveal all cells
        for action in range(env.current_board_width * env.current_board_height):
            state, reward, terminated, truncated, info = env.step(action)
            if terminated:
                # Game should end (either win or lose)
                break

    def test_reveal_action(self):
        """Test reveal action behavior."""
        self.env.reset()
        
        # Test revealing a safe cell
        action = 0
        state, reward, terminated, truncated, info = self.env.step(action)
        
        # Should not terminate unless we hit a mine or win
        if not terminated:
            assert state[0, 0, 0] != CELL_UNREVEALED
        else:
            # If terminated, should be either win or mine hit
            assert info.get('won', False) or reward == REWARD_HIT_MINE

    def test_invalid_actions(self):
        """Test invalid action handling."""
        self.env.reset()
        
        # Test revealing an already revealed cell
        action = 0
        state, reward, terminated, truncated, info = self.env.step(action)
        
        # Try to reveal the same cell again
        state, reward, terminated, truncated, info = self.env.step(action)
        assert reward < 0  # Should get negative reward for invalid action
        
        # Test out of bounds action
        action = self.env.action_space.n
        state, reward, terminated, truncated, info = self.env.step(action)
        assert reward < 0  # Should get negative reward for invalid action

    def test_game_over_condition(self):
        """Test game over condition."""
        self.env.reset()
        
        # Place mine at (0,0) and hit it
        self.env.mines[0, 0] = True
        self.env._update_adjacent_counts()
        self.env.mines_placed = True
        self.env.is_first_cascade = False
        self.env.first_cascade_done = True
        
        action = 0
        state, reward, terminated, truncated, info = self.env.step(action)
        
        assert terminated
        assert reward == REWARD_HIT_MINE
        assert state[0, 0, 0] == CELL_MINE_HIT 