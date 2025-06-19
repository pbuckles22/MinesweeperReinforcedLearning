"""
Test suite for the Minesweeper environment.
See TEST_CHECKLIST.md for comprehensive test coverage plan.
"""

import pytest
import numpy as np
from src.core.minesweeper_env import (
    MinesweeperEnv, REWARD_FIRST_MOVE_HIT_MINE, REWARD_WIN, REWARD_HIT_MINE,
    CELL_UNREVEALED, CELL_MINE, CELL_FLAGGED, CELL_MINE_HIT
)
from src.core.constants import (
    REWARD_SAFE_REVEAL
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
        assert env.max_board_width == 20
        assert env.max_board_height == 35
        assert env.max_mines == 130
        assert env.initial_board_width == 4
        assert env.initial_board_height == 4
        assert env.initial_mines == 2

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
        assert self.env.state.shape == (4, 4)
        assert self.env.flags.shape == (4, 4)

        # Verify board is square
        assert self.env.board.shape[0] == self.env.board.shape[1]
        assert self.env.state.shape[0] == self.env.state.shape[1]
        assert self.env.flags.shape[0] == self.env.flags.shape[1]

        # Check board is properly initialized with hidden cells
        assert np.all(self.env.state == CELL_UNREVEALED)  # All cells should be hidden initially
        assert np.all(self.env.flags == 0)   # No flags should be placed initially

        # Verify board size matches current dimensions
        assert self.env.board.shape[0] == self.env.current_board_height
        assert self.env.board.shape[1] == self.env.current_board_width

    def test_mine_placement(self):
        """Verify mines are placed correctly using only public API and deterministic setup."""
        # Use a fixed seed for deterministic mine placement
        env = MinesweeperEnv(initial_board_size=(4, 4), initial_mines=2)
        env.reset(seed=42)
        # Check mine count using public API (mines are not exposed, so check via state after hitting mines)
        mine_hits = 0
        for action in range(env.current_board_width * env.current_board_height):
            state, reward, terminated, truncated, info = env.step(action)
            if terminated and reward == REWARD_HIT_MINE:
                mine_hits += 1
                env.reset(seed=42)  # Reset to same board
        # We expect to be able to hit 2 mines on this board
        assert mine_hits == 2

    def test_difficulty_levels(self):
        """Test environment with different difficulty levels using public API only."""
        difficulty_configs = [
            ('easy', 9, 9, 10),
            ('normal', 16, 16, 40),
            ('hard', 16, 30, 99),
            ('expert', 18, 24, 115),
            ('chaotic', 20, 35, 130)
        ]
        for name, width, height, mines in difficulty_configs:
            env = MinesweeperEnv(
                max_board_size=(width, height),
                max_mines=mines,
                initial_board_size=(width, height),
                initial_mines=mines,
                mine_spacing=0  # Disable mine spacing for testing
            )
            env.reset(seed=42)
            # Test board and state shapes
            assert env.state.shape == (height, width)
            # Test action space
            expected_actions = width * height * 2
            assert env.action_space.n == expected_actions
            # Test observation space
            assert env.observation_space.shape == (height, width)
            # Test mine count by revealing all cells and counting mine hits
            mine_hits = 0
            for action in range(width * height):
                state, reward, terminated, truncated, info = env.step(action)
                if terminated and reward == REWARD_HIT_MINE:
                    mine_hits += 1
                    env.reset(seed=42)
            assert mine_hits == mines

    def test_rectangular_board_actions(self):
        """Test actions on rectangular boards using public API only."""
        width, height, mines = 16, 30, 99
        env = MinesweeperEnv(
            max_board_size=(width, height),
            max_mines=mines,
            initial_board_size=(width, height),
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
        # Test flag actions (first 5 flag actions)
        board_size = env.current_board_width * env.current_board_height
        for i in range(5):
            action = i + board_size
            state, reward, terminated, truncated, info = env.step(action)
            if terminated:
                env.reset(seed=42)
        # Check board and state shapes
        assert env.state.shape == (height, width)
        assert env.flags.shape == (height, width)
        assert env.observation_space.shape == (height, width)
        assert env.action_space.n == width * height * 2

    def test_safe_cell_reveal(self):
        """Test revealing a safe cell and its effects."""
        # Create a test board with known mine positions
        self.env.board = np.zeros((4, 4), dtype=int)
        self.env.mines = np.zeros((4, 4), dtype=bool)
        self.env.mines[0, 0] = True  # Mine at top-left
        self.env.mines[2, 2] = True  # Mine at center
        
        print("\nInitial mines setup:")
        print("Mines array:")
        print(self.env.mines)
        
        self.env._update_adjacent_counts()  # This will set the correct numbers in board
        
        print("\nAfter _update_adjacent_counts:")
        print("Board array (adjacent counts):")
        print(self.env.board)
        
        # Test revealing a safe cell (1, 1) which should have 2 adjacent mines
        action = 1 * self.env.current_board_width + 1
        print(f"\nRevealing cell (1,1) with action {action}")
        state, reward, terminated, truncated, info = self.env.step(action)
        
        print("\nAfter reveal:")
        print("State array:")
        print(state)
        print(f"Cell (1,1) value: {state[1, 1]}")
        print(f"Terminated: {terminated}")
        print(f"Reward: {reward}")

        # Verify the cell was revealed with correct number
        assert state[1, 1] == 2  # Should show 2 adjacent mines
        assert not terminated
        assert reward >= 0  # Allow 0 reward for first safe reveal

    def test_curriculum_progression(self):
        """Test curriculum learning progression through difficulty levels."""
        # Start with beginner level
        env = MinesweeperEnv(
            max_board_size=(20, 35),
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
            assert env.state.shape == (height, width)
            assert env.flags.shape == (height, width)
            
            # Verify action space
            expected_actions = width * height * 2
            assert env.action_space.n == expected_actions
            
            # Verify observation space
            assert env.observation_space.shape == (height, width)
            
            # Test mine placement
            mine_count = np.sum(env.mines)
            assert mine_count == mines

    def test_win_condition_rectangular(self):
        """Test win condition on rectangular boards."""
        # Test with hard difficulty (16x30)
        env = MinesweeperEnv(
            max_board_size=(16, 30),
            max_mines=99,
            initial_board_size=(16, 30),
            initial_mines=99,
            mine_spacing=0  # Disable mine spacing for testing
        )
        env.reset()
        
        # Flag all mines
        terminated = False
        for y in range(env.current_board_height):
            for x in range(env.current_board_width):
                if env.mines[y, x]:
                    action = y * env.current_board_width + x + env.current_board_width * env.current_board_height
                    state, reward, terminated, truncated, info = env.step(action)
                    if terminated:
                        assert info['won']
                        assert reward == REWARD_WIN
                        break
            if terminated:
                break
        
        # If not terminated, reveal all non-mine cells
        if not terminated:
            for y in range(env.current_board_height):
                for x in range(env.current_board_width):
                    if not env.mines[y, x]:
                        action = y * env.current_board_width + x
                        state, reward, terminated, truncated, info = env.step(action)
                        if terminated:
                            assert info['won']
                            assert reward == REWARD_WIN
                            break
                if terminated:
                    break

    def test_reveal_action(self):
        """Test reveal action behavior."""
        self.env.reset()
        
        # Test first move behavior
        action = 0  # First cell
        state, reward, terminated, truncated, info = self.env.step(action)
        if terminated:
            # If the first move is a mine, verify correct state and reward
            assert state[0, 0] == CELL_MINE_HIT
            assert reward < 0
        else:
            # If the first move is safe, verify correct state and reward
            assert state[0, 0] != CELL_UNREVEALED
            assert reward >= 0
        
        # Test non-first move mine reveal
        # Find a mine that wasn't the first move
        mine_found = False
        for y in range(self.env.current_board_height):
            for x in range(self.env.current_board_width):
                if self.env.mines[y, x] and (y != 0 or x != 0):  # Not the first cell
                    action = y * self.env.current_board_width + x
                    state, reward, terminated, truncated, info = self.env.step(action)
                    assert terminated
                    assert state[y, x] == CELL_MINE_HIT
                    assert reward < 0
                    mine_found = True
                    break
            if mine_found:
                break
                
        # If no mine was found, reset and try again
        if not mine_found:
            self.env.reset()
            # Test first move behavior again
            action = 0
            state, reward, terminated, truncated, info = self.env.step(action)
            if not terminated:  # If first move was safe
                # Now find a mine
                for y in range(self.env.current_board_height):
                    for x in range(self.env.current_board_width):
                        if self.env.mines[y, x]:
                            action = y * self.env.current_board_width + x
                            state, reward, terminated, truncated, info = self.env.step(action)
                            assert terminated
                            assert state[y, x] == CELL_MINE_HIT
                            assert reward < 0
                            mine_found = True
                            break
                    if mine_found:
                        break

    def test_flag_action(self):
        """Test flag action behavior."""
        self.env.reset()
        
        # Test flagging a mine
        self.env.mines[1, 1] = True
        action = 1 * self.env.current_board_width + 1 + self.env.current_board_width * self.env.current_board_height
        state, reward, terminated, truncated, info = self.env.step(action)
        assert not terminated
        assert self.env.flags[1, 1]
        assert reward > 0
        
        # Test flagging a safe cell
        action = 2 * self.env.current_board_width + 2 + self.env.current_board_width * self.env.current_board_height
        state, reward, terminated, truncated, info = self.env.step(action)
        assert not terminated
        assert self.env.flags[2, 2]
        assert reward < 0

    def test_unflag_action(self):
        """Test unflag action behavior."""
        self.env.reset()
        
        # First flag a cell
        action = 1 * self.env.current_board_width + 1 + self.env.current_board_width * self.env.current_board_height
        state, reward, terminated, truncated, info = self.env.step(action)
        assert self.env.flags[1, 1]
        
        # Then unflag it
        state, reward, terminated, truncated, info = self.env.step(action)
        assert not self.env.flags[1, 1]
        assert reward < 0

    def test_invalid_actions(self):
        """Test invalid action handling."""
        self.env.reset()
        
        # Test revealing a flagged cell
        self.env.flags[1, 1] = True
        action = 1 * self.env.current_board_width + 1
        state, reward, terminated, truncated, info = self.env.step(action)
        assert not terminated
        assert reward < 0
        
        # Test revealing an already revealed cell
        self.env.state[2, 2] = 0
        action = 2 * self.env.current_board_width + 2
        state, reward, terminated, truncated, info = self.env.step(action)
        assert not terminated
        assert reward < 0
        
        # Test out of bounds action
        action = self.env.current_board_width * self.env.current_board_height * 2
        state, reward, terminated, truncated, info = self.env.step(action)
        assert not terminated
        assert reward < 0

    def test_board_boundary_actions(self):
        """Test actions at board boundaries."""
        self.env.reset()

        # Test corner cells
        corners = [(0, 0), (0, 3), (3, 0), (3, 3)]
        for y, x in corners:
            # Test reveal action
            action = y * self.env.current_board_width + x
            state, reward, terminated, truncated, info = self.env.step(action)
            
            if terminated:
                # If we hit a mine, verify correct state and reward
                assert state[y, x] == CELL_MINE_HIT
                assert reward < 0
            else:
                # If safe, verify correct state and reward
                assert state[y, x] != CELL_UNREVEALED
                assert reward >= 0  # Safe reveal should give non-negative reward
                
                # Try to reveal the same cell again - should be invalid
                state, reward, terminated, truncated, info = self.env.step(action)
                assert reward == self.env.invalid_action_penalty
                assert not terminated
                
                # Try to flag the revealed cell - should be invalid
                flag_action = y * self.env.current_board_width + x + self.env.current_board_width * self.env.current_board_height
                state, reward, terminated, truncated, info = self.env.step(flag_action)
                assert reward == self.env.invalid_action_penalty
                assert not terminated
                
                # Reset for next corner test
                self.env.reset()

    def test_game_over_condition(self):
        """Test game over condition."""
        self.env.reset()
        
        # Hit a mine
        self.env.mines[1, 1] = True
        self.env.board[1, 1] = CELL_MINE
        self.env.is_first_move = False  # Ensure it's not the first move
        action = 1 * self.env.current_board_width + 1
        state, reward, terminated, truncated, info = self.env.step(action)
        assert terminated
        assert not info['won']
        assert reward == REWARD_HIT_MINE

    def test_win_condition(self):
        """Test win condition."""
        self.env.reset()
        
        # First, reveal all safe cells
        for y in range(self.env.current_board_height):
            for x in range(self.env.current_board_width):
                if not self.env.mines[y, x]:
                    action = y * self.env.current_board_width + x
                    state, reward, terminated, truncated, info = self.env.step(action)
                    if terminated:  # If we hit a mine, reset and try again
                        self.env.reset()
                        break
        
        # Then flag all mines
        for y in range(self.env.current_board_height):
            for x in range(self.env.current_board_width):
                if self.env.mines[y, x]:
                    action = y * self.env.current_board_width + x + self.env.current_board_width * self.env.current_board_height
                    state, reward, terminated, truncated, info = self.env.step(action)
                    if terminated:  # If we won, verify win condition
                        assert info['won']
                        assert reward == self.env.win_reward
                        return
                        
        # If we get here, we should have won
        assert terminated
        assert info['won']
        assert reward == self.env.win_reward

    def test_state_transitions(self):
        """Test state transitions during gameplay."""
        self.env.reset()

        # Find an unrevealed cell
        unrevealed_found = False
        for y in range(self.env.current_board_height):
            for x in range(self.env.current_board_width):
                if self.env.state[y, x] == CELL_UNREVEALED:
                    # Test reveal action
                    action = y * self.env.current_board_width + x
                    state, reward, terminated, truncated, info = self.env.step(action)
                    if not terminated:  # If it was safe
                        assert state[y, x] != CELL_UNREVEALED
                        unrevealed_found = True
                        break
            if unrevealed_found:
                break

        # Find another unrevealed cell for flagging
        if not unrevealed_found:
            self.env.reset()
        for y in range(self.env.current_board_height):
            for x in range(self.env.current_board_width):
                if self.env.state[y, x] == CELL_UNREVEALED:
                    # Test flag action
                    action = y * self.env.current_board_width + x + self.env.current_board_width * self.env.current_board_height
                    state, reward, terminated, truncated, info = self.env.step(action)
                    assert not terminated
                    assert self.env.flags[y, x]
                    assert state[y, x] == CELL_UNREVEALED
                    return

    def test_state_representation(self):
        """Test state representation consistency."""
        self.env.reset()

        # Verify initial state
        assert np.all(self.env.state == CELL_UNREVEALED)
        assert np.all(self.env.flags == False)

        # Find an unrevealed cell
        for y in range(self.env.current_board_height):
            for x in range(self.env.current_board_width):
                if self.env.state[y, x] == CELL_UNREVEALED:
                    # Test reveal action
                    action = y * self.env.current_board_width + x
                    state, reward, terminated, truncated, info = self.env.step(action)
                    if terminated:
                        # If we hit a mine, verify correct state and reward
                        assert state[y, x] == CELL_MINE_HIT
                        assert reward < 0
                    else:
                        # If safe, verify correct state and reward
                        assert state[y, x] != CELL_UNREVEALED
                        assert state[y, x] >= 0 and state[y, x] <= 8
                        assert reward >= 0

                    # Find another unrevealed cell for flagging
                    for y2 in range(self.env.current_board_height):
                        for x2 in range(self.env.current_board_width):
                            if self.env.state[y2, x2] == CELL_UNREVEALED:
                                # Test flag action
                                action = y2 * self.env.current_board_width + x2 + self.env.current_board_width * self.env.current_board_height
                                state, reward, terminated, truncated, info = self.env.step(action)
                                assert not terminated
                                assert self.env.flags[y2, x2]
                                assert state[y2, x2] == CELL_UNREVEALED
                                return 