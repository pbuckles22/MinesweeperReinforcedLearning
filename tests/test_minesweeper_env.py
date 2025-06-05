import pytest
import numpy as np
from src.core.minesweeper_env import MinesweeperEnv

class TestMinesweeperEnv:
    """Test suite for the Minesweeper environment."""

    def setup_method(self):
        """Set up a fresh environment for each test."""
        self.env = MinesweeperEnv(
            max_board_size=10,
            max_mines=10,
            early_learning_mode=False,
            early_learning_threshold=200,
            early_learning_corner_safe=True,
            early_learning_edge_safe=True,
            mine_spacing=1,
            initial_board_size=4,
            initial_mines=2,
            invalid_action_penalty=-0.1
        )
        self.env.reset()

    def test_initialization(self):
        """Verify environment initializes with correct default values."""
        # Check parameters are set correctly
        assert self.env.max_board_size == 10
        assert self.env.max_mines == 10
        assert self.env.early_learning_mode is False
        assert self.env.early_learning_threshold == 200
        assert self.env.early_learning_corner_safe is True
        assert self.env.early_learning_edge_safe is True
        assert self.env.mine_spacing == 1
        assert self.env.initial_board_size == 4
        assert self.env.initial_mines == 2
        assert self.env.invalid_action_penalty == -0.1

        # Check reward values
        assert self.env.mine_penalty == -10.0
        assert self.env.safe_reveal_base == 5.0
        assert self.env.win_reward == 100.0

        # Check state variables are initialized
        assert self.env.current_board_size == 4
        assert self.env.current_mines == 2
        assert self.env.board is not None
        assert self.env.state is not None
        assert self.env.flags is not None
        assert self.env.mines is not None
        assert self.env.won is False
        assert self.env.total_games == 0
        assert self.env.games_at_current_size == 0

        # Verify logger is initialized
        assert self.env.logger is not None

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
        assert np.all(self.env.state == -1)  # All cells should be hidden initially
        assert np.all(self.env.flags == 0)   # No flags should be placed initially

        # Verify board size matches current_board_size
        assert self.env.board.shape[0] == self.env.current_board_size
        assert self.env.state.shape[0] == self.env.current_board_size
        assert self.env.flags.shape[0] == self.env.current_board_size

    def test_mine_placement(self):
        """Verify mines are placed correctly and not in invalid locations."""
        # Reset environment multiple times to check different mine placements
        for _ in range(5):
            self.env.reset()
            
            # Check mine count
            mine_count = np.sum(self.env.board == 9)  # 9 represents a mine
            assert mine_count == self.env.current_mines

            # Check mine spacing
            mine_positions = np.where(self.env.board == 9)
            for i in range(len(mine_positions[0])):
                row, col = mine_positions[0][i], mine_positions[1][i]
                
                # Check surrounding cells for other mines
                for dr in range(-self.env.mine_spacing, self.env.mine_spacing + 1):
                    for dc in range(-self.env.mine_spacing, self.env.mine_spacing + 1):
                        if dr == 0 and dc == 0:
                            continue
                        new_row, new_col = row + dr, col + dc
                        if 0 <= new_row < self.env.current_board_size and 0 <= new_col < self.env.current_board_size:
                            assert self.env.board[new_row, new_col] != 9, \
                                f"Mine spacing violation at ({row}, {col}) and ({new_row}, {new_col})"

    def test_adjacent_mine_counting(self):
        """Verify correct counting of adjacent mines."""
        # Create a test board with known mine positions
        test_board = np.zeros((4, 4), dtype=int)
        test_board[0, 0] = 9  # Mine at top-left
        test_board[2, 2] = 9  # Mine at center

        # Calculate expected adjacent counts
        expected_counts = np.zeros((4, 4), dtype=int)
        # Top-left corner
        expected_counts[0, 1] = 1
        expected_counts[1, 0] = 1
        expected_counts[1, 1] = 2  # Adjacent to both mines
        # Center
        expected_counts[1, 2] = 1  # Adjacent to center mine
        expected_counts[1, 3] = 1
        expected_counts[2, 1] = 1
        expected_counts[2, 3] = 1
        expected_counts[3, 1] = 1
        expected_counts[3, 2] = 1
        expected_counts[3, 3] = 1

        # Set up environment with test board
        self.env.board = test_board
        self.env._update_adjacent_counts()

        # Verify adjacent counts
        for row in range(4):
            for col in range(4):
                if test_board[row, col] != 9:  # Skip mine positions
                    assert self.env.board[row, col] == expected_counts[row, col], \
                        f"Wrong adjacent count at ({row}, {col}): expected {expected_counts[row, col]}, got {self.env.board[row, col]}" 

    def test_safe_cell_reveal(self):
        """Test revealing a safe cell and its effects."""
        # Create a test board with known mine positions
        test_board = np.zeros((4, 4), dtype=int)
        test_board[0, 0] = 9  # Mine at top-left
        test_board[2, 2] = 9  # Mine at center
        self.env.board = test_board
        self.env.mines = {(0, 0), (2, 2)}  # Initialize mines set
        self.env._update_adjacent_counts()

        # Test revealing a safe cell (1, 1) which should have 2 adjacent mines
        # Convert (1, 1) to action value using current_board_size
        action = 1 * self.env.current_board_size + 1
        state, reward, terminated, truncated, info = self.env.step(action)

        # Verify state update
        assert self.env.state[1, 1] == 2  # Cell should be revealed with count 2
        assert reward > 0  # Should get positive reward for safe reveal
        assert not terminated  # Game should not be over
        assert not truncated  # Game should not be truncated
        assert not self.env.won  # Game should not be won

        # Test revealing a cell with no adjacent mines (should trigger cascade)
        # Place a mine at (3, 3) and clear other positions
        test_board = np.zeros((4, 4), dtype=int)
        test_board[3, 3] = 9  # Mine at bottom-right
        self.env.board = test_board
        self.env.mines = {(3, 3)}  # Update mines set
        self.env._update_adjacent_counts()

        # Reveal cell (0, 0) which should trigger cascade
        action = 0 * self.env.current_board_size + 0  # Convert (0, 0) to action value
        state, reward, terminated, truncated, info = self.env.step(action)

        # Verify cascade effect
        # All cells except (3, 3) and its adjacent cells should be revealed
        for row in range(4):
            for col in range(4):
                if row < 2 and col < 2:  # Cells in top-left quadrant
                    assert self.env.state[row, col] != -1  # Should be revealed
                elif row == 3 and col == 3:  # Mine position
                    assert self.env.state[row, col] == -1  # Should still be hidden
                elif abs(row - 3) <= 1 and abs(col - 3) <= 1:  # Adjacent to mine
                    assert self.env.state[row, col] != -1  # Should be revealed with count

        assert reward > 0  # Should get positive reward for safe reveal
        assert not terminated  # Game should not be over
        assert not truncated  # Game should not be truncated
        assert not self.env.won  # Game should not be won

        # Verify info dict contains expected keys
        assert 'revealed_cells' in info
        assert 'adjacent_mines' in info
        assert 'reward_breakdown' in info 