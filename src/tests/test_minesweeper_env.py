import unittest
import numpy as np
from minesweeper_env import MinesweeperEnv

class TestMinesweeperEnv(unittest.TestCase):
    def setUp(self):
        """Set up a test environment before each test."""
        self.env = MinesweeperEnv(board_size=3, num_mines=2)
        self.env.reset(seed=42)  # Use a fixed seed for reproducible tests

    def test_initial_state(self):
        """Test that the initial state is correct."""
        state, _ = self.env.reset()
        # Check that all cells are unrevealed (-1)
        self.assertTrue(np.all(state == -1))
        # Check that the number of mines is correct
        self.assertEqual(len(self.env.mines), 2)

    def test_reveal_safe_cell(self):
        """Test revealing a safe cell."""
        # Find a safe cell
        safe_cell = None
        for x in range(self.env.board_size):
            for y in range(self.env.board_size):
                if (x, y) not in self.env.mines:
                    safe_cell = x * self.env.board_size + y
                    break
            if safe_cell is not None:
                break

        # Reveal the safe cell
        state, reward, terminated, truncated, _ = self.env.step(safe_cell)
        
        # Check that the cell was revealed
        x, y = safe_cell // self.env.board_size, safe_cell % self.env.board_size
        self.assertNotEqual(state[x, y], -1)
        # Check that we got a positive reward
        self.assertGreaterEqual(reward, 0)
        # Check that the game isn't over
        self.assertFalse(terminated)

    def test_reveal_mine(self):
        """Test revealing a mine."""
        # Find a mine
        mine_cell = None
        for x in range(self.env.board_size):
            for y in range(self.env.board_size):
                if (x, y) in self.env.mines:
                    mine_cell = x * self.env.board_size + y
                    break
            if mine_cell is not None:
                break

        # Reveal the mine
        state, reward, terminated, truncated, _ = self.env.step(mine_cell)
        
        # Check that we got a negative reward
        self.assertEqual(reward, -10)
        # Check that the game is over
        self.assertTrue(terminated)
        # Check that the mine was revealed
        x, y = mine_cell // self.env.board_size, mine_cell % self.env.board_size
        self.assertEqual(state[x, y], -2)

    def test_flag_cell(self):
        """Test flagging and unflagging a cell."""
        # Flag a cell
        cell = 0
        flag_action = cell + self.env.board_size * self.env.board_size
        state, reward, terminated, truncated, _ = self.env.step(flag_action)
        
        # Check that the cell is flagged
        x, y = cell // self.env.board_size, cell % self.env.board_size
        self.assertIn((x, y), self.env.flags)
        
        # Unflag the cell
        state, reward, terminated, truncated, _ = self.env.step(flag_action)
        
        # Check that the cell is not flagged
        self.assertNotIn((x, y), self.env.flags)

    def test_reveal_flagged_cell(self):
        """Test trying to reveal a flagged cell."""
        # Flag a cell
        cell = 0
        flag_action = cell + self.env.board_size * self.env.board_size
        self.env.step(flag_action)
        
        # Try to reveal the flagged cell
        state, reward, terminated, truncated, _ = self.env.step(cell)
        
        # Check that we got a negative reward
        self.assertEqual(reward, -1)
        # Check that the game is over
        self.assertTrue(terminated)

    def test_win_condition(self):
        """Test winning the game."""
        # Reveal all safe cells
        for x in range(self.env.board_size):
            for y in range(self.env.board_size):
                if (x, y) not in self.env.mines:
                    cell = x * self.env.board_size + y
                    state, reward, terminated, truncated, _ = self.env.step(cell)
                    if terminated:
                        break
            if terminated:
                break
        
        # Check that we won
        self.assertTrue(self.env.won)
        # Check that we got a positive reward
        self.assertEqual(reward, 10)

    def test_invalid_move_after_game_over(self):
        """Test making a move after the game is over."""
        # End the game by revealing a mine
        mine_cell = None
        for x in range(self.env.board_size):
            for y in range(self.env.board_size):
                if (x, y) in self.env.mines:
                    mine_cell = x * self.env.board_size + y
                    break
            if mine_cell is not None:
                break
        
        self.env.step(mine_cell)
        
        # Try to make another move
        state, reward, terminated, truncated, _ = self.env.step(0)
        
        # Check that we got zero reward
        self.assertEqual(reward, 0)
        # Check that the game is still over
        self.assertTrue(terminated)

    def test_auto_reveal_zero_cells(self):
        """Test that revealing a cell with no adjacent mines automatically reveals surrounding cells."""
        # Create a new environment with a known mine layout
        self.env = MinesweeperEnv(board_size=5, num_mines=1)
        self.env.reset(seed=42)
        
        # Place mine in a corner
        self.env.mines = {(0, 0)}
        self.env.board[0, 0] = -1
        
        # Calculate numbers for all cells
        for x in range(self.env.board_size):
            for y in range(self.env.board_size):
                if (x, y) not in self.env.mines:
                    self.env.board[x, y] = self.env._count_adjacent_mines(x, y)
        
        # Find a cell that has no adjacent mines (should be far from the corner)
        zero_cell = None
        for x in range(self.env.board_size):
            for y in range(self.env.board_size):
                if self.env.board[x, y] == 0:
                    zero_cell = x * self.env.board_size + y
                    break
            if zero_cell is not None:
                break
        
        # Reveal the zero cell
        state, reward, terminated, truncated, _ = self.env.step(zero_cell)
        
        # Count how many cells were revealed
        revealed_cells = np.sum(state != -1)
        
        # The number of revealed cells should be greater than 1
        # (the zero cell plus its surrounding cells)
        self.assertGreater(revealed_cells, 1)
        
        # Check that all revealed cells are either:
        # 1. The original zero cell
        # 2. Other zero cells
        # 3. Numbered cells (1-8)
        x, y = zero_cell // self.env.board_size, zero_cell % self.env.board_size
        for i in range(self.env.board_size):
            for j in range(self.env.board_size):
                if state[i, j] != -1:  # If cell is revealed
                    self.assertGreaterEqual(state[i, j], 0)  # Should be 0 or a positive number
                    self.assertLessEqual(state[i, j], 8)  # Should not be greater than 8

if __name__ == '__main__':
    unittest.main() 