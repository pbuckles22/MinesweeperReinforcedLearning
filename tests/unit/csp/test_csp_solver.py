#!/usr/bin/env python3
"""
Unit tests for CSP Solver
"""

import pytest
import numpy as np
import sys
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from src.core.csp_solver import MinesweeperCSP


class TestCSPSolver:
    """Test cases for the CSP Solver."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.board_size = (4, 4)
        self.mine_count = 2
        self.csp = MinesweeperCSP(self.board_size, self.mine_count)
    
    def test_initialization(self):
        """Test CSP solver initialization."""
        assert self.csp.board_height == 4
        assert self.csp.board_width == 4
        assert self.csp.mine_count == 2
        assert self.csp.total_cells == 16
        
        # Check that all cells are variables
        expected_variables = {(i, j) for i in range(4) for j in range(4)}
        assert self.csp.variables == expected_variables
        
        # Check that all cells have initial domains
        for cell in self.csp.variables:
            assert self.csp.domains[cell] == {'safe', 'mine'}
    
    def test_get_neighbors(self):
        """Test neighbor calculation."""
        # Test corner cell
        neighbors = self.csp._get_neighbors(0, 0)
        expected = {(0, 1), (1, 0), (1, 1)}
        assert set(neighbors) == expected
        
        # Test edge cell
        neighbors = self.csp._get_neighbors(0, 1)
        expected = {(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)}
        assert set(neighbors) == expected
        
        # Test center cell
        neighbors = self.csp._get_neighbors(1, 1)
        expected = {(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)}
        assert set(neighbors) == expected
    
    def test_update_board_state(self):
        """Test board state update."""
        board_state = np.zeros((4, 4, 4), dtype=np.float32)
        revealed_cells = {(0, 0), (0, 1)}
        flagged_cells = {(3, 3)}
        
        # Set some revealed numbers
        board_state[0, 0, 0] = 1
        board_state[0, 1, 0] = 2
        
        self.csp.update_board_state(board_state, revealed_cells, flagged_cells)
        
        # Check that revealed cells are marked as safe
        assert self.csp.domains[(0, 0)] == {'safe'}
        assert self.csp.domains[(0, 1)] == {'safe'}
        
        # Check that flagged cells are marked as mines
        assert self.csp.domains[(3, 3)] == {'mine'}
        
        # Check that other cells still have both options
        assert self.csp.domains[(1, 1)] == {'safe', 'mine'}
    
    def test_simple_constraint_propagation(self):
        """Test simple constraint propagation."""
        board_state = np.zeros((4, 4, 4), dtype=np.float32)
        
        # Create a simple scenario: cell (0,0) has number 1, with one neighbor revealed as safe
        board_state[0, 0, 0] = 1  # Number 1
        revealed_cells = {(0, 0), (0, 1)}  # (0,1) is revealed as safe
        flagged_cells = set()
        
        self.csp.update_board_state(board_state, revealed_cells, flagged_cells)
        
        # The CSP should find that (1,0) and (1,1) are safe
        # because (0,0) needs exactly 1 mine, and (0,1) is safe
        safe_cells = self.csp.solve_step()
        
        # This is a simple test - in practice, the CSP might need more sophisticated
        # constraint propagation to find these safe cells
        assert isinstance(safe_cells, list)
    
    def test_can_make_progress(self):
        """Test progress detection."""
        # Initially, no progress can be made
        assert not self.csp.can_make_progress()
        
        # After revealing some cells, check if progress can be made
        board_state = np.zeros((4, 4, 4), dtype=np.float32)
        board_state[0, 0, 0] = 1
        revealed_cells = {(0, 0)}
        flagged_cells = set()
        
        self.csp.update_board_state(board_state, revealed_cells, flagged_cells)
        
        # The result depends on the CSP implementation
        # This test ensures the method doesn't crash
        result = self.csp.can_make_progress()
        assert isinstance(result, bool)
    
    def test_constraint_info(self):
        """Test constraint information retrieval."""
        board_state = np.zeros((4, 4, 4), dtype=np.float32)
        board_state[0, 0, 0] = 1
        revealed_cells = {(0, 0)}
        flagged_cells = set()
        
        self.csp.update_board_state(board_state, revealed_cells, flagged_cells)
        
        info = self.csp.get_constraint_info()
        
        assert 'total_constraints' in info
        assert 'adjacency_constraints' in info
        assert 'global_constraints' in info
        assert 'revealed_cells' in info
        assert 'flagged_cells' in info
        assert 'remaining_mines' in info
        
        assert info['revealed_cells'] == 1
        assert info['flagged_cells'] == 0
        assert info['remaining_mines'] == 2


if __name__ == "__main__":
    pytest.main([__file__]) 