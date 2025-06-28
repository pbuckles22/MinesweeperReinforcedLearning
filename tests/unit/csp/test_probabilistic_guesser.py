#!/usr/bin/env python3
"""
Unit tests for Probabilistic Guesser
"""

import pytest
import numpy as np
import sys
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from src.core.probabilistic_guesser import ProbabilisticGuesser


class TestProbabilisticGuesser:
    """Test cases for the Probabilistic Guesser."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.board_size = (4, 4)
        self.mine_count = 2
        self.guesser = ProbabilisticGuesser(self.board_size, self.mine_count)
    
    def test_initialization(self):
        """Test probabilistic guesser initialization."""
        assert self.guesser.board_height == 4
        assert self.guesser.board_width == 4
        assert self.guesser.mine_count == 2
        assert self.guesser.total_cells == 16
        
        # Check that weights are set
        assert 'global_density' in self.guesser.weights
        assert 'edge_factor' in self.guesser.weights
        assert 'corner_factor' in self.guesser.weights
        assert 'adjacency_factor' in self.guesser.weights
        
        # Check that weights sum to approximately 1.0
        total_weight = sum(self.guesser.weights.values())
        assert abs(total_weight - 1.0) < 0.01
    
    def test_get_neighbors(self):
        """Test neighbor calculation."""
        # Test corner cell
        neighbors = self.guesser._get_neighbors(0, 0)
        expected = {(0, 1), (1, 0), (1, 1)}
        assert set(neighbors) == expected
        
        # Test edge cell
        neighbors = self.guesser._get_neighbors(0, 1)
        expected = {(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)}
        assert set(neighbors) == expected
        
        # Test center cell
        neighbors = self.guesser._get_neighbors(1, 1)
        expected = {(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)}
        assert set(neighbors) == expected
    
    def test_calculate_global_density_probability(self):
        """Test global density probability calculation."""
        unrevealed_cells = [(0, 0), (0, 1), (1, 0), (1, 1)]
        remaining_mines = 2
        
        probabilities = self.guesser.calculate_global_density_probability(
            unrevealed_cells, remaining_mines
        )
        
        # Should have probability for each unrevealed cell
        assert len(probabilities) == len(unrevealed_cells)
        
        # All cells should have the same probability (uniform)
        expected_prob = remaining_mines / len(unrevealed_cells)  # 2/4 = 0.5
        for cell, prob in probabilities.items():
            assert abs(prob - expected_prob) < 0.001
    
    def test_calculate_edge_probability(self):
        """Test edge probability calculation."""
        unrevealed_cells = [(0, 0), (0, 1), (1, 0), (1, 1)]
        
        probabilities = self.guesser.calculate_edge_probability(unrevealed_cells)
        
        # Should have probability for each unrevealed cell
        assert len(probabilities) == len(unrevealed_cells)
        
        # Corner cells should have higher edge factors
        corner_cell = (0, 0)
        center_cell = (1, 1)
        
        assert probabilities[corner_cell] > probabilities[center_cell]
        
        # All probabilities should be positive
        for prob in probabilities.values():
            assert prob > 0
    
    def test_calculate_corner_probability(self):
        """Test corner probability calculation."""
        unrevealed_cells = [(0, 0), (0, 1), (1, 0), (1, 1)]
        
        probabilities = self.guesser.calculate_corner_probability(unrevealed_cells)
        
        # Should have probability for each unrevealed cell
        assert len(probabilities) == len(unrevealed_cells)
        
        # Corner cells should have higher corner factors
        corner_cell = (0, 0)
        non_corner_cell = (1, 1)
        
        assert probabilities[corner_cell] > probabilities[non_corner_cell]
        
        # All probabilities should be positive
        for prob in probabilities.values():
            assert prob > 0
    
    def test_calculate_adjacency_probability(self):
        """Test adjacency probability calculation."""
        unrevealed_cells = [(0, 0), (0, 1), (1, 0), (1, 1)]
        revealed_cells = {(2, 0), (2, 1)}  # Some revealed cells
        board_state = np.zeros((4, 4, 4), dtype=np.float32)
        
        # Set some revealed numbers
        board_state[0, 2, 0] = 1  # Number 1
        board_state[0, 2, 1] = 2  # Number 2
        
        probabilities = self.guesser.calculate_adjacency_probability(
            unrevealed_cells, revealed_cells, board_state
        )
        
        # Should have probability for each unrevealed cell
        assert len(probabilities) == len(unrevealed_cells)
        
        # All probabilities should be positive
        for prob in probabilities.values():
            assert prob > 0
    
    def test_get_guessing_candidates(self):
        """Test guessing candidate selection."""
        unrevealed_cells = [(0, 0), (0, 1), (1, 0), (1, 1)]
        revealed_cells = {(2, 0), (2, 1)}
        flagged_cells = set()
        board_state = np.zeros((4, 4, 4), dtype=np.float32)
        
        # Set some revealed numbers
        board_state[0, 2, 0] = 1
        board_state[0, 2, 1] = 2
        
        candidates = self.guesser.get_guessing_candidates(
            unrevealed_cells, revealed_cells, flagged_cells, board_state
        )
        
        # Should return all unrevealed cells in some order
        assert len(candidates) == len(unrevealed_cells)
        assert set(candidates) == set(unrevealed_cells)
        
        # Should be sorted by safety (safest first)
        # This is a basic check - the actual ordering depends on the probability model
    
    def test_select_best_guess(self):
        """Test best guess selection."""
        unrevealed_cells = [(0, 0), (0, 1), (1, 0), (1, 1)]
        revealed_cells = {(2, 0), (2, 1)}
        flagged_cells = set()
        board_state = np.zeros((4, 4, 4), dtype=np.float32)
        
        # Set some revealed numbers
        board_state[0, 2, 0] = 1
        board_state[0, 2, 1] = 2
        
        best_guess = self.guesser.select_best_guess(
            unrevealed_cells, revealed_cells, flagged_cells, board_state
        )
        
        # Should return one of the unrevealed cells
        assert best_guess in unrevealed_cells
    
    def test_get_probability_info(self):
        """Test detailed probability information."""
        cell = (0, 0)
        unrevealed_cells = [(0, 0), (0, 1), (1, 0), (1, 1)]
        revealed_cells = {(2, 0), (2, 1)}
        flagged_cells = set()
        board_state = np.zeros((4, 4, 4), dtype=np.float32)
        
        # Set some revealed numbers
        board_state[0, 2, 0] = 1
        board_state[0, 2, 1] = 2
        
        info = self.guesser.get_probability_info(
            cell, unrevealed_cells, revealed_cells, flagged_cells, board_state
        )
        
        # Should contain all probability components
        assert 'cell' in info
        assert 'global_density' in info
        assert 'edge_factor' in info
        assert 'corner_factor' in info
        assert 'adjacency_factor' in info
        assert 'combined_probability' in info
        assert 'weights' in info
        
        assert info['cell'] == cell
        assert isinstance(info['combined_probability'], float)
        assert info['combined_probability'] > 0


if __name__ == "__main__":
    pytest.main([__file__]) 