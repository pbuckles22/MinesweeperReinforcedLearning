#!/usr/bin/env python3
"""
Probabilistic Guessing for Minesweeper

When CSP can't make logical progress, use probability-based guessing
to select the safest cell to reveal.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProbabilisticGuesser:
    """
    Probability-based guessing for Minesweeper.
    
    Uses various probability models to estimate mine probability
    for each unrevealed cell when logical moves are exhausted.
    """
    
    def __init__(self, board_size: Tuple[int, int], mine_count: int):
        """
        Initialize the probabilistic guesser.
        
        Args:
            board_size: (height, width) of the board
            mine_count: Total number of mines on the board
        """
        self.board_height, self.board_width = board_size
        self.mine_count = mine_count
        self.total_cells = self.board_height * self.board_width
        
        # Probability model weights
        self.weights = {
            'global_density': 0.4,
            'edge_factor': 0.2,
            'corner_factor': 0.1,
            'adjacency_factor': 0.3
        }
        
        # Cache for calculations
        self.adjacency_cache = {}
    
    def _get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get valid neighboring cells."""
        if (row, col) in self.adjacency_cache:
            return self.adjacency_cache[(row, col)]
        
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = row + di, col + dj
                if (0 <= ni < self.board_height and 
                    0 <= nj < self.board_width):
                    neighbors.append((ni, nj))
        
        self.adjacency_cache[(row, col)] = neighbors
        return neighbors
    
    def _get_revealed_neighbors(self, row: int, col: int, revealed_cells: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Get revealed neighboring cells."""
        neighbors = self._get_neighbors(row, col)
        return [n for n in neighbors if n in revealed_cells]
    
    def calculate_global_density_probability(self, unrevealed_cells: List[Tuple[int, int]], 
                                           remaining_mines: int) -> Dict[Tuple[int, int], float]:
        """
        Calculate mine probability based on global mine density.
        
        Args:
            unrevealed_cells: List of unrevealed cell positions
            remaining_mines: Number of mines not yet flagged
            
        Returns:
            Dictionary mapping cell positions to mine probabilities
        """
        if not unrevealed_cells:
            return {}
        
        # Simple uniform probability
        base_probability = remaining_mines / len(unrevealed_cells)
        
        probabilities = {}
        for cell in unrevealed_cells:
            probabilities[cell] = base_probability
        
        logger.debug(f"Global density probability: {base_probability:.3f}")
        return probabilities
    
    def calculate_edge_probability(self, unrevealed_cells: List[Tuple[int, int]]) -> Dict[Tuple[int, int], float]:
        """
        Calculate mine probability based on edge proximity.
        
        Mines are often more likely near edges due to game design.
        """
        probabilities = {}
        
        for cell in unrevealed_cells:
            row, col = cell
            
            # Calculate distance to nearest edge
            dist_to_edge = min(row, col, 
                              self.board_height - 1 - row, 
                              self.board_width - 1 - col)
            
            # Edge factor: closer to edge = higher mine probability
            edge_factor = 1.0 + (0.5 * (1.0 - dist_to_edge / max(self.board_height, self.board_width)))
            
            probabilities[cell] = edge_factor
        
        logger.debug(f"Edge probability calculated for {len(unrevealed_cells)} cells")
        return probabilities
    
    def calculate_corner_probability(self, unrevealed_cells: List[Tuple[int, int]]) -> Dict[Tuple[int, int], float]:
        """
        Calculate mine probability based on corner proximity.
        
        Corners often have different mine probabilities.
        """
        probabilities = {}
        
        for cell in unrevealed_cells:
            row, col = cell
            
            # Check if cell is in a corner
            is_corner = ((row == 0 or row == self.board_height - 1) and 
                        (col == 0 or col == self.board_width - 1))
            
            # Corner factor: corners might have different probabilities
            corner_factor = 1.2 if is_corner else 1.0
            
            probabilities[cell] = corner_factor
        
        logger.debug(f"Corner probability calculated for {len(unrevealed_cells)} cells")
        return probabilities
    
    def calculate_adjacency_probability(self, unrevealed_cells: List[Tuple[int, int]], 
                                      revealed_cells: Set[Tuple[int, int]], 
                                      board_state: np.ndarray) -> Dict[Tuple[int, int], float]:
        """
        Calculate mine probability based on adjacency to revealed numbers.
        
        Uses revealed numbers to estimate local mine density.
        """
        probabilities = {}
        
        for cell in unrevealed_cells:
            row, col = cell
            revealed_neighbors = self._get_revealed_neighbors(row, col, revealed_cells)
            
            if not revealed_neighbors:
                # No revealed neighbors, use neutral probability
                probabilities[cell] = 1.0
                continue
            
            # Calculate average mine density from revealed neighbors
            total_mine_indicators = 0
            total_neighbors = 0
            
            for neighbor in revealed_neighbors:
                ni, nj = neighbor
                # Get the number from the game state (first channel)
                number = board_state[0, ni, nj]
                
                if number >= 0:  # Valid number (not a mine)
                    # Count unrevealed neighbors of this revealed cell
                    neighbor_unrevealed = [n for n in self._get_neighbors(ni, nj) 
                                         if n not in revealed_cells]
                    
                    if neighbor_unrevealed:
                        # Estimate local mine density
                        local_mine_density = number / len(neighbor_unrevealed)
                        total_mine_indicators += local_mine_density
                        total_neighbors += 1
            
            if total_neighbors > 0:
                avg_mine_density = total_mine_indicators / total_neighbors
                # Convert to probability factor (normalize around 1.0)
                adjacency_factor = 0.5 + avg_mine_density
            else:
                adjacency_factor = 1.0
            
            probabilities[cell] = adjacency_factor
        
        logger.debug(f"Adjacency probability calculated for {len(unrevealed_cells)} cells")
        return probabilities
    
    def get_guessing_candidates(self, unrevealed_cells: List[Tuple[int, int]], 
                               revealed_cells: Set[Tuple[int, int]], 
                               flagged_cells: Set[Tuple[int, int]], 
                               board_state: np.ndarray) -> List[Tuple[int, int]]:
        """
        Get ranked list of guessing candidates.
        
        Args:
            unrevealed_cells: List of unrevealed cell positions
            revealed_cells: Set of revealed cell positions
            flagged_cells: Set of flagged cell positions
            board_state: Current board state
            
        Returns:
            List of cell positions ranked by safety (safest first)
        """
        if not unrevealed_cells:
            return []
        
        remaining_mines = self.mine_count - len(flagged_cells)
        
        # Calculate individual probability components
        global_probs = self.calculate_global_density_probability(unrevealed_cells, remaining_mines)
        edge_probs = self.calculate_edge_probability(unrevealed_cells)
        corner_probs = self.calculate_corner_probability(unrevealed_cells)
        adjacency_probs = self.calculate_adjacency_probability(unrevealed_cells, revealed_cells, board_state)
        
        # Combine probabilities using weighted average
        combined_probs = {}
        for cell in unrevealed_cells:
            combined_prob = (
                self.weights['global_density'] * global_probs.get(cell, 1.0) +
                self.weights['edge_factor'] * edge_probs.get(cell, 1.0) +
                self.weights['corner_factor'] * corner_probs.get(cell, 1.0) +
                self.weights['adjacency_factor'] * adjacency_probs.get(cell, 1.0)
            )
            combined_probs[cell] = combined_prob
        
        # Rank by safety (lower probability = safer)
        ranked_candidates = sorted(unrevealed_cells, key=lambda cell: combined_probs[cell])
        
        logger.info(f"Ranked {len(ranked_candidates)} guessing candidates")
        for i, cell in enumerate(ranked_candidates[:5]):  # Log top 5
            prob = combined_probs[cell]
            logger.info(f"  {i+1}. Cell {cell}: probability {prob:.3f}")
        
        return ranked_candidates
    
    def select_best_guess(self, unrevealed_cells: List[Tuple[int, int]], 
                         revealed_cells: Set[Tuple[int, int]], 
                         flagged_cells: Set[Tuple[int, int]], 
                         board_state: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Select the best cell to guess.
        
        Returns:
            Cell position to reveal, or None if no candidates
        """
        candidates = self.get_guessing_candidates(unrevealed_cells, revealed_cells, 
                                                flagged_cells, board_state)
        
        if candidates:
            best_guess = candidates[0]
            logger.info(f"Selected best guess: {best_guess}")
            return best_guess
        
        logger.warning("No guessing candidates available")
        return None
    
    def get_probability_info(self, cell: Tuple[int, int], 
                           unrevealed_cells: List[Tuple[int, int]], 
                           revealed_cells: Set[Tuple[int, int]], 
                           flagged_cells: Set[Tuple[int, int]], 
                           board_state: np.ndarray) -> Dict:
        """
        Get detailed probability information for a specific cell.
        
        Returns:
            Dictionary with probability breakdown
        """
        remaining_mines = self.mine_count - len(flagged_cells)
        
        global_prob = self.calculate_global_density_probability(unrevealed_cells, remaining_mines).get(cell, 1.0)
        edge_prob = self.calculate_edge_probability(unrevealed_cells).get(cell, 1.0)
        corner_prob = self.calculate_corner_probability(unrevealed_cells).get(cell, 1.0)
        adjacency_prob = self.calculate_adjacency_probability(unrevealed_cells, revealed_cells, board_state).get(cell, 1.0)
        
        combined_prob = (
            self.weights['global_density'] * global_prob +
            self.weights['edge_factor'] * edge_prob +
            self.weights['corner_factor'] * corner_prob +
            self.weights['adjacency_factor'] * adjacency_prob
        )
        
        return {
            'cell': cell,
            'global_density': global_prob,
            'edge_factor': edge_prob,
            'corner_factor': corner_prob,
            'adjacency_factor': adjacency_prob,
            'combined_probability': combined_prob,
            'weights': self.weights.copy()
        }


def test_probabilistic_guesser():
    """Test the probabilistic guesser with a simple example."""
    # Create a 4x4 board with 2 mines
    guesser = ProbabilisticGuesser((4, 4), 2)
    
    # Simulate a board state
    board_state = np.zeros((4, 4, 4), dtype=np.float32)
    revealed_cells = {(0, 0), (0, 1)}
    flagged_cells = set()
    unrevealed_cells = [(i, j) for i in range(4) for j in range(4) 
                        if (i, j) not in revealed_cells]
    
    # Set some revealed numbers
    board_state[0, 0, 0] = 1
    board_state[0, 1, 0] = 2
    
    # Get guessing candidates
    candidates = guesser.get_guessing_candidates(unrevealed_cells, revealed_cells, 
                                               flagged_cells, board_state)
    
    print(f"Guessing candidates: {candidates[:5]}")  # Show top 5
    
    # Get detailed info for first candidate
    if candidates:
        info = guesser.get_probability_info(candidates[0], unrevealed_cells, 
                                          revealed_cells, flagged_cells, board_state)
        print(f"Probability info for {candidates[0]}: {info}")


if __name__ == "__main__":
    test_probabilistic_guesser() 