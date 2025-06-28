#!/usr/bin/env python3
"""
Constraint Satisfaction Problem (CSP) Solver for Minesweeper

Implements a CSP-based approach to find logically safe moves in Minesweeper.
Uses constraint propagation to identify cells that can be safely revealed.
"""

import numpy as np
from typing import List, Tuple, Set, Dict, Optional
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MinesweeperCSP:
    """
    Constraint Satisfaction Problem solver for Minesweeper.
    
    Variables: Each cell (i, j) on the board
    Domains: {safe, mine} for each cell
    Constraints: Adjacency constraints from revealed numbers
    """
    
    def __init__(self, board_size: Tuple[int, int], mine_count: int):
        """
        Initialize the CSP solver.
        
        Args:
            board_size: (height, width) of the board
            mine_count: Total number of mines on the board
        """
        self.board_height, self.board_width = board_size
        self.mine_count = mine_count
        self.total_cells = self.board_height * self.board_width
        
        # CSP state
        self.variables = set()  # All cell positions
        self.domains = {}       # Domain for each variable
        self.constraints = []   # List of constraints
        
        # Board state tracking
        self.revealed = set()   # Revealed cells
        self.flagged = set()    # Flagged mines
        self.adjacency_cache = {}  # Cache for adjacency calculations
        
        # Initialize variables and domains
        self._initialize_csp()
    
    def _initialize_csp(self):
        """Initialize CSP variables and domains."""
        # Create variables for all cells
        for i in range(self.board_height):
            for j in range(self.board_width):
                cell = (i, j)
                self.variables.add(cell)
                # Initial domain: {safe, mine} for all cells
                self.domains[cell] = {'safe', 'mine'}
    
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
    
    def _get_unrevealed_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get unrevealed neighboring cells."""
        neighbors = self._get_neighbors(row, col)
        return [n for n in neighbors if n not in self.revealed]
    
    def _get_flagged_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get flagged neighboring cells."""
        neighbors = self._get_neighbors(row, col)
        return [n for n in neighbors if n in self.flagged]
    
    def update_board_state(self, board_state: np.ndarray, revealed_cells: Set[Tuple[int, int]], 
                          flagged_cells: Set[Tuple[int, int]]):
        """
        Update the CSP with current board state.
        
        Args:
            board_state: Current board state (4-channel representation)
            revealed_cells: Set of revealed cell positions
            flagged_cells: Set of flagged cell positions
        """
        self.revealed = revealed_cells.copy()
        self.flagged = flagged_cells.copy()
        
        # Update domains based on revealed information
        self._update_domains_from_revealed()
        
        # Generate constraints from revealed numbers
        self._generate_constraints(board_state)
        
        logger.debug(f"Updated CSP state: {len(self.revealed)} revealed, {len(self.flagged)} flagged")
    
    def _update_domains_from_revealed(self):
        """Update domains based on revealed cells."""
        # Revealed cells are definitely safe
        for cell in self.revealed:
            self.domains[cell] = {'safe'}
        
        # Flagged cells are definitely mines
        for cell in self.flagged:
            self.domains[cell] = {'mine'}
    
    def _generate_constraints(self, board_state: np.ndarray):
        """Generate constraints from revealed numbers."""
        self.constraints = []
        
        # Extract game state from first channel
        game_state = board_state[0]
        
        for i in range(self.board_height):
            for j in range(self.board_width):
                cell = (i, j)
                if cell in self.revealed:
                    # This is a revealed number
                    number = game_state[i, j]
                    if number >= 0:  # Not a mine
                        # Create constraint: sum of adjacent mines = number
                        unrevealed_neighbors = self._get_unrevealed_neighbors(i, j)
                        flagged_neighbors = self._get_flagged_neighbors(i, j)
                        
                        if unrevealed_neighbors:  # Only create constraint if there are unrevealed neighbors
                            constraint = {
                                'type': 'adjacency',
                                'center': cell,
                                'number': number,
                                'unrevealed': unrevealed_neighbors,
                                'flagged': flagged_neighbors,
                                'required_mines': number - len(flagged_neighbors)
                            }
                            self.constraints.append(constraint)
        
        # Add global mine count constraint
        remaining_mines = self.mine_count - len(self.flagged)
        unrevealed_cells = [cell for cell in self.variables 
                           if cell not in self.revealed and cell not in self.flagged]
        
        if unrevealed_cells:
            global_constraint = {
                'type': 'global_mine_count',
                'cells': unrevealed_cells,
                'required_mines': remaining_mines
            }
            self.constraints.append(global_constraint)
        
        logger.debug(f"Generated {len(self.constraints)} constraints")
    
    def solve_step(self) -> List[Tuple[int, int]]:
        """
        Find safe moves using constraint satisfaction.
        
        Returns:
            List of cell positions that can be safely revealed
        """
        safe_cells = []
        
        # Apply constraint propagation
        self._propagate_constraints()
        
        # Find cells that are definitely safe
        for cell in self.variables:
            if cell not in self.revealed and cell not in self.flagged:
                if self.domains[cell] == {'safe'}:
                    safe_cells.append(cell)
        
        logger.info(f"Found {len(safe_cells)} safe cells: {safe_cells}")
        return safe_cells
    
    def _propagate_constraints(self):
        """Apply constraint propagation to reduce domains."""
        changed = True
        iterations = 0
        max_iterations = 100  # Prevent infinite loops
        
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            
            for constraint in self.constraints:
                if constraint['type'] == 'adjacency':
                    changed |= self._propagate_adjacency_constraint(constraint)
                elif constraint['type'] == 'global_mine_count':
                    changed |= self._propagate_global_constraint(constraint)
        
        if iterations >= max_iterations:
            logger.warning("Constraint propagation reached max iterations")
    
    def _propagate_adjacency_constraint(self, constraint: Dict) -> bool:
        """Propagate adjacency constraint."""
        changed = False
        unrevealed = constraint['unrevealed']
        required_mines = constraint['required_mines']
        
        # Count current mine assignments
        mine_count = 0
        safe_cells = []
        unknown_cells = []
        
        for cell in unrevealed:
            if cell in self.domains:
                if self.domains[cell] == {'mine'}:
                    mine_count += 1
                elif self.domains[cell] == {'safe'}:
                    safe_cells.append(cell)
                else:
                    unknown_cells.append(cell)
        
        # If we have exactly the required mines, all unknown cells are safe
        if mine_count == required_mines:
            for cell in unknown_cells:
                if self.domains[cell] != {'safe'}:
                    self.domains[cell] = {'safe'}
                    changed = True
        
        # If we have the maximum possible mines, remaining cells are safe
        elif mine_count + len(unknown_cells) == required_mines:
            for cell in unknown_cells:
                if self.domains[cell] != {'mine'}:
                    self.domains[cell] = {'mine'}
                    changed = True
        
        return changed
    
    def _propagate_global_constraint(self, constraint: Dict) -> bool:
        """Propagate global mine count constraint."""
        # This is a simple implementation - could be enhanced
        return False
    
    def get_constraint_info(self) -> Dict:
        """Get information about current constraints."""
        return {
            'total_constraints': len(self.constraints),
            'adjacency_constraints': len([c for c in self.constraints if c['type'] == 'adjacency']),
            'global_constraints': len([c for c in self.constraints if c['type'] == 'global_mine_count']),
            'revealed_cells': len(self.revealed),
            'flagged_cells': len(self.flagged),
            'remaining_mines': self.mine_count - len(self.flagged)
        }
    
    def can_make_progress(self) -> bool:
        """Check if CSP can make logical progress."""
        safe_cells = self.solve_step()
        return len(safe_cells) > 0


def test_csp_solver():
    """Test the CSP solver with a simple example."""
    # Create a simple 4x4 board with 2 mines
    csp = MinesweeperCSP((4, 4), 2)
    
    # Simulate a board state where we have some revealed cells
    board_state = np.zeros((4, 4, 4), dtype=np.float32)
    
    # Reveal some cells (simulate a game in progress)
    revealed_cells = {(0, 0), (0, 1), (1, 0)}
    flagged_cells = set()
    
    # Set some revealed numbers
    board_state[0, 0, 0] = 1  # Number 1
    board_state[0, 1, 0] = 2  # Number 2
    board_state[1, 0, 0] = 1  # Number 1
    
    csp.update_board_state(board_state, revealed_cells, flagged_cells)
    
    # Try to find safe moves
    safe_cells = csp.solve_step()
    print(f"Safe cells found: {safe_cells}")
    
    # Get constraint info
    info = csp.get_constraint_info()
    print(f"Constraint info: {info}")


if __name__ == "__main__":
    test_csp_solver() 