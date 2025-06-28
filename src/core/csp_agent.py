#!/usr/bin/env python3
"""
CSP + Probabilistic Hybrid Agent for Minesweeper

Combines constraint satisfaction problem solving with probability-based guessing
when logical moves are exhausted.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
import logging
from .csp_solver import MinesweeperCSP
from .probabilistic_guesser import ProbabilisticGuesser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CSPAgent:
    """
    Hybrid agent that combines CSP solving with probabilistic guessing.
    
    Strategy:
    1. Try to find logical moves using CSP
    2. If no logical moves, use probability-based guessing
    3. Update state and repeat
    """
    
    def __init__(self, board_size: Tuple[int, int], mine_count: int):
        """
        Initialize the CSP agent.
        
        Args:
            board_size: (height, width) of the board
            mine_count: Total number of mines on the board
        """
        self.board_size = board_size
        self.mine_count = mine_count
        
        # Initialize components
        self.csp_solver = MinesweeperCSP(board_size, mine_count)
        self.probabilistic_guesser = ProbabilisticGuesser(board_size, mine_count)
        
        # Game state tracking
        self.revealed_cells = set()
        self.flagged_cells = set()
        self.current_board_state = None
        
        # Statistics
        self.stats = {
            'total_moves': 0,
            'csp_moves': 0,
            'probability_moves': 0,
            'wins': 0,
            'losses': 0,
            'games_played': 0
        }
    
    def reset(self):
        """Reset the agent for a new game."""
        self.revealed_cells = set()
        self.flagged_cells = set()
        self.current_board_state = None
        
        # Reset CSP solver
        self.csp_solver = MinesweeperCSP(self.board_size, self.mine_count)
    
    def update_state(self, board_state: np.ndarray, revealed_cells: Set[Tuple[int, int]], 
                    flagged_cells: Set[Tuple[int, int]]):
        """
        Update the agent's state with current board information.
        
        Args:
            board_state: Current board state (4-channel representation)
            revealed_cells: Set of revealed cell positions
            flagged_cells: Set of flagged cell positions
        """
        self.current_board_state = board_state.copy()
        self.revealed_cells = revealed_cells.copy()
        self.flagged_cells = flagged_cells.copy()
        
        # Update CSP solver
        self.csp_solver.update_board_state(board_state, revealed_cells, flagged_cells)
        
        logger.debug(f"Updated state: {len(revealed_cells)} revealed, {len(flagged_cells)} flagged")
    
    def choose_action(self) -> Optional[Tuple[int, int]]:
        """
        Choose the next action to take.
        
        Returns:
            Cell position to reveal, or None if no valid action
        """
        if self.current_board_state is None:
            logger.error("No board state available")
            return None
        
        # Get unrevealed cells
        unrevealed_cells = self._get_unrevealed_cells()
        
        if not unrevealed_cells:
            logger.warning("No unrevealed cells available")
            return None
        
        # Try CSP first
        safe_cells = self.csp_solver.solve_step()
        
        if safe_cells:
            # CSP found logical moves
            action = safe_cells[0]  # Take the first safe cell
            self.stats['csp_moves'] += 1
            logger.info(f"CSP move: {action}")
            return action
        
        else:
            # No logical moves, use probability-based guessing
            action = self.probabilistic_guesser.select_best_guess(
                unrevealed_cells, self.revealed_cells, self.flagged_cells, 
                self.current_board_state
            )
            
            if action:
                self.stats['probability_moves'] += 1
                logger.info(f"Probability move: {action}")
                return action
        
        logger.error("No action found")
        return None
    
    def _get_unrevealed_cells(self) -> List[Tuple[int, int]]:
        """Get list of unrevealed cells."""
        unrevealed = []
        for i in range(self.board_size[0]):
            for j in range(self.board_size[1]):
                cell = (i, j)
                if cell not in self.revealed_cells and cell not in self.flagged_cells:
                    unrevealed.append(cell)
        return unrevealed
    
    def get_action_breakdown(self) -> Dict:
        """
        Get breakdown of action types used.
        
        Returns:
            Dictionary with action statistics
        """
        total_moves = self.stats['csp_moves'] + self.stats['probability_moves']
        
        if total_moves == 0:
            return {
                'total_moves': 0,
                'csp_percentage': 0.0,
                'probability_percentage': 0.0,
                'csp_moves': 0,
                'probability_moves': 0
            }
        
        return {
            'total_moves': total_moves,
            'csp_percentage': (self.stats['csp_moves'] / total_moves) * 100,
            'probability_percentage': (self.stats['probability_moves'] / total_moves) * 100,
            'csp_moves': self.stats['csp_moves'],
            'probability_moves': self.stats['probability_moves']
        }
    
    def get_csp_info(self) -> Dict:
        """Get CSP solver information."""
        return self.csp_solver.get_constraint_info()
    
    def get_probability_info(self, cell: Tuple[int, int]) -> Dict:
        """Get detailed probability information for a cell."""
        unrevealed_cells = self._get_unrevealed_cells()
        
        return self.probabilistic_guesser.get_probability_info(
            cell, unrevealed_cells, self.revealed_cells, 
            self.flagged_cells, self.current_board_state
        )
    
    def record_game_result(self, won: bool):
        """Record the result of a game."""
        self.stats['games_played'] += 1
        if won:
            self.stats['wins'] += 1
        else:
            self.stats['losses'] += 1
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics."""
        stats = self.stats.copy()
        
        # Add win rate
        if stats['games_played'] > 0:
            stats['win_rate'] = (stats['wins'] / stats['games_played']) * 100
        else:
            stats['win_rate'] = 0.0
        
        # Add action breakdown
        stats.update(self.get_action_breakdown())
        
        return stats
    
    def can_make_progress(self) -> bool:
        """Check if the agent can make any progress."""
        return self.csp_solver.can_make_progress() or len(self._get_unrevealed_cells()) > 0


def test_csp_agent():
    """Test the CSP agent with a simple example."""
    # Create a 4x4 board with 2 mines
    agent = CSPAgent((4, 4), 2)
    
    # Simulate a board state
    board_state = np.zeros((4, 4, 4), dtype=np.float32)
    revealed_cells = {(0, 0), (0, 1)}
    flagged_cells = set()
    
    # Set some revealed numbers
    board_state[0, 0, 0] = 1
    board_state[0, 1, 0] = 2
    
    # Update agent state
    agent.update_state(board_state, revealed_cells, flagged_cells)
    
    # Choose action
    action = agent.choose_action()
    print(f"Chosen action: {action}")
    
    # Get statistics
    stats = agent.get_stats()
    print(f"Agent stats: {stats}")
    
    # Get CSP info
    csp_info = agent.get_csp_info()
    print(f"CSP info: {csp_info}")


if __name__ == "__main__":
    test_csp_agent() 