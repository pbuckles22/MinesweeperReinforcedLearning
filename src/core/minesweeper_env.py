import gymnasium as gym
import numpy as np
from gymnasium import spaces
import time
from datetime import datetime
from collections import deque
import logging
import os
import sys
import pygame
import warnings
from typing import Tuple, Dict, Optional, List, Set
from .constants import (
    CELL_UNREVEALED,
    CELL_MINE,
    CELL_MINE_HIT,
    UNKNOWN_SAFETY,
    REWARD_FIRST_CASCADE_SAFE,
    REWARD_FIRST_CASCADE_HIT_MINE,
    REWARD_SAFE_REVEAL,
    REWARD_WIN,
    REWARD_HIT_MINE,
    REWARD_INVALID_ACTION,
    DIFFICULTY_LEVELS
)

class MinesweeperEnv(gym.Env):
    """
    A Minesweeper environment for reinforcement learning with enhanced state representation and fixed observation/action space for curriculum learning.
    Supports multiple difficulty levels from easy to chaotic.
    """
    def __init__(self, max_board_size=(35, 20), max_mines=130, render_mode=None,
                 early_learning_mode=False, early_learning_threshold=200,
                 early_learning_corner_safe=True, early_learning_edge_safe=True,
                 mine_spacing=1, initial_board_size=4, initial_mines=2,
                 invalid_action_penalty=REWARD_INVALID_ACTION, mine_penalty=REWARD_HIT_MINE,
                 safe_reveal_base=REWARD_SAFE_REVEAL, win_reward=REWARD_WIN,
                 first_cascade_safe_reward=REWARD_FIRST_CASCADE_SAFE, first_cascade_hit_mine_reward=REWARD_FIRST_CASCADE_HIT_MINE,
                 learnable_only=True, max_learnable_attempts=1000):
        """Initialize the Minesweeper environment.
        
        Args:
            max_board_size: Maximum board dimensions (height, width)
            max_mines: Maximum number of mines
            render_mode: Rendering mode ('human' or None)
            early_learning_mode: Enable early learning mode
            early_learning_threshold: Threshold for early learning mode
            early_learning_corner_safe: Make corners safe in early learning
            early_learning_edge_safe: Make edges safe in early learning
            mine_spacing: Minimum distance between mines
            initial_board_size: Initial board size (height, width) or single dimension
            initial_mines: Initial number of mines
            invalid_action_penalty: Penalty for invalid actions
            mine_penalty: Penalty for hitting mines
            safe_reveal_base: Base reward for safe reveals
            win_reward: Reward for winning
            first_cascade_safe_reward: Reward for first cascade safe
            first_cascade_hit_mine_reward: Reward for first cascade hit mine
            learnable_only: Only generate board configurations that require 2+ moves
            max_learnable_attempts: Maximum attempts to find learnable configuration
        """
        super().__init__()
        
        # Validate parameters
        if isinstance(max_board_size, int):
            if max_board_size <= 0:
                raise ValueError("Board size must be positive")
            if max_board_size > 100:
                raise ValueError("Board dimensions too large")
            max_board_area = max_board_size * max_board_size
            self.max_board_size = (max_board_size, max_board_size)
        else:
            if max_board_size[0] <= 0 or max_board_size[1] <= 0:
                raise ValueError("Board dimensions must be positive")
            if max_board_size[0] > 100 or max_board_size[1] > 100:
                raise ValueError("Board dimensions too large")
            max_board_area = max_board_size[0] * max_board_size[1]
            self.max_board_size = max_board_size
        if max_mines <= 0:
            raise ValueError("Mine count must be positive")
        if max_mines > max_board_area:
            raise ValueError("Mine count cannot exceed board size area (height*width)")
        
        # Initial parameters
        if isinstance(initial_board_size, int):
            if initial_board_size <= 0:
                raise ValueError("Initial board size must be positive")
            if isinstance(max_board_size, int):
                if initial_board_size > max_board_size:
                    raise ValueError("Initial board size cannot exceed max board size")
            else:
                if initial_board_size > max_board_size[0] or initial_board_size > max_board_size[1]:
                    raise ValueError("Initial board size cannot exceed max board size")
            self.initial_board_size = (initial_board_size, initial_board_size)
        else:
            if initial_board_size[0] <= 0 or initial_board_size[1] <= 0:
                raise ValueError("Initial board dimensions must be positive")
            if initial_board_size[0] > self.max_board_size[0] or initial_board_size[1] > self.max_board_size[1]:
                raise ValueError("Initial board size cannot exceed max board size")
            self.initial_board_size = initial_board_size
        if initial_mines <= 0:
            raise ValueError("Initial mine count must be positive")
        if initial_mines > self.initial_board_size[0] * self.initial_board_size[1]:
            raise ValueError("Initial mine count cannot exceed initial board area (height*width)")
        
        # Validate reward parameters
        if invalid_action_penalty is None or mine_penalty is None or safe_reveal_base is None or win_reward is None:
            raise TypeError("'>=' not supported between instances of 'NoneType' and 'int'")
        
        self.max_mines = max_mines
        self.mine_spacing = mine_spacing
        self.initial_mines = initial_mines
        
        # Current parameters (can change during curriculum learning)
        self.current_board_height, self.current_board_width = self.initial_board_size
        self.current_mines = self.initial_mines
        
        # Early learning parameters
        self.early_learning_mode = early_learning_mode
        self.early_learning_threshold = early_learning_threshold
        self.early_learning_corner_safe = early_learning_corner_safe
        self.early_learning_edge_safe = early_learning_edge_safe
        
        # Reward parameters
        self.invalid_action_penalty = invalid_action_penalty
        self.mine_penalty = mine_penalty
        self.safe_reveal_base = safe_reveal_base
        self.win_reward = win_reward
        self.first_cascade_safe_reward = first_cascade_safe_reward
        self.first_cascade_hit_mine_reward = first_cascade_hit_mine_reward
        self.reward_invalid_action = invalid_action_penalty
        
        # Learnable configuration parameters
        self.learnable_only = learnable_only
        self.max_learnable_attempts = max_learnable_attempts
        
        # Game state
        self.board = None
        self.mines = None
        self.revealed = None
        self.terminated = False
        self.truncated = False
        self.mines_placed = False
        
        # Pre-cascade tracking
        self.is_first_cascade = True
        self.in_cascade = False
        
        # Statistics tracking - Dual system
        # Real-life statistics (what would happen in actual Minesweeper)
        self.real_life_games_played = 0
        self.real_life_games_won = 0
        self.real_life_games_lost = 0
        
        # RL training statistics (excluding pre-cascade games)
        self.rl_games_played = 0
        self.rl_games_won = 0
        self.rl_games_lost = 0
        
        # Current game tracking
        self.current_game_was_pre_cascade = False
        self.current_game_ended_pre_cascade = False
        
        # Move counting for current game
        self.move_count = 0
        self.total_moves_across_games = 0
        self.games_with_move_counts = []
        
        # Repeated actions and revealed cell clicks
        self.repeated_actions = set()
        self.repeated_action_count = 0
        self.revealed_cell_click_count = 0
        self._actions_taken_this_game = set()
        
        # Reset invalid action and guaranteed mine click counters
        self.invalid_action_count = 0
        
        # Action space and observation space
        self.action_space = gym.spaces.Discrete(self.current_board_width * self.current_board_height)
        self.observation_space = gym.spaces.Box(
            low=-1, high=9, 
            shape=(4, self.current_board_height, self.current_board_width), 
            dtype=np.float32
        )
        
        # State representation
        self.state = np.zeros((4, self.current_board_height, self.current_board_width), dtype=np.float32)
        
        # Rendering
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.cell_size = 30
        
        # Initialize the environment
        self.reset()

        # Initialize pygame if render mode is set
        if self.render_mode == "human":
            pygame.init()
            self.cell_size = 40
            self.screen = pygame.display.set_mode((self.current_board_width * self.cell_size, 
                                                 self.current_board_height * self.cell_size))
            pygame.display.set_caption("Minesweeper")
            self.clock = pygame.time.Clock()

    @property
    def max_board_height(self):
        """Get the maximum board height."""
        return self.max_board_size[0]

    @property
    def max_board_width(self):
        """Get the maximum board width."""
        return self.max_board_size[1]

    @property
    def initial_board_height(self):
        """Get the initial board height."""
        return self.initial_board_size[0]

    @property
    def initial_board_width(self):
        """Get the initial board width."""
        return self.initial_board_size[1]

    # Backward compatibility properties
    @property
    def max_board_size_int(self):
        """Get max board size as integer for backward compatibility."""
        if self.max_board_size[0] == self.max_board_size[1]:
            return self.max_board_size[0]
        return self.max_board_size[0]  # Return width as default

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Ensure deterministic numpy RNG if seed is provided
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize or update action space based on current board size
        self.action_space = spaces.Discrete(self.current_board_width * self.current_board_height)
        
        # Initialize enhanced state space with 4 channels for better pattern recognition
        low_bounds = np.full((4, self.current_board_height, self.current_board_width), -1, dtype=np.float32)
        low_bounds[0] = -4  # Channel 0: game state can go as low as -4 (mine hit)
        low_bounds[1] = -1  # Channel 1: safety hints can go as low as -1 (unknown)
        low_bounds[2] = 0   # Channel 2: revealed cell count (always >= 0)
        low_bounds[3] = 0   # Channel 3: game progress indicators (always >= 0)
        
        high_bounds = np.full((4, self.current_board_height, self.current_board_width), 8, dtype=np.float32)
        high_bounds[2] = self.current_board_height * self.current_board_width  # Max revealed cells
        high_bounds[3] = 1  # Binary indicators
        
        self.observation_space = spaces.Box(
            low=low_bounds,
            high=high_bounds,
            shape=(4, self.current_board_height, self.current_board_width),
            dtype=np.float32
        )
        
        # Initialize board state
        self.board = np.zeros((self.current_board_height, self.current_board_width), dtype=np.int8)
        self.mines = np.zeros((self.current_board_height, self.current_board_width), dtype=bool)
        self.revealed = np.zeros((self.current_board_height, self.current_board_width), dtype=bool)
        
        # Initialize enhanced state with 4 channels
        self.state = np.zeros((4, self.current_board_height, self.current_board_width), dtype=np.float32)
        
        # Channel 0: Game state (all unrevealed initially)
        self.state[0] = CELL_UNREVEALED
        
        # Channel 1: Safety hints (all unknown initially)
        self.state[1] = UNKNOWN_SAFETY
        
        # Channel 2: Revealed cell count (0 initially)
        self.state[2] = 0
        
        # Channel 3: Game progress indicators (0 initially)
        self.state[3] = 0
        
        # Reset game state variables
        self.revealed_count = 0
        self.won = False
        self.terminated = False
        self.truncated = False
        self.is_first_cascade = True
        self.first_cascade_done = False
        self.in_cascade = False  # Track if we're currently in a cascade
        
        # Reset move counting for new game
        self.move_count = 0
        
        # Reset repeated action and revealed cell click counters
        self.repeated_actions = set()
        self.repeated_action_count = 0
        self.revealed_cell_click_count = 0
        self._actions_taken_this_game = set()
        
        # Initialize info dict
        self.info = {
            "won": False,
            "learnable": False
        }
        
        # Place mines immediately (before first move)
        self._place_mines()
        
        # Update enhanced state after mine placement
        self._update_enhanced_state()
        
        return self.state, self.info

    def _place_mines(self):
        """Place mines on the board, ensuring learnable configuration if requested."""
        if self.learnable_only:
            self._place_mines_learnable()
        else:
            self._place_mines_random()
    
    def _place_mines_learnable(self):
        """Place mines ensuring configuration requires 2+ moves and first move is safe."""
        for attempt in range(self.max_learnable_attempts):
            # Generate random mine positions
            mine_positions = self._generate_random_mine_positions()
            
            # Check if configuration is learnable
            if self._is_learnable_configuration(mine_positions):
                # Place the mines
                self._place_mines_at_positions(mine_positions)
                
                # Check if first move is safe (no mine adjustment, pure filtering)
                if self._has_safe_first_move():
                    # Store learnable status
                    self.info['learnable'] = True
                    return
                else:
                    # First move is not safe, try again
                    continue
        
        # Fallback to random if no learnable config found
        warnings.warn(f"Could not find learnable configuration after {self.max_learnable_attempts} attempts, using random placement")
        self._place_mines_random()
        # Mark as not learnable
        self.info['learnable'] = False
    
    def _has_safe_first_move(self):
        """Check if there's at least one safe first move (corners or edges)."""
        # Check corners first (common first moves)
        corners = [(0, 0), (0, self.current_board_width-1), 
                  (self.current_board_height-1, 0), (self.current_board_height-1, self.current_board_width-1)]
        
        for row, col in corners:
            if not self.mines[row, col]:
                return True
        
        # Check edges if no safe corners
        for row in [0, self.current_board_height-1]:
            for col in range(self.current_board_width):
                if not self.mines[row, col]:
                    return True
        
        for col in [0, self.current_board_width-1]:
            for row in range(self.current_board_height):
                if not self.mines[row, col]:
                    return True
        
        return False
    
    def _get_safe_positions(self):
        """Get positions that would be safe for first moves."""
        safe_positions = []
        
        # Add corners
        safe_positions.extend([
            (0, 0), (0, self.current_board_width-1),
            (self.current_board_height-1, 0), (self.current_board_height-1, self.current_board_width-1)
        ])
        
        # Add edges
        for row in [0, self.current_board_height-1]:
            for col in range(self.current_board_width):
                if (row, col) not in safe_positions:
                    safe_positions.append((row, col))
        
        for col in [0, self.current_board_width-1]:
            for row in range(self.current_board_height):
                if (row, col) not in safe_positions:
                    safe_positions.append((row, col))
        
        return safe_positions
    
    def _place_mines_random(self):
        """Place mines randomly (original implementation)."""
        # Create list of valid positions
        valid_positions = []
        for y in range(self.current_board_height):
            for x in range(self.current_board_width):
                # Skip positions that would violate mine spacing
                if self.mine_spacing > 0:
                    valid = True
                    for dy in range(-self.mine_spacing, self.mine_spacing + 1):
                        for dx in range(-self.mine_spacing, self.mine_spacing + 1):
                            ny, nx = y + dy, x + dx
                            if (0 <= ny < self.current_board_height and 
                                0 <= nx < self.current_board_width and 
                                self.mines[ny, nx]):
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                valid_positions.append((y, x))

        # Shuffle valid positions
        np.random.shuffle(valid_positions)

        # Place mines
        mines_placed = 0
        for y, x in valid_positions:
            if mines_placed >= self.current_mines:
                break
            if not self.mines[y, x]:  # Ensure no mine is already placed at this position
                self.mines[y, x] = True
                mines_placed += 1

        # Update current_mines if we couldn't place all mines
        if mines_placed < self.current_mines:
            warnings.warn(f"Could only place {mines_placed} mines due to spacing constraints")
            self.current_mines = mines_placed

        # Update adjacent counts
        self._update_adjacent_counts()
        
        # For random placement, check if it's learnable
        mine_positions = [(y, x) for y in range(self.current_board_height) 
                         for x in range(self.current_board_width) if self.mines[y, x]]
        self.info['learnable'] = self._is_learnable_configuration(mine_positions)
    
    def _generate_random_mine_positions(self):
        """Generate random mine positions respecting spacing constraints."""
        # Create list of valid positions
        valid_positions = []
        for y in range(self.current_board_height):
            for x in range(self.current_board_width):
                # Skip positions that would violate mine spacing
                if self.mine_spacing > 0:
                    valid = True
                    for dy in range(-self.mine_spacing, self.mine_spacing + 1):
                        for dx in range(-self.mine_spacing, self.mine_spacing + 1):
                            ny, nx = y + dy, x + dx
                            if (0 <= ny < self.current_board_height and 
                                0 <= nx < self.current_board_width and 
                                self.mines[ny, nx]):
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                valid_positions.append((y, x))

        # Shuffle and select positions
        np.random.shuffle(valid_positions)
        return valid_positions[:self.current_mines]
    
    def _place_mines_at_positions(self, mine_positions):
        """Place mines at specific positions."""
        # Clear existing mines
        self.mines.fill(False)
        
        # Place mines at specified positions
        for y, x in mine_positions:
            self.mines[y, x] = True
        
        # Update adjacent counts
        self._update_adjacent_counts()
    
    def _is_learnable_configuration(self, mine_positions):
        """Check if mine placement creates a learnable scenario (requires 2+ moves)."""
        if len(mine_positions) == 1:
            return self._is_single_mine_learnable(mine_positions[0])
        else:
            return self._is_multi_mine_learnable(mine_positions)
    
    def _is_single_mine_learnable(self, mine_pos):
        """Check if single mine placement requires 2+ moves using cascade simulation."""
        # Create a temporary board with the mine at the specified position
        temp_board = np.zeros((self.current_board_height, self.current_board_width), dtype=int)
        row, col = mine_pos
        temp_board[row, col] = 9  # Place mine
        
        # Fill in adjacent counts
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = row + dr, col + dc
                if (0 <= nr < self.current_board_height and 
                    0 <= nc < self.current_board_width and 
                    (nr, nc) != mine_pos):
                    temp_board[nr, nc] += 1
        
        # Simulate cascade from a non-mine cell
        revealed_cells = self._simulate_cascade(temp_board, mine_pos)
        total_cells = self.current_board_height * self.current_board_width
        
        # If cascade reveals (total_cells - 1) cells, it's a 1-move win (not learnable)
        # Otherwise, it requires strategic play (learnable)
        return revealed_cells < (total_cells - 1)
    
    def _simulate_cascade(self, board, mine_pos):
        """Simulate a cascade from a non-mine cell and return number of revealed cells."""
        h, w = board.shape
        max_revealed = 0
        
        # Try cascading from every non-mine cell and find the maximum
        for start_r in range(h):
            for start_c in range(w):
                if (start_r, start_c) == mine_pos:
                    continue
                
                # Simulate cascade from this start position
                revealed = np.zeros_like(board, dtype=bool)
                queue = [(start_r, start_c)]
                
                while queue:
                    r, c = queue.pop(0)
                    if revealed[r, c]:
                        continue
                    
                    revealed[r, c] = True
                    
                    # If this cell has no adjacent mines (value 0), cascade to neighbors
                    if board[r, c] == 0:
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr == 0 and dc == 0:
                                    continue
                                nr, nc = r + dr, c + dc
                                if (0 <= nr < h and 
                                    0 <= nc < w and 
                                    not revealed[nr, nc] and 
                                    (nr, nc) != mine_pos):
                                    queue.append((nr, nc))
                
                # Update maximum revealed
                revealed_count = revealed.sum()
                if revealed_count > max_revealed:
                    max_revealed = revealed_count
        
        return max_revealed
    
    def _simulate_cascade_multi_mine(self, board, mine_positions):
        """Simulate a cascade from a non-mine cell for multi-mine boards."""
        h, w = board.shape
        max_revealed = 0
        
        # Try cascading from every non-mine cell and find the maximum
        for start_r in range(h):
            for start_c in range(w):
                if (start_r, start_c) in mine_positions:
                    continue
                
                # Simulate cascade from this start position
                revealed = np.zeros_like(board, dtype=bool)
                queue = [(start_r, start_c)]
                
                while queue:
                    r, c = queue.pop(0)
                    if revealed[r, c]:
                        continue
                    
                    revealed[r, c] = True
                    
                    # If this cell has no adjacent mines (value 0), cascade to neighbors
                    if board[r, c] == 0:
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr == 0 and dc == 0:
                                    continue
                                nr, nc = r + dr, c + dc
                                if (0 <= nr < h and 
                                    0 <= nc < w and 
                                    not revealed[nr, nc] and 
                                    (nr, nc) not in mine_positions):
                                    queue.append((nr, nc))
                
                # Update maximum revealed
                revealed_count = revealed.sum()
                if revealed_count > max_revealed:
                    max_revealed = revealed_count
        
        return max_revealed
    
    def _is_multi_mine_learnable(self, mine_positions):
        """Check if multi-mine placement requires 2+ moves."""
        # For multi-mine games, we need to check if any first move can cause an instant win
        # or if the first move is guaranteed to be safe
        
        # First, check if there's a safe first move
        if not self._has_safe_first_move():
            return False
        
        # Create a temporary board with the mines
        temp_board = np.zeros((self.current_board_height, self.current_board_width), dtype=int)
        
        # Place mines
        for row, col in mine_positions:
            temp_board[row, col] = 9
        
        # Fill in adjacent counts
        for row, col in mine_positions:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = row + dr, col + dc
                    if (0 <= nr < self.current_board_height and 
                        0 <= nc < self.current_board_width and 
                        (nr, nc) not in mine_positions):
                        temp_board[nr, nc] += 1
        
        # Test each non-mine position as a first move for instant wins
        total_cells = self.current_board_height * self.current_board_width
        mine_count = len(mine_positions)
        safe_cells = total_cells - mine_count
        
        for start_row in range(self.current_board_height):
            for start_col in range(self.current_board_width):
                if (start_row, start_col) in mine_positions:
                    continue  # Skip mine positions
                
                # Simulate clicking this position and any cascading cells
                revealed = np.zeros_like(temp_board, dtype=bool)
                queue = [(start_row, start_col)]
                
                while queue:
                    r, c = queue.pop(0)
                    if revealed[r, c]:
                        continue
                    
                    revealed[r, c] = True
                    
                    # If this cell has no adjacent mines (value 0), cascade to neighbors
                    if temp_board[r, c] == 0:
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr == 0 and dc == 0:
                                    continue
                                nr, nc = r + dr, c + dc
                                if (0 <= nr < self.current_board_height and 
                                    0 <= nc < self.current_board_width and 
                                    not revealed[nr, nc] and 
                                    (nr, nc) not in mine_positions):
                                    queue.append((nr, nc))
                
                # Check if this first move reveals all safe cells (instant win)
                revealed_count = revealed.sum()
                
                # If this first move reveals all safe cells, it's an instant win (not learnable)
                if revealed_count >= safe_cells:
                    return False  # This configuration allows instant win
        
        # If no instant wins found, the configuration is learnable
        return True

    def _update_adjacent_counts(self):
        """Update the board with the count of adjacent mines for each cell."""
        # Reset the board to zeros
        self.board.fill(0)
        
        # For each mine, increment the count of adjacent cells
        for i in range(self.current_board_height):
            for j in range(self.current_board_width):
                if self.mines[i, j]:
                    # Set the mine cell to 9 (representing a mine)
                    self.board[i, j] = 9
                    # Increment count for all adjacent cells
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue  # Skip the mine cell itself
                            ni, nj = i + di, j + dj
                            if (0 <= ni < self.current_board_height and 
                                0 <= nj < self.current_board_width):
                                self.board[ni, nj] += 1

    def _reveal_cell(self, row: int, col: int) -> None:
        """Reveal a cell and its neighbors if it's empty."""
        if not (0 <= row < self.current_board_height and 0 <= col < self.current_board_width):
            return
        if self.revealed[row, col]:
            return

        self.revealed[row, col] = True
        cell_value = self._get_cell_value(row, col)
        self.state[0, row, col] = cell_value

        # Check if this is a cascade (cell with value 0)
        if cell_value == 0:
            # This is a cascade - mark that we're in a cascade
            self.in_cascade = True
            # Note: We don't set is_first_cascade = False here anymore
            # It will be set after the win check in the step function
            # Reveal all neighbors
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    self._reveal_cell(row + dr, col + dc)

    def _get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get all valid neighbors of a cell.
        Args:
            row: Row coordinate
            col: Column coordinate
        Returns:
            List of (row, col) tuples for valid neighbors
        """
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if (0 <= nr < self.current_board_height and 
                    0 <= nc < self.current_board_width):
                    neighbors.append((nr, nc))
        return neighbors

    def _check_win(self) -> bool:
        """Check if the game is won.
        Win condition: All non-mine cells must be revealed.
        Returns:
            bool: True if all non-mine cells are revealed, False otherwise.
        """
        # For each cell that is not a mine, it must be revealed
        for i in range(self.current_board_height):
            for j in range(self.current_board_width):
                if not self.mines[i, j] and not self.revealed[i, j]:
                    return False
        return True

    def step(self, action):
        # Initialize info dict with 'won' key
        info = {'won': self._check_win()}

        # Convert action to integer if it's a numpy array
        if hasattr(action, 'item'):
            action = action.item()

        # If game is over, all actions are invalid and return negative reward
        if self.terminated or self.truncated:
            return self.state, self.invalid_action_penalty, True, False, info

        # Terminate if no valid actions left
        if not np.any(self.action_masks):
            self.terminated = True
            info['won'] = self._check_win()
            return self.state, 0.0, True, False, info

        # Check if action is within bounds first
        if action < 0 or action >= self.action_space.n:
            self.invalid_action_count += 1
            return self.state, self.invalid_action_penalty, False, False, info

        # Track repeated actions
        if action in self._actions_taken_this_game:
            self.repeated_action_count += 1
            self.repeated_actions.add(action)
        else:
            self._actions_taken_this_game.add(action)

        # Check if action is valid using action masks
        if not self.action_masks[action]:
            self.invalid_action_count += 1
            # If the cell is already revealed, increment revealed_cell_click_count
            col = action % self.current_board_width
            row = action // self.current_board_width
            if self.revealed[row, col]:
                self.revealed_cell_click_count += 1
            return self.state, self.invalid_action_penalty, False, False, info

        # Increment move count for valid actions
        self.move_count += 1

        # Convert action to (x, y) coordinates
        col = action % self.current_board_width
        row = action // self.current_board_width

        # Handle cell reveal
        if self.mines[row, col]:  # Hit a mine
            # Game always terminates on mine hit
            self.state[0, row, col] = CELL_MINE_HIT
            self.revealed[row, col] = True
            self.terminated = True
            info['won'] = False
            
            # Track if this game ended pre-cascade
            game_ended_pre_cascade = self.is_first_cascade
            
            # Update statistics
            self._update_statistics(game_won=False, game_ended_pre_cascade=game_ended_pre_cascade)
            
            # Mine hit penalty - immediate negative feedback
            return self.state, self.mine_penalty, True, False, info

        # Reveal the cell (safe cell)
        self._reveal_cell(row, col)

        # Update enhanced state after revealing cells
        self._update_enhanced_state()

        # Always check for win after all reveals (including cascades)
        if self._check_win():
            # Check if this win happened during the first cascade period
            win_during_first_cascade_period = self.is_first_cascade
            
            self.is_first_cascade = False
            self.terminated = True
            info['won'] = True
            
            # Track if this game ended pre-cascade
            game_ended_pre_cascade = win_during_first_cascade_period
            
            # Update statistics
            self._update_statistics(game_won=True, game_ended_pre_cascade=game_ended_pre_cascade)
            
            # Win reward - always give full win reward
            return self.state, self.win_reward, True, False, info

        # Safe reveal reward - immediate positive feedback
        reward = self.safe_reveal_base
        
        # If we had a cascade in this step and no win occurred, exit pre-cascade period
        if self.in_cascade and self.is_first_cascade:
            self.is_first_cascade = False
        
        # Reset cascade flag for next step
        self.in_cascade = False
        
        info['won'] = False
        return self.state, reward, False, False, info

    @property
    def action_masks(self):
        """Return a boolean mask indicating which actions are valid, including smart masking for obviously bad moves."""
        # If game is over, all actions are invalid
        if self.terminated or self.truncated:
            return np.zeros(self.action_space.n, dtype=bool)
        
        masks = np.ones(self.action_space.n, dtype=bool)
        
        for i in range(self.current_board_height):
            for j in range(self.current_board_width):
                # Reveal action
                reveal_idx = i * self.current_board_width + j
                
                # Basic masking: can't reveal already revealed cells
                if self.revealed[i, j]:
                    masks[reveal_idx] = False
                    continue
                
                # Smart masking: prefer cells that are guaranteed to be safe
                # (This is optional - we could prioritize safe cells but still allow others)
                # For now, we'll just avoid guaranteed mines
                
        return masks
    
    def render(self):
        """Render the environment."""
        if self.render_mode != "human":
            return

        self.screen.fill((192, 192, 192))  # Gray background

        for y in range(self.current_board_height):
            for x in range(self.current_board_width):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                                 self.cell_size, self.cell_size)
                
                # Use channel 0 (game state) for rendering
                cell_value = self.state[0, y, x]
                
                if cell_value == CELL_UNREVEALED:
                    pygame.draw.rect(self.screen, (128, 128, 128), rect)  # Gray for unrevealed
                elif cell_value == CELL_MINE_HIT:
                    pygame.draw.rect(self.screen, (255, 0, 0), rect)  # Red for mine hit
                else:
                    pygame.draw.rect(self.screen, (255, 255, 255), rect)  # White for revealed
                    if cell_value > 0:
                        # Draw number
                        font = pygame.font.Font(None, 36)
                        text = font.render(str(int(cell_value)), True, (0, 0, 0))
                        text_rect = text.get_rect(center=rect.center)
                        self.screen.blit(text, text_rect)
                
                # Draw grid lines
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)

        pygame.display.flip()
        self.clock.tick(60)

    def _is_valid_action(self, action):
        """Check if an action is valid."""
        # Check if action is within bounds
        if action < 0 or action >= self.action_space.n:
            return False

        # Convert action to (x, y) coordinates
        col = action % self.current_board_width
        row = action // self.current_board_width

        # Check if coordinates are valid
        if not (0 <= row < self.current_board_height and 0 <= col < self.current_board_width):
            return False

        # Handle reveal actions
        if self.revealed[row, col]:  # Can't reveal already revealed cells
            return False
        return True

    def _get_cell_value(self, row: int, col: int) -> int:
        """Get the value of a cell (number of adjacent mines).
        Args:
            row (int): Row index of the cell.
            col (int): Column index of the cell.
        Returns:
            int: The value of the cell (number of adjacent mines).
        """
        return self.board[row, col]

    def _update_enhanced_state(self):
        """Update the enhanced state representation with 4 channels for better pattern recognition."""
        # Channel 0: Game state (revealed cells with numbers, unrevealed as -1, mine hits as -4)
        for i in range(self.current_board_height):
            for j in range(self.current_board_width):
                if self.revealed[i, j]:
                    if self.mines[i, j]:
                        self.state[0, i, j] = CELL_MINE_HIT
                    else:
                        self.state[0, i, j] = self.board[i, j]
                else:
                    self.state[0, i, j] = CELL_UNREVEALED
        
        # Channel 1: Safety hints (number of adjacent mines for unrevealed cells, -1 for unknown)
        for i in range(self.current_board_height):
            for j in range(self.current_board_width):
                if self.revealed[i, j]:
                    self.state[1, i, j] = UNKNOWN_SAFETY  # Revealed cells don't need safety hints
                else:
                    # Count adjacent mines for unrevealed cells
                    adjacent_mines = 0
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if (0 <= ni < self.current_board_height and 
                                0 <= nj < self.current_board_width and 
                                self.mines[ni, nj]):
                                adjacent_mines += 1
                    self.state[1, i, j] = adjacent_mines
        
        # Channel 2: Revealed cell count (total number of revealed cells across the board)
        total_revealed = np.sum(self.revealed)
        self.state[2] = total_revealed
        
        # Channel 3: Game progress indicators (binary flags for important game states)
        for i in range(self.current_board_height):
            for j in range(self.current_board_width):
                # Set to 1 if this cell is a "safe bet" (adjacent to revealed cells with 0 mines)
                is_safe_bet = 0
                if not self.revealed[i, j]:  # Only for unrevealed cells
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if (0 <= ni < self.current_board_height and 
                                0 <= nj < self.current_board_width and 
                                self.revealed[ni, nj] and 
                                not self.mines[ni, nj] and 
                                self.board[ni, nj] == 0):  # Adjacent to revealed cell with 0 mines
                                is_safe_bet = 1
                                break
                        if is_safe_bet:
                            break
                self.state[3, i, j] = is_safe_bet

    def get_real_life_statistics(self):
        """Get real-life statistics (what would happen in actual Minesweeper gameplay).
        
        Returns:
            dict: Real-life statistics including games played, won, lost, and win rate
        """
        total_games = self.real_life_games_played
        if total_games == 0:
            return {
                'games_played': 0,
                'games_won': 0,
                'games_lost': 0,
                'win_rate': 0.0
            }
        
        return {
            'games_played': total_games,
            'games_won': self.real_life_games_won,
            'games_lost': self.real_life_games_lost,
            'win_rate': self.real_life_games_won / total_games
        }
    
    def get_rl_training_statistics(self):
        """Get RL training statistics (excluding pre-cascade games).
        
        Returns:
            dict: RL training statistics including games played, won, lost, and win rate
        """
        total_games = self.rl_games_played
        if total_games == 0:
            return {
                'games_played': 0,
                'games_won': 0,
                'games_lost': 0,
                'win_rate': 0.0
            }
        
        return {
            'games_played': total_games,
            'games_won': self.rl_games_won,
            'games_lost': self.rl_games_lost,
            'win_rate': self.rl_games_won / total_games
        }
    
    def get_combined_statistics(self):
        """Get both real-life and RL training statistics.
        
        Returns:
            dict: Combined statistics with both real-life and RL metrics
        """
        return {
            'real_life': self.get_real_life_statistics(),
            'rl_training': self.get_rl_training_statistics()
        }
    
    def _update_statistics(self, game_won, game_ended_pre_cascade):
        """Update both real-life and RL training statistics.
        
        Args:
            game_won (bool): Whether the game was won
            game_ended_pre_cascade (bool): Whether the game ended during pre-cascade period
        """
        # Always update real-life statistics
        self.real_life_games_played += 1
        if game_won:
            self.real_life_games_won += 1
        else:
            self.real_life_games_lost += 1
        
        # Only update RL training statistics if game didn't end pre-cascade
        if not game_ended_pre_cascade:
            self.rl_games_played += 1
            if game_won:
                self.rl_games_won += 1
            else:
                self.rl_games_lost += 1
        
        # Record move count for this game
        self._record_game_moves()

    def get_move_statistics(self):
        """Get statistics about moves made in the current game and across all games.
        
        Returns:
            dict: Dictionary containing move statistics
        """
        average_moves = self.total_moves_across_games / len(self.games_with_move_counts) if self.games_with_move_counts else 0
        min_moves = min(self.games_with_move_counts) if self.games_with_move_counts else 0
        max_moves = max(self.games_with_move_counts) if self.games_with_move_counts else 0
        return {
            'current_game_moves': self.move_count,
            'total_moves_across_games': self.total_moves_across_games,
            'games_with_move_counts': self.games_with_move_counts.copy(),
            'average_moves_per_game': average_moves,
            'min_moves_in_game': min_moves,
            'max_moves_in_game': max_moves,
            'repeated_action_count': self.repeated_action_count,
            'repeated_actions': list(self.repeated_actions),
            'revealed_cell_click_count': self.revealed_cell_click_count,
            'invalid_action_count': self.invalid_action_count
        }
    
    def get_board_statistics(self):
        """Get statistics about the current board configuration.
        
        Returns:
            dict: Dictionary containing board configuration statistics
        """
        # Get current mine positions
        mine_positions = []
        for i in range(self.current_board_height):
            for j in range(self.current_board_width):
                if self.mines[i, j]:
                    mine_positions.append((i, j))
        
        # Check if current configuration is learnable
        is_learnable = self._is_learnable_configuration(mine_positions)
        
        # Calculate board metrics
        total_cells = self.current_board_height * self.current_board_width
        mine_density = self.current_mines / total_cells
        
        return {
            'board_size': (self.current_board_height, self.current_board_width),
            'mines_placed': self.current_mines,
            'mine_positions': mine_positions,
            'learnable_configuration': is_learnable,
            'learnable_only_mode': self.learnable_only,
            'total_cells': total_cells,
            'mine_density': mine_density,
            'safe_cells': total_cells - self.current_mines,
            'safe_cell_ratio': (total_cells - self.current_mines) / total_cells
        }
    
    def _record_game_moves(self):
        """Record the move count for the current game when it ends."""
        if self.move_count > 0:  # Only record if moves were made
            self.games_with_move_counts.append(self.move_count)
            self.total_moves_across_games += self.move_count

def main():
    # Create and test the environment
    env = MinesweeperEnv(max_board_size=8, max_mines=12)
    state, _ = env.reset()
    
    # Take a random action
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    
    return state, reward, terminated, truncated, info

if __name__ == "__main__":
    main() 