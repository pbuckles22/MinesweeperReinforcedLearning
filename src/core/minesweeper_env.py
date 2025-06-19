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
from src.core.constants import (
    CELL_UNREVEALED,
    CELL_MINE,
    CELL_MINE_HIT,
    UNKNOWN_SAFETY,
    REWARD_FIRST_MOVE_SAFE,
    REWARD_FIRST_MOVE_HIT_MINE,
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
    def __init__(self, max_board_size=(20, 35), max_mines=130, render_mode=None,
                 early_learning_mode=False, early_learning_threshold=200,
                 early_learning_corner_safe=True, early_learning_edge_safe=True,
                 mine_spacing=1, initial_board_size=4, initial_mines=2,
                 invalid_action_penalty=REWARD_INVALID_ACTION, mine_penalty=REWARD_HIT_MINE,
                 safe_reveal_base=REWARD_SAFE_REVEAL, win_reward=REWARD_WIN):
        """Initialize the Minesweeper environment."""
        super().__init__()
        
        # Initialize progress tracking variables
        self.last_progress_update = time.time()
        self.progress_interval = 1.0  # Update every second
        self.win_count = 0
        self.total_games = 0
        self.recent_rewards = deque(maxlen=100)
        self.recent_episode_lengths = deque(maxlen=100)
        self.last_win_rate = 0
        self.last_avg_reward = 0
        self.last_avg_length = 0
        self.games_at_current_size = 0
        
        # Initialize training health variables
        self.min_win_rate = 0.1  # Minimum expected win rate
        self.consecutive_mine_hits = 0
        self.max_consecutive_mine_hits = 5  # Maximum allowed consecutive mine hits
        
        # Handle tuple board sizes
        if isinstance(max_board_size, tuple):
            self.max_board_width, self.max_board_height = max_board_size
            self.max_board_size = max(max_board_size)
        else:
            self.max_board_width = self.max_board_height = max_board_size
            self.max_board_size = max_board_size
            
        # Validate board size
        if self.max_board_width <= 0 or self.max_board_height <= 0:
            raise ValueError("Board size must be positive")
        if self.max_board_width > 100 or self.max_board_height > 100:
            raise ValueError("Board dimensions too large")
            
        # Validate mine count
        if max_mines <= 0:
            raise ValueError("Mine count must be positive")
        if max_mines > self.max_board_width * self.max_board_height:
            raise ValueError("Mine count cannot exceed board size squared")
            
        # Validate initial board size
        if isinstance(initial_board_size, tuple):
            initial_width, initial_height = initial_board_size
        else:
            initial_width = initial_height = initial_board_size
            
        if initial_width > self.max_board_width or initial_height > self.max_board_height:
            raise ValueError("Initial board size cannot exceed max board size")
            
        # Validate reward parameters
        if mine_penalty >= 0:
            raise ValueError("Mine penalty must be negative")
            
        # Store parameters
        self.max_board_width = self.max_board_width
        self.max_board_height = self.max_board_height
        self.max_mines = max_mines
        self.render_mode = render_mode
        self.early_learning_mode = early_learning_mode
        self.early_learning_threshold = early_learning_threshold
        self.early_learning_corner_safe = early_learning_corner_safe
        self.early_learning_edge_safe = early_learning_edge_safe
        self.mine_spacing = mine_spacing
        self.initial_board_width = initial_width
        self.initial_board_height = initial_height
        self.initial_mines = initial_mines
        self.reward_invalid_action = invalid_action_penalty
        self.reward_hit_mine = mine_penalty
        self.reward_safe_reveal = safe_reveal_base
        self.reward_win = win_reward
        
        # Initialize game state
        self.current_board_width = initial_width
        self.current_board_height = initial_height
        self.current_mines = initial_mines
        self.state = None
        self.board = None
        self.mines = None
        self.revealed = None
        self.revealed_count = 0
        self.won = False
        self.terminated = False
        self.truncated = False
        self.is_first_move = True
        self.mines_placed = False
        self.first_move_done = False
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.current_board_width * self.current_board_height)
        
        # Initialize enhanced state space with 2 channels
        low_bounds = np.full((2, self.current_board_height, self.current_board_width), -1, dtype=np.float32)
        low_bounds[0] = -4  # Channel 0: game state can go as low as -4 (mine hit)
        low_bounds[1] = -1  # Channel 1: safety hints can go as low as -1 (unknown)
        
        high_bounds = np.full((2, self.current_board_height, self.current_board_width), 8, dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=low_bounds,
            high=high_bounds,
            shape=(2, self.current_board_height, self.current_board_width),
            dtype=np.float32
        )
        
        # Initialize info dictionary
        self.info = {}
        
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

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Ensure deterministic numpy RNG if seed is provided
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize or update action space based on current board size
        self.action_space = spaces.Discrete(self.current_board_width * self.current_board_height)
        
        # Initialize enhanced state space with 2 channels
        low_bounds = np.full((2, self.current_board_height, self.current_board_width), -1, dtype=np.float32)
        low_bounds[0] = -4  # Channel 0: game state can go as low as -4 (mine hit)
        low_bounds[1] = -1  # Channel 1: safety hints can go as low as -1 (unknown)
        
        high_bounds = np.full((2, self.current_board_height, self.current_board_width), 8, dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=low_bounds,
            high=high_bounds,
            shape=(2, self.current_board_height, self.current_board_width),
            dtype=np.float32
        )
        
        # Initialize board state
        self.board = np.zeros((self.current_board_height, self.current_board_width), dtype=np.int8)
        self.mines = np.zeros((self.current_board_height, self.current_board_width), dtype=bool)
        self.revealed = np.zeros((self.current_board_height, self.current_board_width), dtype=bool)
        
        # Initialize enhanced state with 2 channels
        self.state = np.zeros((2, self.current_board_height, self.current_board_width), dtype=np.float32)
        
        # Channel 0: Game state (all unrevealed initially)
        self.state[0] = CELL_UNREVEALED
        
        # Channel 1: Safety hints (unknown initially)
        self.state[1] = UNKNOWN_SAFETY
        
        # Reset game state
        self.terminated = False
        self.truncated = False
        self.is_first_move = True
        self.first_move_done = False
        self.mines_placed = False
        
        # Place mines unconditionally
        self._place_mines()
        
        # Update enhanced state after mine placement
        self._update_enhanced_state()
        
        # Initialize info dict
        self.info = {
            "won": False
        }
        
        return self.state, self.info

    def _place_mines(self, first_x=None, first_y=None):
        """Place mines on the board, avoiding the first revealed cell."""
        # Create list of valid positions
        valid_positions = []
        for y in range(self.current_board_height):
            for x in range(self.current_board_width):
                # Skip first revealed cell
                if x == first_x and y == first_y:
                    continue
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

    def handle_mine_hit(self, col, row, is_first_move):
        """Handle a mine hit."""
        if is_first_move:
            self._place_mines(col, row)
            return 0, False
        else:
            self.state[row, col] = CELL_MINE_HIT  # Set to -4 for mine hit
            return REWARD_HIT_MINE, True

    def _reveal_cell(self, row: int, col: int) -> None:
        """Reveal a cell and its neighbors if it's empty."""
        if not (0 <= row < self.current_board_height and 0 <= col < self.current_board_width):
            return
        if self.revealed[row, col]:
            return

        self.revealed[row, col] = True
        self.state[0, row, col] = self._get_cell_value(row, col)

        if self._get_cell_value(row, col) == 0:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    self._reveal_cell(row + dr, col + dc)

    def _relocate_mine_from_position(self, row: int, col: int) -> None:
        """Relocate a mine from the given position to a safe location."""
        # Remove mine from current position
        self.mines[row, col] = False
        
        # Find a safe location for the mine (not the first move position)
        safe_positions = []
        for i in range(self.current_board_height):
            for j in range(self.current_board_width):
                if not self.mines[i, j] and (i != row or j != col):
                    safe_positions.append((i, j))
        
        if safe_positions:
            # Choose a random safe position
            import random
            new_row, new_col = random.choice(safe_positions)
            self.mines[new_row, new_col] = True
        else:
            # Fallback: if no safe positions, just remove the mine
            pass
        
        # Update adjacent counts after mine relocation
        self._update_adjacent_counts()

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

        # If game is over, all actions are invalid and return negative reward
        if self.terminated or self.truncated:
            return self.state, REWARD_INVALID_ACTION, True, False, info

        # Check if action is within bounds first
        if action < 0 or action >= self.action_space.n:
            return self.state, REWARD_INVALID_ACTION, False, False, info

        # Check if action is valid using action masks
        if not self.action_masks[action]:
            # Check if ALL actions are invalid - if so, terminate the game
            if not np.any(self.action_masks):
                self.terminated = True
                info['won'] = False
                return self.state, REWARD_INVALID_ACTION, True, False, info
            return self.state, REWARD_INVALID_ACTION, False, False, info

        # Convert action to (x, y) coordinates
        col = action % self.current_board_width
        row = action // self.current_board_width

        # Handle cell reveal
        if self.mines[row, col]:  # Hit a mine
            if self.is_first_move:
                # First move safety: relocate the mine and reveal the intended cell
                self._relocate_mine_from_position(row, col)
                # Now reveal the cell (which should be safe)
                self._reveal_cell(row, col)
                # Update enhanced state after revealing cells
                self._update_enhanced_state()
                # Check for win after all reveals (including cascades)
                if self._check_win():
                    self.is_first_move = False
                    self.terminated = True
                    info['won'] = True
                    return self.state, REWARD_WIN, True, False, info
                # Return first move safe reward since we relocated the mine
                reward = REWARD_FIRST_MOVE_SAFE
                self.is_first_move = False
                info['won'] = False
                return self.state, reward, False, False, info
            else:
                # Game over - hit a mine (not first move)
                self.state[0, row, col] = CELL_MINE_HIT
                self.revealed[row, col] = True
                self.terminated = True
                info['won'] = False
                return self.state, REWARD_HIT_MINE, True, False, info

        # Reveal the cell (safe cell)
        self._reveal_cell(row, col)

        # Update enhanced state after revealing cells
        self._update_enhanced_state()

        # Always check for win after all reveals (including cascades)
        if self._check_win():
            self.is_first_move = False
            self.terminated = True
            info['won'] = True
            return self.state, REWARD_WIN, True, False, info

        # Determine reward based on whether this is the first move
        reward = REWARD_FIRST_MOVE_SAFE if self.is_first_move else REWARD_SAFE_REVEAL
        self.is_first_move = False
        info['won'] = False
        return self.state, reward, False, False, info

    @property
    def action_masks(self):
        """Return a boolean mask indicating which actions are valid."""
        # If game is over, all actions are invalid
        if self.terminated or self.truncated:
            return np.zeros(self.action_space.n, dtype=bool)
        
        masks = np.ones(self.action_space.n, dtype=bool)
        for i in range(self.current_board_height):
            for j in range(self.current_board_width):
                # Reveal action
                reveal_idx = i * self.current_board_width + j
                if self.revealed[i, j]:  # Can't reveal revealed cells
                    masks[reveal_idx] = False
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
        """Update the enhanced state representation."""
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