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
from typing import Tuple, Dict, Optional, List, Set
from src.core.constants import (
    CELL_UNREVEALED,
    CELL_MINE,
    CELL_FLAGGED,
    CELL_MINE_HIT,
    REWARD_FIRST_MOVE_SAFE,
    REWARD_FIRST_MOVE_HIT_MINE,
    REWARD_SAFE_REVEAL,
    REWARD_WIN,
    REWARD_HIT_MINE,
    REWARD_FLAG_PLACED,
    REWARD_FLAG_REMOVED,
    REWARD_FLAG_MINE,
    REWARD_FLAG_SAFE,
    REWARD_UNFLAG,
    REWARD_INVALID_ACTION,
    DIFFICULTY_LEVELS
)

class MinesweeperEnv(gym.Env):
    """
    A Minesweeper environment for reinforcement learning with flagging, realistic win condition, and fixed observation/action space for curriculum learning.
    Supports multiple difficulty levels from easy to chaotic.
    """
    def __init__(self, max_board_size=(20, 35), max_mines=130, render_mode=None,
                 early_learning_mode=False, early_learning_threshold=200,
                 early_learning_corner_safe=True, early_learning_edge_safe=True,
                 mine_spacing=1, initial_board_size=4, initial_mines=2,
                 invalid_action_penalty=REWARD_INVALID_ACTION, mine_penalty=REWARD_HIT_MINE,
                 flag_mine_reward=REWARD_FLAG_MINE, flag_safe_penalty=REWARD_FLAG_SAFE,
                 unflag_penalty=REWARD_UNFLAG, safe_reveal_base=REWARD_SAFE_REVEAL, win_reward=REWARD_WIN):
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
        self.reward_flag_mine = flag_mine_reward
        self.reward_flag_safe = flag_safe_penalty
        self.reward_unflag = unflag_penalty
        self.reward_safe_reveal = safe_reveal_base
        self.reward_win = win_reward
        
        # Initialize game state
        self.current_board_width = initial_width
        self.current_board_height = initial_height
        self.current_mines = initial_mines
        self.state = None
        self.board = None
        self.mines = None
        self.flags = None
        self.revealed = None
        self.flags_remaining = initial_mines
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
        self.action_space = spaces.Discrete(self.current_board_width * self.current_board_height * 2)
        self.observation_space = spaces.Box(
            low=-4,  # CELL_MINE_HIT
            high=8,  # Maximum number of adjacent mines
            shape=(self.current_board_height, self.current_board_width),
            dtype=np.int8
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

    def reset(self, seed=None):
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Initialize or update action space based on current board size
        self.action_space = spaces.Discrete(self.current_board_width * self.current_board_height * 2)
        
        # Initialize state space
        self.observation_space = spaces.Box(
            low=-4,  # CELL_MINE_HIT
            high=8,  # Maximum number of adjacent mines
            shape=(self.current_board_height, self.current_board_width),
            dtype=np.int8
        )
        
        # Initialize board state
        self.state = np.full((self.current_board_height, self.current_board_width), CELL_UNREVEALED, dtype=np.int8)
        self.board = np.zeros((self.current_board_height, self.current_board_width), dtype=np.int8)
        self.mines = np.zeros((self.current_board_height, self.current_board_width), dtype=bool)
        self.revealed = np.zeros((self.current_board_height, self.current_board_width), dtype=bool)
        self.flags = np.zeros((self.current_board_height, self.current_board_width), dtype=bool)
        
        # Reset game state
        self.terminated = False
        self.truncated = False
        self.first_move_done = False
        self.mines_placed = False
        self.flags_remaining = self.initial_mines
        
        # Initialize info dict
        self.info = {
            "flags_remaining": self.flags_remaining,
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
        """Handle a mine hit.
        Args:
            col: Column coordinate
            row: Row coordinate
            is_first_move: Whether this is the first move
        Returns:
            Tuple of (reward, terminated)
        """
        print(f"\nHandling mine hit at ({row}, {col})")
        print(f"is_first_move: {is_first_move}")
        
        if is_first_move:
            print("First move mine hit - resetting board")
            # Reset the board if the first move is a mine
            self._place_mines(col, row)
            self._update_adjacent_counts()
            return REWARD_FIRST_MOVE_HIT_MINE, False
        else:
            print("Subsequent move mine hit - ending game")
            # Update state to indicate mine hit
            self.state[row, col] = CELL_MINE_HIT
            self.revealed[row, col] = True
            return REWARD_HIT_MINE, True

    def _reveal_cell(self, row: int, col: int) -> None:
        """Reveal a cell and cascade through empty cells."""
        if self.revealed[row, col] or self.state[row, col] == CELL_FLAGGED:
            return

        print(f"\nStarting reveal at ({row}, {col})")
        print(f"Initial board value: {self.board[row, col]}")
        
        # Initialize queue with the starting cell
        queue = [(row, col)]
        visited = {(row, col)}

        while queue:
            r, c = queue.pop(0)
            print(f"\nProcessing cell ({r}, {c})")
            
            # Skip if already revealed or flagged
            if self.revealed[r, c] or self.state[r, c] == CELL_FLAGGED:
                continue
                
            # Reveal the cell
            self.revealed[r, c] = True
            self.state[r, c] = self.board[r, c]
            print(f"Revealed cell ({r}, {c}) with value {self.board[r, c]}")
            
            # If the cell is empty (value == 0), cascade to neighbors
            if self.board[r, c] == 0:
                # Get all neighbors
                for nr, nc in self._get_neighbors(r, c):
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
                        print(f"Adding neighbor ({nr}, {nc}) to queue")

        print("\nFinal state after cascade:")
        print(self.state)
        print("\nRevealed cells:")
        print(self.revealed)

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
        Flag placement is not required for winning."""
        # Game is won if all non-mine cells are revealed
        # A cell is "done" if it's either:
        # 1. Revealed (self.revealed)
        # 2. A mine (self.mines)
        # 3. Flagged (self.state == CELL_FLAGGED)
        return np.all(self.revealed | self.mines | (self.state == CELL_FLAGGED))

    def step(self, action):
        """Take a step in the environment."""
        # Convert action to (x, y) coordinates and action type
        action_type = action // (self.current_board_width * self.current_board_height)
        action_index = action % (self.current_board_width * self.current_board_height)
        col = action_index % self.current_board_width
        row = action_index // self.current_board_width

        print(f"\nStep: action={action}, type={action_type}, pos=({row}, {col})")
        print(f"is_first_move: {self.is_first_move}")

        # Check if action is valid
        if not self._is_valid_action(action):
            return self.state, REWARD_INVALID_ACTION, False, False, {}

        # Handle flag placement/removal
        if action_type == 1:  # Flag action
            if self.flags[row, col]:
                # Remove flag
                self.flags[row, col] = False
                self.flags_remaining += 1
                return self.state, REWARD_UNFLAG, False, False, {'flags_remaining': self.flags_remaining}
            else:
                # Place flag
                if self.flags_remaining > 0:
                    self.flags[row, col] = True
                    self.flags_remaining -= 1
                    reward = REWARD_FLAG_MINE if self.mines[row, col] else REWARD_FLAG_SAFE
                    return self.state, reward, False, False, {'flags_remaining': self.flags_remaining}
                else:
                    return self.state, REWARD_INVALID_ACTION, False, False, {'flags_remaining': self.flags_remaining}

        # Handle cell reveal
        if self.flags[row, col]:
            return self.state, REWARD_INVALID_ACTION, False, False, {'flags_remaining': self.flags_remaining}

        # Check if it's a mine
        if self.mines[row, col]:
            print("Mine hit detected")
            reward, terminated = self.handle_mine_hit(col, row, self.is_first_move)
            print(f"After handle_mine_hit: reward={reward}, terminated={terminated}")
            return self.state, reward, terminated, False, {'flags_remaining': self.flags_remaining, 'won': False}

        # Reveal the cell
        self._reveal_cell(col, row)

        # Check for win
        if self._check_win():
            self.is_first_move = False
            return self.state, REWARD_WIN, True, False, {'flags_remaining': self.flags_remaining, 'won': True}

        # Determine reward based on whether this is the first move
        reward = REWARD_FIRST_MOVE_SAFE if self.is_first_move else REWARD_SAFE_REVEAL
        self.is_first_move = False
        return self.state, reward, False, False, {'flags_remaining': self.flags_remaining, 'won': False}

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
                if self.state[i, j] != CELL_UNREVEALED:
                    masks[reveal_idx] = False
                
                # Flag action
                flag_idx = (self.current_board_width * self.current_board_height) + reveal_idx
                if self.state[i, j] != CELL_UNREVEALED and self.state[i, j] != CELL_FLAGGED:
                    masks[flag_idx] = False
                if self.state[i, j] == CELL_FLAGGED and self.flags_remaining == 0:
                    masks[flag_idx] = False
                if self.flags_remaining == 0 and self.state[i, j] == CELL_UNREVEALED:
                    masks[flag_idx] = False
                    
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
                
                if self.flags[y, x]:
                    pygame.draw.rect(self.screen, (255, 0, 0), rect)  # Red for flags
                elif not self.revealed[y, x]:
                    pygame.draw.rect(self.screen, (128, 128, 128), rect)  # Gray for unrevealed
                else:
                    if self.mines[y, x]:
                        pygame.draw.rect(self.screen, (0, 0, 0), rect)  # Black for mines
                    else:
                        pygame.draw.rect(self.screen, (255, 255, 255), rect)  # White for revealed
                        if self.board[y, x] > 0:
                            font = pygame.font.Font(None, 36)
                            text = font.render(str(self.board[y, x]), True, (0, 0, 0))
                            text_rect = text.get_rect(center=rect.center)
                            self.screen.blit(text, text_rect)

        pygame.display.flip()
        self.clock.tick(30)

    def _is_valid_action(self, action):
        """Check if an action is valid."""
        # Check if action is within bounds
        if action < 0 or action >= self.action_space.n:
            return False

        # Convert action to (x, y) coordinates and action type
        action_type = action // (self.current_board_width * self.current_board_height)
        action_index = action % (self.current_board_width * self.current_board_height)
        col = action_index % self.current_board_width
        row = action_index // self.current_board_width

        # Check if coordinates are valid
        if not (0 <= row < self.current_board_height and 0 <= col < self.current_board_width):
            return False

        # Check if cell is already revealed
        if self.state[row, col] != CELL_UNREVEALED:
            return False

        # Check if cell is flagged
        if self.flags[row, col]:
            return False

        return True

def main():
    # Create and test the environment
    env = MinesweeperEnv(max_board_size=8, max_mines=12)
    state, _ = env.reset()
    
    print("Initial state:")
    env.render()
    
    # Take a random action
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    
    print("\nAfter action:", action)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    env.render()

if __name__ == "__main__":
    main() 