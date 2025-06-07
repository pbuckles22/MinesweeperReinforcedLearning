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
            raise ValueError("Board dimensions must be positive")
        if self.max_board_width > 100 or self.max_board_height > 100:
            raise ValueError("Board dimensions too large")
            
        # Validate mine count
        if max_mines <= 0:
            raise ValueError("Mine count must be positive")
        if max_mines > self.max_board_width * self.max_board_height:
            raise ValueError("Mine count cannot exceed board area")
            
        # Validate mine spacing
        if mine_spacing < 0:
            raise ValueError("Mine spacing must be non-negative")
        if mine_spacing >= self.max_board_size:
            raise ValueError("Mine spacing too large for board size")
            
        # Validate initial parameters
        if isinstance(initial_board_size, tuple):
            self.initial_board_width, self.initial_board_height = initial_board_size
        else:
            self.initial_board_width = self.initial_board_height = initial_board_size
            
        if self.initial_board_width > self.max_board_width or self.initial_board_height > self.max_board_height:
            raise ValueError("Initial board size cannot exceed max board size")
            
        if initial_mines > max_mines:
            raise ValueError("Initial mine count cannot exceed max mines")
        if initial_mines > self.initial_board_width * self.initial_board_height:
            raise ValueError("Initial mine count cannot exceed initial board area")
            
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
        self.initial_board_width = self.initial_board_width
        self.initial_board_height = self.initial_board_height
        self.initial_mines = initial_mines
        self.invalid_action_penalty = invalid_action_penalty
        self.mine_penalty = mine_penalty
        self.flag_mine_reward = flag_mine_reward
        self.flag_safe_penalty = flag_safe_penalty
        self.unflag_penalty = unflag_penalty
        self.safe_reveal_base = safe_reveal_base
        self.win_reward = win_reward
        
        # Initialize game state
        self.current_board_width = self.initial_board_width
        self.current_board_height = self.initial_board_height
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
        """Place mines on the board with minimum spacing, avoiding (first_y, first_x) if provided."""
        self.mines = np.zeros((self.current_board_height, self.current_board_width), dtype=bool)
        self.board = np.full((self.current_board_height, self.current_board_width), CELL_UNREVEALED, dtype=np.int8)
        mines_placed = 0
        attempts = 0
        max_attempts = self.current_board_width * self.current_board_height * 10

        while mines_placed < self.current_mines and attempts < max_attempts:
            row = np.random.randint(0, self.current_board_height)
            col = np.random.randint(0, self.current_board_width)

            # Avoid placing a mine at the first revealed cell
            if first_x is not None and first_y is not None:
                if row == first_y and col == first_x:
                    attempts += 1
                    continue

            # Check if position is valid (not already a mine and respects spacing)
            if not self.mines[row, col]:
                valid_position = True
                for dr in range(-self.mine_spacing, self.mine_spacing + 1):
                    for dc in range(-self.mine_spacing, self.mine_spacing + 1):
                        r, c = row + dr, col + dc
                        if (0 <= r < self.current_board_height and 
                            0 <= c < self.current_board_width and 
                            self.mines[r, c]):
                            valid_position = False
                            break
                    if not valid_position:
                        break

                if valid_position:
                    self.mines[row, col] = True
                    mines_placed += 1

            attempts += 1

        if mines_placed < self.current_mines:
            print(f"Warning: Could not place {self.current_mines} mines with spacing {self.mine_spacing} on a {self.current_board_width}x{self.current_board_height} board.")
            print(f"Placed {mines_placed} mines instead.")
            self.current_mines = mines_placed

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

    def _handle_mine_hit(self, x: int, y: int, first_move: bool) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Handle mine hit consistently across all cases."""
        if first_move:
            # On first move mine hit, reset the board
            state, _ = self.reset()
            return state, REWARD_FIRST_MOVE_HIT_MINE, False, False, {"won": False}
        else:
            self.revealed[y, x] = True
            self.state[y, x] = CELL_MINE_HIT
            self.terminated = True
            self.info['won'] = False
            return self.state, REWARD_HIT_MINE, True, False, self.info

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

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment."""
        if self.terminated or self.truncated:
            return self.state, REWARD_INVALID_ACTION, True, False, {"flags_remaining": self.flags_remaining, "won": False}

        # Calculate row and column from action
        is_flag = action >= (self.current_board_width * self.current_board_height)
        action = action % (self.current_board_width * self.current_board_height)
        row = action // self.current_board_width
        col = action % self.current_board_width

        # Check if this is a first move before any state changes
        is_first_move = not self.first_move_done and not is_flag

        # Handle first move
        if is_first_move:  # Only place mines on first reveal
            self._place_mines(first_x=col, first_y=row)
            self._update_adjacent_counts()
            self.first_move_done = True
            self.mines_placed = True

        # Handle flag placement/removal
        if is_flag:
            if self.state[row, col] == CELL_UNREVEALED and self.flags_remaining > 0:
                self.state[row, col] = CELL_FLAGGED
                self.flags_remaining -= 1
                reward = REWARD_FLAG_PLACED
            elif self.state[row, col] == CELL_FLAGGED:
                self.state[row, col] = CELL_UNREVEALED
                self.flags_remaining += 1
                reward = REWARD_FLAG_REMOVED
            else:
                reward = REWARD_INVALID_ACTION
        else:
            # Handle cell reveal
            if self.state[row, col] == CELL_UNREVEALED:
                if self.mines[row, col]:
                    # Handle mine hit
                    if is_first_move:
                        # Reset on first move mine hit
                        state, _ = self.reset()
                        return state, REWARD_FIRST_MOVE_HIT_MINE, False, False, {"won": False}
                    else:
                        # Normal mine hit
                        self.revealed[row, col] = True
                        self.state[row, col] = CELL_MINE_HIT
                        self.terminated = True
                        reward = REWARD_HIT_MINE
                else:
                    # Handle safe cell reveal
                    self._reveal_cell(row, col)
                    reward = REWARD_FIRST_MOVE_SAFE if is_first_move else REWARD_SAFE_REVEAL
                    # Check win condition after revealing a cell
                    if self._check_win():
                        self.terminated = True
                        reward = REWARD_WIN
            else:
                reward = REWARD_INVALID_ACTION

        return self.state, reward, self.terminated, False, {
            "flags_remaining": self.flags_remaining,
            "won": self.terminated and reward == REWARD_WIN
        }

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