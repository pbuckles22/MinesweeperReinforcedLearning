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
            low=CELL_MINE_HIT,  # Lowest cell state value
            high=8,  # Highest possible adjacent mine count
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
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        self.state = np.full((self.current_board_height, self.current_board_width), CELL_UNREVEALED, dtype=np.int8)
        self.mines = np.zeros((self.current_board_height, self.current_board_width), dtype=bool)
        self.board = np.zeros((self.current_board_height, self.current_board_width), dtype=np.int8)
        self.revealed = np.zeros((self.current_board_height, self.current_board_width), dtype=bool)
        self.flags = np.zeros((self.current_board_height, self.current_board_width), dtype=bool)
        self.first_move_done = False
        self.flags_remaining = self.initial_mines
        self.action_space = spaces.Discrete(self.current_board_width * self.current_board_height * 2)
        return self.state, {'flags_remaining': self.flags_remaining, 'won': False}

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
                            ni, nj = i + di, j + dj
                            if (0 <= ni < self.current_board_height and 
                                0 <= nj < self.current_board_width and 
                                not self.mines[ni, nj]):
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

    def _reveal_cell(self, x, y):
        print(f"\nStarting reveal at ({x}, {y})")
        if self.revealed[y, x] or self.flags[y, x]:
            print(f"Cell ({x}, {y}) already revealed or flagged, skipping.")
            return
        initial_value = self.board[y, x]
        print(f"Initial board value: {initial_value}")
        self.revealed[y, x] = True
        self.state[y, x] = self.board[y, x]
        if initial_value != 0:
            print(f"Initial cell ({x}, {y}) is not empty, stopping cascade")
            return
        print(f"Revealed initial cell ({x}, {y}) with value 0")
        print(f"Starting cascade from ({x}, {y})")
        queue = [(x, y)]
        visited = set([(x, y)])
        while queue:
            cx, cy = queue.pop(0)
            print(f"\nProcessing cell ({cx}, {cy})")
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < self.current_board_width and 0 <= ny < self.current_board_height:
                        if (nx, ny) in visited:
                            continue
                        if self.revealed[ny, nx] or self.flags[ny, nx]:
                            continue
                        self.revealed[ny, nx] = True
                        self.state[ny, nx] = self.board[ny, nx]
                        if self.board[ny, nx] == 0:
                            print(f"Adding empty neighbor ({nx}, {ny}) to queue")
                            queue.append((nx, ny))
                        visited.add((nx, ny))
        print("\nFinal state after cascade:")
        print(self.state)
        print("\nRevealed cells:")
        print(self.revealed)

    def _get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get all valid neighbors of a cell."""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.current_board_width and 
                    0 <= ny < self.current_board_height):
                    neighbors.append((nx, ny))
        return neighbors

    def _check_win(self) -> bool:
        """Check if the game is won.
        Win condition: All non-mine cells must be revealed.
        Flag placement is not required for winning."""
        # All non-mine cells must be revealed
        return np.all((self.state == CELL_UNREVEALED) == self.mines)

    def _check_win_condition(self):
        """Check if all non-mine cells have been revealed."""
        # Count unrevealed non-mine cells
        unrevealed_safe = np.sum((~self.revealed) & (~self.mines))
        if unrevealed_safe == 0:
            self.terminated = True
            self.info['won'] = True

    def step(self, action):
        """Execute one time step within the environment."""
        # Decode action
        action_type = action // (self.current_board_width * self.current_board_height)
        action = action % (self.current_board_width * self.current_board_height)
        row = action // self.current_board_width
        col = action % self.current_board_width

        # Check if this is the first move
        if not self.first_move_done and action_type == 0:  # Reveal action
            self._place_mines(row, col)
            self._update_adjacent_counts()
            self.first_move_done = True

        # Handle cell reveal
        if action_type == 0:  # Reveal action
            # Check if cell is already revealed or flagged
            if self.state[row, col] != CELL_UNREVEALED:
                return self.state, REWARD_INVALID_ACTION, False, False, {'flags_remaining': self.flags_remaining, 'won': False}

            # Check if cell is a mine
            if self.mines[row, col]:
                self.state[row, col] = CELL_MINE_HIT
                return self.state, REWARD_HIT_MINE, True, False, {'flags_remaining': self.flags_remaining, 'won': False}

            # Reveal cell and cascade if needed
            self._reveal_cell(row, col)

            # Check win condition
            if np.all((self.state == CELL_UNREVEALED) == self.mines):
                return self.state, REWARD_WIN, True, False, {'flags_remaining': self.flags_remaining, 'won': True}

            return self.state, REWARD_SAFE_REVEAL, False, False, {'flags_remaining': self.flags_remaining, 'won': False}

        # Handle flag placement/removal
        elif action_type == 1:  # Flag action
            # Check if cell is already revealed
            if self.state[row, col] != CELL_UNREVEALED and self.state[row, col] != CELL_FLAGGED:
                return self.state, REWARD_INVALID_ACTION, False, False, {'flags_remaining': self.flags_remaining, 'won': False}

            # Toggle flag
            if self.state[row, col] == CELL_FLAGGED:
                self.state[row, col] = CELL_UNREVEALED
                self.flags_remaining += 1
                return self.state, REWARD_FLAG_REMOVED, False, False, {'flags_remaining': self.flags_remaining, 'won': False}
            elif self.flags_remaining > 0:
                self.state[row, col] = CELL_FLAGGED
                self.flags_remaining -= 1
                reward = REWARD_FLAG_MINE if self.mines[row, col] else REWARD_FLAG_SAFE
                return self.state, reward, False, False, {'flags_remaining': self.flags_remaining, 'won': False}
            else:
                return self.state, REWARD_INVALID_ACTION, False, False, {'flags_remaining': self.flags_remaining, 'won': False}

        return self.state, REWARD_INVALID_ACTION, False, False, {'flags_remaining': self.flags_remaining, 'won': False}

    @property
    def action_masks(self):
        """Return a boolean mask indicating which actions are valid."""
        return self.get_action_masks()

    def get_action_masks(self):
        """Return a boolean mask indicating which actions are valid."""
        masks = np.ones(self.action_space.n, dtype=bool)
        
        # For each cell
        for i in range(self.current_board_height):
            for j in range(self.current_board_width):
                # Get action indices for reveal and flag
                reveal_action = i * self.current_board_width + j
                flag_action = (self.current_board_width * self.current_board_height) + reveal_action
                
                # If cell is already revealed, mask both reveal and flag actions
                if self.state[i, j] != CELL_UNREVEALED:
                    masks[reveal_action] = False
                    masks[flag_action] = False
                # If cell is flagged, mask reveal action
                elif self.state[i, j] == CELL_FLAGGED:
                    masks[reveal_action] = False
                    # If no flags remaining, mask flag action
                    if self.flags_remaining <= 0:
                        masks[flag_action] = False
                # If no flags remaining, mask flag action
                elif self.flags_remaining <= 0:
                    masks[flag_action] = False
        
        # If game is over, mask all actions
        if self.terminated:
            masks.fill(False)
            
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