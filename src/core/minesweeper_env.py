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
from .constants import (
    CELL_UNREVEALED,
    CELL_MINE,
    CELL_FLAGGED,
    CELL_MINE_HIT,
    REWARD_FIRST_MOVE_SAFE,
    REWARD_FIRST_MOVE_HIT_MINE,
    REWARD_SAFE_REVEAL,
    REWARD_WIN,
    REWARD_HIT_MINE,
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

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        self.revealed_count = 0
        self.won = False
        self.terminated = False
        self.is_first_move = True
        self.flags_remaining = self.current_mines
        
        # Initialize all arrays with consistent values
        self.state = np.full((self.current_board_height, self.current_board_width), CELL_UNREVEALED, dtype=np.int8)
        self.board = np.full((self.current_board_height, self.current_board_width), CELL_UNREVEALED, dtype=np.int8)
        self.mines = np.zeros((self.current_board_height, self.current_board_width), dtype=bool)
        self.flags = np.zeros((self.current_board_height, self.current_board_width), dtype=bool)
        self.revealed = np.zeros((self.current_board_height, self.current_board_width), dtype=bool)
        
        # Place mines and update adjacent counts
        self._place_mines()
        
        # Update action space to match current board size
        self.action_space = spaces.Discrete(self.current_board_width * self.current_board_height * 2)
        # Update observation space to match current board size
        self.observation_space = spaces.Box(
            low=CELL_MINE_HIT,  # Lowest cell state value
            high=8,  # Highest possible adjacent mine count
            shape=(self.current_board_height, self.current_board_width),
            dtype=np.int8
        )
        
        return self.state, {}

    def _place_mines(self):
        """Place mines on the board with minimum spacing."""
        self.mines = np.zeros((self.current_board_height, self.current_board_width), dtype=bool)
        self.board = np.full((self.current_board_height, self.current_board_width), CELL_UNREVEALED, dtype=np.int8)
        mines_placed = 0
        attempts = 0
        max_attempts = self.current_board_width * self.current_board_height * 10

        while mines_placed < self.current_mines and attempts < max_attempts:
            row = np.random.randint(0, self.current_board_height)
            col = np.random.randint(0, self.current_board_width)

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

    def _reveal_cell(self, x: int, y: int) -> None:
        """Reveal a cell and its neighbors if it's empty."""
        if not (0 <= x < self.current_board_width and 0 <= y < self.current_board_height):
            return
        if self.revealed[y, x] or self.flags[y, x]:
            return

        print(f"\nRevealing cell ({x}, {y})")
        print(f"Current board value: {self.board[y, x]}")
        
        # Use a stack for depth-first traversal
        stack = [(x, y)]
        visited = set()
        
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))
            
            # Reveal current cell
            self.revealed[cy, cx] = True
            self.revealed_count += 1
            # Set state to board value (0-8) for revealed cells
            self.state[cy, cx] = self.board[cy, cx]
            
            print(f"State after reveal: {self.state[cy, cx]}")
            
            # If this is an empty cell, add neighbors to stack
            if self.board[cy, cx] == 0:
                print(f"Empty cell detected at ({cx}, {cy}), adding neighbors to stack")
                # Add all neighbors to stack
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = cx + dx, cy + dy
                        if (0 <= nx < self.current_board_width and 
                            0 <= ny < self.current_board_height and
                            not self.revealed[ny, nx] and 
                            not self.flags[ny, nx]):
                            print(f"Adding neighbor to stack: ({nx}, {ny})")
                            stack.append((nx, ny))

    def _check_win(self) -> bool:
        """Check if the game is won.
        Win condition: All non-mine cells must be revealed.
        Flag placement is not required for winning."""
        # All non-mine cells must be revealed
        return np.all((self.state == CELL_UNREVEALED) == self.mines)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step in the environment.
        action: integer representing the action to take
        """
        # Reset info at the start of each step
        self.info = {}
        
        # Convert action to (x, y, action_type)
        action_type = action // (self.current_board_width * self.current_board_height)
        pos = action % (self.current_board_width * self.current_board_height)
        x = pos % self.current_board_width
        y = pos // self.current_board_width

        print(f"\nAction: {action}")
        print(f"Action type: {action_type}")
        print(f"Position: ({x}, {y})")

        # Handle invalid actions
        if not (0 <= x < self.current_board_width and 0 <= y < self.current_board_height):
            return self.state, REWARD_INVALID_ACTION, False, False, {"error": "Invalid position"}

        if self.revealed[y, x]:
            return self.state, REWARD_INVALID_ACTION, False, False, {"error": "Cell already revealed"}

        # Place mines on first move
        if self.is_first_move:
            self._place_mines()
            self.is_first_move = False

        if action_type == 0:  # Reveal
            if self.flags[y, x]:
                return self.state, REWARD_INVALID_ACTION, False, False, {"error": "Cannot reveal flagged cell"}
            
            if self.mines[y, x]:
                self.terminated = True
                self.revealed[y, x] = True
                self.state[y, x] = CELL_MINE_HIT  # Use constant for mine hit
                reward = REWARD_FIRST_MOVE_HIT_MINE if self.is_first_move else REWARD_HIT_MINE
                self.info['won'] = False
                return self.state, reward, True, False, self.info

            self._reveal_cell(x, y)
            
            # Check for win after reveal
            if self._check_win():
                self.won = True
                self.terminated = True
                self.info['won'] = True
                return self.state, REWARD_WIN, True, False, self.info
            
            # Set reward based on first move
            reward = REWARD_FIRST_MOVE_SAFE if self.is_first_move else REWARD_SAFE_REVEAL
            self.is_first_move = False
            return self.state, reward, False, False, {}

        elif action_type == 1:  # Flag
            if self.revealed[y, x]:
                return self.state, REWARD_INVALID_ACTION, False, False, {"error": "Cannot flag revealed cell"}
            
            if self.flags[y, x]:
                self.flags[y, x] = False
                self.flags_remaining += 1
                self.state[y, x] = CELL_UNREVEALED  # Use constant for unrevealed
                self.info['flags_remaining'] = self.flags_remaining
                # Check for win after unflag
                if self._check_win():
                    self.won = True
                    self.terminated = True
                    self.info['won'] = True
                    return self.state, REWARD_WIN, True, False, self.info
                return self.state, REWARD_UNFLAG, False, False, self.info
            
            if self.flags_remaining <= 0:
                return self.state, REWARD_INVALID_ACTION, False, False, {"error": "No flags remaining"}
            
            self.flags[y, x] = True
            self.flags_remaining -= 1
            self.state[y, x] = CELL_FLAGGED  # Use constant for flagged
            reward = REWARD_FLAG_MINE if self.mines[y, x] else REWARD_FLAG_SAFE
            self.info['flags_remaining'] = self.flags_remaining
            # Check for win after flag
            if self._check_win():
                self.won = True
                self.terminated = True
                self.info['won'] = True
                return self.state, REWARD_WIN, True, False, self.info
            return self.state, reward, False, False, self.info

        return self.state, REWARD_INVALID_ACTION, False, False, {"error": "Invalid action type"}

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