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
    REWARD_FIRST_MOVE_SAFE,
    REWARD_SAFE_REVEAL,
    REWARD_WIN,
    REWARD_HIT_MINE,
    REWARD_FIRST_MOVE_HIT_MINE,
    CELL_UNREVEALED,
    CELL_MINE,
    CELL_FLAGGED,
    CELL_MINE_HIT
)

# Difficulty level constants
DIFFICULTY_LEVELS = {
    'easy': {'size': 9, 'mines': 10},
    'normal': {'size': 16, 'mines': 40},
    'hard': {'size': (16, 30), 'mines': 99},
    'expert': {'size': (18, 24), 'mines': 115},
    'chaotic': {'size': (20, 35), 'mines': 130}
}

class MinesweeperEnv(gym.Env):
    """
    A Minesweeper environment for reinforcement learning with flagging, realistic win condition, and fixed observation/action space for curriculum learning.
    Supports multiple difficulty levels from easy to chaotic.
    """
    def __init__(self, max_board_size=(20, 35), max_mines=130, render_mode=None,
                 early_learning_mode=False, early_learning_threshold=200,
                 early_learning_corner_safe=True, early_learning_edge_safe=True,
                 mine_spacing=1, initial_board_size=4, initial_mines=2,
                 invalid_action_penalty=-0.1, mine_penalty=-10.0,
                 flag_mine_reward=5.0, flag_safe_penalty=-1.0,
                 unflag_penalty=-0.1, safe_reveal_base=5.0, win_reward=100.0):
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
            raise ValueError("Initial mine count cannot exceed initial board size squared")
            
        # Validate reward parameters
        if mine_penalty >= 0:
            raise ValueError("Mine penalty must be negative")
        if flag_safe_penalty >= 0:
            raise ValueError("Flag safe penalty must be negative")
        if unflag_penalty >= 0:
            raise ValueError("Unflag penalty must be negative")
        
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
        
        # Handle initial board size
        if isinstance(initial_board_size, tuple):
            self.initial_board_width, self.initial_board_height = initial_board_size
        else:
            self.initial_board_width = self.initial_board_height = initial_board_size
            
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
        self.revealed_count = 0
        self.won = False
        self.terminated = False  # Track if the game is over
        self.is_first_move = True  # Track if this is the first move
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Define action and observation spaces
        # Action space: [0, width*height-1] for reveal, [width*height, 2*width*height-1] for flag
        self.action_space = spaces.Discrete(self.current_board_width * self.current_board_height * 2)
        self.observation_space = spaces.Box(
            low=-2,
            high=8,
            shape=(self.current_board_height, self.current_board_width),
            dtype=np.int8
        )
        
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
        self.terminated = False  # Reset game over state
        self.is_first_move = True  # Reset first move flag
        self.state = np.full((self.current_board_height, self.current_board_width), CELL_UNREVEALED, dtype=int)
        self.board = np.zeros((self.current_board_height, self.current_board_width), dtype=int)
        self.mines = np.zeros((self.current_board_height, self.current_board_width), dtype=bool)
        self.flags = np.zeros((self.current_board_height, self.current_board_width), dtype=bool)
        self._place_mines()
        self._update_adjacent_counts()
        # Ensure state is all -1 after all operations
        self.state = np.full((self.current_board_height, self.current_board_width), CELL_UNREVEALED, dtype=int)
        
        # Update action space to match current board size
        self.action_space = spaces.Discrete(self.current_board_width * self.current_board_height * 2)
        # Update observation space to match current board size
        self.observation_space = spaces.Box(
            low=-2,
            high=8,
            shape=(self.current_board_height, self.current_board_width),
            dtype=np.int8
        )
        
        return self.state, {}

    def _place_mines(self):
        """Place mines on the board with minimum spacing."""
        self.mines = np.zeros((self.current_board_height, self.current_board_width), dtype=bool)
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
                    self.board[row, col] = CELL_MINE  # Mark mine
                    mines_placed += 1

            attempts += 1

        if mines_placed < self.current_mines:
            print(f"Warning: Could not place {self.current_mines} mines with spacing {self.mine_spacing} on a {self.current_board_width}x{self.current_board_height} board.")
            print(f"Placed {mines_placed} mines instead.")
            self.current_mines = mines_placed

        self._update_adjacent_counts()

    def _update_adjacent_counts(self):
        """Update the adjacent mine counts for each cell."""
        for y in range(self.current_board_height):
            for x in range(self.current_board_width):
                if not self.mines[y, x]:
                    count = 0
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = y + dy, x + dx
                            if (0 <= ny < self.current_board_height and 
                                0 <= nx < self.current_board_width and 
                                self.mines[ny, nx]):
                                count += 1
                    self.board[y, x] = count

    def _reveal_cell(self, x: int, y: int) -> None:
        """Reveal a cell and handle cascading."""
        if self.state[y, x] != CELL_UNREVEALED or self.flags[y, x]:
            return

        if self.mines[y, x]:
            self.state[y, x] = CELL_MINE_HIT  # Use -2 for hit mine
            self.terminated = True
            self.won = False
            return

        # Reveal the cell
        self.state[y, x] = self.board[y, x]
        self.revealed_count += 1  # Increment revealed count

        # If it's a safe cell (0), cascade to adjacent cells
        if self.board[y, x] == 0:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    new_x, new_y = x + dx, y + dy
                    if 0 <= new_x < self.current_board_width and 0 <= new_y < self.current_board_height:
                        if self.state[new_y, new_x] == CELL_UNREVEALED and not self.flags[new_y, new_x]:
                            self._reveal_cell(new_x, new_y)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one time step within the environment."""
        if self.terminated:
            return self.state, 0, True, False, {"won": self.won}
            
        # Check if action is within bounds
        if action < 0 or action >= self.current_board_width * self.current_board_height * 2:
            return self.state, self.invalid_action_penalty, False, False, {}
            
        # Convert action to coordinates and type (reveal or flag)
        is_flag_action = action >= self.current_board_width * self.current_board_height
        if is_flag_action:
            action = action - self.current_board_width * self.current_board_height
            
        # Convert 1D action to 2D coordinates
        y = action // self.current_board_width
        x = action % self.current_board_width
        
        # Debug logging
        self.logger.debug(f"Action: {action}, Is Flag: {is_flag_action}, Coords: ({y}, {x})")
        self.logger.debug(f"Board: {self.current_board_width}x{self.current_board_height}")
        self.logger.debug(f"Mines: {self.mines[y, x]}, State: {self.state[y, x]}")
        
        # Check if the action is valid
        if not (0 <= y < self.current_board_height and 0 <= x < self.current_board_width):
            return self.state, self.invalid_action_penalty, False, False, {}
            
        # Handle flag actions
        if is_flag_action:
            # Check if cell is already revealed
            if self.state[y, x] != CELL_UNREVEALED:
                return self.state, self.invalid_action_penalty, False, False, {}

            # Toggle flag
            self.flags[y, x] = not self.flags[y, x]
            # Keep state as CELL_UNREVEALED for flagged cells
            self.state[y, x] = CELL_UNREVEALED
            
            if self.flags[y, x]:  # Placing flag
                if self.mines[y, x]:
                    reward = self.flag_mine_reward
                else:
                    reward = self.flag_safe_penalty
            else:  # Removing flag
                reward = self.unflag_penalty

            # Check for win after flag action
            if self._check_win():
                reward = self.win_reward
                self.terminated = True
                self.won = True
                return self.state, reward, True, False, {"won": self.won}
        else:
            # Handle reveal actions
            if self.flags[y, x]:
                return self.state, self.unflag_penalty, False, False, {}
                
            if self.state[y, x] != CELL_UNREVEALED:
                return self.state, self.invalid_action_penalty, False, False, {}
                
            if self.mines[y, x]:
                # Update the state array to show the mine hit
                self.state = np.copy(self.state)  # Create a copy to ensure the array is updated
                self.state[y, x] = CELL_MINE_HIT
                reward = REWARD_FIRST_MOVE_HIT_MINE if self.is_first_move else REWARD_HIT_MINE
                self.terminated = True
                self.won = False
                self.is_first_move = False
                return self.state, reward, True, False, {"won": False}
            else:
                self._reveal_cell(x, y)
                if self.is_first_move:
                    reward = REWARD_FIRST_MOVE_SAFE
                else:
                    reward = REWARD_SAFE_REVEAL
                    
                # Check for win
                if self._check_win():
                    reward = self.win_reward
                    self.terminated = True
                    self.won = True
                    
        self.is_first_move = False
        
        # Update progress display if in human render mode
        if self.render_mode == "human":
            self._update_progress_display()
            
        return self.state, reward, self.terminated, False, {"won": self.won}

    def _check_win(self) -> bool:
        """Check if the game is won."""
        # Count unrevealed safe cells and incorrectly flagged cells
        unrevealed_safe_cells = 0
        incorrect_flags = 0
        
        for y in range(self.current_board_height):
            for x in range(self.current_board_width):
                if not self.mines[y, x]:  # Safe cell
                    if self.state[y, x] == CELL_UNREVEALED:
                        unrevealed_safe_cells += 1
                else:  # Mine cell
                    if not self.flags[y, x]:
                        incorrect_flags += 1
        
        # Win if all safe cells are revealed and all mines are flagged
        return unrevealed_safe_cells == 0 and incorrect_flags == 0

    def _update_progress_display(self):
        """Update the progress display with current training stats."""
        current_time = time.time()
        if current_time - self.last_progress_update >= self.progress_interval:
            # Calculate current stats
            win_rate = self.win_count / self.total_games if self.total_games > 0 else 0
            avg_reward = sum(self.recent_rewards) / len(self.recent_rewards) if self.recent_rewards else 0
            avg_length = sum(self.recent_episode_lengths) / len(self.recent_episode_lengths) if self.recent_episode_lengths else 0
            
            # Only update if there are significant changes
            if (abs(win_rate - self.last_win_rate) > 0.05 or  # Increased threshold from 0.01 to 0.05
                abs(avg_reward - self.last_avg_reward) > 0.5 or  # Increased threshold from 0.1 to 0.5
                abs(avg_length - self.last_avg_length) > 5):  # Increased threshold from 1 to 5
                
                # Clear previous progress display
                print("\033[2J\033[H", end="")  # Clear screen and move cursor to top
                
                # Add some padding at the top
                print("\n" * 2)
                
                self.logger.info("\n" + "="*50)  # Reduced from 70 to 50
                self.logger.info(f"Training Progress at {datetime.now().strftime('%H:%M:%S')}")
                self.logger.info("="*50)
                
                # Game Configuration
                self.logger.info(f"Board: {self.current_board_width}x{self.current_board_height} | Mines: {self.current_mines}")
                
                # Performance Metrics
                self.logger.info(f"Win Rate: {win_rate:.1%} | Avg Reward: {avg_reward:.1f} | Avg Length: {avg_length:.0f}")
                
                # Game Statistics
                self.logger.info(f"Games: {self.total_games} | Wins: {self.win_count} | Current Size Games: {self.games_at_current_size}")
                
                # Add some padding at the bottom
                self.logger.info("="*50 + "\n")
                
                self.last_win_rate = win_rate
                self.last_avg_reward = avg_reward
                self.last_avg_length = avg_length
            
            self.last_progress_update = current_time

    def _check_training_health(self):
        """Check if training is going well and log warnings if not."""
        current_time = time.time()
        warnings = []
        
        # Check win rate with smaller window
        if self.total_games > 10:  # Reduced from 20
            win_rate = self.win_count / self.total_games
            if win_rate < self.min_win_rate:
                warnings.append(f"Low win rate: {win_rate:.1%}")
        
        # Check consecutive mine hits with lower threshold
        if self.consecutive_mine_hits > self.max_consecutive_mine_hits:
            warnings.append(f"{self.consecutive_mine_hits} consecutive mine hits")
        
        if warnings:
            self.logger.warning("\nTraining Health Warnings:")
            for warning in warnings:
                self.logger.warning(f"- {warning}")
            self.logger.warning("Consider adjusting learning parameters or restarting training.")
        
        return len(warnings) == 0
        
    def render(self):
        """Render the current state of the environment."""
        for y in range(self.current_board_height):
            for x in range(self.current_board_width):
                if self.flags[y, x] and self.state[y, x] == CELL_UNREVEALED:
                    print('⚑', end=' ')
                elif self.state[y, x] == CELL_UNREVEALED:
                    print('□', end=' ')
                elif self.state[y, x] == CELL_MINE:
                    print('💣', end=' ')
                else:
                    print(self.state[y, x], end=' ')
            print()

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