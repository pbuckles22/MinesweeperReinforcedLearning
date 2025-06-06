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
    CELL_FLAGGED
)

class MinesweeperEnv(gym.Env):
    """
    A Minesweeper environment for reinforcement learning with flagging, realistic win condition, and fixed observation/action space for curriculum learning.
    """
    def __init__(self, max_board_size=10, max_mines=10, render_mode=None,
                 early_learning_mode=False, early_learning_threshold=200,
                 early_learning_corner_safe=True, early_learning_edge_safe=True,
                 mine_spacing=1, initial_board_size=4, initial_mines=2,
                 invalid_action_penalty=-0.1, mine_penalty=-10.0,
                 flag_mine_reward=5.0, flag_safe_penalty=-1.0,
                 unflag_penalty=-0.1, safe_reveal_base=5.0, win_reward=100.0):
        """Initialize the Minesweeper environment."""
        super().__init__()
        
        # Validate board size
        if max_board_size <= 0:
            raise ValueError("Board size must be positive")
        if max_board_size > 100:
            raise ValueError("Board size too large")
            
        # Validate mine count
        if max_mines <= 0:
            raise ValueError("Mine count must be positive")
        if max_mines > max_board_size * max_board_size:
            raise ValueError("Mine count cannot exceed board size squared")
            
        # Validate mine spacing
        if mine_spacing < 0:
            raise ValueError("Mine spacing must be non-negative")
        if mine_spacing >= max_board_size:
            raise ValueError("Mine spacing too large for board size")
            
        # Validate initial parameters
        if initial_board_size > max_board_size:
            raise ValueError("Initial board size cannot exceed max board size")
        if initial_mines > max_mines:
            raise ValueError("Initial mine count cannot exceed max mines")
        if initial_mines > initial_board_size * initial_board_size:
            raise ValueError("Initial mine count cannot exceed initial board size squared")
            
        # Validate reward parameters
        if mine_penalty >= 0:
            raise ValueError("Mine penalty must be negative")
        if flag_safe_penalty >= 0:
            raise ValueError("Flag safe penalty must be negative")
        if unflag_penalty >= 0:
            raise ValueError("Unflag penalty must be negative")
        
        # Store parameters
        self.max_board_size = max_board_size
        self.max_mines = max_mines
        self.render_mode = render_mode
        self.early_learning_mode = early_learning_mode
        self.early_learning_threshold = early_learning_threshold
        self.early_learning_corner_safe = early_learning_corner_safe
        self.early_learning_edge_safe = early_learning_edge_safe
        self.mine_spacing = mine_spacing
        self.initial_board_size = initial_board_size
        self.initial_mines = initial_mines
        self.invalid_action_penalty = invalid_action_penalty
        self.mine_penalty = mine_penalty
        self.flag_mine_reward = flag_mine_reward
        self.flag_safe_penalty = flag_safe_penalty
        self.unflag_penalty = unflag_penalty
        self.safe_reveal_base = safe_reveal_base
        self.win_reward = win_reward
        
        # Initialize game state
        self.current_board_size = initial_board_size
        self.current_mines = initial_mines
        self.state = None
        self.board = None
        self.mines = None
        self.flags = None
        self.revealed_count = 0
        self.won = False
        self.total_games = 0
        self.games_at_current_size = 0
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
        self.action_space = spaces.Discrete(self.current_board_size * self.current_board_size * 2)
        self.observation_space = spaces.Box(
            low=-2,
            high=8,
            shape=(self.current_board_size, self.current_board_size),
            dtype=np.int8
        )
        
        # Initialize the environment
        self.reset()

        # Initialize pygame if render mode is set
        if self.render_mode == "human":
            pygame.init()
            self.cell_size = 40
            self.screen = pygame.display.set_mode((self.current_board_size * self.cell_size, self.current_board_size * self.cell_size))
            pygame.display.set_caption("Minesweeper")
            self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        self.revealed_count = 0
        self.won = False
        self.terminated = False  # Reset game over state
        self.is_first_move = True  # Reset first move flag
        self.state = np.full((self.current_board_size, self.current_board_size), CELL_UNREVEALED, dtype=int)
        self.board = np.zeros((self.current_board_size, self.current_board_size), dtype=int)
        self.mines = np.zeros((self.current_board_size, self.current_board_size), dtype=bool)
        self.flags = np.zeros((self.current_board_size, self.current_board_size), dtype=bool)
        self._place_mines()
        self._update_adjacent_counts()
        # Ensure state is all -1 after all operations
        self.state = np.full((self.current_board_size, self.current_board_size), CELL_UNREVEALED, dtype=int)
        return self.state, {}

    def _place_mines(self):
        """Place mines on the board with minimum spacing."""
        self.mines = np.zeros((self.current_board_size, self.current_board_size), dtype=bool)
        mines_placed = 0
        attempts = 0
        max_attempts = self.current_board_size * self.current_board_size * 10

        while mines_placed < self.current_mines and attempts < max_attempts:
            row = np.random.randint(0, self.current_board_size)
            col = np.random.randint(0, self.current_board_size)

            # Check if position is valid (not already a mine and respects spacing)
            if not self.mines[row, col]:
                valid_position = True
                for dr in range(-self.mine_spacing, self.mine_spacing + 1):
                    for dc in range(-self.mine_spacing, self.mine_spacing + 1):
                        r, c = row + dr, col + dc
                        if (0 <= r < self.current_board_size and 
                            0 <= c < self.current_board_size and 
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
            print(f"Warning: Could not place {self.current_mines} mines with spacing {self.mine_spacing} on a {self.current_board_size}x{self.current_board_size} board.")
            print(f"Placed {mines_placed} mines instead.")
            self.current_mines = mines_placed

        self._update_adjacent_counts()

    def _update_adjacent_counts(self):
        """Update the adjacent mine counts for each cell."""
        # Debug print: Log adjacent counts update
        print("Updating adjacent mine counts...")
        for y in range(self.current_board_size):
            for x in range(self.current_board_size):
                if not self.mines[y, x]:
                    count = 0
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < self.current_board_size and 0 <= nx < self.current_board_size and self.mines[ny, nx]:
                                count += 1
                    self.board[y, x] = count
                    print(f"Cell at (x, y): ({x}, {y}) has {count} adjacent mines")

    def _reveal_cell(self, y: int, x: int, info: Dict) -> None:
        """Reveal a cell and handle cascade effect."""
        # Debug print: Log cell reveal
        print(f"Revealing cell at (x, y): ({x}, {y})")
        if self.state[y, x] != CELL_UNREVEALED or self.flags[y, x]:
            print(f"Cell at (x, y): ({x}, {y}) already revealed or flagged, skipping")
            return

        self.state[y, x] = self.board[y, x]
        info['revealed_cells'].add((y, x))
        print(f"Cell at (x, y): ({x}, {y}) revealed with value {self.board[y, x]}")
        if self.board[y, x] == 0:
            info['adjacent_mines'].add((y, x))
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.current_board_size and 0 <= nx < self.current_board_size:
                        # Debug print: Log cascade attempt
                        print(f"Attempting to cascade to cell at (x, y): ({nx}, {ny})")
                        # Check if the cell is adjacent to a mine
                        is_adjacent_to_mine = False
                        for my in range(max(0, ny - 1), min(self.current_board_size, ny + 2)):
                            for mx in range(max(0, nx - 1), min(self.current_board_size, nx + 2)):
                                if self.mines[my, mx]:
                                    is_adjacent_to_mine = True
                                    break
                            if is_adjacent_to_mine:
                                break
                        if not is_adjacent_to_mine:
                            print(f"Cascading to cell at (x, y): ({nx}, {ny})")
                            self._reveal_cell(ny, nx, info)
                        else:
                            print(f"Cell at (x, y): ({nx}, {ny}) adjacent to mine, revealing with count")
                            self.state[ny, nx] = self.board[ny, nx]
                            info['revealed_cells'].add((ny, nx))
                            info['adjacent_mines'].add((ny, nx))
        else:
            info['adjacent_mines'].add((y, x))

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment."""
        # Debug print: Log action
        print(f"Action: {action}, is_first_move: {self.is_first_move}")
        # Convert action to coordinates
        y, x = action // self.current_board_size, action % self.current_board_size
        # Debug print: Log coordinates
        print(f"Coordinates: (x, y): ({x}, {y})")
        # Check if the action is valid
        if not (0 <= y < self.current_board_size and 0 <= x < self.current_board_size):
            raise ValueError(f"Invalid action: {action}")
        # Check if the cell is already revealed or flagged
        if self.state[y, x] != CELL_UNREVEALED or self.flags[y, x]:
            raise ValueError(f"Cell already revealed or flagged: {action}")
        # Check if the cell is a mine
        if self.mines[y, x]:
            self.state[y, x] = CELL_MINE  # Mark mine
            self.won = False
            # First move hit mine gets 0 reward
            if self.is_first_move:
                self.is_first_move = False
                return self.state, REWARD_FIRST_MOVE_HIT_MINE, True, False, {'reward_breakdown': {'first_move_hit_mine': REWARD_FIRST_MOVE_HIT_MINE}}
            return self.state, REWARD_HIT_MINE, True, False, {'reward_breakdown': {'hit_mine': REWARD_HIT_MINE}}
        # Reveal the cell
        info = {'revealed_cells': set(), 'adjacent_mines': set(), 'reward_breakdown': {}}
        self._reveal_cell(y, x, info)
        # Debug print: Log state array after move
        print("State array after move:")
        print(self.state)
        # Check if the game is won
        if self._check_win():
            self.won = True
            return self.state, REWARD_WIN, True, False, {'reward_breakdown': {'win': REWARD_WIN}}
        # First move safe reveal
        if self.is_first_move:
            self.is_first_move = False
            return self.state, REWARD_FIRST_MOVE_SAFE, False, False, {'reward_breakdown': {'first_move_safe_reveal': REWARD_FIRST_MOVE_SAFE}}
        return self.state, REWARD_SAFE_REVEAL, False, False, {'reward_breakdown': {'safe_reveal': REWARD_SAFE_REVEAL}}

    def _handle_flag_action(self, x, y):
        """Handle flag placement/removal action."""
        # Check if cell is already revealed
        if self.state[y, x] != CELL_UNREVEALED:
            return self.state, self.invalid_action_penalty, False, False, {
                'reward_breakdown': {'invalid_action': self.invalid_action_penalty},
                'revealed_cells': set(),
                'adjacent_mines': set()
            }

        # Toggle flag
        self.flags[y, x] = not self.flags[y, x]
        
        if self.flags[y, x]:  # Placing flag
            if self.mines[y, x]:
                reward = self.flag_mine_reward
                info = {
                    'reward_breakdown': {'correct_flag': self.flag_mine_reward},
                    'revealed_cells': set(),
                    'adjacent_mines': set()
                }
            else:
                reward = self.flag_safe_penalty
                info = {
                    'reward_breakdown': {'incorrect_flag': self.flag_safe_penalty},
                    'revealed_cells': set(),
                    'adjacent_mines': set()
                }
        else:  # Removing flag
            reward = self.unflag_penalty
            info = {
                'reward_breakdown': {'unflag': self.unflag_penalty},
                'revealed_cells': set(),
                'adjacent_mines': set()
            }

        # Check for win after flagging
        if self._check_win():
            reward += self.win_reward
            info['reward_breakdown']['win'] = self.win_reward
            self.won = True
            self.total_games += 1
            self.terminated = True
            return self.state, reward, True, False, info

        return self.state, reward, False, False, info

    def _check_win(self) -> bool:
        """Check if the game is won."""
        print("\n=== Checking Win Condition ===")
        print("Current state array:")
        print(self.state)
        print("\nMines array:")
        print(self.mines)
        
        unrevealed_safe_cells = []
        for y in range(self.current_board_size):
            for x in range(self.current_board_size):
                if not self.mines[y, x] and self.state[y, x] == CELL_UNREVEALED:
                    unrevealed_safe_cells.append((x, y))
                    print(f"Found unrevealed safe cell at (x, y): ({x}, {y})")
        
        if unrevealed_safe_cells:
            print(f"\nGame not won: Found {len(unrevealed_safe_cells)} unrevealed safe cells")
            return False
        
        print("\nGame won: All safe cells revealed")
        return True

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
                self.logger.info(f"Board: {self.current_board_size}x{self.current_board_size} | Mines: {self.current_mines}")
                
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
        for x in range(self.current_board_size):
            for y in range(self.current_board_size):
                if self.flags[x, y] and self.state[x, y] == CELL_UNREVEALED:
                    print('⚑', end=' ')
                elif self.state[x, y] == CELL_UNREVEALED:
                    print('□', end=' ')
                elif self.state[x, y] == CELL_MINE:
                    print('💣', end=' ')
                else:
                    print(self.state[x, y], end=' ')
            print()

def main():
    # Create and test the environment
    env = MinesweeperEnv(board_size=8, num_mines=12)
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