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
from typing import Tuple, Dict

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
        self.state = np.full((self.current_board_size, self.current_board_size), -1, dtype=np.int8)
        self.board = np.zeros((self.current_board_size, self.current_board_size), dtype=np.int8)
        self.mines = np.zeros((self.current_board_size, self.current_board_size), dtype=bool)
        self.flags = np.zeros((self.current_board_size, self.current_board_size), dtype=bool)
        self._place_mines()
        self._update_adjacent_counts()
        # Ensure state is all -1 after all operations
        self.state = np.full((self.current_board_size, self.current_board_size), -1, dtype=np.int8)
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
                    self.board[row, col] = 9  # 9 represents a mine
                    mines_placed += 1

            attempts += 1

        if mines_placed < self.current_mines:
            print(f"Warning: Could not place {self.current_mines} mines with spacing {self.mine_spacing} on a {self.current_board_size}x{self.current_board_size} board.")
            print(f"Placed {mines_placed} mines instead.")
            self.current_mines = mines_placed

        self._update_adjacent_counts()

    def _update_adjacent_counts(self):
        """Update the board with counts of adjacent mines."""
        self.board = np.zeros((self.current_board_size, self.current_board_size), dtype=np.int8)
        for y in range(self.current_board_size):
            for x in range(self.current_board_size):
                if self.mines[y, x]:
                    self.board[y, x] = 9  # Mark mine
                    # Update adjacent cells
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = y + dy, x + dx
                            if (0 <= ny < self.current_board_size and 
                                0 <= nx < self.current_board_size and 
                                not self.mines[ny, nx]):
                                self.board[ny, nx] += 1

    def _reveal_cell(self, row, col, info=None):
        """Reveal a cell and handle cascading reveals."""
        if info is None:
            info = {'revealed_cells': set(), 'adjacent_mines': set()}
        
        if (row < 0 or row >= self.current_board_size or 
            col < 0 or col >= self.current_board_size or 
            self.state[row, col] != -1 or 
            self.flags[row, col]):
            return

        self.state[row, col] = self.board[row, col]
        self.revealed_count += 1
        info['revealed_cells'].add((row, col))

        # If this is a mine, add to adjacent mines
        if self.mines[row, col]:
            info['adjacent_mines'].add((row, col))
            return

        # If this cell has adjacent mines, add them to info
        if self.board[row, col] > 0:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    r, c = row + dr, col + dc
                    if (0 <= r < self.current_board_size and 
                        0 <= c < self.current_board_size and 
                        self.mines[r, c]):
                        info['adjacent_mines'].add((r, c))

        # If this is a 0, cascade to adjacent cells
        if self.board[row, col] == 0:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    self._reveal_cell(row + dr, col + dc, info)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment."""
        # Validate action
        if not isinstance(action, (int, np.integer)):
            raise ValueError(f"Action must be an integer, got {type(action)}")
        if action < 0 or action >= self.action_space.n:
            raise ValueError(f"Invalid action {action}. Must be between 0 and {self.action_space.n-1}")

        if self.terminated:
            # If game is terminated, return invalid action penalty
            return self.state, self.invalid_action_penalty, True, False, {
                'reward_breakdown': {'invalid_action': self.invalid_action_penalty},
                'revealed_cells': set(),
                'adjacent_mines': set()
            }
        
        # Convert action to (x, y) coordinates
        board_size = self.current_board_size
        if action < board_size * board_size:
            # Reveal action
            x = action % board_size
            y = action // board_size
            
            # Check if cell is already revealed or flagged
            if self.state[y, x] != -1 or self.flags[y, x]:
                return self.state, self.invalid_action_penalty, False, False, {
                    'reward_breakdown': {'invalid_action': self.invalid_action_penalty},
                    'revealed_cells': set(),
                    'adjacent_mines': set()
                }
            
            # Handle first move separately
            if self.is_first_move:
                if self.mines[y, x]:
                    # First move hit a mine - reset the game
                    self.reset()
                    # Force state to be all -1 after reset
                    self.state = np.full((self.current_board_size, self.current_board_size), -1, dtype=np.int8)
                    self.flags = np.zeros((self.current_board_size, self.current_board_size), dtype=bool)
                    self.is_first_move = False  # Set flag to False after move is complete
                    return self.state, 0, False, False, {
                        'reward_breakdown': {'first_move_mine_hit_reset': 0},
                        'revealed_cells': set(),
                        'adjacent_mines': set()
                    }
                else:
                    # First move safe reveal - reveal cell and continue
                    info = {'revealed_cells': set(), 'adjacent_mines': set()}
                    self._reveal_cell(y, x, info)
                    # Check for win after first move reveal
                    if self._check_win():
                        self.won = True
                        self.total_games += 1
                        self.terminated = True
                        self.is_first_move = False  # Set flag to False after move is complete
                        return self.state, 0, True, False, {
                            'reward_breakdown': {'first_move_safe_reveal': 0},
                            'revealed_cells': info['revealed_cells'],
                            'adjacent_mines': info['adjacent_mines']
                        }
                    self.is_first_move = False  # Set flag to False after move is complete
                    return self.state, 0, False, False, {
                        'reward_breakdown': {'first_move_safe_reveal': 0},
                        'revealed_cells': info['revealed_cells'],
                        'adjacent_mines': info['adjacent_mines']
                    }
            
            # Handle reveal action for non-first moves
            return self._handle_reveal_action(x, y)
        else:
            # Flag action
            action = action - (board_size * board_size)
            x = action % board_size
            y = action // board_size
            
            # Check if cell is already revealed
            if self.state[y, x] != -1:
                return self.state, self.invalid_action_penalty, False, False, {
                    'reward_breakdown': {'invalid_action': self.invalid_action_penalty},
                    'revealed_cells': set(),
                    'adjacent_mines': set()
                }
            
            return self._handle_flag_action(x, y)

    def _handle_reveal_action(self, x, y):
        """Handle reveal action and return reward and termination status."""
        row, col = y, x
        
        # Check if cell is already revealed
        if self.state[row, col] != -1:
            return self.state, self.invalid_action_penalty, False, False, {
                'reward_breakdown': {'invalid_action': self.invalid_action_penalty},
                'revealed_cells': set(),
                'adjacent_mines': set()
            }
        
        # Check if cell is flagged
        if self.flags[row, col]:
            return self.state, self.invalid_action_penalty, False, False, {
                'reward_breakdown': {'invalid_action': self.invalid_action_penalty},
                'revealed_cells': set(),
                'adjacent_mines': set()
            }
        
        # Handle mine hit
        if self.mines[row, col]:
            if self.is_first_move:
                # First move hit a mine - reset the game
                self.reset()
                # Force state to be all -1 after reset
                self.state = np.full((self.current_board_size, self.current_board_size), -1, dtype=np.int8)
                self.flags = np.zeros((self.current_board_size, self.current_board_size), dtype=bool)
                self.is_first_move = False  # Set flag to False after move is complete
                return self.state, 0, False, False, {
                    'reward_breakdown': {'first_move_mine_hit_reset': 0},
                    'revealed_cells': set(),
                    'adjacent_mines': set()
                }
            else:
                # Non-first move hit a mine - game over
                self.terminated = True
                self.state[row, col] = -2  # Mark mine as hit
                # Reveal all mines
                for y in range(self.current_board_size):
                    for x in range(self.current_board_size):
                        if self.mines[y, x]:
                            self.state[y, x] = -2
                return self.state, self.mine_penalty, True, False, {
                    'reward_breakdown': {'mine_hit': self.mine_penalty},
                    'revealed_cells': {(row, col)},
                    'adjacent_mines': {(row, col)}
                }
        
        # Handle safe cell reveal
        info = {'revealed_cells': set(), 'adjacent_mines': set()}
        self._reveal_cell(row, col, info)
        
        # Check for win after reveal
        if self._check_win():
            self.won = True
            self.total_games += 1
            self.terminated = True
            # If this is a first move win, return 0 reward
            if self.is_first_move:
                return self.state, 0, True, False, {
                    'reward_breakdown': {'first_move_safe_reveal': 0},
                    'revealed_cells': info['revealed_cells'],
                    'adjacent_mines': info['adjacent_mines']
                }
            return self.state, self.win_reward, True, False, {
                'reward_breakdown': {'win': self.win_reward},
                'revealed_cells': info['revealed_cells'],
                'adjacent_mines': info['adjacent_mines']
            }
        
        # Normal safe reveal
        return self.state, self.safe_reveal_base, False, False, {
            'reward_breakdown': {'safe_reveal': self.safe_reveal_base},
            'revealed_cells': info['revealed_cells'],
            'adjacent_mines': info['adjacent_mines']
        }

    def _handle_flag_action(self, x, y):
        """Handle flag placement/removal action."""
        # Check if cell is already revealed
        if self.state[y, x] != -1:
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

    def _check_win(self):
        """Check if all non-mine cells have been revealed."""
        # Count revealed cells
        revealed_count = np.sum(self.state != -1)
        # Total cells minus mines should equal revealed cells
        total_safe_cells = self.current_board_size * self.current_board_size - np.sum(self.mines)
        return revealed_count >= total_safe_cells

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

    def _check_win_condition(self):
        """Check if the game is won"""
        # All mines must be flagged
        for mine in self.mines:
            if not self.flags[mine]:
                return False

        # All safe cells must be revealed
        for x in range(self.current_board_size):
            for y in range(self.current_board_size):
                if (x, y) not in self.mines and self.state[x, y] == -1:
                    return False

        return True
        
    def render(self):
        for x in range(self.current_board_size):
            for y in range(self.current_board_size):
                if self.flags[x, y] and self.state[x, y] == -1:
                    print('⚑', end=' ')
                elif self.state[x, y] == -1:
                    print('□', end=' ')
                elif self.state[x, y] == -2:
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