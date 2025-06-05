import gymnasium as gym
import numpy as np
from gymnasium import spaces
import time
from datetime import datetime
from collections import deque
import logging
import os
import sys

class MinesweeperEnv(gym.Env):
    """
    A Minesweeper environment for reinforcement learning with flagging, realistic win condition, and fixed observation/action space for curriculum learning.
    """
    def __init__(self, max_board_size=10, max_mines=10, early_learning_mode=True,
                 early_learning_threshold=200, early_learning_corner_safe=True,
                 early_learning_edge_safe=True, mine_spacing=2, initial_board_size=4,
                 initial_mines=2, invalid_action_penalty=-0.1):
        """Initialize the Minesweeper environment"""
        super().__init__()
        
        # Store parameters
        self.max_board_size = max_board_size
        self.max_mines = max_mines
        self.early_learning_mode = early_learning_mode
        self.early_learning_threshold = early_learning_threshold
        self.early_learning_corner_safe = early_learning_corner_safe
        self.early_learning_edge_safe = early_learning_edge_safe
        self.mine_spacing = mine_spacing
        self.initial_board_size = initial_board_size
        self.initial_mines = initial_mines
        self.invalid_action_penalty = invalid_action_penalty

        # Reward values
        self.mine_penalty = -10.0
        self.safe_reveal_base = 5.0
        self.win_reward = 100.0

        # Initialize state
        self.current_board_size = initial_board_size
        self.current_mines = initial_mines
        self.board = None
        self.state = None
        self.flags = None
        self.mines = None
        self.won = False
        self.total_games = 0
        self.games_at_current_size = 0
        self.revealed_count = 0  # Track number of revealed cells

        # Curriculum learning parameters
        self.curriculum_mode = True
        self.mine_density = 0.15
        self.mines_increment = 1
        self.mines_increment_threshold = 0.25

        # Define action and observation spaces
        self.action_space = spaces.Discrete(max_board_size * max_board_size * 2)  # *2 for reveal/flag actions
        self.observation_space = spaces.Box(
            low=-2,  # -2 for mines, -1 for hidden, 0-8 for revealed cells
            high=8,
            shape=(max_board_size, max_board_size),
            dtype=np.int8
        )

        # Log environment configuration
        self.logger = logging.getLogger(__name__)
        self.logger.info("\n🔧 Environment Configuration:")
        self.logger.info(f"- Early Learning Mode: {self.early_learning_mode}")
        self.logger.info(f"- Early Learning Threshold: {self.early_learning_threshold}")
        self.logger.info(f"- Corner Safety: {self.early_learning_corner_safe}")
        self.logger.info(f"- Edge Safety: {self.early_learning_edge_safe}")
        self.logger.info(f"- Mine Spacing: {self.mine_spacing}")
        self.logger.info(f"- Initial Board Size: {self.initial_board_size}")
        self.logger.info(f"- Initial Mines: {self.initial_mines}")

        # Initialize the game
        self.reset()

    def reset(self, seed=None, options=None):
        """Reset the environment to start a new game."""
        super().reset(seed=seed)
        
        # Initialize board arrays
        self.board = np.zeros((self.current_board_size, self.current_board_size), dtype=np.int8)
        self.state = np.full((self.current_board_size, self.current_board_size), -1, dtype=np.int8)
        self.flags = np.zeros((self.current_board_size, self.current_board_size), dtype=np.int8)
        self.mines = set()
        self.won = False

        # Place mines
        self._place_mines()
        
        # Update adjacent mine counts
        self._update_adjacent_counts()

        return self.state, {}  # Return observation and info dict

    def _place_mines(self):
        """Place mines on the board according to the current configuration."""
        # Clear existing mines
        self.mines.clear()
        self.board.fill(0)

        # Calculate number of mines based on current configuration
        num_mines = self.current_mines

        # Calculate maximum possible mines with current spacing
        max_possible_mines = (self.current_board_size // (self.mine_spacing + 1)) ** 2
        if num_mines > max_possible_mines:
            self.logger.warning(f"Requested {num_mines} mines with spacing {self.mine_spacing} on {self.current_board_size}x{self.current_board_size} board is impossible. Reducing to {max_possible_mines} mines.")
            num_mines = max_possible_mines

        # Place mines
        attempts = 0
        max_attempts = 1000  # Prevent infinite loops
        while len(self.mines) < num_mines and attempts < max_attempts:
            row = np.random.randint(0, self.current_board_size)
            col = np.random.randint(0, self.current_board_size)
            
            # Skip if position already has a mine
            if (row, col) in self.mines:
                attempts += 1
                continue
                
            # Check mine spacing
            if self._check_mine_spacing(row, col):
                self.mines.add((row, col))
                self.board[row, col] = 9  # 9 represents a mine
            attempts += 1

        if len(self.mines) < num_mines:
            self.logger.warning(f"Could not place all {num_mines} mines with spacing {self.mine_spacing}. Placed {len(self.mines)} mines instead.")

    def _check_mine_spacing(self, row, col):
        """Check if a mine can be placed at the given position respecting spacing rules."""
        for r in range(max(0, row - self.mine_spacing), min(self.current_board_size, row + self.mine_spacing + 1)):
            for c in range(max(0, col - self.mine_spacing), min(self.current_board_size, col + self.mine_spacing + 1)):
                if (r, c) in self.mines:
                    return False
        return True

    def _update_adjacent_counts(self):
        """Update the board with counts of adjacent mines."""
        for row in range(self.current_board_size):
            for col in range(self.current_board_size):
                if self.board[row, col] != 9:  # Skip mines
                    count = 0
                    for r in range(max(0, row - 1), min(self.current_board_size, row + 2)):
                        for c in range(max(0, col - 1), min(self.current_board_size, col + 2)):
                            if self.board[r, c] == 9:  # If adjacent cell is a mine
                                count += 1
                    self.board[row, col] = count

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
        
    def step(self, action):
        """Execute one time step within the environment"""
        if self.state is None:
            raise ValueError("Environment not initialized. Call reset() first.")

        # Parse action
        if action < self.current_board_size * self.current_board_size:
            # Reveal cell
            row = action // self.current_board_size
            col = action % self.current_board_size
            is_flag = False
        else:
            # Flag cell
            action = action - self.current_board_size * self.current_board_size
            row = action // self.current_board_size
            col = action % self.current_board_size
            is_flag = True

        # Validate action
        if not (0 <= row < self.current_board_size and 0 <= col < self.current_board_size):
            return self.state, self.invalid_action_penalty, False, False, {}

        # Check if cell is already revealed
        if not is_flag and self.state[row, col] != -1:
            return self.state, self.invalid_action_penalty, False, False, {}

        # Track revealed cells for this step
        revealed_cells = set()
        adjacent_mines = set()
        reward_breakdown = {}

        # Execute action
        if is_flag:
            # Toggle flag
            if self.flags[row, col]:
                self.flags[row, col] = False
                reward = -0.1  # Small penalty for unflagging
                reward_breakdown['unflag'] = -0.1
            else:
                self.flags[row, col] = True
                if (row, col) in self.mines:
                    reward = 5.0  # Reward for correctly flagging a mine
                    reward_breakdown['correct_flag'] = 5.0
                else:
                    reward = -1.0  # Penalty for incorrectly flagging a safe cell
                    reward_breakdown['incorrect_flag'] = -1.0
        else:
            # Reveal cell
            if (row, col) in self.mines:
                self.state[row, col] = -2  # Mark mine
                self.revealed_count += 1  # Increment revealed count for mine hit
                self.won = False
                reward = self.mine_penalty
                reward_breakdown['mine_hit'] = self.mine_penalty
                return self.state, reward, True, False, {
                    'revealed_cells': revealed_cells,
                    'adjacent_mines': adjacent_mines,
                    'reward_breakdown': reward_breakdown
                }
            else:
                # Reveal safe cell and handle cascade effect
                self._reveal_cell(row, col, revealed_cells)
                # Add adjacent mines to the info
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = row + dx, col + dy
                        if 0 <= nx < self.current_board_size and 0 <= ny < self.current_board_size:
                            if (nx, ny) in self.mines:
                                adjacent_mines.add((nx, ny))
                reward = self.safe_reveal_base  # Reward for revealing a safe cell
                reward_breakdown['safe_reveal'] = self.safe_reveal_base

        # Check win condition
        if self._check_win_condition():
            self.won = True
            reward = self.win_reward
            reward_breakdown['win'] = self.win_reward
            return self.state, reward, True, False, {
                'revealed_cells': revealed_cells,
                'adjacent_mines': adjacent_mines,
                'reward_breakdown': reward_breakdown
            }

        return self.state, reward, False, False, {
            'revealed_cells': revealed_cells,
            'adjacent_mines': adjacent_mines,
            'reward_breakdown': reward_breakdown
        }

    def _count_adjacent_mines(self, x, y):
        count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.current_board_size and 0 <= ny < self.current_board_size:
                    if (nx, ny) in self.mines:
                        count += 1
        return count

    def _reveal_cell(self, x, y, revealed_cells=None):
        """Reveal a cell and handle cascade effect."""
        if revealed_cells is None:
            revealed_cells = set()
        queue = [(x, y)]
        while queue:
            x, y = queue.pop(0)
            if not (0 <= x < self.current_board_size and 0 <= y < self.current_board_size):
                continue
            if self.state[x, y] != -1 or self.flags[x, y]:
                continue
            if (x, y) in self.mines:
                continue  # Do not reveal mines during cascade
            self.state[x, y] = self.board[x, y]
            self.revealed_count += 1
            revealed_cells.add((x, y))
            # Only cascade if this cell has no adjacent mines
            if self.board[x, y] == 0:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        queue.append((x + dx, y + dy))

    def _check_win(self):
        for x in range(self.current_board_size):
            for y in range(self.current_board_size):
                if (x, y) in self.mines:
                    if not self.flags[x, y]:
                        return False
                else:
                    if self.state[x, y] == -1:
                        return False
        return True

    def _get_obs(self):
        """Get the current observation of the environment"""
        # Create observation array of max size
        obs = np.full((self.max_board_size, self.max_board_size), -3, dtype=np.int8)
        # Fill in the current board state
        obs[:self.current_board_size, :self.current_board_size] = self.state
        return obs

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