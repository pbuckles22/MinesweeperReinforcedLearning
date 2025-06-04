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
    def __init__(self, board_size=8, num_mines=12, curriculum_mode=True):
        super(MinesweeperEnv, self).__init__()
        
        # Curriculum learning parameters
        self.curriculum_mode = curriculum_mode
        self.max_board_size = board_size
        self.max_mines = num_mines
        self.current_board_size = 8  # Back to original size
        self.mine_density = 0.15  # Back to original density
        self.current_mines = 12 if curriculum_mode else num_mines  # Back to original mine count
        self.mines_increment = 1
        self.mines_increment_threshold = 0.3  # Back to original threshold
        self.min_games_before_size_increase = 200  # Back to original
        self.min_win_rate_for_size_increase = 0.3  # Back to original
        
        # Training health parameters
        self.consecutive_mine_hits = 0
        self.max_consecutive_mine_hits = 30  # Back to original
        self.min_win_rate = 0.1  # Back to original
        self.win_rate_window = 500  # Back to original
        
        # Reward scaling
        self.win_reward = 20  # Back to original
        self.mine_penalty = -5  # Back to original
        self.progress_multiplier = 3  # Back to original
        self.safe_reveal_base = 1.0  # Back to original
        self.number_reveal_bonus = 0.5  # Back to original
        self.correct_flag_reward = 2  # Back to original
        self.incorrect_flag_penalty = -0.5  # Back to original
        
        # Setup logging
        self.setup_logging()
        
        # Curriculum tracking
        self.games_at_current_size = 0
        self.wins_at_current_size = 0
        self.board_size_increment_threshold = 0.4
        self.win_count = 0
        self.total_games = 0
        
        # Scaling parameters
        self.max_steps = min(100 + (self.max_board_size - 6) * 20, 200)  # Scale steps with board size
        self.max_mines = min(int(self.max_board_size * self.max_board_size * self.mine_density), 30)  # Cap at 30 mines
        
        # Performance monitoring
        self.recent_rewards = deque(maxlen=100)  # Track last 100 rewards
        self.recent_episode_lengths = deque(maxlen=100)  # Track last 100 episode lengths
        self.recent_mine_hits = deque(maxlen=100)  # Track last 100 mine hits
        self.last_win_time = None  # Track time since last win
        self.episode_start_time = None
        self.last_action_time = None
        self.steps = 0
        
        # Progress display
        self.last_progress_update = time.time()
        self.progress_interval = 5  # Update progress every 5 seconds
        self.last_win_rate = 0
        self.last_avg_reward = 0
        self.last_avg_length = 0
        
        # Warning thresholds
        self.max_time_without_win = 300  # 5 minutes
        self.min_win_rate_threshold = 0.1  # 10%
        self.max_avg_episode_length = 50  # steps
        
        # Always use max size for obs/action space
        self.action_space = spaces.Discrete(2 * self.max_board_size * self.max_board_size)
        self.observation_space = spaces.Box(
            low=-3,  # -3: unused, -2: revealed mine, -1: unrevealed, 0-8: revealed safe, 9: flagged
            high=9,
            shape=(self.max_board_size, self.max_board_size),
            dtype=np.int8
        )
        self.board = None
        self.state = None
        self.flags = None
        self.mines = None
        self.game_over = False
        self.won = False
        self.revealed_count = 0
        self.flagged_mines = 0
        self.incorrect_flags = 0

    def setup_logging(self):
        """Setup logging to both file and console with different levels."""
        # Create logs directory if it doesn't exist
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Create a timestamp for the log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"minesweeper_{timestamp}.log")
        
        # Setup file handler for detailed logging
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Setup console handler for progress display
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        
        # Setup logger
        self.logger = logging.getLogger('MinesweeperEnv')
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _update_progress_display(self):
        """Update the progress display with current training stats."""
        current_time = time.time()
        if current_time - self.last_progress_update >= self.progress_interval:
            # Calculate current stats
            win_rate = self.win_count / self.total_games if self.total_games > 0 else 0
            avg_reward = sum(self.recent_rewards) / len(self.recent_rewards) if self.recent_rewards else 0
            avg_length = sum(self.recent_episode_lengths) / len(self.recent_episode_lengths) if self.recent_episode_lengths else 0
            
            # Only update if there are significant changes
            if (abs(win_rate - self.last_win_rate) > 0.01 or
                abs(avg_reward - self.last_avg_reward) > 0.1 or
                abs(avg_length - self.last_avg_length) > 1):
                
                # Clear previous progress display
                print("\033[2J\033[H", end="")  # Clear screen and move cursor to top
                
                # Add some padding at the top
                print("\n" * 2)
                
                self.logger.info("\n" + "="*70)
                self.logger.info(f"Training Progress at {datetime.now().strftime('%H:%M:%S')}")
                self.logger.info("="*70)
                
                # Game Configuration
                self.logger.info("\nGame Configuration:")
                self.logger.info(f"Board Size: {self.current_board_size}x{self.current_board_size}")
                self.logger.info(f"Mines: {self.current_mines} (Density: {self.current_mines/(self.current_board_size*self.current_board_size):.1%})")
                self.logger.info(f"Max Steps: {self.max_steps}")
                
                # Performance Metrics
                self.logger.info("\nPerformance Metrics:")
                self.logger.info(f"Win Rate: {win_rate:.1%}")
                self.logger.info(f"Average Reward: {avg_reward:.2f}")
                self.logger.info(f"Average Game Length: {avg_length:.1f} steps")
                
                # Game Statistics
                self.logger.info("\nGame Statistics:")
                self.logger.info(f"Total Games: {self.total_games}")
                self.logger.info(f"Total Wins: {self.win_count}")
                self.logger.info(f"Games at Current Size: {self.games_at_current_size}")
                self.logger.info(f"Consecutive Mine Hits: {self.consecutive_mine_hits}")
                
                # Recent Performance
                if len(self.recent_rewards) > 0:
                    self.logger.info("\nRecent Performance:")
                    self.logger.info(f"Last 10 Rewards: {[f'{r:.1f}' for r in list(self.recent_rewards)[-10:]]}")
                    self.logger.info(f"Last 10 Game Lengths: {[f'{l:.0f}' for l in list(self.recent_episode_lengths)[-10:]]}")
                
                # Add some padding at the bottom
                self.logger.info("\n" + "="*70 + "\n")
                
                self.last_win_rate = win_rate
                self.last_avg_reward = avg_reward
                self.last_avg_length = avg_length
            
            self.last_progress_update = current_time

    def _check_training_health(self):
        """Check if training is going well and log warnings if not."""
        current_time = time.time()
        warnings = []
        
        # Check win rate
        if self.total_games > 20:  # Only check after some games
            win_rate = self.win_count / self.total_games
            if win_rate < self.min_win_rate_threshold:
                warnings.append(f"Low win rate: {win_rate:.1%}")
        
        # Check time since last win
        if self.last_win_time and (current_time - self.last_win_time) > self.max_time_without_win:
            warnings.append(f"No wins in {int((current_time - self.last_win_time)/60)} minutes")
        
        # Check consecutive mine hits
        if self.consecutive_mine_hits >= self.max_consecutive_mine_hits:
            warnings.append(f"{self.consecutive_mine_hits} consecutive mine hits")
        
        # Check average episode length
        if len(self.recent_episode_lengths) >= 20:
            avg_length = sum(self.recent_episode_lengths) / len(self.recent_episode_lengths)
            if avg_length > self.max_avg_episode_length:
                warnings.append(f"Long average episode length: {avg_length:.1f} steps")
        
        # Check recent rewards
        if len(self.recent_rewards) >= 20:
            avg_reward = sum(self.recent_rewards) / len(self.recent_rewards)
            if avg_reward < -5:  # Consistently negative rewards
                warnings.append(f"Poor average reward: {avg_reward:.1f}")
        
        if warnings:
            self.logger.warning("\nTraining Health Warnings:")
            for warning in warnings:
                self.logger.warning(f"- {warning}")
            self.logger.warning("Consider adjusting learning parameters or restarting training.")

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.steps = 0
        self.episode_start_time = time.time()
        self.last_action_time = self.episode_start_time
        
        if self.curriculum_mode and self.total_games > 0:
            # Track games at current size
            self.games_at_current_size += 1
            # Check if we should increase board size
            if (self.games_at_current_size >= self.min_games_before_size_increase and 
                self.current_board_size < self.max_board_size):
                win_rate_at_size = self.wins_at_current_size / self.games_at_current_size
                if win_rate_at_size >= self.min_win_rate_for_size_increase:
                    self.current_board_size = min(self.current_board_size + 1, self.max_board_size)
                    # Adjust mine count based on new board size
                    target_mines = int(self.current_board_size * self.current_board_size * self.mine_density)
                    self.current_mines = min(target_mines, self.max_mines)
                    self.logger.info("\n🎯 Curriculum Update:")
                    self.logger.info(f"- Increased board size to {self.current_board_size}x{self.current_board_size}")
                    self.logger.info(f"- Adjusted mines to {self.current_mines} (density: {self.mine_density:.1%})")
                    self.logger.info(f"- Win rate at previous size: {win_rate_at_size:.1%}")
                    self.logger.info(f"- Games played at previous size: {self.games_at_current_size}")
                    # Reset size-specific counters
                    self.games_at_current_size = 0
                    self.wins_at_current_size = 0
            # Check if we should increase mines
            if self.current_mines < self.max_mines:
                win_rate = self.win_count / self.total_games
                if win_rate >= self.mines_increment_threshold:
                    self.current_mines = min(self.current_mines + self.mines_increment, self.max_mines)
                    self.logger.info(f"\n💣 Increased mines to {self.current_mines}")
        # Initialize arrays
        self.board = np.zeros((self.current_board_size, self.current_board_size), dtype=np.int8)
        self.state = np.full((self.current_board_size, self.current_board_size), -1, dtype=np.int8)
        self.flags = np.zeros((self.current_board_size, self.current_board_size), dtype=bool)
        self.revealed = np.zeros((self.current_board_size, self.current_board_size), dtype=np.int8)
        # Place mines as a set of (x, y) tuples
        self.mines = set()
        while len(self.mines) < self.current_mines:
            x = np.random.randint(0, self.current_board_size)
            y = np.random.randint(0, self.current_board_size)
            self.mines.add((x, y))
        # Fill board with adjacent mine counts
        for x in range(self.current_board_size):
            for y in range(self.current_board_size):
                if (x, y) in self.mines:
                    self.board[x, y] = -1
                else:
                    self.board[x, y] = self._count_adjacent_mines(x, y)
        self.game_over = False
        self.won = False
        self.revealed_count = 0
        self.flagged_mines = 0
        self.incorrect_flags = 0
        # Log episode start with debug level
        self.logger.debug(f"\nStarting new episode:")
        self.logger.debug(f"- Board size: {self.current_board_size}x{self.current_board_size}")
        self.logger.debug(f"- Mines: {self.current_mines} (density: {self.current_mines/(self.current_board_size*self.current_board_size):.1%})")
        self.logger.debug(f"- Win rate: {(self.win_count/self.total_games*100 if self.total_games > 0 else 0):.1f}%")
        self.logger.debug(f"- Games at current size: {self.games_at_current_size}")
        self.logger.debug(f"- Max steps: {self.max_steps}")
        self._check_training_health()
        self._update_progress_display()
        return self._get_obs(), {}

    def step(self, action):
        current_time = time.time()
        time_since_last_action = current_time - self.last_action_time
        self.last_action_time = current_time
        if self.game_over:
            self.recent_episode_lengths.append(self.steps)
            if self.won:
                self.wins_at_current_size += 1
            return self._get_obs(), 0, True, True, {}
        self.steps += 1
        if self.steps >= self.max_steps:
            episode_duration = current_time - self.episode_start_time
            self.logger.debug(f"Episode timeout after {episode_duration:.1f}s")
            self.recent_episode_lengths.append(self.steps)
            return self._get_obs(), -5, True, False, {}
        N = self.max_board_size * self.max_board_size
        is_flag = action >= N
        idx = action % N
        x = idx // self.max_board_size
        y = idx % self.max_board_size
        if time_since_last_action > 1.0:
            self.logger.warning(f"Slow action detected: {time_since_last_action:.1f}s")
        if x >= self.current_board_size or y >= self.current_board_size:
            return self._get_obs(), -1, False, False, {}
        if not is_flag:
            if self.flags[x, y]:
                return self._get_obs(), -0.5, False, False, {}
            if self.state[x, y] != -1:
                return self._get_obs(), -0.2, False, False, {}
            if (x, y) in self.mines:
                self.state[x, y] = -2
                self.game_over = True
                self.total_games += 1
                self.consecutive_mine_hits += 1
                self.recent_mine_hits.append(True)
                episode_duration = current_time - self.episode_start_time
                self.logger.debug(f"Hit mine after {episode_duration:.1f}s")
                self.recent_episode_lengths.append(self.steps)
                return self._get_obs(), self.mine_penalty, True, False, {}
            self._reveal_cell(x, y)
            safe_reward = self.safe_reveal_base
            if self.board[x, y] > 0:
                safe_reward += self.number_reveal_bonus
            if self._check_win():
                self.won = True
                self.game_over = True
                self.win_count += 1
                self.total_games += 1
                self.consecutive_mine_hits = 0
                self.last_win_time = current_time
                self.recent_mine_hits.append(False)
                episode_duration = current_time - self.episode_start_time
                self.logger.info(f"🎉 Won game after {episode_duration:.1f}s")
                self.recent_episode_lengths.append(self.steps)
                return self._get_obs(), self.win_reward, True, False, {}
            revealed_cells = np.sum((self.state != -1) & (~self.flags))
            total_cells = self.current_board_size * self.current_board_size
            progress_reward = (revealed_cells / total_cells) * self.progress_multiplier
            reward = safe_reward + progress_reward
            self.recent_rewards.append(reward)
            return self._get_obs(), reward, False, False, {}
        else:
            if self.state[x, y] != -1:
                return self._get_obs(), -0.2, False, False, {}
            if self.flags[x, y]:
                return self._get_obs(), -0.2, False, False, {}
            self.flags[x, y] = True
            if (x, y) in self.mines:
                self.flagged_mines += 1
                reward = self.correct_flag_reward
            else:
                self.incorrect_flags += 1
                reward = self.incorrect_flag_penalty
            if self._check_win():
                self.won = True
                self.game_over = True
                self.win_count += 1
                self.total_games += 1
                self.consecutive_mine_hits = 0
                self.last_win_time = current_time
                self.recent_mine_hits.append(False)
                episode_duration = current_time - self.episode_start_time
                self.logger.info(f"🎉 Won game after {episode_duration:.1f}s")
                self.recent_episode_lengths.append(self.steps)
                return self._get_obs(), reward, False, False, {}
            self.recent_rewards.append(reward)
            return self._get_obs(), reward, False, False, {}

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

    def _reveal_cell(self, x, y):
        queue = [(x, y)]
        while queue:
            x, y = queue.pop(0)
            if not (0 <= x < self.current_board_size and 0 <= y < self.current_board_size):
                continue
            if self.state[x, y] != -1 or self.flags[x, y]:
                continue
            self.state[x, y] = self.board[x, y]
            self.revealed_count += 1
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
        obs = np.full((self.current_board_size, self.current_board_size), -3, dtype=np.int8)
        obs[:self.current_board_size, :self.current_board_size] = self.state
        mask = (self.flags & (self.state == -1))
        obs[:self.current_board_size, :self.current_board_size][mask] = 9
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