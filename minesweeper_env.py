import gymnasium as gym
import numpy as np
from gymnasium import spaces

class MinesweeperEnv(gym.Env):
    """
    A Minesweeper environment for reinforcement learning.
    """
    def __init__(self, board_size=8, num_mines=10):
        super(MinesweeperEnv, self).__init__()
        
        # Game parameters
        self.board_size = board_size
        self.num_mines = num_mines
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(2 * board_size * board_size)  # Each cell has reveal and flag actions
        self.observation_space = spaces.Box(
            low=-1,  # -1 for unrevealed cells
            high=8,  # 0-8 for revealed cells (0-8 adjacent mines)
            shape=(board_size, board_size),
            dtype=np.int8
        )
        
        # Initialize game state
        self.board = None  # The actual board with mines
        self.state = None  # The visible state to the agent
        self.mines = None  # Positions of mines
        self.flags = set()  # Positions of flagged cells
        self.game_over = False
        self.won = False
        
    def reset(self, seed=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Initialize empty board
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.state = np.full((self.board_size, self.board_size), -1, dtype=np.int8)
        
        # Place mines randomly
        self.mines = set()
        while len(self.mines) < self.num_mines:
            x = self.np_random.integers(0, self.board_size)
            y = self.np_random.integers(0, self.board_size)
            if (x, y) not in self.mines:
                self.mines.add((x, y))
                self.board[x, y] = -1  # -1 represents a mine
        
        # Calculate numbers for non-mine cells
        for x in range(self.board_size):
            for y in range(self.board_size):
                if (x, y) not in self.mines:
                    self.board[x, y] = self._count_adjacent_mines(x, y)
        
        self.flags = set()
        self.game_over = False
        self.won = False
        
        return self.state, {}
    
    def step(self, action):
        """Take a step in the environment."""
        if self.game_over:
            return self.state, 0, True, True, {}
        
        # Convert action to coordinates and type (reveal or flag)
        cell_action = action % (self.board_size * self.board_size)
        x = cell_action // self.board_size
        y = cell_action % self.board_size
        is_flag = action >= self.board_size * self.board_size
        
        # Handle flagging
        if is_flag:
            if (x, y) in self.flags:
                self.flags.remove((x, y))
            else:
                self.flags.add((x, y))
            return self.state, 0, False, False, {}
        
        # Check if action is valid
        if self.state[x, y] != -1 or (x, y) in self.flags:
            return self.state, -1, True, False, {}  # Invalid move
        
        # Reveal the cell
        if (x, y) in self.mines:
            # Hit a mine
            self.state[x, y] = -2  # -2 represents a revealed mine
            self.game_over = True
            return self.state, -10, True, False, {}
        
        # Reveal the cell and its neighbors if it's a 0
        self._reveal_cell(x, y)
        
        # Check if won
        if self._check_win():
            self.won = True
            self.game_over = True
            return self.state, 10, True, False, {}
        
        return self.state, 1, False, False, {}
    
    def _count_adjacent_mines(self, x, y):
        """Count the number of mines adjacent to a cell."""
        count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                    if (nx, ny) in self.mines:
                        count += 1
        return count
    
    def _reveal_cell(self, x, y):
        """Reveal a cell and its neighbors if it's a 0."""
        if not (0 <= x < self.board_size and 0 <= y < self.board_size):
            return
        if self.state[x, y] != -1:
            return
        
        self.state[x, y] = self.board[x, y]
        
        if self.board[x, y] == 0:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    self._reveal_cell(x + dx, y + dy)
    
    def _check_win(self):
        """Check if all non-mine cells are revealed."""
        for x in range(self.board_size):
            for y in range(self.board_size):
                if (x, y) not in self.mines and self.state[x, y] == -1:
                    return False
        return True
    
    def render(self):
        """Render the current state of the environment."""
        for x in range(self.board_size):
            for y in range(self.board_size):
                if (x, y) in self.flags:
                    print('🚩', end=' ')
                elif self.state[x, y] == -1:
                    print('□', end=' ')
                elif self.state[x, y] == -2:
                    print('💣', end=' ')
                else:
                    print(self.state[x, y], end=' ')
            print()

def main():
    # Create and test the environment
    env = MinesweeperEnv(board_size=8, num_mines=10)
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