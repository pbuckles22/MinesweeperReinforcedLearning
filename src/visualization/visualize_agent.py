import os
import sys
import time
import argparse
import numpy as np
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.core.minesweeper_env import MinesweeperEnv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (192, 192, 192)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
COLORS = {
    1: (0, 0, 255),    # Blue
    2: (0, 128, 0),    # Green
    3: (255, 0, 0),    # Red
    4: (0, 0, 128),    # Dark Blue
    5: (128, 0, 0),    # Dark Red
    6: (0, 128, 128),  # Teal
    7: (0, 0, 0),      # Black
    8: (128, 128, 128) # Gray
}

class MinesweeperVisualizer:
    def __init__(self, board_size=5, num_mines=4, cell_size=60, speed=2):
        self.board_size = board_size
        self.num_mines = num_mines
        self.cell_size = cell_size
        self.speed = speed
        self.window_size = board_size * cell_size
        self.padding = 50  # Space for stats
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size + self.padding))
        pygame.display.set_caption("Minesweeper RL Agent")
        
        # Initialize environment and model
        self.env = DummyVecEnv([lambda: MinesweeperEnv(
            max_board_size=(board_size, board_size), 
            max_mines=num_mines,
            initial_board_size=(board_size, board_size),
            initial_mines=num_mines,
            learnable_only=True,  # Only generate learnable board configurations
            max_learnable_attempts=1000  # Maximum attempts to find learnable configuration
        )])
        self.model = PPO("MlpPolicy", self.env, verbose=0)
        
        # Game state
        self.episode = 0
        self.score = 0
        self.paused = False
        self.running = True
        
        # Fonts
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

    def draw_board(self, obs):
        """Draw the current state of the board"""
        self.screen.fill(WHITE)
        
        # Draw grid
        for i in range(self.board_size):
            for j in range(self.board_size):
                rect = pygame.Rect(j * self.cell_size, i * self.cell_size, 
                                 self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, GRAY, rect, 1)
                
                # Get cell state from channel 0 (game state)
                cell_value = obs[0, i, j]
                if cell_value == -1:  # Unrevealed
                    pygame.draw.rect(self.screen, GRAY, rect)
                elif cell_value == -4:  # Mine hit
                    pygame.draw.circle(self.screen, RED, 
                                    (j * self.cell_size + self.cell_size//2,
                                     i * self.cell_size + self.cell_size//2),
                                    self.cell_size//3)
                elif cell_value >= 0:  # Revealed with number
                    pygame.draw.rect(self.screen, WHITE, rect)
                    if cell_value > 0:
                        text = self.font.render(str(int(cell_value)), True, COLORS.get(int(cell_value), BLACK))
                        text_rect = text.get_rect(center=(j * self.cell_size + self.cell_size//2,
                                                        i * self.cell_size + self.cell_size//2))
                        self.screen.blit(text, text_rect)

        # Draw stats
        stats_text = f"Episode: {self.episode} | Score: {self.score}"
        text = self.small_font.render(stats_text, True, BLACK)
        self.screen.blit(text, (10, self.window_size + 10))

    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.reset_episode()
                elif event.key == pygame.K_n:
                    self.next_episode()
                elif event.key == pygame.K_q:
                    self.running = False

    def reset_episode(self):
        """Reset the current episode"""
        self.env.reset()
        self.score = 0

    def next_episode(self):
        """Start a new episode"""
        self.episode += 1
        self.reset_episode()

    def run(self, episodes=1):
        """Run the visualization"""
        self.episode = 1
        obs, _ = self.env.reset()
        
        while self.running and self.episode <= episodes:
            self.handle_events()
            
            if not self.paused:
                # Get agent's action
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.env.step(action)
                self.score += reward[0]
                
                # Draw current state
                self.draw_board(obs)
                pygame.display.flip()
                
                # Handle episode completion
                if done[0] or truncated[0]:
                    time.sleep(1)  # Pause to show final state
                    if self.episode < episodes:
                        self.next_episode()
                        obs, _ = self.env.reset()
                    else:
                        self.running = False
                
                # Control speed
                time.sleep(1/self.speed)
            else:
                # Still need to handle events and update display when paused
                self.draw_board(obs)
                pygame.display.flip()
                time.sleep(0.1)
        
        pygame.quit()

def main():
    parser = argparse.ArgumentParser(description='Visualize Minesweeper RL Agent')
    parser.add_argument('--model-path', type=str, help='Path to trained model')
    parser.add_argument('--board-size', type=int, default=5, help='Size of the board')
    parser.add_argument('--num-mines', type=int, default=4, help='Number of mines')
    parser.add_argument('--speed', type=int, default=2, help='Game speed (FPS)')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to play')
    args = parser.parse_args()

    visualizer = MinesweeperVisualizer(
        board_size=args.board_size,
        num_mines=args.num_mines,
        speed=args.speed
    )
    
    if args.model_path and os.path.exists(args.model_path):
        visualizer.model = PPO.load(args.model_path)
    
    visualizer.run(episodes=args.episodes)

if __name__ == '__main__':
    main() 