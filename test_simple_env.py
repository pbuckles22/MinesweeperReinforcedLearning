#!/usr/bin/env python3
"""
Simple environment test and training script
"""

from src.core.minesweeper_env import MinesweeperEnv
import numpy as np

print("=== Simple Environment Test ===")
env = MinesweeperEnv(max_board_size=(4, 4), max_mines=2)
print(f"Board size: 4x4")
print(f"Mine count: 2")
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")

# Test a few random episodes
wins = 0
total_episodes = 10
total_steps = 0

for i in range(total_episodes):
    obs, info = env.reset()
    done = False
    steps = 0
    while not done and steps < 20:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        if info.get('won', False):
            wins += 1
            break
    total_steps += steps

print(f"Random play win rate: {wins/total_episodes*100:.1f}%")
print(f"Average steps per episode: {total_steps/total_episodes:.1f}")
print("=== Environment test completed ===") 