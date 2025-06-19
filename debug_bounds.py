#!/usr/bin/env python3
"""Debug observation space bounds."""

import sys
sys.path.append('src')

import numpy as np
from core.minesweeper_env import MinesweeperEnv

def debug_bounds():
    """Debug the observation space bounds."""
    print("🔍 Debugging observation space bounds...")
    
    env = MinesweeperEnv(max_board_size=4, max_mines=2)
    state, info = env.reset()
    
    obs_space = env.observation_space
    print(f"✅ Observation space shape: {obs_space.shape}")
    print(f"✅ Observation space low shape: {obs_space.low.shape}")
    print(f"✅ Observation space high shape: {obs_space.high.shape}")
    
    print(f"✅ Channel 0 low: {obs_space.low[0, 0, 0]}")
    print(f"✅ Channel 0 high: {obs_space.high[0, 0, 0]}")
    print(f"✅ Channel 1 low: {obs_space.low[1, 0, 0]}")
    print(f"✅ Channel 1 high: {obs_space.high[1, 0, 0]}")
    
    # Check if bounds are correct
    assert obs_space.low[0, 0, 0] == -4, f"Channel 0 low should be -4, got {obs_space.low[0, 0, 0]}"
    assert obs_space.high[0, 0, 0] == 8, f"Channel 0 high should be 8, got {obs_space.high[0, 0, 0]}"
    assert obs_space.low[1, 0, 0] == -1, f"Channel 1 low should be -1, got {obs_space.low[1, 0, 0]}"
    assert obs_space.high[1, 0, 0] == 8, f"Channel 1 high should be 8, got {obs_space.high[1, 0, 0]}"
    
    print("🎉 Bounds are correct!")

if __name__ == "__main__":
    debug_bounds() 