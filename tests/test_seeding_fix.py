#!/usr/bin/env python3
"""
Test script to verify that seeding works correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.core.minesweeper_env import MinesweeperEnv

def test_seeding_consistency():
    """Test that seeding produces consistent results."""
    print("🔍 Testing Seeding Consistency")
    print("=" * 50)
    
    # Test 1: Same seed should produce same board
    print("\nTest 1: Same seed = Same board")
    env1 = MinesweeperEnv(max_board_size=(4, 4), max_mines=1)
    env2 = MinesweeperEnv(max_board_size=(4, 4), max_mines=1)
    
    # Set same seed
    seed = 42
    env1.seed(seed)
    env2.seed(seed)
    
    # Reset both environments
    obs1, info1 = env1.reset()
    obs2, info2 = env2.reset()
    
    print(f"Environment 1 mines: {env1.mines}")
    print(f"Environment 2 mines: {env2.mines}")
    print(f"Mines identical: {np.array_equal(env1.mines, env2.mines)}")
    print(f"Observations identical: {np.array_equal(obs1, obs2)}")
    
    # Test 2: Different seeds should produce different boards
    print("\nTest 2: Different seeds = Different boards")
    env3 = MinesweeperEnv(max_board_size=(4, 4), max_mines=1)
    env4 = MinesweeperEnv(max_board_size=(4, 4), max_mines=1)
    
    env3.seed(42)
    env4.seed(123)
    
    obs3, info3 = env3.reset()
    obs4, info4 = env4.reset()
    
    print(f"Environment 3 mines (seed 42): {env3.mines}")
    print(f"Environment 4 mines (seed 123): {env4.mines}")
    print(f"Mines different: {not np.array_equal(env3.mines, env4.mines)}")
    print(f"Observations different: {not np.array_equal(obs3, obs4)}")
    
    # Test 3: Reset with seed parameter
    print("\nTest 3: Reset with seed parameter")
    env5 = MinesweeperEnv(max_board_size=(4, 4), max_mines=1)
    
    obs5a, info5a = env5.reset(seed=42)
    obs5b, info5b = env5.reset(seed=42)
    
    print(f"Reset 1 mines: {env5.mines}")
    obs5c, info5c = env5.reset(seed=42)
    print(f"Reset 2 mines: {env5.mines}")
    print(f"Resets identical: {np.array_equal(obs5a, obs5c)}")
    
    # Test 4: Multiple resets with same seed
    print("\nTest 4: Multiple resets with same seed")
    env6 = MinesweeperEnv(max_board_size=(4, 4), max_mines=1)
    
    boards = []
    for i in range(5):
        obs, info = env6.reset(seed=42)
        boards.append(env6.mines.copy())
    
    all_identical = all(np.array_equal(boards[0], board) for board in boards)
    print(f"All resets identical: {all_identical}")
    
    if all_identical:
        print("✅ Seeding is working correctly!")
    else:
        print("❌ Seeding is not working correctly!")

if __name__ == "__main__":
    test_seeding_consistency() 