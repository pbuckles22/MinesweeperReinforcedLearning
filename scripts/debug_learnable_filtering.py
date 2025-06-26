#!/usr/bin/env python3
"""
Debug Learnable Filtering

This script helps debug why the learnable configuration filtering isn't working as expected.
"""

import numpy as np
from pathlib import Path

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.minesweeper_env import MinesweeperEnv

def debug_learnable_filtering():
    """Debug the learnable filtering logic."""
    
    print("🔍 Debugging Learnable Configuration Filtering")
    print("=" * 50)
    
    # Test 4×4 (1 mine) with learnable_only=True
    print("\n📊 Testing 4×4 (1 mine) with learnable_only=True")
    print("-" * 40)
    
    learnable_count = 0
    total_tests = 50
    
    for i in range(total_tests):
        env = MinesweeperEnv(
            initial_board_size=(4, 4),
            initial_mines=1,
            learnable_only=True
        )
        
        # Get board statistics
        stats = env.get_board_statistics()
        
        if stats['learnable_configuration']:
            learnable_count += 1
        else:
            print(f"Non-learnable config found: {stats['mine_positions']}")
    
    learnable_percentage = (learnable_count / total_tests) * 100
    print(f"Learnable configurations: {learnable_count}/{total_tests} ({learnable_percentage:.1f}%)")
    
    # Test 4×4 (1 mine) with learnable_only=False
    print("\n📊 Testing 4×4 (1 mine) with learnable_only=False")
    print("-" * 40)
    
    random_learnable_count = 0
    total_random_tests = 50
    
    for i in range(total_random_tests):
        env = MinesweeperEnv(
            initial_board_size=(4, 4),
            initial_mines=1,
            learnable_only=False
        )
        
        # Get board statistics
        stats = env.get_board_statistics()
        
        if stats['learnable_configuration']:
            random_learnable_count += 1
        else:
            print(f"Non-learnable config found: {stats['mine_positions']}")
    
    random_learnable_percentage = (random_learnable_count / total_random_tests) * 100
    print(f"Learnable configurations: {random_learnable_count}/{total_random_tests} ({random_learnable_percentage:.1f}%)")
    
    # Test individual positions manually
    print("\n🔍 Testing Individual Positions Manually")
    print("-" * 40)
    
    # Test corner positions
    corner_positions = [(0, 0), (0, 3), (3, 0), (3, 3)]
    for pos in corner_positions:
        env = MinesweeperEnv(
            initial_board_size=(4, 4),
            initial_mines=1,
            learnable_only=False
        )
        
        # Manually place mine at corner position
        env.mines.fill(False)
        env.mines[pos[0], pos[1]] = True
        env._update_adjacent_counts()
        
        stats = env.get_board_statistics()
        print(f"Corner position {pos}: {'❌' if not stats['learnable_configuration'] else '✅'} (should be ❌)")
    
    # Test edge positions
    edge_positions = [(0, 1), (1, 0), (2, 3), (3, 2)]
    for pos in edge_positions:
        env = MinesweeperEnv(
            initial_board_size=(4, 4),
            initial_mines=1,
            learnable_only=False
        )
        
        # Manually place mine at edge position
        env.mines.fill(False)
        env.mines[pos[0], pos[1]] = True
        env._update_adjacent_counts()
        
        stats = env.get_board_statistics()
        print(f"Edge position {pos}: {'✅' if stats['learnable_configuration'] else '❌'} (should be ✅)")
    
    # Test center positions
    center_positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for pos in center_positions:
        env = MinesweeperEnv(
            initial_board_size=(4, 4),
            initial_mines=1,
            learnable_only=False
        )
        
        # Manually place mine at center position
        env.mines.fill(False)
        env.mines[pos[0], pos[1]] = True
        env._update_adjacent_counts()
        
        stats = env.get_board_statistics()
        print(f"Center position {pos}: {'✅' if stats['learnable_configuration'] else '❌'} (should be ✅)")

if __name__ == "__main__":
    debug_learnable_filtering() 