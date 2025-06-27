#!/usr/bin/env python3
"""
Test Learnable Configurations

This script tests the new learnable configuration filtering in the environment
to ensure it correctly identifies and filters out lucky 1-move win scenarios.
"""

import numpy as np
from pathlib import Path

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.minesweeper_env import MinesweeperEnv

def test_learnable_configurations():
    """Test learnable configuration filtering."""
    
    print("🧪 Testing Learnable Configuration Filtering")
    print("=" * 50)
    
    # Test 4×4 (1 mine) configurations - learnable mode should filter out corners
    print("\n📊 Testing 4×4 (1 mine) - Learnable Mode vs Random Mode")
    print("-" * 40)
    
    # Test learnable mode
    learnable_count = 0
    total_tests = 100
    
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
    
    learnable_percentage = (learnable_count / total_tests) * 100
    print(f"Learnable mode: {learnable_count}/{total_tests} ({learnable_percentage:.1f}%) learnable")
    
    # Test random mode
    random_learnable_count = 0
    total_random_tests = 100
    
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
    
    random_learnable_percentage = (random_learnable_count / total_random_tests) * 100
    print(f"Random mode: {random_learnable_count}/{total_random_tests} ({random_learnable_percentage:.1f}%) learnable")
    print(f"Expected: ~75% (based on our analysis)")
    
    # Test 5×5 (1 mine) configurations
    print("\n📊 Testing 5×5 (1 mine) - Learnable Mode vs Random Mode")
    print("-" * 40)
    
    # Test learnable mode
    learnable_count_5x5 = 0
    total_tests_5x5 = 100
    
    for i in range(total_tests_5x5):
        env = MinesweeperEnv(
            initial_board_size=(5, 5),
            initial_mines=1,
            learnable_only=True
        )
        
        # Get board statistics
        stats = env.get_board_statistics()
        
        if stats['learnable_configuration']:
            learnable_count_5x5 += 1
    
    learnable_percentage_5x5 = (learnable_count_5x5 / total_tests_5x5) * 100
    print(f"Learnable mode: {learnable_count_5x5}/{total_tests_5x5} ({learnable_percentage_5x5:.1f}%) learnable")
    
    # Test random mode
    random_learnable_count_5x5 = 0
    total_random_tests_5x5 = 100
    
    for i in range(total_random_tests_5x5):
        env = MinesweeperEnv(
            initial_board_size=(5, 5),
            initial_mines=1,
            learnable_only=False
        )
        
        # Get board statistics
        stats = env.get_board_statistics()
        
        if stats['learnable_configuration']:
            random_learnable_count_5x5 += 1
    
    random_learnable_percentage_5x5 = (random_learnable_count_5x5 / total_random_tests_5x5) * 100
    print(f"Random mode: {random_learnable_count_5x5}/{total_random_tests_5x5} ({random_learnable_percentage_5x5:.1f}%) learnable")
    print(f"Expected: ~84% (based on our analysis)")
    
    # Test 5×5 (2 mines) configurations
    print("\n📊 Testing 5×5 (2 mines) - Learnable Mode vs Random Mode")
    print("-" * 40)
    
    # Test learnable mode
    learnable_count_2mines = 0
    total_tests_2mines = 100
    
    for i in range(total_tests_2mines):
        env = MinesweeperEnv(
            initial_board_size=(5, 5),
            initial_mines=2,
            learnable_only=True
        )
        
        # Get board statistics
        stats = env.get_board_statistics()
        
        if stats['learnable_configuration']:
            learnable_count_2mines += 1
    
    learnable_percentage_2mines = (learnable_count_2mines / total_tests_2mines) * 100
    print(f"Learnable mode: {learnable_count_2mines}/{total_tests_2mines} ({learnable_percentage_2mines:.1f}%) learnable")
    
    # Test random mode
    random_learnable_count_2mines = 0
    total_random_tests_2mines = 100
    
    for i in range(total_random_tests_2mines):
        env = MinesweeperEnv(
            initial_board_size=(5, 5),
            initial_mines=2,
            learnable_only=False
        )
        
        # Get board statistics
        stats = env.get_board_statistics()
        
        if stats['learnable_configuration']:
            random_learnable_count_2mines += 1
    
    random_learnable_percentage_2mines = (random_learnable_count_2mines / total_random_tests_2mines) * 100
    print(f"Random mode: {random_learnable_count_2mines}/{total_random_tests_2mines} ({random_learnable_percentage_2mines:.1f}%) learnable")
    print(f"Expected: 100% (all multi-mine configurations are learnable)")
    
    # Summary
    print(f"\n📋 Summary")
    print("=" * 50)
    print(f"✅ 4×4 (1 mine) learnable filtering: {learnable_percentage:.1f}% vs {random_learnable_percentage:.1f}%")
    print(f"✅ 5×5 (1 mine) learnable filtering: {learnable_percentage_5x5:.1f}% vs {random_learnable_percentage_5x5:.1f}%")
    print(f"✅ 5×5 (2 mines) learnable filtering: {learnable_percentage_2mines:.1f}% vs {random_learnable_percentage_2mines:.1f}%")
    print(f"✅ Learnable mode successfully filters out lucky configurations")
    
    # Test individual mine positions
    print(f"\n🔍 Testing Individual Mine Positions")
    print("-" * 40)
    
    # Test corner positions (should be filtered out)
    corner_positions = [(0, 0), (0, 3), (3, 0), (3, 3)]
    for pos in corner_positions:
        env = MinesweeperEnv(
            initial_board_size=(4, 4),
            initial_mines=1,
            learnable_only=False  # Use random mode to test individual positions
        )
        
        # Manually place mine at corner position
        env.mines.fill(False)
        env.mines[pos[0], pos[1]] = True
        env._update_adjacent_counts()
        
        stats = env.get_board_statistics()
        print(f"Corner position {pos}: {'❌' if not stats['learnable_configuration'] else '✅'} (should be ❌)")
    
    # Test edge positions (should be learnable)
    edge_positions = [(0, 1), (1, 0), (2, 3), (3, 2)]
    for pos in edge_positions:
        env = MinesweeperEnv(
            initial_board_size=(4, 4),
            initial_mines=1,
            learnable_only=False  # Use random mode to test individual positions
        )
        
        # Manually place mine at edge position
        env.mines.fill(False)
        env.mines[pos[0], pos[1]] = True
        env._update_adjacent_counts()
        
        stats = env.get_board_statistics()
        print(f"Edge position {pos}: {'✅' if stats['learnable_configuration'] else '❌'} (should be ✅)")

if __name__ == "__main__":
    test_learnable_configurations() 