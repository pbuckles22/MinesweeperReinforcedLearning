#!/usr/bin/env python3
"""
Focused test to understand learnable environment behavior.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from core.minesweeper_env import MinesweeperEnv

def test_learnable_behavior():
    """Test what the learnable environment actually does."""
    print("🔍 Testing Learnable Environment Behavior")
    print("=" * 50)
    
    board_size = (4, 4)
    mines = 1
    num_tests = 50
    
    print(f"\nTesting {num_tests} boards with learnable_only=True:")
    
    instant_wins = 0
    first_move_mine_hits = 0
    normal_games = 0
    
    for i in range(num_tests):
        env = MinesweeperEnv(
            max_board_size=board_size,
            initial_board_size=board_size,
            max_mines=mines,
            initial_mines=mines,
            learnable_only=True,
            max_learnable_attempts=1000
        )
        
        obs, info = env.reset()
        
        if not info.get('learnable', False):
            print(f"   Board {i+1}: Not learnable (fallback to random)")
            continue
        
        # Test first move
        action = 0  # Corner (0,0)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if info.get('won', False):
            instant_wins += 1
            print(f"   Board {i+1}: Instant win (should not happen in learnable)")
        elif terminated and reward == -20:
            first_move_mine_hits += 1
            print(f"   Board {i+1}: First move mine hit (should not happen if environment adjusts)")
        else:
            normal_games += 1
    
    print(f"\n📊 Results:")
    print(f"   Instant wins: {instant_wins}")
    print(f"   First-move mine hits: {first_move_mine_hits}")
    print(f"   Normal games: {normal_games}")
    print(f"   Total learnable boards: {instant_wins + first_move_mine_hits + normal_games}")
    
    print(f"\n🔍 Analysis:")
    if instant_wins == 0:
        print(f"   ✅ Environment correctly filters instant wins")
    else:
        print(f"   ❌ Environment allows instant wins")
    
    if first_move_mine_hits == 0:
        print(f"   ✅ Environment ensures safe first moves (adjusts mine placement)")
    else:
        print(f"   ⚠️  Environment allows first-move mine hits (doesn't adjust)")
    
    return instant_wins == 0, first_move_mine_hits == 0

def test_random_behavior():
    """Test random environment for comparison."""
    print(f"\n🔍 Testing Random Environment Behavior")
    print("=" * 50)
    
    board_size = (4, 4)
    mines = 1
    num_tests = 50
    
    print(f"\nTesting {num_tests} boards with learnable_only=False:")
    
    instant_wins = 0
    first_move_mine_hits = 0
    normal_games = 0
    
    for i in range(num_tests):
        env = MinesweeperEnv(
            max_board_size=board_size,
            initial_board_size=board_size,
            max_mines=mines,
            initial_mines=mines,
            learnable_only=False
        )
        
        obs, info = env.reset()
        
        # Test first move
        action = 0  # Corner (0,0)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if info.get('won', False):
            instant_wins += 1
        elif terminated and reward == -20:
            first_move_mine_hits += 1
        else:
            normal_games += 1
    
    print(f"\n📊 Results:")
    print(f"   Instant wins: {instant_wins}")
    print(f"   First-move mine hits: {first_move_mine_hits}")
    print(f"   Normal games: {normal_games}")
    
    expected_mine_hit_rate = mines / (board_size[0] * board_size[1]) * 100
    actual_mine_hit_rate = (first_move_mine_hits / num_tests) * 100
    
    print(f"   Expected mine hit rate: {expected_mine_hit_rate:.1f}%")
    print(f"   Actual mine hit rate: {actual_mine_hit_rate:.1f}%")

def main():
    """Run the behavior tests."""
    print("🧪 Learnable Environment Behavior Analysis")
    print("=" * 60)
    
    # Test learnable environment
    filters_instant_wins, ensures_safe_moves = test_learnable_behavior()
    
    # Test random environment
    test_random_behavior()
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    print(f"Learnable Environment:")
    print(f"  - Filters instant wins: {'✅ Yes' if filters_instant_wins else '❌ No'}")
    print(f"  - Ensures safe first moves: {'✅ Yes' if ensures_safe_moves else '❌ No'}")
    
    if filters_instant_wins and ensures_safe_moves:
        print(f"\n🎯 Current behavior: Environment does BOTH filtering and mine adjustment")
        print(f"   This is more robust than just filtering instant wins")
    elif filters_instant_wins and not ensures_safe_moves:
        print(f"\n🎯 Current behavior: Environment only filters instant wins")
        print(f"   First-move mine hits are possible and should be discarded by training")
    else:
        print(f"\n⚠️  Current behavior: Environment has issues")

if __name__ == "__main__":
    main() 