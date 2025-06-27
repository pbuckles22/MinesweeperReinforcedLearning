#!/usr/bin/env python3
"""
Test the wrapped environment with FirstMoveDiscardWrapper.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from core.minesweeper_env import MinesweeperEnv
from core.train_agent import FirstMoveDiscardWrapper

def test_wrapped_environment():
    """Test the wrapped environment with first-move discard logic."""
    print("🔍 Testing Wrapped Environment with FirstMoveDiscardWrapper")
    print("=" * 60)
    
    board_size = (4, 4)
    mines = 1
    episodes = 100
    
    print(f"\nTesting {episodes} episodes with wrapped environment:")
    
    # Create base environment
    base_env = MinesweeperEnv(
        max_board_size=board_size,
        initial_board_size=board_size,
        max_mines=mines,
        initial_mines=mines,
        learnable_only=True,
        max_learnable_attempts=1000
    )
    
    # Wrap with FirstMoveDiscardWrapper
    wrapped_env = FirstMoveDiscardWrapper(base_env, learnable_only=True)
    
    # Simulate training episodes
    for episode in range(episodes):
        obs, info = wrapped_env.reset()
        
        if not info.get('learnable', False):
            continue
        
        # Simulate first move
        action = 0  # Corner (0,0)
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        
        # Continue with random play if not terminated
        if not terminated:
            steps = 1  # Already made first move
            while steps < 10:  # Limit steps
                action = wrapped_env.action_space.sample()
                obs, reward, terminated, truncated, info = wrapped_env.step(action)
                steps += 1
                
                if info.get('won', False) or terminated or truncated:
                    break
    
    # Get discard statistics
    stats = wrapped_env.get_discard_stats()
    
    print(f"\n📊 Wrapped Environment Results:")
    print(f"   Episodes discarded: {stats['episodes_discarded']}")
    print(f"   Episodes counted: {stats['episodes_counted']}")
    print(f"   First-move mine hits: {stats['first_move_mine_hits']}")
    print(f"   Discard rate: {stats['discard_rate']:.1f}%")
    
    if stats['episodes_discarded'] > 0:
        print(f"   ✅ PASSED: Wrapper correctly discarded first-move mine hits")
    else:
        print(f"   ⚠️  WARNING: No episodes discarded (environment may be generating safe boards)")
    
    return stats['episodes_discarded'] > 0

def test_raw_vs_wrapped():
    """Compare raw environment vs wrapped environment."""
    print(f"\n🔍 Testing Raw vs Wrapped Environment")
    print("=" * 50)
    
    board_size = (4, 4)
    mines = 1
    episodes = 50
    
    print(f"\nTesting {episodes} episodes each:")
    
    # Test raw environment
    raw_env = MinesweeperEnv(
        max_board_size=board_size,
        initial_board_size=board_size,
        max_mines=mines,
        initial_mines=mines,
        learnable_only=True,
        max_learnable_attempts=1000
    )
    
    raw_first_move_mine_hits = 0
    raw_episodes = 0
    
    for episode in range(episodes):
        obs, info = raw_env.reset()
        
        if not info.get('learnable', False):
            continue
        
        raw_episodes += 1
        
        # Test first move
        action = 0  # Corner (0,0)
        obs, reward, terminated, truncated, info = raw_env.step(action)
        
        if terminated and reward == -20:  # Mine hit
            raw_first_move_mine_hits += 1
    
    # Test wrapped environment
    wrapped_env = FirstMoveDiscardWrapper(raw_env, learnable_only=True)
    
    wrapped_first_move_mine_hits = 0
    wrapped_episodes = 0
    
    for episode in range(episodes):
        obs, info = wrapped_env.reset()
        
        if not info.get('learnable', False):
            continue
        
        wrapped_episodes += 1
        
        # Test first move
        action = 0  # Corner (0,0)
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        
        if terminated and reward == -20:  # Mine hit
            wrapped_first_move_mine_hits += 1
    
    print(f"\n📊 Comparison Results:")
    print(f"   Raw environment first-move mine hits: {raw_first_move_mine_hits}")
    print(f"   Wrapped environment first-move mine hits: {wrapped_first_move_mine_hits}")
    print(f"   Raw episodes: {raw_episodes}")
    print(f"   Wrapped episodes: {wrapped_episodes}")
    
    if raw_first_move_mine_hits > 0 and wrapped_first_move_mine_hits == 0:
        print(f"   ✅ PASSED: Wrapper correctly prevents first-move mine hits from being counted")
    elif raw_first_move_mine_hits == 0:
        print(f"   ⚠️  WARNING: Raw environment generates no first-move mine hits")
    else:
        print(f"   ❌ FAILED: Wrapper not working correctly")

def main():
    """Run the wrapped environment tests."""
    print("🧪 Wrapped Environment Test")
    print("=" * 60)
    
    # Test wrapped environment
    wrapper_working = test_wrapped_environment()
    
    # Test raw vs wrapped
    test_raw_vs_wrapped()
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    if wrapper_working:
        print("✅ FirstMoveDiscardWrapper is working correctly")
        print("   Training script will discard first-move mine hits")
    else:
        print("⚠️  FirstMoveDiscardWrapper may not be needed")
        print("   Environment appears to generate safe first moves")
    
    print("\n🎯 Current System:")
    print("   1. Environment: Pure filtering (learnable + safe first moves)")
    print("   2. Training: FirstMoveDiscardWrapper (safety net for rare cases)")

if __name__ == "__main__":
    main() 