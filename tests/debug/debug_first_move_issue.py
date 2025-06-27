#!/usr/bin/env python3
"""
Debug script to investigate first-move mine hits when _has_safe_first_move() returns True.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from core.minesweeper_env import MinesweeperEnv

def debug_first_move_issue():
    """Debug why first-move mine hits occur when _has_safe_first_move() returns True."""
    print("🔍 Debugging First-Move Mine Hit Issue")
    print("=" * 60)
    
    # Test configuration that showed issues in audit
    board_size = (4, 4)
    mine_count = 2
    num_tests = 100
    
    print(f"Testing {board_size[0]}x{board_size[1]} with {mine_count} mines")
    print(f"Running {num_tests} tests to find problematic cases")
    
    problematic_cases = []
    
    for test_num in range(num_tests):
        env = MinesweeperEnv(
            max_board_size=board_size,
            initial_board_size=board_size,
            max_mines=mine_count,
            initial_mines=mine_count,
            learnable_only=True,
            max_learnable_attempts=1000
        )
        
        obs, info = env.reset()
        
        if not info.get('learnable', False):
            continue
        
        # Check if first move is safe
        has_safe_first_move = env._has_safe_first_move()
        
        # Test first move (action 0 = corner 0,0)
        action = 0
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check if first move hit a mine
        first_move_hit_mine = terminated and reward == -20
        
        if first_move_hit_mine:
            # This is a problematic case
            mine_positions = [(y, x) for y in range(env.current_board_height) 
                             for x in range(env.current_board_width) if env.mines[y, x]]
            
            case = {
                'test_num': test_num,
                'has_safe_first_move': has_safe_first_move,
                'mine_positions': mine_positions,
                'first_move_position': (0, 0),
                'first_move_hit_mine': first_move_hit_mine
            }
            problematic_cases.append(case)
            
            print(f"\n❌ Problematic Case #{test_num}:")
            print(f"   _has_safe_first_move(): {has_safe_first_move}")
            print(f"   Mine positions: {mine_positions}")
            print(f"   First move position: (0, 0)")
            print(f"   First move hit mine: {first_move_hit_mine}")
            
            # Check what safe positions are available
            safe_positions = env._get_safe_positions()
            actual_safe_positions = [pos for pos in safe_positions if not env.mines[pos[0], pos[1]]]
            print(f"   Available safe positions: {actual_safe_positions}")
            
            # Check if (0,0) specifically is safe
            corner_0_0_safe = not env.mines[0, 0]
            print(f"   Corner (0,0) safe: {corner_0_0_safe}")
            
            if len(problematic_cases) >= 5:  # Limit output
                break
    
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}")
    
    if problematic_cases:
        print(f"Found {len(problematic_cases)} problematic cases")
        print("\nRoot cause analysis:")
        
        # Analyze the problematic cases
        for case in problematic_cases:
            if case['has_safe_first_move'] and case['first_move_hit_mine']:
                print(f"   ❌ Case #{case['test_num']}: _has_safe_first_move() returned True but first move hit mine")
                print(f"      This suggests a logic error in _has_safe_first_move() or mine placement")
                
                # Check if the issue is that (0,0) is not in the safe positions
                mine_positions = case['mine_positions']
                if (0, 0) in mine_positions:
                    print(f"      Issue: (0,0) is a mine, but _has_safe_first_move() returned True")
                    print(f"      This means _has_safe_first_move() found a safe position elsewhere")
                    print(f"      But the audit always uses action 0 (0,0) as first move")
    else:
        print("No problematic cases found in this sample")
        print("The issue might be rare or the sample size is too small")

def test_safe_move_logic():
    """Test the safe move logic more thoroughly."""
    print(f"\n{'='*60}")
    print("Testing Safe Move Logic")
    print(f"{'='*60}")
    
    env = MinesweeperEnv(
        max_board_size=(4, 4),
        initial_board_size=(4, 4),
        max_mines=2,
        initial_mines=2,
        learnable_only=False  # Don't use learnable filtering for this test
    )
    
    # Test specific mine configurations
    test_configs = [
        {"mines": [(0, 0), (1, 1)], "name": "Corner + Center"},
        {"mines": [(0, 0), (0, 1)], "name": "Two corners"},
        {"mines": [(1, 1), (2, 2)], "name": "Two centers"},
        {"mines": [(0, 1), (1, 0)], "name": "Adjacent to corner"},
    ]
    
    for config in test_configs:
        print(f"\nTesting: {config['name']}")
        print(f"Mine positions: {config['mines']}")
        
        # Place mines manually
        env._place_mines_at_positions(config['mines'])
        
        # Check safe first move
        has_safe_first_move = env._has_safe_first_move()
        print(f"_has_safe_first_move(): {has_safe_first_move}")
        
        # Get safe positions
        safe_positions = env._get_safe_positions()
        actual_safe_positions = [pos for pos in safe_positions if not env.mines[pos[0], pos[1]]]
        print(f"Available safe positions: {actual_safe_positions}")
        
        # Test first move
        env.reset()
        action = 0  # Corner (0,0)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if info.get('won', False):
            print(f"First move result: Instant win")
        elif terminated and reward == -20:
            print(f"First move result: Mine hit")
        else:
            print(f"First move result: Safe")
        
        # Check if (0,0) is safe
        corner_0_0_safe = not env.mines[0, 0]
        print(f"Corner (0,0) safe: {corner_0_0_safe}")

def main():
    """Run the debug tests."""
    debug_first_move_issue()
    test_safe_move_logic()
    
    print(f"\n{'='*60}")
    print("DEBUG SUMMARY")
    print(f"{'='*60}")
    print("The issue is likely that _has_safe_first_move() checks for ANY safe position")
    print("but the audit always uses action 0 (0,0) as the first move.")
    print("If (0,0) is a mine but other corners/edges are safe, this creates a mismatch.")

if __name__ == "__main__":
    main() 