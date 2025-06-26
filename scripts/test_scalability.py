#!/usr/bin/env python3
"""
Test Scalability of Cascade Simulation

This script tests our cascade simulation function across different board sizes
and shapes to ensure it works reliably for squares and rectangles.
"""

import numpy as np
import time
from pathlib import Path

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.minesweeper_env import MinesweeperEnv

def test_board_scalability():
    """Test cascade simulation across different board sizes and shapes."""
    
    print("🧪 Testing Cascade Simulation Scalability")
    print("=" * 60)
    
    # Test configurations: (height, width, description)
    test_configs = [
        # Square boards
        (4, 4, "4×4 Square"),
        (5, 5, "5×5 Square"), 
        (6, 6, "6×6 Square"),
        (7, 7, "7×7 Square"),
        (8, 8, "8×8 Square"),
        (9, 9, "9×9 Square"),
        
        # Rectangular boards
        (4, 5, "4×5 Rectangle"),
        (4, 6, "4×6 Rectangle"),
        (4, 7, "4×7 Rectangle"),
        (5, 4, "5×4 Rectangle"),
        (5, 6, "5×6 Rectangle"),
        (5, 7, "5×7 Rectangle"),
        (6, 4, "6×4 Rectangle"),
        (6, 5, "6×5 Rectangle"),
        (7, 4, "7×4 Rectangle"),
        (7, 5, "7×5 Rectangle"),
    ]
    
    results = {}
    
    for height, width, desc in test_configs:
        print(f"\n📊 Testing {desc} ({height}×{width})")
        print("-" * 40)
        
        start_time = time.time()
        
        # Test a few key positions for each board
        test_positions = [
            (0, 0),  # Top-left corner
            (0, width-1),  # Top-right corner
            (height-1, 0),  # Bottom-left corner
            (height-1, width-1),  # Bottom-right corner
            (height//2, width//2),  # Center
        ]
        
        board_results = {}
        
        for mine_pos in test_positions:
            try:
                # Create environment with this board size
                env = MinesweeperEnv(
                    initial_board_size=(height, width),
                    initial_mines=1,
                    learnable_only=False  # Use random mode to test specific positions
                )
                
                # Manually place mine at specific position
                env.mines.fill(False)
                env.mines[mine_pos[0], mine_pos[1]] = True
                env._update_adjacent_counts()
                
                # Test learnable configuration
                stats = env.get_board_statistics()
                is_learnable = stats['learnable_configuration']
                
                board_results[mine_pos] = is_learnable
                
                status = "✅ LEARNABLE" if is_learnable else "❌ 1-MOVE WIN"
                print(f"  Mine at {mine_pos}: {status}")
                
            except Exception as e:
                print(f"  Mine at {mine_pos}: ❌ ERROR - {e}")
                board_results[mine_pos] = None
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        results[desc] = {
            'board_size': (height, width),
            'test_positions': board_results,
            'execution_time': execution_time,
            'success': all(v is not None for v in board_results.values())
        }
        
        print(f"  Execution time: {execution_time:.3f}s")
        print(f"  Success: {'✅' if results[desc]['success'] else '❌'}")
    
    # Summary
    print(f"\n📋 Scalability Test Summary")
    print("=" * 60)
    
    successful_tests = 0
    total_tests = len(test_configs)
    
    for desc, result in results.items():
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"{desc}: {status} ({result['execution_time']:.3f}s)")
        if result['success']:
            successful_tests += 1
    
    print(f"\nOverall: {successful_tests}/{total_tests} tests passed")
    
    # Test edge cases
    print(f"\n🔍 Testing Edge Cases")
    print("=" * 60)
    
    edge_cases = [
        (2, 2, "2×2 Tiny"),
        (3, 3, "3×3 Small"),
        (10, 10, "10×10 Large"),
        (3, 5, "3×5 Tall Rectangle"),
        (5, 3, "5×3 Wide Rectangle"),
    ]
    
    for height, width, desc in edge_cases:
        try:
            env = MinesweeperEnv(
                initial_board_size=(height, width),
                initial_mines=1,
                learnable_only=True
            )
            stats = env.get_board_statistics()
            print(f"{desc}: ✅ SUCCESS - {stats['learnable_configuration']} learnable")
        except Exception as e:
            print(f"{desc}: ❌ ERROR - {e}")

def test_performance():
    """Test performance with larger boards."""
    
    print(f"\n⚡ Performance Test")
    print("=" * 60)
    
    # Test larger boards to see performance impact
    large_boards = [(8, 8), (10, 10), (12, 12), (15, 15)]
    
    for height, width in large_boards:
        print(f"\nTesting {height}×{width} board:")
        
        start_time = time.time()
        
        # Test 10 random configurations
        for i in range(10):
            env = MinesweeperEnv(
                initial_board_size=(height, width),
                initial_mines=1,
                learnable_only=True
            )
            stats = env.get_board_statistics()
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 10
        
        print(f"  Average time per configuration: {avg_time:.3f}s")
        print(f"  Total time for 10 configs: {end_time - start_time:.3f}s")

if __name__ == "__main__":
    test_board_scalability()
    test_performance() 