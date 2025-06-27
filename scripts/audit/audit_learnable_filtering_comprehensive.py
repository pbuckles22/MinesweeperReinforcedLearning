#!/usr/bin/env python3
"""
Comprehensive audit of learnable environment filtering.
Tests that learnable environment correctly filters out instant wins and first-move mine hits.

Audits all board sizes from 4×4 to 9×9 with mine counts from 1 to 7.
Estimated time: 6-9 hours for complete audit.
"""
import sys
import os
import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Add the project root to the path so we can import src modules
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.core.minesweeper_env import MinesweeperEnv

def audit_learnable_filtering(board_size, mine_count, num_boards=3000):
    """Audit a specific board size and mine count configuration."""
    print(f"  🎯 Testing {board_size[0]}×{board_size[1]} with {mine_count} mines...")
    
    instant_wins = 0
    non_learnable = 0
    total = 0
    start_time = time.time()
    
    # Track failures for pattern analysis
    failures = []
    
    for i in range(num_boards):
        if i % 500 == 0 and i > 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            remaining = (num_boards - i) / rate
            print(f"    Progress: {i}/{num_boards} ({i/num_boards*100:.1f}%) - ETA: {remaining/60:.1f}min")
        
        try:
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
                non_learnable += 1
                continue
            
            # Test every non-mine position as a first move
            has_instant_win = False
            instant_win_action = None
            instant_win_mine_positions = None
            
            for row in range(env.current_board_height):
                for col in range(env.current_board_width):
                    if not env.mines[row, col]:  # Only test non-mine positions
                        action = row * env.current_board_width + col
                        
                        # Reset to get fresh board
                        obs, info = env.reset()
                        
                        # Make the move
                        obs, reward, terminated, truncated, info = env.step(action)
                        
                        # Check if this move caused an instant win
                        if info.get('won', False):
                            has_instant_win = True
                            instant_win_action = action
                            instant_win_mine_positions = np.where(env.mines)[0].tolist()
                            break
                
                if has_instant_win:
                    break
            
            if has_instant_win:
                instant_wins += 1
                # Record failure details for pattern analysis
                failure_info = {
                    'board_index': i,
                    'board_size': board_size,
                    'mine_count': mine_count,
                    'action': instant_win_action,
                    'action_coords': (instant_win_action // board_size[1], instant_win_action % board_size[1]),
                    'mine_positions': instant_win_mine_positions,
                    'mine_coords': [(pos // board_size[1], pos % board_size[1]) for pos in instant_win_mine_positions],
                    'failure_type': 'instant_win'
                }
                failures.append(failure_info)
            else:
                total += 1
                
        except Exception as e:
            # Record error for pattern analysis
            failure_info = {
                'board_index': i,
                'board_size': board_size,
                'mine_count': mine_count,
                'error': str(e),
                'failure_type': 'exception'
            }
            failures.append(failure_info)
            continue
    
    elapsed_time = time.time() - start_time
    
    return {
        'board_size': board_size,
        'mine_count': mine_count,
        'total': total,
        'instant_wins': instant_wins,
        'non_learnable': non_learnable,
        'elapsed_time': elapsed_time,
        'boards_per_second': num_boards / elapsed_time if elapsed_time > 0 else 0,
        'failures': failures
    }

def analyze_failures(all_failures):
    """Analyze failure patterns across all configurations."""
    if not all_failures:
        return {}
    
    analysis = {
        'total_failures': len(all_failures),
        'failure_types': {},
        'board_size_failures': {},
        'mine_count_failures': {},
        'action_patterns': {},
        'mine_position_patterns': {},
        'error_patterns': {}
    }
    
    for failure in all_failures:
        # Count failure types
        failure_type = failure['failure_type']
        analysis['failure_types'][failure_type] = analysis['failure_types'].get(failure_type, 0) + 1
        
        # Count by board size
        board_size = failure['board_size']
        board_key = f"{board_size[0]}×{board_size[1]}"
        analysis['board_size_failures'][board_key] = analysis['board_size_failures'].get(board_key, 0) + 1
        
        # Count by mine count
        mine_count = failure['mine_count']
        analysis['mine_count_failures'][mine_count] = analysis['mine_count_failures'].get(mine_count, 0) + 1
        
        # Analyze action patterns for instant wins
        if failure_type == 'instant_win':
            action_coords = failure['action_coords']
            analysis['action_patterns'][action_coords] = analysis['action_patterns'].get(action_coords, 0) + 1
            
            # Analyze mine position patterns
            mine_coords = failure['mine_coords']
            for mine_pos in mine_coords:
                analysis['mine_position_patterns'][mine_pos] = analysis['mine_position_patterns'].get(mine_pos, 0) + 1
        
        # Analyze error patterns
        elif failure_type == 'exception':
            error = failure['error']
            analysis['error_patterns'][error] = analysis['error_patterns'].get(error, 0) + 1
    
    return analysis

def main():
    """Main audit function for comprehensive testing."""
    
    # All configurations to test: 4×4 to 9×9 with 1-7 mines
    configs = []
    for board_size in [(4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]:
        max_mines = min(7, board_size[0] * board_size[1] - 1)  # Can't have more mines than cells-1
        for mine_count in range(1, max_mines + 1):
            configs.append((board_size, mine_count))
    
    num_boards = 3000  # Comprehensive audit: 3000 boards per configuration
    total_configs = len(configs)
    
    print(f"🚀 Comprehensive Learnable Filtering Audit")
    print(f"📊 Testing {total_configs} configurations with {num_boards:,} boards each")
    print(f"🎯 Board sizes: 4×4 to 9×9")
    print(f"💣 Mine counts: 1 to 7")
    print(f"⏱️  Estimated time: 6-9 hours")
    print(f"🔍 Testing EVERY position on EVERY board for instant wins")
    print(f"🎯 Goal: Verify learnable_only=True eliminates ALL instant wins")
    print("=" * 80)
    
    # Create results directory
    results_dir = Path("audit_results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"comprehensive_audit_{timestamp}.json"
    failures_file = results_dir / f"audit_failures_{timestamp}.json"
    
    all_results = []
    all_failures = []
    start_time = time.time()
    
    for i, (board_size, mine_count) in enumerate(configs):
        print(f"\n📋 Configuration {i+1}/{total_configs}")
        print(f"   Board: {board_size[0]}×{board_size[1]}, Mines: {mine_count}")
        
        result = audit_learnable_filtering(board_size, mine_count, num_boards)
        all_results.append(result)
        
        # Collect all failures
        all_failures.extend(result['failures'])
        
        # Print results for this configuration
        print(f"   ✅ Total truly learnable boards: {result['total']}")
        print(f"   ❌ Non-learnable boards: {result['non_learnable']}")
        print(f"   ⚠️  Boards with instant wins: {result['instant_wins']}")
        if result['total'] > 0:
            print(f"   📈 Instant win rate: {result['instant_wins']/result['total']*100:.2f}%")
        print(f"   ⏱️  Time: {result['elapsed_time']/60:.1f} minutes")
        print(f"   🚀 Rate: {result['boards_per_second']:.1f} boards/second")
        print(f"   🐛 Failures in this config: {len(result['failures'])}")
        
        # Save intermediate results
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Save failures separately for analysis
        with open(failures_file, 'w') as f:
            json.dump(all_failures, f, indent=2, default=str)
        
        # Estimate remaining time
        elapsed_total = time.time() - start_time
        if i > 0:
            avg_time_per_config = elapsed_total / i
            remaining_configs = total_configs - i - 1
            estimated_remaining = avg_time_per_config * remaining_configs
            print(f"   🕐 Estimated remaining time: {estimated_remaining/60:.1f} minutes")
    
    # Analyze failure patterns
    print(f"\n🔍 Analyzing failure patterns...")
    failure_analysis = analyze_failures(all_failures)
    
    # Final summary
    total_time = time.time() - start_time
    total_boards_tested = sum(r['total'] + r['instant_wins'] + r['non_learnable'] for r in all_results)
    total_instant_wins = sum(r['instant_wins'] for r in all_results)
    total_non_learnable = sum(r['non_learnable'] for r in all_results)
    
    print(f"\n🎉 COMPREHENSIVE AUDIT COMPLETE")
    print("=" * 80)
    print(f"⏱️  Total time: {total_time/3600:.1f} hours")
    print(f"📊 Total boards tested: {total_boards_tested:,}")
    print(f"✅ Truly learnable boards: {total_boards_tested - total_instant_wins - total_non_learnable:,}")
    print(f"❌ Non-learnable boards: {total_non_learnable:,}")
    print(f"⚠️  Boards with instant wins: {total_instant_wins:,}")
    print(f"🐛 Total failures recorded: {len(all_failures):,}")
    
    if total_instant_wins > 0:
        print(f"🚨 CRITICAL: Found {total_instant_wins} boards with instant wins!")
        print(f"   This indicates a bug in the learnable filtering logic.")
        print(f"   Check failure analysis for patterns.")
    else:
        print(f"✅ SUCCESS: No instant wins found! Learnable filtering is working correctly.")
    
    # Print failure analysis summary
    if failure_analysis['total_failures'] > 0:
        print(f"\n🔍 FAILURE ANALYSIS SUMMARY")
        print(f"   Total failures: {failure_analysis['total_failures']}")
        print(f"   Failure types: {failure_analysis['failure_types']}")
        print(f"   Most common board size failures: {dict(sorted(failure_analysis['board_size_failures'].items(), key=lambda x: x[1], reverse=True)[:3])}")
        print(f"   Most common mine count failures: {dict(sorted(failure_analysis['mine_count_failures'].items(), key=lambda x: x[1], reverse=True)[:3])}")
        
        if failure_analysis['action_patterns']:
            print(f"   Most common instant win actions: {dict(sorted(failure_analysis['action_patterns'].items(), key=lambda x: x[1], reverse=True)[:5])}")
        
        if failure_analysis['error_patterns']:
            print(f"   Most common errors: {dict(sorted(failure_analysis['error_patterns'].items(), key=lambda x: x[1], reverse=True)[:3])}")
    
    print(f"\n📁 Results saved to: {results_file}")
    print(f"📁 Failures saved to: {failures_file}")
    
    # Save final summary
    summary = {
        'timestamp': timestamp,
        'total_time_hours': total_time / 3600,
        'total_configurations': total_configs,
        'total_boards_tested': total_boards_tested,
        'total_instant_wins': total_instant_wins,
        'total_non_learnable': total_non_learnable,
        'total_failures': len(all_failures),
        'success': total_instant_wins == 0,
        'failure_analysis': failure_analysis,
        'configurations': all_results
    }
    
    summary_file = results_dir / f"audit_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"📁 Summary saved to: {summary_file}")

if __name__ == "__main__":
    main() 