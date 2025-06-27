#!/usr/bin/env python3
"""
Comprehensive audit script to test that training scripts properly handle first move mine hits.
Verifies that first move mine hits are not counted in RL training statistics.

Tests all scenarios:
- Different first move strategies (corner, edge, center, random)
- All board configurations (learnable and non-learnable)
- Edge cases (max mines, single mine, etc.)
- Statistics handling across all scenarios
"""
import sys
import os
import time
import json
from datetime import datetime
from pathlib import Path
import pytest

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))

import numpy as np
from src.core.minesweeper_env import MinesweeperEnv

@pytest.fixture
def board_size():
    """Fixture for board size."""
    return (4, 4)

@pytest.fixture
def mine_count():
    """Fixture for mine count."""
    return 2

def test_first_move_strategies(board_size, mine_count, num_games=500):
    """Test different first move strategies comprehensively."""
    
    print(f"  🎯 Testing {board_size[0]}×{board_size[1]} with {mine_count} mines...")
    
    # Track different scenarios
    scenarios = {
        'corner_moves': {'mine_hits': 0, 'instant_wins': 0, 'safe': 0},
        'edge_moves': {'mine_hits': 0, 'instant_wins': 0, 'safe': 0},
        'center_moves': {'mine_hits': 0, 'instant_wins': 0, 'safe': 0},
        'random_moves': {'mine_hits': 0, 'instant_wins': 0, 'safe': 0}
    }
    
    # Track statistics
    real_life_stats = {'games_played': 0, 'games_won': 0, 'games_lost': 0}
    rl_stats = {'games_played': 0, 'games_won': 0, 'games_lost': 0}
    
    # Track learnable vs non-learnable
    learnable_configs = 0
    non_learnable_configs = 0
    
    # Track pre-cascade games
    pre_cascade_games = 0
    multi_move_games = 0
    
    start_time = time.time()
    
    for i in range(num_games):
        if i % 100 == 0 and i > 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            remaining = (num_games - i) / rate
            print(f"    Progress: {i}/{num_games} ({i/num_games*100:.1f}%) - ETA: {remaining/60:.1f}min")
        
        try:
            # Create environment with learnable_only=False to test all scenarios
            env = MinesweeperEnv(
                max_board_size=board_size,
                initial_board_size=board_size,
                max_mines=mine_count,
                initial_mines=mine_count,
                learnable_only=False,  # Allow all board types for testing
                max_learnable_attempts=1000
            )
            
            # Reset and get initial state
            obs, info = env.reset()
            
            # Check if this is a learnable configuration
            is_learnable = info.get('learnable', False)
            if is_learnable:
                learnable_configs += 1
            else:
                non_learnable_configs += 1
            
            # Test different first move strategies
            height, width = board_size
            
            # Define move strategies
            move_strategies = {
                'corner_moves': [
                    0,  # Top-left
                    width - 1,  # Top-right
                    (height - 1) * width,  # Bottom-left
                    height * width - 1  # Bottom-right
                ],
                'edge_moves': [
                    width // 2,  # Top edge center
                    (height - 1) * width + width // 2,  # Bottom edge center
                    height // 2 * width,  # Left edge center
                    height // 2 * width + width - 1  # Right edge center
                ],
                'center_moves': [
                    (height // 2) * width + (width // 2)  # Center
                ],
                'random_moves': [
                    np.random.randint(0, height * width)  # Random position
                ]
            }
            
            # Test each strategy
            for strategy_name, moves in move_strategies.items():
                for move in moves:
                    # Reset to get fresh board
                    obs, info = env.reset()
                    
                    # Make the move
                    obs, reward, terminated, truncated, info = env.step(move)
                    
                    # Determine what happened
                    if info.get('won', False):
                        scenarios[strategy_name]['instant_wins'] += 1
                    elif terminated and not info.get('won', False):
                        scenarios[strategy_name]['mine_hits'] += 1
                    else:
                        scenarios[strategy_name]['safe'] += 1
                        
                        # Continue game if not terminated
                        moves_made = 1
                        while not terminated and not truncated and moves_made < 10:
                            # Choose a random valid action
                            valid_actions = [i for i in range(env.action_space.n) 
                                           if not env.revealed[i // env.current_board_width, i % env.current_board_width]]
                            if valid_actions:
                                action = np.random.choice(valid_actions)
                                obs, reward, terminated, truncated, info = env.step(action)
                                moves_made += 1
                            else:
                                break
                        
                        if moves_made > 1:
                            multi_move_games += 1
            
            # Check if any game ended pre-cascade
            game_ended_pre_cascade = info.get('game_ended_pre_cascade', False)
            if game_ended_pre_cascade:
                pre_cascade_games += 1
            
            # Get final statistics
            final_real_life = env.get_real_life_statistics()
            final_rl = env.get_rl_training_statistics()
            
            # Update our tracking
            real_life_stats['games_played'] = final_real_life['games_played']
            real_life_stats['games_won'] = final_real_life['games_won']
            real_life_stats['games_lost'] = final_real_life['games_lost']
            
            rl_stats['games_played'] = final_rl['games_played']
            rl_stats['games_won'] = final_rl['games_won']
            rl_stats['games_lost'] = final_rl['games_lost']
            
        except Exception as e:
            print(f"    ❌ Error on game {i}: {e}")
            continue
    
    elapsed_time = time.time() - start_time
    
    return {
        'board_size': board_size,
        'mine_count': mine_count,
        'total_games': num_games,
        'scenarios': scenarios,
        'learnable_configs': learnable_configs,
        'non_learnable_configs': non_learnable_configs,
        'multi_move_games': multi_move_games,
        'pre_cascade_games': pre_cascade_games,
        'real_life_stats': real_life_stats,
        'rl_stats': rl_stats,
        'elapsed_time': elapsed_time,
        'games_per_second': num_games / elapsed_time if elapsed_time > 0 else 0
    }

def test_edge_cases():
    """Test specific edge cases that might cause issues."""
    
    print(f"\n🔍 Testing Edge Cases")
    print("=" * 50)
    
    edge_cases = [
        # Single mine configurations
        ((4, 4), 1, "4×4 with 1 mine"),
        ((5, 5), 1, "5×5 with 1 mine"),
        ((9, 9), 1, "9×9 with 1 mine"),
        
        # High mine density
        ((4, 4), 15, "4×4 with 15 mines (max)"),
        ((5, 5), 24, "5×5 with 24 mines (max)"),
        ((9, 9), 80, "9×9 with 80 mines (max)"),
        
        # Rectangular boards
        ((4, 5), 1, "4×5 with 1 mine"),
        ((5, 4), 1, "5×4 with 1 mine"),
        ((4, 5), 19, "4×5 with 19 mines (max)"),
    ]
    
    edge_case_results = []
    
    for board_size, mine_count, description in edge_cases:
        print(f"  Testing {description}...")
        
        try:
            env = MinesweeperEnv(
                max_board_size=board_size,
                initial_board_size=board_size,
                max_mines=mine_count,
                initial_mines=mine_count,
                learnable_only=False
            )
            
            # Test a few games
            mine_hits = 0
            instant_wins = 0
            safe_moves = 0
            
            for _ in range(100):
                obs, info = env.reset()
                
                # Try different first moves
                for move in [0, board_size[1]//2, board_size[0]*board_size[1]//2]:
                    obs, info = env.reset()
                    obs, reward, terminated, truncated, info = env.step(move)
                    
                    if info.get('won', False):
                        instant_wins += 1
                    elif terminated and not info.get('won', False):
                        mine_hits += 1
                    else:
                        safe_moves += 1
            
            result = {
                'description': description,
                'board_size': board_size,
                'mine_count': mine_count,
                'mine_hits': mine_hits,
                'instant_wins': instant_wins,
                'safe_moves': safe_moves,
                'success': True
            }
            
        except Exception as e:
            result = {
                'description': description,
                'board_size': board_size,
                'mine_count': mine_count,
                'error': str(e),
                'success': False
            }
        
        edge_case_results.append(result)
        print(f"    ✅ {description}: {result.get('mine_hits', 0)} mine hits, {result.get('instant_wins', 0)} instant wins")
    
    return edge_case_results

def analyze_statistics(results):
    """Analyze if statistics are being handled correctly."""
    
    analysis = {
        'total_configurations': len(results),
        'total_games': sum(r['total_games'] for r in results),
        'total_learnable_configs': sum(r['learnable_configs'] for r in results),
        'total_non_learnable_configs': sum(r['non_learnable_configs'] for r in results),
        'statistics_issues': [],
        'scenario_analysis': {}
    }
    
    # Aggregate scenario data
    all_scenarios = {}
    for result in results:
        for strategy, data in result['scenarios'].items():
            if strategy not in all_scenarios:
                all_scenarios[strategy] = {'mine_hits': 0, 'instant_wins': 0, 'safe': 0}
            all_scenarios[strategy]['mine_hits'] += data['mine_hits']
            all_scenarios[strategy]['instant_wins'] += data['instant_wins']
            all_scenarios[strategy]['safe'] += data['safe']
    
    analysis['scenario_analysis'] = all_scenarios
    
    for result in results:
        board_key = f"{result['board_size'][0]}×{result['board_size'][1]}×{result['mine_count']}"
        
        # Check if RL stats exclude first move issues
        total_first_move_issues = sum(
            data['mine_hits'] + data['instant_wins'] 
            for data in result['scenarios'].values()
        )
        rl_games = result['rl_stats']['games_played']
        real_life_games = result['real_life_stats']['games_played']
        
        # RL games should be less than real life games if there were first move issues
        if total_first_move_issues > 0 and rl_games >= real_life_games:
            analysis['statistics_issues'].append({
                'configuration': board_key,
                'issue': 'RL stats not excluding first move issues',
                'first_move_issues': total_first_move_issues,
                'rl_games': rl_games,
                'real_life_games': real_life_games
            })
        
        # Check if pre-cascade games are being handled correctly
        if result['pre_cascade_games'] > 0:
            if result['rl_stats']['games_played'] > 0:
                analysis['statistics_issues'].append({
                    'configuration': board_key,
                    'issue': 'Pre-cascade games being counted in RL stats',
                    'pre_cascade_games': result['pre_cascade_games'],
                    'rl_games': result['rl_stats']['games_played']
                })
    
    return analysis

def main():
    """Main audit function for comprehensive first move testing."""
    
    # Test configurations: 4×4 to 9×9 with 1-7 mines
    configs = []
    for board_size in [(4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]:
        max_mines = min(7, board_size[0] * board_size[1] - 1)
        for mine_count in range(1, max_mines + 1):
            configs.append((board_size, mine_count))
    
    num_games = 500  # Fewer games per config for faster testing
    total_configs = len(configs)
    
    print(f"🔍 Comprehensive First Move Mine Hit Handling Audit")
    print(f"📊 Testing {total_configs} configurations with {num_games:,} games each")
    print(f"🎯 Board sizes: 4×4 to 9×9")
    print(f"💣 Mine counts: 1 to 7")
    print(f"🎮 Testing multiple first move strategies")
    print(f"🔍 Testing edge cases and all scenarios")
    print("=" * 80)
    
    # Create results directory
    results_dir = Path("audit_results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"comprehensive_first_move_audit_{timestamp}.json"
    
    all_results = []
    start_time = time.time()
    
    for i, (board_size, mine_count) in enumerate(configs):
        print(f"\n📋 Configuration {i+1}/{total_configs}")
        print(f"   Board: {board_size[0]}×{board_size[1]}, Mines: {mine_count}")
        
        result = test_first_move_strategies(board_size, mine_count, num_games)
        all_results.append(result)
        
        # Print results for this configuration
        print(f"   🎮 Total games: {result['total_games']}")
        print(f"   📊 Learnable configs: {result['learnable_configs']}")
        print(f"   ❌ Non-learnable configs: {result['non_learnable_configs']}")
        
        # Print scenario breakdown
        for strategy, data in result['scenarios'].items():
            total = data['mine_hits'] + data['instant_wins'] + data['safe']
            if total > 0:
                print(f"   {strategy}: {data['mine_hits']} hits, {data['instant_wins']} wins, {data['safe']} safe")
        
        print(f"   🔄 Multi-move games: {result['multi_move_games']}")
        print(f"   ⏭️  Pre-cascade games: {result['pre_cascade_games']}")
        print(f"   📊 Real life games: {result['real_life_stats']['games_played']}")
        print(f"   🤖 RL training games: {result['rl_stats']['games_played']}")
        print(f"   ⏱️  Time: {result['elapsed_time']/60:.1f} minutes")
        print(f"   🚀 Rate: {result['games_per_second']:.1f} games/second")
        
        # Save intermediate results
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Estimate remaining time
        elapsed_total = time.time() - start_time
        if i > 0:
            avg_time_per_config = elapsed_total / i
            remaining_configs = total_configs - i - 1
            estimated_remaining = avg_time_per_config * remaining_configs
            print(f"   🕐 Estimated remaining time: {estimated_remaining/60:.1f} minutes")
    
    # Test edge cases
    edge_case_results = test_edge_cases()
    
    # Analyze statistics
    print(f"\n🔍 Analyzing statistics handling...")
    analysis = analyze_statistics(all_results)
    
    # Final summary
    total_time = time.time() - start_time
    
    print(f"\n🎉 COMPREHENSIVE FIRST MOVE AUDIT COMPLETE")
    print("=" * 80)
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"📊 Total games tested: {analysis['total_games']:,}")
    print(f"✅ Learnable configurations: {analysis['total_learnable_configs']:,}")
    print(f"❌ Non-learnable configurations: {analysis['total_non_learnable_configs']:,}")
    
    # Print scenario analysis
    print(f"\n📈 SCENARIO ANALYSIS:")
    for strategy, data in analysis['scenario_analysis'].items():
        total = data['mine_hits'] + data['instant_wins'] + data['safe']
        if total > 0:
            print(f"   {strategy}: {data['mine_hits']} hits ({data['mine_hits']/total*100:.1f}%), "
                  f"{data['instant_wins']} wins ({data['instant_wins']/total*100:.1f}%), "
                  f"{data['safe']} safe ({data['safe']/total*100:.1f}%)")
    
    if analysis['statistics_issues']:
        print(f"\n🚨 STATISTICS ISSUES FOUND:")
        for issue in analysis['statistics_issues']:
            print(f"   ❌ {issue['configuration']}: {issue['issue']}")
    else:
        print(f"\n✅ SUCCESS: All statistics handled correctly!")
        print(f"   First move issues properly excluded from RL training stats")
    
    print(f"\n📁 Results saved to: {results_file}")
    
    # Save final summary
    summary = {
        'timestamp': timestamp,
        'total_time_minutes': total_time / 60,
        'total_configurations': total_configs,
        'analysis': analysis,
        'edge_cases': edge_case_results,
        'configurations': all_results
    }
    
    summary_file = results_dir / f"comprehensive_first_move_audit_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"📁 Summary saved to: {summary_file}")

if __name__ == "__main__":
    main() 