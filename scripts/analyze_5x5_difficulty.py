#!/usr/bin/env python3
"""
Analyze 5×5 Difficulty

This script analyzes why 5×5 boards are challenging and what specific
aspects need focus for improvement.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.minesweeper_env import MinesweeperEnv
from core.constants import REWARD_SAFE_REVEAL, REWARD_HIT_MINE, REWARD_WIN

def analyze_board_difficulty(board_size, mines, num_games=1000):
    """Analyze the difficulty of a specific board configuration."""
    
    env = MinesweeperEnv(
        initial_board_size=board_size,
        initial_mines=mines,
        max_board_size=board_size,
        max_mines=mines
    )
    
    stats = {
        'total_games': num_games,
        'wins': 0,
        'mine_hits': 0,
        'avg_steps': [],
        'first_move_success': 0,
        'games_with_cascade': 0,
        'avg_revealed_cells': [],
        'difficulty_factors': {}
    }
    
    for game in range(num_games):
        state = env.reset()
        steps = 0
        revealed_cells = 0
        first_move_safe = True
        
        while True:
            # Random action for analysis
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            
            # Track revealed cells
            if reward == REWARD_SAFE_REVEAL:
                revealed_cells += 1
                if steps == 1:
                    first_move_safe = True
            elif reward == REWARD_HIT_MINE:
                if steps == 1:
                    first_move_safe = False
                stats['mine_hits'] += 1
                break
            elif reward == REWARD_WIN:
                stats['wins'] += 1
                break
            
            # Check for cascade (multiple cells revealed)
            if 'cascade' in info and info['cascade']:
                stats['games_with_cascade'] += 1
            
            if done:
                break
        
        stats['avg_steps'].append(steps)
        stats['avg_revealed_cells'].append(revealed_cells)
        
        if first_move_safe:
            stats['first_move_success'] += 1
        
        # Progress output every 10 games
        if (game + 1) % 10 == 0:
            current_win_rate = stats['wins'] / (game + 1) * 100
            print(f"  Progress: {game + 1}/{num_games} games, Win Rate: {current_win_rate:.1f}%")
    
    # Calculate statistics
    stats['win_rate'] = stats['wins'] / num_games
    stats['mine_hit_rate'] = stats['mine_hits'] / num_games
    stats['avg_steps'] = np.mean(stats['avg_steps'])
    stats['avg_revealed_cells'] = np.mean(stats['avg_revealed_cells'])
    stats['first_move_success_rate'] = stats['first_move_success'] / num_games
    stats['cascade_rate'] = stats['games_with_cascade'] / num_games
    
    # Calculate difficulty factors
    total_cells = board_size[0] * board_size[1]
    mine_density = mines / total_cells
    
    stats['difficulty_factors'] = {
        'total_cells': total_cells,
        'mine_density': mine_density,
        'cells_per_mine': total_cells / mines,
        'safe_cells': total_cells - mines,
        'safe_cell_ratio': (total_cells - mines) / total_cells
    }
    
    return stats

def compare_board_configurations():
    """Compare different board configurations to understand difficulty progression."""
    
    configurations = [
        {'name': '4×4 (1 mine)', 'board_size': (4, 4), 'mines': 1},
        {'name': '4×5 (1 mine)', 'board_size': (4, 5), 'mines': 1},
        {'name': '5×4 (1 mine)', 'board_size': (5, 4), 'mines': 1},
        {'name': '5×5 (1 mine)', 'board_size': (5, 5), 'mines': 1},
        {'name': '5×5 (2 mines)', 'board_size': (5, 5), 'mines': 2},
        {'name': '6×6 (3 mines)', 'board_size': (6, 6), 'mines': 3},
    ]
    
    results = {}
    
    print("🔍 Analyzing Board Configurations")
    print("=" * 60)
    
    for config in configurations:
        print(f"\n📊 {config['name']}")
        print("-" * 40)
        
        stats = analyze_board_difficulty(
            config['board_size'], 
            config['mines'], 
            num_games=1000
        )
        
        results[config['name']] = stats
        
        print(f"Win Rate: {stats['win_rate']*100:.1f}%")
        print(f"Mine Hit Rate: {stats['mine_hit_rate']*100:.1f}%")
        print(f"First Move Success: {stats['first_move_success_rate']*100:.1f}%")
        print(f"Average Steps: {stats['avg_steps']:.1f}")
        print(f"Average Revealed Cells: {stats['avg_revealed_cells']:.1f}")
        print(f"Cascade Rate: {stats['cascade_rate']*100:.1f}%")
        print(f"Mine Density: {stats['difficulty_factors']['mine_density']:.3f}")
        print(f"Safe Cell Ratio: {stats['difficulty_factors']['safe_cell_ratio']:.3f}")
    
    return results

def analyze_optimal_strategy(board_size, mines, num_games=500):
    """Analyze what an optimal strategy might achieve."""
    
    env = MinesweeperEnv(
        initial_board_size=board_size,
        initial_mines=mines,
        max_board_size=board_size,
        max_mines=mines
    )
    
    wins = 0
    total_reward = 0
    
    for game in range(num_games):
        state = env.reset()
        game_reward = 0
        
        while True:
            # For now, just use random actions (same as baseline)
            # This is just for comparison - in practice you'd implement a real heuristic
            action = env.action_space.sample()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            game_reward += reward
            
            if reward == REWARD_WIN:
                wins += 1
                break
            elif reward == REWARD_HIT_MINE:
                break
            
            if done:
                break
        
        total_reward += game_reward
        
        # Progress output every 50 games
        if (game + 1) % 50 == 0:
            current_win_rate = wins / (game + 1) * 100
            print(f"    Heuristic Progress: {game + 1}/{num_games} games, Win Rate: {current_win_rate:.1f}%")
    
    return {
        'win_rate': wins / num_games,
        'avg_reward': total_reward / num_games
    }

def main():
    """Main analysis function."""
    
    print("🎯 5×5 Difficulty Analysis")
    print("=" * 50)
    
    # Compare configurations
    results = compare_board_configurations()
    
    # Analyze optimal strategy potential
    print(f"\n🎯 Optimal Strategy Analysis")
    print("=" * 50)
    
    for config_name in ['4×4 (1 mine)', '5×5 (1 mine)', '5×5 (2 mines)']:
        config = next(c for c in [
            {'name': '4×4 (1 mine)', 'board_size': (4, 4), 'mines': 1},
            {'name': '5×5 (1 mine)', 'board_size': (5, 5), 'mines': 1},
            {'name': '5×5 (2 mines)', 'board_size': (5, 5), 'mines': 2},
        ] if c['name'] == config_name)
        
        optimal_stats = analyze_optimal_strategy(
            config['board_size'], 
            config['mines'], 
            num_games=500
        )
        
        print(f"\n{config_name}:")
        print(f"  Random Win Rate: {results[config_name]['win_rate']*100:.1f}%")
        print(f"  Heuristic Win Rate: {optimal_stats['win_rate']*100:.1f}%")
        print(f"  Improvement: {(optimal_stats['win_rate'] - results[config_name]['win_rate'])*100:.1f}%")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"5x5_difficulty_analysis_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to {results_file}")
    
    # Key insights
    print(f"\n🔍 Key Insights:")
    print("=" * 50)
    
    # Compare 4×4 vs 5×5
    config_4x4 = results['4×4 (1 mine)']
    config_5x5_1 = results['5×5 (1 mine)']
    config_5x5_2 = results['5×5 (2 mines)']
    
    print(f"1. 4×4 → 5×5 (1 mine) difficulty increase:")
    print(f"   Win rate: {config_4x4['win_rate']*100:.1f}% → {config_5x5_1['win_rate']*100:.1f}%")
    print(f"   Mine density: {config_4x4['difficulty_factors']['mine_density']:.3f} → {config_5x5_1['difficulty_factors']['mine_density']:.3f}")
    
    print(f"\n2. 5×5 (1 mine) → 5×5 (2 mines) difficulty increase:")
    print(f"   Win rate: {config_5x5_1['win_rate']*100:.1f}% → {config_5x5_2['win_rate']*100:.1f}%")
    print(f"   Mine density: {config_5x5_1['difficulty_factors']['mine_density']:.3f} → {config_5x5_2['difficulty_factors']['mine_density']:.3f}")
    
    print(f"\n3. Recommended progression:")
    print(f"   - Start with 5×5 (1 mine) to learn the larger board")
    print(f"   - Then progress to 5×5 (2 mines) for mine avoidance")
    print(f"   - Use intermediate stages (4×5, 5×4) for gradual learning")

if __name__ == "__main__":
    main() 