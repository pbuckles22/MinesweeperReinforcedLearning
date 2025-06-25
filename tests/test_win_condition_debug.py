#!/usr/bin/env python3
"""
Comprehensive test to debug win condition logic
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from src.core.minesweeper_env import MinesweeperEnv
from src.core.train_agent import make_env

def test_win_condition_logic():
    """Test the win condition logic step by step."""
    print("🔍 Testing Win Condition Logic")
    print("=" * 60)
    
    # Test with 4x4 board, 1 mine (should be winnable)
    env = MinesweeperEnv(
        max_board_size=(4, 4),
        max_mines=1,
        initial_board_size=(4, 4),
        initial_mines=1,
        render_mode=None
    )
    
    print("✅ Environment created: 4x4 board with 1 mine")
    
    # Reset and get initial state
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    # Print initial board state
    print("\n📋 Initial Board State:")
    print(f"Board shape: {env.board.shape}")
    print(f"Mines shape: {env.mines.shape}")
    print(f"Revealed shape: {env.revealed.shape}")
    
    # Find mine location
    mine_locations = np.where(env.mines)
    print(f"Mine locations: {list(zip(mine_locations[0], mine_locations[1]))}")
    
    # Count total cells and mines
    total_cells = env.current_board_height * env.current_board_width
    total_mines = env.current_mines
    safe_cells = total_cells - total_mines
    print(f"Total cells: {total_cells}, Mines: {total_mines}, Safe cells: {safe_cells}")
    
    # Test win condition before any moves
    initial_win = env._check_win()
    print(f"Win condition before any moves: {initial_win}")
    
    # Try to reveal all safe cells manually
    print("\n🎮 Manually revealing all safe cells...")
    revealed_count = 0
    
    for row in range(env.current_board_height):
        for col in range(env.current_board_width):
            if not env.mines[row, col] and not env.revealed[row, col]:
                action = row * env.current_board_width + col
                print(f"  Revealing cell ({row}, {col}) with action {action}")
                
                obs, reward, terminated, truncated, info = env.step(action)
                revealed_count += 1
                
                print(f"    Reward: {reward}, Terminated: {terminated}, Won: {info.get('won', False)}")
                print(f"    Revealed count: {np.sum(env.revealed)}")
                
                if terminated:
                    print(f"    Game ended! Final reward: {reward}")
                    print(f"    Final info: {info}")
                    break
        
        if env.terminated:
            break
    
    print(f"\n📊 Final Results:")
    print(f"  Total cells revealed: {revealed_count}")
    print(f"  Final revealed count: {np.sum(env.revealed)}")
    print(f"  Safe cells that should be revealed: {safe_cells}")
    print(f"  Win condition check: {env._check_win()}")
    print(f"  Game terminated: {env.terminated}")
    print(f"  Final info: {info}")

def test_action_masks():
    """Test if action masks are preventing valid moves."""
    print("\n🔍 Testing Action Masks")
    print("=" * 60)
    
    env = MinesweeperEnv(
        max_board_size=(4, 4),
        max_mines=1,
        initial_board_size=(4, 4),
        initial_mines=1,
        render_mode=None
    )
    
    obs, info = env.reset()
    
    print("✅ Environment created")
    
    # Check initial action masks
    masks = env.action_masks
    print(f"Action masks shape: {masks.shape}")
    print(f"Valid actions: {np.sum(masks)} out of {len(masks)}")
    print(f"Valid action indices: {np.where(masks)[0]}")
    
    # Try a few actions and see how masks change
    for step in range(5):
        valid_actions = np.where(masks)[0]
        if len(valid_actions) == 0:
            print(f"Step {step}: No valid actions!")
            break
        
        action = valid_actions[0]
        print(f"Step {step}: Taking action {action}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Reward: {reward}, Terminated: {terminated}, Won: {info.get('won', False)}")
        
        if terminated:
            print(f"  Game ended!")
            break
        
        # Check masks after action
        masks = env.action_masks
        print(f"  Valid actions after: {np.sum(masks)} out of {len(masks)}")

def test_simple_win_scenario():
    """Test a simple scenario that should definitely result in a win."""
    print("\n🔍 Testing Simple Win Scenario")
    print("=" * 60)
    
    # Create a 3x3 board with 1 mine in corner
    env = MinesweeperEnv(
        max_board_size=(3, 3),
        max_mines=1,
        initial_board_size=(3, 3),
        initial_mines=1,
        render_mode=None
    )
    
    obs, info = env.reset()
    
    print("✅ Environment created: 3x3 board with 1 mine")
    
    # Find mine location
    mine_locations = np.where(env.mines)
    mine_row, mine_col = mine_locations[0][0], mine_locations[1][0]
    print(f"Mine at: ({mine_row}, {mine_col})")
    
    # Try to reveal all non-mine cells
    total_cells = 9
    safe_cells = 8
    revealed_count = 0
    
    print("🎮 Revealing all safe cells...")
    
    for row in range(3):
        for col in range(3):
            if (row, col) != (mine_row, mine_col):
                action = row * 3 + col
                print(f"  Revealing ({row}, {col}) with action {action}")
                
                obs, reward, terminated, truncated, info = env.step(action)
                revealed_count += 1
                
                print(f"    Reward: {reward}, Terminated: {terminated}, Won: {info.get('won', False)}")
                print(f"    Revealed: {np.sum(env.revealed)}/{safe_cells}")
                
                if terminated:
                    print(f"    Game ended!")
                    break
        
        if env.terminated:
            break
    
    print(f"\n📊 Final Results:")
    print(f"  Cells revealed: {revealed_count}")
    print(f"  Total revealed: {np.sum(env.revealed)}")
    print(f"  Win condition: {env._check_win()}")
    print(f"  Game terminated: {env.terminated}")
    print(f"  Final info: {info}")

def test_random_actions_with_debug():
    """Test random actions with detailed debugging."""
    print("\n🔍 Testing Random Actions with Debug")
    print("=" * 60)
    
    env = MinesweeperEnv(
        max_board_size=(4, 4),
        max_mines=1,
        initial_board_size=(4, 4),
        initial_mines=1,
        render_mode=None
    )
    
    obs, info = env.reset()
    
    print("✅ Environment created")
    
    # Play 10 games with random actions
    wins = 0
    total_games = 10
    
    for game in range(total_games):
        print(f"\n🎮 Game {game + 1}:")
        obs, info = env.reset()
        
        step_count = 0
        total_reward = 0
        
        while step_count < 20 and not env.terminated:
            step_count += 1
            
            # Get valid actions
            masks = env.action_masks
            valid_actions = np.where(masks)[0]
            
            if len(valid_actions) == 0:
                print(f"  No valid actions at step {step_count}")
                break
            
            # Take random valid action
            action = np.random.choice(valid_actions)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"  Step {step_count}: Action {action}, Reward {reward}, Won {info.get('won', False)}")
            
            if terminated:
                print(f"  Game ended after {step_count} steps")
                if info.get('won', False):
                    wins += 1
                    print(f"  🎉 WIN!")
                else:
                    print(f"  ❌ LOSS")
                break
    
    print(f"\n📊 Random Action Results:")
    print(f"  Wins: {wins}/{total_games} ({wins/total_games*100:.1f}%)")

if __name__ == "__main__":
    test_win_condition_logic()
    test_action_masks()
    test_simple_win_scenario()
    test_random_actions_with_debug() 