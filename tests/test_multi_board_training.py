#!/usr/bin/env python3
"""
Test script to verify multi-board training functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.core.minesweeper_env import MinesweeperEnv
from src.core.train_agent import make_env, ActionMaskingWrapper, MultiBoardTrainingWrapper

def test_multi_board_training():
    """Test that multi-board training generates different board layouts."""
    print("🔍 Testing Multi-Board Training")
    print("=" * 50)
    
    # Create base environment
    base_env = DummyVecEnv([make_env(max_board_size=(4, 4), max_mines=1)])
    
    # Apply wrappers
    env = ActionMaskingWrapper(base_env)
    env = MultiBoardTrainingWrapper(env, board_variations=5)
    
    print(f"✅ Created multi-board training environment")
    print(f"   Board variations: 5")
    print(f"   Board size: 4x4")
    print(f"   Mines: 1")
    
    # Test different board layouts
    print("\n🎯 Testing Board Layout Variations:")
    board_layouts = []
    
    for i in range(5):
        obs = env.reset()
        board_info = env.get_board_variation_info()
        
        # Get the mine positions from the underlying environment
        # Navigate through wrappers to get the base environment
        base_env_instance = env.venv.envs[0]
        if hasattr(base_env_instance, 'env'):
            base_env_instance = base_env_instance.env
        mine_positions = np.where(base_env_instance.mines)
        
        print(f"  Variation {i+1}:")
        print(f"    Seed: {board_info['variation_seed']}")
        print(f"    Mine positions: {list(zip(mine_positions[0], mine_positions[1]))}")
        print(f"    Board layout:")
        print(f"      {base_env_instance.mines}")
        
        board_layouts.append(base_env_instance.mines.copy())
    
    # Check if boards are different
    unique_boards = []
    for board in board_layouts:
        is_unique = True
        for existing_board in unique_boards:
            if np.array_equal(board, existing_board):
                is_unique = False
                break
        if is_unique:
            unique_boards.append(board)
    
    print(f"\n📊 Results:")
    print(f"  Total boards generated: {len(board_layouts)}")
    print(f"  Unique boards: {len(unique_boards)}")
    print(f"  Diversity: {len(unique_boards)/len(board_layouts)*100:.1f}%")
    
    if len(unique_boards) == len(board_layouts):
        print("✅ Multi-board training is working correctly!")
        print("   All board layouts are unique.")
    else:
        print("⚠️  Some board layouts are identical.")
        print("   This might indicate a seeding issue.")
    
    # Test training with multi-board environment
    print("\n🚀 Testing Training with Multi-Board Environment:")
    
    # Create a simple PPO model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0003,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        verbose=0
    )
    
    print("✅ Created PPO model")
    
    # Train for a few steps
    print("🎯 Training for 1000 timesteps...")
    model.learn(total_timesteps=1000, progress_bar=False)
    print("✅ Training completed")
    
    # Test evaluation on a fixed board
    print("\n🔍 Testing Evaluation on Fixed Board:")
    
    # Create evaluation environment with fixed seed
    eval_env = DummyVecEnv([make_env(max_board_size=(4, 4), max_mines=1)])
    eval_env = ActionMaskingWrapper(eval_env)
    
    # Set fixed seed for evaluation
    for env_instance in eval_env.envs:
        if hasattr(env_instance, 'env'):
            env_instance.env.seed(42)
        else:
            env_instance.seed(42)
    
    obs = eval_env.reset()
    print(f"✅ Created evaluation environment with fixed seed")
    
    # Run a few evaluation episodes
    wins = 0
    total_episodes = 5
    
    for episode in range(total_episodes):
        obs = eval_env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done and steps < 20:
            action, _ = model.predict(obs, deterministic=True)
            step_result = eval_env.step(action)
            
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = np.any(terminated) or np.any(truncated)
            else:
                obs, reward, done, info = step_result
            
            episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            steps += 1
            
            # Check for win
            if info and len(info) > 0 and info[0].get('won', False):
                wins += 1
                break
        
        print(f"  Episode {episode+1}: Reward = {episode_reward:.1f}, Steps = {steps}")
    
    win_rate = (wins / total_episodes) * 100
    print(f"\n📊 Evaluation Results:")
    print(f"  Episodes: {total_episodes}")
    print(f"  Wins: {wins}")
    print(f"  Win rate: {win_rate:.1f}%")
    
    if win_rate > 0:
        print("✅ Agent learned to win on fixed board!")
    else:
        print("⚠️  Agent needs more training to generalize.")

if __name__ == "__main__":
    test_multi_board_training() 