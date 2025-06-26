#!/usr/bin/env python3
"""
Test script to analyze 4x4 with 2 mines difficulty
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.core.minesweeper_env import MinesweeperEnv
from src.core.train_agent import make_env, ActionMaskingWrapper, MultiBoardTrainingWrapper

def test_4x4_2mines_difficulty():
    """Test the difficulty of 4x4 with 2 mines."""
    print("🔍 Testing 4x4 with 2 Mines Difficulty")
    print("=" * 50)
    
    # Create environment
    base_env = make_env(max_board_size=(4, 4), max_mines=2)()
    base_env = ActionMaskingWrapper(base_env)
    base_env = MultiBoardTrainingWrapper(base_env, board_variations=10)
    env = DummyVecEnv([lambda: base_env])
    
    print("✅ Created 4x4 with 2 mines environment")
    print("   Board size: 4x4")
    print("   Mines: 2")
    print("   Mine density: 12.5%")
    
    # Test random play performance
    print("\n🎲 Testing Random Play Performance:")
    random_wins = 0
    random_episodes = 100
    
    for episode in range(random_episodes):
        obs = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 20:
            # Random action
            action = env.action_space.sample()
            step_result = env.step([action])  # Pass as list for DummyVecEnv
            
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = np.any(terminated) or np.any(truncated)
            else:
                obs, reward, done, info = step_result
            
            steps += 1
            
            # Check for win
            if info and len(info) > 0 and info[0].get('won', False):
                random_wins += 1
                break
    
    random_win_rate = (random_wins / random_episodes) * 100
    print(f"Random Play Results:")
    print(f"  Episodes: {random_episodes}")
    print(f"  Wins: {random_wins}")
    print(f"  Win rate: {random_win_rate:.1f}%")
    
    # Test training performance
    print("\n🎯 Testing Training Performance:")
    
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
    
    # Train for more timesteps
    print("🎯 Training for 10000 timesteps...")
    model.learn(total_timesteps=10000, progress_bar=False)
    print("✅ Training completed")
    
    # Test trained model performance
    print("\n📊 Testing Trained Model Performance:")
    trained_wins = 0
    trained_episodes = 50
    
    for episode in range(trained_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done and steps < 20:
            action, _ = model.predict(obs, deterministic=True)
            step_result = env.step([action])  # Pass as list for DummyVecEnv
            
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = np.any(terminated) or np.any(truncated)
            else:
                obs, reward, done, info = step_result
            
            episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            steps += 1
            
            # Check for win
            if info and len(info) > 0 and info[0].get('won', False):
                trained_wins += 1
                break
        
        if episode < 10:  # Show first 10 episodes
            print(f"  Episode {episode+1}: Reward = {episode_reward:.1f}, Steps = {steps}")
    
    trained_win_rate = (trained_wins / trained_episodes) * 100
    print(f"\nTrained Model Results:")
    print(f"  Episodes: {trained_episodes}")
    print(f"  Wins: {trained_wins}")
    print(f"  Win rate: {trained_win_rate:.1f}%")
    
    # Analysis
    print(f"\n📊 Difficulty Analysis:")
    print(f"  Random win rate: {random_win_rate:.1f}%")
    print(f"  Trained win rate: {trained_win_rate:.1f}%")
    print(f"  Improvement: {trained_win_rate - random_win_rate:.1f}%")
    
    if trained_win_rate > 15:
        print("✅ Agent can achieve 15% target with sufficient training!")
    elif trained_win_rate > 5:
        print("⚠️  Agent shows some learning but needs more training")
    else:
        print("❌ 4x4 with 2 mines may be too difficult for current approach")
    
    # Test different board configurations
    print(f"\n🎲 Testing Different Board Configurations:")
    
    for mines in [1, 2, 3]:
        base_test_env = make_env(max_board_size=(4, 4), max_mines=mines)()
        base_test_env = ActionMaskingWrapper(base_test_env)
        test_env = DummyVecEnv([lambda: base_test_env])
        
        wins = 0
        episodes = 20
        
        for episode in range(episodes):
            obs = test_env.reset()
            done = False
            steps = 0
            
            while not done and steps < 20:
                action, _ = model.predict(obs, deterministic=True)
                step_result = test_env.step([action])
                
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = np.any(terminated) or np.any(truncated)
                else:
                    obs, reward, done, info = step_result
                
                steps += 1
                
                if info and len(info) > 0 and info[0].get('won', False):
                    wins += 1
                    break
        
        win_rate = (wins / episodes) * 100
        mine_density = (mines / 16) * 100
        print(f"  {mines} mine(s): {win_rate:.1f}% win rate ({mine_density:.1f}% density)")

if __name__ == "__main__":
    test_4x4_2mines_difficulty() 