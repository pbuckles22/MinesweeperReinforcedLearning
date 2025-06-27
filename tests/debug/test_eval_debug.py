#!/usr/bin/env python3
"""
Debug script to test evaluation vs training discrepancy
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from core.minesweeper_env import MinesweeperEnv
from core.train_agent import make_env

def test_evaluation_win_detection():
    """Test if evaluation win detection is working correctly."""
    print("🔍 Testing Evaluation Win Detection")
    print("=" * 50)
    
    # Create environment
    env = DummyVecEnv([make_env(max_board_size=(4, 4), max_mines=2)])
    
    print("✅ Environment created")
    
    # Test 50 episodes with random actions to see if we can get any wins
    print("\n🎮 Testing 50 episodes with random actions...")
    random_wins = 0
    random_rewards = []
    
    for episode in range(50):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_won = False
        
        while not done:
            # Take random actions
            action = np.array([np.random.randint(0, 16)])
            step_result = env.step(action)
            
            if len(step_result) == 4:
                obs, reward, terminated, truncated = step_result
                info = {}
            else:
                obs, reward, terminated, truncated, info = step_result
            
            done = terminated or truncated
            episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            
            # Check for win
            if info and isinstance(info, list) and len(info) > 0:
                if info[0].get('won', False):
                    episode_won = True
            elif info and isinstance(info, dict):
                if info.get('won', False):
                    episode_won = True
        
        if episode_won:
            random_wins += 1
            print(f"  Episode {episode + 1}: WIN (reward: {episode_reward:.2f})")
        else:
            print(f"  Episode {episode + 1}: LOSS (reward: {episode_reward:.2f})")
        
        random_rewards.append(episode_reward)
    
    random_win_rate = (random_wins / 50) * 100
    random_avg_reward = np.mean(random_rewards)
    
    print(f"\n📊 Random Action Results:")
    print(f"  Random wins: {random_wins}/50 ({random_win_rate:.1f}%)")
    print(f"  Random avg reward: {random_avg_reward:.2f}")
    
    if random_wins == 0:
        print(f"  ⚠️  Even random actions got 0 wins!")
        print(f"  💡 This suggests the game might be too hard or there's a bug")
    else:
        print(f"  ✅ Random actions can win, so the game is winnable")
        print(f"  💡 The agent should be able to learn to win")

def test_mine_density_learning():
    """Test if different mine densities affect learning ability."""
    print("🔍 Testing Mine Density Learning")
    print("=" * 50)
    
    # Test different mine densities
    test_configs = [
        {"size": 4, "mines": 1, "density": "6.25%", "description": "Very Easy"},
        {"size": 4, "mines": 2, "density": "12.5%", "description": "Easy"},
        {"size": 4, "mines": 3, "density": "18.75%", "description": "Medium"},
        {"size": 4, "mines": 4, "density": "25%", "description": "Hard"},
    ]
    
    for config in test_configs:
        print(f"\n🎮 Testing {config['description']}: 4x4 with {config['mines']} mines ({config['density']} density)")
        
        # Create environment
        env = DummyVecEnv([make_env(max_board_size=(4, 4), max_mines=config['mines'])])
        
        # Test 20 episodes with random actions
        random_wins = 0
        random_rewards = []
        
        for episode in range(20):
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_won = False
            
            while not done:
                action = np.array([np.random.randint(0, 16)])
                step_result = env.step(action)
                
                if len(step_result) == 4:
                    obs, reward, terminated, truncated = step_result
                    info = {}
                else:
                    obs, reward, terminated, truncated, info = step_result
                
                done = terminated or truncated
                episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
                
                # Check for win
                if info and isinstance(info, list) and len(info) > 0:
                    if info[0].get('won', False):
                        episode_won = True
                elif info and isinstance(info, dict):
                    if info.get('won', False):
                        episode_won = True
            
            if episode_won:
                random_wins += 1
            
            random_rewards.append(episode_reward)
        
        random_win_rate = (random_wins / 20) * 100
        random_avg_reward = np.mean(random_rewards)
        
        print(f"  Random wins: {random_wins}/20 ({random_win_rate:.1f}%)")
        print(f"  Random avg reward: {random_avg_reward:.2f}")
        
        if random_wins > 0:
            print(f"  ✅ {config['description']} is winnable with random actions")
        else:
            print(f"  ❌ {config['description']} is too hard for random actions")

def test_training_buffer_vs_evaluation():
    """Test the actual discrepancy between training buffer and evaluation."""
    print("🔍 Testing Training Buffer vs Evaluation Discrepancy")
    print("=" * 60)
    
    # Create environment
    env = DummyVecEnv([make_env(max_board_size=(4, 4), max_mines=2)])
    
    # Create model
    model = PPO("MlpPolicy", env, verbose=0)
    
    print("✅ Environment and model created")
    
    # Train for a short period to populate the buffer
    print("\n🚀 Training for 1000 timesteps to populate buffer...")
    model.learn(total_timesteps=1000, progress_bar=False)
    
    # Check training buffer
    print(f"\n📊 Training Buffer Analysis:")
    print(f"  Buffer size: {len(model.ep_info_buffer)} episodes")
    
    if len(model.ep_info_buffer) > 0:
        buffer_episodes = list(model.ep_info_buffer)  # Convert to list for slicing
        buffer_wins = sum(1 for ep_info in buffer_episodes if ep_info.get("won", False))
        buffer_total = len(buffer_episodes)
        buffer_win_rate = (buffer_wins / buffer_total) * 100 if buffer_total > 0 else 0
        buffer_avg_reward = np.mean([ep_info["r"] for ep_info in buffer_episodes])
        
        print(f"  Buffer wins: {buffer_wins}/{buffer_total} ({buffer_win_rate:.1f}%)")
        print(f"  Buffer avg reward: {buffer_avg_reward:.2f}")
        
        # Show some example episodes
        print(f"\n📋 Sample episodes from buffer:")
        for i, ep_info in enumerate(buffer_episodes[:5]):
            print(f"  Episode {i+1}: won={ep_info.get('won', False)}, reward={ep_info['r']:.2f}, length={ep_info['l']}")
    else:
        print("  ⚠️  Buffer is empty!")
    
    # Now run evaluation
    print(f"\n🎮 Running evaluation (20 episodes)...")
    eval_wins = 0
    eval_rewards = []
    
    for episode in range(20):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_won = False
        
        while not done:
            action = model.predict(obs, deterministic=True)[0]
            step_result = env.step(action)
            
            if len(step_result) == 4:
                obs, reward, terminated, truncated = step_result
                info = {}
            else:
                obs, reward, terminated, truncated, info = step_result
            
            done = terminated or truncated
            episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            
            # Check for win
            if info and isinstance(info, list) and len(info) > 0:
                if info[0].get('won', False):
                    episode_won = True
            elif info and isinstance(info, dict):
                if info.get('won', False):
                    episode_won = True
        
        if episode_won:
            eval_wins += 1
            print(f"  Episode {episode + 1}: WIN (reward: {episode_reward:.2f})")
        else:
            print(f"  Episode {episode + 1}: LOSS (reward: {episode_reward:.2f})")
        
        eval_rewards.append(episode_reward)
    
    eval_win_rate = (eval_wins / 20) * 100
    eval_avg_reward = np.mean(eval_rewards)
    
    print(f"\n📊 Final Comparison:")
    print(f"  Training Buffer: {buffer_win_rate:.1f}% ({buffer_wins}/{buffer_total} wins)")
    print(f"  Final Evaluation: {eval_win_rate:.1f}% ({eval_wins}/20 wins)")
    print(f"  Buffer Avg Reward: {buffer_avg_reward:.2f}")
    print(f"  Final Avg Reward: {eval_avg_reward:.2f}")
    print(f"  Win Rate Difference: {abs(buffer_win_rate - eval_win_rate):.1f}%")
    print(f"  Reward Difference: {abs(buffer_avg_reward - eval_avg_reward):.2f}")
    
    if abs(buffer_win_rate - eval_win_rate) > 10:
        print(f"  ⚠️  Large discrepancy detected!")
        print(f"  💡 Possible causes:")
        print(f"     - Training buffer contains older episodes")
        print(f"     - Agent performance degraded during training")
        print(f"     - Evaluation environment differences")
    else:
        print(f"  ✅ Results are consistent")

def debug_single_game():
    """Debug a single game to see what's happening with win detection."""
    print("🔍 Debugging Single Game")
    print("=" * 50)
    
    # Create environment
    env = DummyVecEnv([make_env(max_board_size=(4, 4), max_mines=1)])
    
    print("✅ Environment created")
    
    # Play one game step by step
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    step_count = 0
    total_reward = 0
    
    while step_count < 20:  # Limit to 20 steps to avoid infinite loop
        step_count += 1
        
        # Take random action
        action = np.array([np.random.randint(0, 16)])
        step_result = env.step(action)
        
        if len(step_result) == 4:
            obs, reward, terminated, truncated = step_result
            info = {}
        else:
            obs, reward, terminated, truncated, info = step_result
        
        total_reward += reward[0] if isinstance(reward, np.ndarray) else reward
        
        print(f"Step {step_count}: Action={action[0]}, Reward={reward[0] if isinstance(reward, np.ndarray) else reward}, Terminated={terminated}, Truncated={truncated}")
        
        # Check info for win
        if info:
            if isinstance(info, list) and len(info) > 0:
                print(f"  Info: {info[0]}")
                if info[0].get('won', False):
                    print(f"  🎉 WIN DETECTED in info!")
            elif isinstance(info, dict):
                print(f"  Info: {info}")
                if info.get('won', False):
                    print(f"  🎉 WIN DETECTED in info!")
        
        if terminated or truncated:
            print(f"Game ended after {step_count} steps")
            print(f"Total reward: {total_reward}")
            
            # Check if this should be a win based on reward
            if total_reward >= 500:
                print(f"🎉 WIN DETECTED by reward (>=500)")
            else:
                print(f"❌ LOSS detected by reward (<500)")
            break

if __name__ == "__main__":
    debug_single_game()
    print("\n" + "="*60 + "\n")
    test_evaluation_win_detection()
    print("\n" + "="*60 + "\n")
    test_mine_density_learning()
    print("\n" + "="*60 + "\n")
    test_training_buffer_vs_evaluation() 