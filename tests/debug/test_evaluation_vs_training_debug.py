#!/usr/bin/env python3
"""
Detailed debug script to compare training vs evaluation environments
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from core.minesweeper_env import MinesweeperEnv
from core.train_agent import make_env, ActionMaskingWrapper, MultiBoardTrainingWrapper

def debug_training_vs_evaluation():
    """Debug the exact differences between training and evaluation environments."""
    print("🔍 Debugging Training vs Evaluation Environments")
    print("=" * 60)
    
    # Create training environment (with all wrappers)
    print("🏗️  Creating Training Environment:")
    training_env = DummyVecEnv([make_env(max_board_size=(4, 4), max_mines=1)])
    training_env = ActionMaskingWrapper(training_env)
    training_env = MultiBoardTrainingWrapper(training_env, board_variations=5)
    print("✅ Training environment created with wrappers")
    
    # Create evaluation environment (without multi-board wrapper)
    print("\n🏗️  Creating Evaluation Environment:")
    eval_env = DummyVecEnv([make_env(max_board_size=(4, 4), max_mines=1)])
    eval_env = ActionMaskingWrapper(eval_env)
    print("✅ Evaluation environment created (no multi-board wrapper)")
    
    # Train a model
    print("\n🎯 Training Model:")
    model = PPO(
        policy="MlpPolicy",
        env=training_env,
        learning_rate=0.0003,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        verbose=0
    )
    
    print("🎯 Training for 5000 timesteps...")
    model.learn(total_timesteps=5000, progress_bar=False)
    print("✅ Training completed")
    
    # Test training environment performance
    print("\n📊 Testing Training Environment Performance:")
    training_wins = 0
    training_episodes = 10
    
    for episode in range(training_episodes):
        obs = training_env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done and steps < 20:
            action, _ = model.predict(obs, deterministic=True)
            step_result = training_env.step(action)
            
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = np.any(terminated) or np.any(truncated)
            else:
                obs, reward, done, info = step_result
            
            episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            steps += 1
            
            # Check for win
            if info and len(info) > 0 and info[0].get('won', False):
                training_wins += 1
                break
        
        print(f"  Training Episode {episode+1}: Reward = {episode_reward:.1f}, Steps = {steps}")
    
    training_win_rate = (training_wins / training_episodes) * 100
    print(f"Training Win Rate: {training_win_rate:.1f}%")
    
    # Test evaluation environment performance
    print("\n📊 Testing Evaluation Environment Performance:")
    eval_wins = 0
    eval_episodes = 10
    
    for episode in range(eval_episodes):
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
                eval_wins += 1
                break
        
        print(f"  Eval Episode {episode+1}: Reward = {episode_reward:.1f}, Steps = {steps}")
    
    eval_win_rate = (eval_wins / eval_episodes) * 100
    print(f"Evaluation Win Rate: {eval_win_rate:.1f}%")
    
    # Compare environments in detail
    print("\n🔍 Detailed Environment Comparison:")
    
    # Get base environments
    training_base = training_env.venv.envs[0]
    if hasattr(training_base, 'env'):
        training_base = training_base.env
    
    eval_base = eval_env.envs[0]
    if hasattr(eval_base, 'env'):
        eval_base = eval_base.env
    
    print(f"Training Environment Type: {type(training_base)}")
    print(f"Evaluation Environment Type: {type(eval_base)}")
    print(f"Same Environment Class: {type(training_base) == type(eval_base)}")
    
    # Compare action spaces
    print(f"\nAction Spaces:")
    print(f"Training: {training_env.action_space}")
    print(f"Evaluation: {eval_env.action_space}")
    print(f"Same Action Space: {training_env.action_space == eval_env.action_space}")
    
    # Compare observation spaces
    print(f"\nObservation Spaces:")
    print(f"Training: {training_env.observation_space}")
    print(f"Evaluation: {eval_env.observation_space}")
    print(f"Same Observation Space: {training_env.observation_space == eval_env.observation_space}")
    
    # Test board generation
    print(f"\n🎲 Board Generation Test:")
    
    # Reset both environments and compare initial states
    training_obs = training_env.reset()
    eval_obs = eval_env.reset()
    
    print(f"Training obs shape: {training_obs.shape}")
    print(f"Evaluation obs shape: {eval_obs.shape}")
    print(f"Same initial obs: {np.array_equal(training_obs, eval_obs)}")
    
    # Get board states
    training_mines = training_base.mines.copy()
    eval_mines = eval_base.mines.copy()
    
    print(f"Training mines: {np.where(training_mines)}")
    print(f"Evaluation mines: {np.where(eval_mines)}")
    print(f"Same mine layout: {np.array_equal(training_mines, eval_mines)}")
    
    # Test action masking
    print(f"\n🎭 Action Masking Test:")
    
    if hasattr(training_base, 'get_action_mask'):
        training_mask = training_base.get_action_mask()
        eval_mask = eval_base.get_action_mask()
        print(f"Training mask: {training_mask}")
        print(f"Evaluation mask: {eval_mask}")
        print(f"Same action mask: {np.array_equal(training_mask, eval_mask)}")
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"Training Win Rate: {training_win_rate:.1f}%")
    print(f"Evaluation Win Rate: {eval_win_rate:.1f}%")
    print(f"Discrepancy: {training_win_rate - eval_win_rate:.1f}%")
    
    if eval_win_rate > 0:
        print("✅ Agent can generalize to evaluation environment!")
    else:
        print("❌ Agent cannot generalize to evaluation environment")
        print("💡 This suggests fundamental differences between training and evaluation")

if __name__ == "__main__":
    debug_training_vs_evaluation() 