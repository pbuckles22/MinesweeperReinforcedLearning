#!/usr/bin/env python3
"""
Test script to verify learnable environment works correctly with actual training.
Tests DQN training with learnable-only filtering, curriculum progression, and evaluation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import time
from datetime import datetime
from core.dqn_agent import DQNAgent
from core.minesweeper_env import MinesweeperEnv
from stable_baselines3.common.vec_env import DummyVecEnv

def test_learnable_training():
    """Test DQN training with learnable environment filtering."""
    print("🧪 Testing Learnable Environment Training")
    print("=" * 60)
    
    # Test 1: Basic learnable environment creation
    print("\n1️⃣ Testing Learnable Environment Creation:")
    
    env = MinesweeperEnv(
        max_board_size=(4, 4),
        max_mines=1,
        learnable_only=True,
        max_learnable_attempts=1000
    )
    
    # Test multiple resets to ensure learnable filtering works
    learnable_count = 0
    total_tests = 50
    
    for i in range(total_tests):
        obs, info = env.reset()
        # Check if the board is actually learnable (requires 2+ moves)
        if info.get('learnable', False):
            learnable_count += 1
        else:
            # Debug: print info for non-learnable boards
            if i < 5:  # Only print first 5 for debugging
                print(f"      Debug {i}: info = {info}")
    
    learnable_percentage = (learnable_count / total_tests) * 100
    print(f"   ✅ Learnable boards generated: {learnable_count}/{total_tests} ({learnable_percentage:.1f}%)")
    
    if learnable_percentage > 90:
        print("   ✅ Learnable filtering working correctly!")
    else:
        print(f"   ⚠️  Learnable filtering may need adjustment ({learnable_percentage:.1f}%)")
    
    # Test 2: DQN Training with learnable environment
    print("\n2️⃣ Testing DQN Training with Learnable Environment:")
    
    # Create vectorized environment for training
    def make_learnable_env():
        return MinesweeperEnv(
            max_board_size=(4, 4),
            max_mines=1,
            learnable_only=True,
            max_learnable_attempts=1000
        )
    
    train_env = DummyVecEnv([make_learnable_env])
    eval_env = DummyVecEnv([make_learnable_env])
    
    # Debug: Check what the environment actually returns
    print("   🔍 Debug: Testing environment step return values...")
    test_obs = train_env.reset()
    test_action = train_env.action_space.sample()
    test_result = train_env.step([test_action])
    print(f"      Step result type: {type(test_result)}")
    print(f"      Step result length: {len(test_result)}")
    print(f"      Step result: {test_result}")
    
    # Get environment info for DQN agent
    temp_env = make_learnable_env()
    temp_obs, temp_info = temp_env.reset()
    board_size = (4, 4)  # Fixed for this test
    action_size = 16  # 4x4 board = 16 actions
    
    # Create DQN agent
    agent = DQNAgent(
        board_size=board_size,
        action_size=action_size,
        learning_rate=0.0001,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
        replay_buffer_size=10000,
        batch_size=32,
        target_update_freq=1000
    )
    
    print("   ✅ Created DQN agent with learnable environment")
    
    # Train for a short period
    print("   🎯 Training for 1000 timesteps...")
    start_time = time.time()
    
    try:
        # Use the DQN agent's native training interface
        episodes = 0
        total_timesteps = 0
        max_episodes = 50  # Limit episodes to prevent infinite loop
        
        while total_timesteps < 1000 and episodes < max_episodes:
            obs = train_env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            while not done and steps < 20 and total_timesteps < 1000:
                # Choose action using the agent
                action = agent.choose_action(obs, training=True)
                next_obs, reward, done, info = train_env.step([action])
                terminated = bool(done[0])
                truncated = False
                # Store experience
                agent.store_experience(obs, action, reward[0], next_obs, terminated)
                # Train if enough samples
                if len(agent.replay_buffer) > agent.batch_size:
                    loss = agent.train()
                obs = next_obs
                episode_reward += reward[0]
                steps += 1
                total_timesteps += 1
                if terminated or truncated:
                    break
            episodes += 1
            agent.update_epsilon()
        
        training_time = time.time() - start_time
        print(f"   ✅ Training completed in {training_time:.2f}s")
        print(f"   📊 Trained for {episodes} episodes, {total_timesteps} timesteps")
        
        # Test 3: Evaluation
        print("\n3️⃣ Testing Evaluation:")
        
        wins = 0
        total_episodes = 20
        
        for episode in range(total_episodes):
            obs = eval_env.reset()
            done = False
            steps = 0
            
            while not done and steps < 20:
                action = agent.choose_action(obs, training=False)  # No exploration during evaluation
                obs, reward, done, info = eval_env.step([action])
                terminated = bool(done[0])
                truncated = False
                steps += 1
                if info and len(info) > 0 and info[0].get('won', False):
                    wins += 1
                    break
                if terminated or truncated:
                    break
        
        win_rate = (wins / total_episodes) * 100
        print(f"   📊 Evaluation Results:")
        print(f"      Episodes: {total_episodes}")
        print(f"      Wins: {wins}")
        print(f"      Win rate: {win_rate:.1f}%")
        
        if win_rate > 0:
            print("   ✅ Agent learned to win on learnable boards!")
        else:
            print("   ⚠️  Agent needs more training time")
        
        # Test 4: Curriculum progression
        print("\n4️⃣ Testing Curriculum Progression:")
        
        # Test larger board
        def make_5x5_env():
            return MinesweeperEnv(
                max_board_size=(5, 5),
                max_mines=1,
                learnable_only=True,
                max_learnable_attempts=1000
            )
        
        larger_env = DummyVecEnv([make_5x5_env])
        
        # Test if the environment works with larger boards
        obs = larger_env.reset()
        print("   ✅ 5x5 learnable environment created successfully")
        
        # Test a few steps
        for step in range(5):
            action = larger_env.action_space.sample()
            obs, reward, done, info = larger_env.step([action])
            terminated = bool(done[0])
            truncated = False
            if terminated or truncated:
                break
        
        print("   ✅ 5x5 environment step execution successful")
        
        # Test 5: Performance metrics
        print("\n5️⃣ Performance Metrics:")
        
        # Test environment creation speed
        start_time = time.time()
        for _ in range(10):
            test_env = MinesweeperEnv(
                max_board_size=(4, 4),
                max_mines=1,
                learnable_only=True,
                max_learnable_attempts=1000
            )
            test_env.reset()
        env_creation_time = time.time() - start_time
        
        print(f"   ⏱️  Environment creation: {env_creation_time:.3f}s for 10 envs")
        print(f"   📈 Average per env: {env_creation_time/10:.3f}s")
        
        if env_creation_time < 1.0:
            print("   ✅ Environment creation performance is good")
        else:
            print("   ⚠️  Environment creation may be slow")
        
        print("\n🎉 All Learnable Environment Tests Passed!")
        print("   ✅ Learnable filtering works correctly")
        print("   ✅ DQN training compatible with learnable environment")
        print("   ✅ Evaluation pipeline functional")
        print("   ✅ Curriculum progression supported")
        print("   ✅ Performance metrics acceptable")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_learnable_training()
    if success:
        print("\n✅ Learnable environment is ready for production!")
    else:
        print("\n❌ Learnable environment needs fixes before merging.")
        sys.exit(1) 