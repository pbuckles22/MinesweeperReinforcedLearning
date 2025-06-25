#!/usr/bin/env python3
"""
Comprehensive evaluation debug script
Tests training vs evaluation differences, environment states, and action selection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.core.minesweeper_env import MinesweeperEnv
from src.core.train_agent import make_env

def compare_training_vs_evaluation_step_by_step():
    """Compare training vs evaluation step by step to find differences."""
    print("🔍 Comparing Training vs Evaluation Step by Step")
    print("=" * 70)
    
    # Create environment
    env = DummyVecEnv([make_env(max_board_size=(4, 4), max_mines=1)])
    
    # Create and train a model briefly
    print("🚀 Training model for 1000 timesteps...")
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=1000, progress_bar=False)
    
    print("✅ Model trained")
    
    # Get a sample episode from training buffer
    if len(model.ep_info_buffer) > 0:
        buffer_episodes = list(model.ep_info_buffer)
        wins = [ep for ep in buffer_episodes if ep.get('won', False)]
        
        if len(wins) > 0:
            print(f"📊 Found {len(wins)} wins in training buffer")
            print("🎯 Let's try to reproduce one of these wins...")
            
            # Try to reproduce a win by following the same actions
            reproduce_win_with_same_actions(model, env)
        else:
            print("❌ No wins found in training buffer")
    else:
        print("❌ Training buffer is empty")

def reproduce_win_with_same_actions(model, env):
    """Try to reproduce a win by following the same actions as training."""
    print("\n🎮 Attempting to reproduce training win...")
    
    # Reset environment
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    step_count = 0
    total_reward = 0
    
    while step_count < 20:
        step_count += 1
        
        # Get action from model (deterministic)
        action, _ = model.predict(obs, deterministic=True)
        print(f"\nStep {step_count}:")
        print(f"  Action: {action}")
        print(f"  Action shape: {action.shape if hasattr(action, 'shape') else 'scalar'}")
        
        # Take step (handle both 4-value and 5-value returns)
        step_result = env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = np.any(terminated) or np.any(truncated)
        else:
            obs, reward, done, info = step_result
            terminated = done
        total_reward += reward[0] if isinstance(reward, np.ndarray) else reward
        
        print(f"  Reward: {reward}")
        print(f"  Terminated: {terminated}")
        print(f"  Info: {info}")
        print(f"  Total reward: {total_reward}")
        
        if (len(step_result) == 5 and np.any(terminated)) or (len(step_result) == 4 and done):
            print(f"  Game ended after {step_count} steps")
            if info and len(info) > 0 and info[0].get('won', False):
                print(f"  🎉 WIN REPRODUCED!")
            else:
                print(f"  ❌ LOSS - couldn't reproduce win")
            break

def test_environment_state_consistency():
    """Test if environment states are consistent between training and evaluation."""
    print("\n🔍 Testing Environment State Consistency")
    print("=" * 70)
    
    # Create two identical environments
    env1 = DummyVecEnv([make_env(max_board_size=(4, 4), max_mines=1)])
    env2 = DummyVecEnv([make_env(max_board_size=(4, 4), max_mines=1)])
    
    print("✅ Created two identical environments")
    
    # Reset both with same seed
    seed = 42
    obs1 = env1.reset()
    obs2 = env2.reset()
    
    print(f"Environment 1 initial obs shape: {obs1.shape}")
    print(f"Environment 2 initial obs shape: {obs2.shape}")
    
    # Compare initial states
    print(f"Initial observations identical: {np.array_equal(obs1, obs2)}")
    
    # Take same action on both environments
    action = np.array([0])
    print(f"\nTaking action {action} on both environments...")
    
    # Robustly handle both 4-value and 5-value returns
    step_result1 = env1.step(action)
    step_result2 = env2.step(action)
    if len(step_result1) == 5:
        obs1_new, reward1, terminated1, truncated1, info1_new = step_result1
        obs2_new, reward2, terminated2, truncated2, info2_new = step_result2
        done1 = np.any(terminated1) or np.any(truncated1)
        done2 = np.any(terminated2) or np.any(truncated2)
    else:
        obs1_new, reward1, done1, info1_new = step_result1
        obs2_new, reward2, done2, info2_new = step_result2
        terminated1 = done1
        terminated2 = done2
    
    print(f"Environment 1: reward={reward1}, terminated={terminated1}, won={info1_new[0].get('won', False) if info1_new else False}")
    print(f"Environment 2: reward={reward2}, terminated={terminated2}, won={info2_new[0].get('won', False) if info2_new else False}")
    print(f"Rewards identical: {np.array_equal(reward1, reward2)}")
    print(f"Termination identical: {np.array_equal(terminated1, terminated2)}")
    print(f"Win status identical: {info1_new[0].get('won', False) == info2_new[0].get('won', False) if info1_new and info2_new else False}")

def test_action_selection_differences():
    """Test differences between stochastic and deterministic action selection."""
    print("\n🔍 Testing Action Selection Differences")
    print("=" * 70)
    
    # Create environment
    env = DummyVecEnv([make_env(max_board_size=(4, 4), max_mines=1)])
    
    # Create model
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=500, progress_bar=False)
    
    print("✅ Model trained")
    
    # Test same observation with different action selection methods
    obs = env.reset()
    
    print(f"Testing action selection on observation shape: {obs.shape}")
    
    # Get deterministic action
    action_det, _ = model.predict(obs, deterministic=True)
    print(f"Deterministic action: {action_det}")
    
    # Get stochastic action
    action_stoch, _ = model.predict(obs, deterministic=False)
    print(f"Stochastic action: {action_stoch}")
    
    # Test multiple stochastic actions
    print("\nTesting multiple stochastic actions:")
    for i in range(5):
        action, _ = model.predict(obs, deterministic=False)
        print(f"  Stochastic {i+1}: {action}")
    
    # Check if actions are different
    actions_different = not np.array_equal(action_det, action_stoch)
    print(f"\nActions different (stochastic vs deterministic): {actions_different}")

def test_evaluation_process():
    """Test the exact evaluation process used in training."""
    print("\n🔍 Testing Evaluation Process")
    print("=" * 70)
    
    # Create environment
    env = DummyVecEnv([make_env(max_board_size=(4, 4), max_mines=1)])
    
    # Create and train model
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=1000, progress_bar=False)
    
    print("✅ Model trained")
    
    # Run evaluation exactly like the training script does
    eval_episodes = 20
    eval_wins = 0
    eval_rewards = []
    
    print(f"🎮 Running evaluation with {eval_episodes} episodes...")
    
    for episode in range(eval_episodes):
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
            
            done = np.any(terminated) or np.any(truncated)
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
    
    eval_win_rate = (eval_wins / eval_episodes) * 100
    eval_avg_reward = np.mean(eval_rewards)
    
    print(f"\n📊 Evaluation Results:")
    print(f"  Wins: {eval_wins}/{eval_episodes} ({eval_win_rate:.1f}%)")
    print(f"  Average reward: {eval_avg_reward:.2f}")
    
    # Compare with training buffer
    if len(model.ep_info_buffer) > 0:
        buffer_episodes = list(model.ep_info_buffer)
        buffer_wins = sum(1 for ep in buffer_episodes if ep.get("won", False))
        buffer_total = len(buffer_episodes)
        buffer_win_rate = (buffer_wins / buffer_total) * 100 if buffer_total > 0 else 0
        
        print(f"\n📊 Training Buffer Comparison:")
        print(f"  Training Buffer: {buffer_win_rate:.1f}% ({buffer_wins}/{buffer_total} wins)")
        print(f"  Evaluation: {eval_win_rate:.1f}% ({eval_wins}/{eval_episodes} wins)")
        print(f"  Difference: {abs(buffer_win_rate - eval_win_rate):.1f}%")

def test_manual_win_reproduction():
    """Manually try to win a game to verify the process works."""
    print("\n🔍 Testing Manual Win Reproduction")
    print("=" * 70)
    
    # Create environment
    env = MinesweeperEnv(
        max_board_size=(4, 4),
        max_mines=1,
        initial_board_size=(4, 4),
        initial_mines=1,
        render_mode=None
    )
    
    obs, info = env.reset()
    
    print("✅ Environment created")
    
    # Find mine location
    mine_locations = np.where(env.mines)
    mine_row, mine_col = mine_locations[0][0], mine_locations[1][0]
    print(f"Mine at: ({mine_row}, {mine_col})")
    
    # Manually reveal all safe cells
    safe_cells = 15
    revealed_count = 0
    
    print("🎮 Manually revealing all safe cells...")
    
    for row in range(4):
        for col in range(4):
            if (row, col) != (mine_row, mine_col):
                action = row * 4 + col
                print(f"  Revealing ({row}, {col}) with action {action}")
                
                obs, reward, terminated, truncated, info = env.step(action)
                revealed_count += 1
                
                print(f"    Reward: {reward}, Terminated: {terminated}, Won: {info.get('won', False)}")
                print(f"    Revealed: {np.sum(env.revealed)}/{safe_cells}")
                
                if terminated:
                    print(f"    Game ended!")
                    if info.get('won', False):
                        print(f"    🎉 MANUAL WIN SUCCESSFUL!")
                    else:
                        print(f"    ❌ Manual win failed")
                    break
        
        if env.terminated:
            break
    
    print(f"\n📊 Manual Test Results:")
    print(f"  Cells revealed: {revealed_count}")
    print(f"  Total revealed: {np.sum(env.revealed)}")
    print(f"  Win condition: {env._check_win()}")
    print(f"  Game terminated: {env.terminated}")
    print(f"  Final info: {info}")

def make_seeded_env(seed=42, **kwargs):
    def _init():
        env = MinesweeperEnv(**kwargs)
        env.seed(seed)
        return env
    return _init

def print_board_and_mask(env, obs):
    # Print the underlying board and action mask for debugging
    if hasattr(env, 'envs'):
        base_env = env.envs[0]
    else:
        base_env = env
    print("Current board (mines):")
    print(base_env.mines)
    print("Revealed:")
    print(base_env.revealed)
    print("Action mask:")
    if hasattr(base_env, 'get_action_mask'):
        print(base_env.get_action_mask())
    elif hasattr(base_env, 'action_mask'):
        print(base_env.action_mask)
    else:
        print("No action mask available.")
    print("Observation:")
    print(obs)

def test_seeded_training_and_evaluation():
    print("\n🔍 Testing Seeded Training and Evaluation (Fixed Board)")
    print("=" * 70)
    seed = 123
    env_kwargs = dict(max_board_size=(4, 4), max_mines=1, initial_board_size=(4, 4), initial_mines=1, render_mode=None)
    env = DummyVecEnv([make_seeded_env(seed=seed, **env_kwargs)])
    eval_env = DummyVecEnv([make_seeded_env(seed=seed, **env_kwargs)])

    print("Training and evaluation will use the same board (seeded)")
    print("Training board:")
    print_board_and_mask(env, env.reset())

    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=1000, progress_bar=False)
    print("✅ Model trained on fixed board")

    print("\nEvaluating on the same fixed board:")
    obs = eval_env.reset()
    print_board_and_mask(eval_env, obs)
    done = False
    step = 0
    total_reward = 0
    while not done and step < 20:
        step += 1
        action, _ = model.predict(obs, deterministic=True)
        print(f"\nStep {step}: Action {action}")
        
        # Handle both 4-value and 5-value returns
        step_result = eval_env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = np.any(terminated) or np.any(truncated)
        else:
            obs, reward, done, info = step_result
            terminated = done
            
        total_reward += reward[0] if isinstance(reward, np.ndarray) else reward
        print_board_and_mask(eval_env, obs)
        print(f"  Reward: {reward}, Terminated: {terminated}, Info: {info}")
        
        if done:
            if info and len(info) > 0 and info[0].get('won', False):
                print(f"  🎉 WIN reproduced on fixed board!")
            else:
                print(f"  ❌ LOSS on fixed board.")
            break
    print(f"Total reward: {total_reward}")

if __name__ == "__main__":
    print("🚀 Starting Comprehensive Evaluation Debug")
    print("=" * 80)
    
    # Test 1: Manual win reproduction (baseline)
    test_manual_win_reproduction()
    
    # Test 2: Environment state consistency
    test_environment_state_consistency()
    
    # Test 3: Action selection differences
    test_action_selection_differences()
    
    # Test 4: Training vs evaluation comparison
    compare_training_vs_evaluation_step_by_step()
    
    # Test 5: Exact evaluation process
    test_evaluation_process()
    
    # Test 6: Seeded training and evaluation
    test_seeded_training_and_evaluation()
    
    print("\n" + "=" * 80)
    print("✅ Comprehensive evaluation debug completed") 