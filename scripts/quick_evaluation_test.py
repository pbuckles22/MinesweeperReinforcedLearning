#!/usr/bin/env python3
"""
Quick test of the evaluation function to ensure it doesn't get stuck.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.train_agent_modular import train_modular, make_modular_env
from stable_baselines3.common.vec_env import DummyVecEnv


def evaluate_model(model, env, n_eval_episodes=10, verbose=True):
    """
    Quick evaluation function with safety measures.
    """
    if verbose:
        print(f"🔍 Quick evaluation with {n_eval_episodes} episodes...")
    
    rewards = []
    wins = 0
    
    for episode in range(n_eval_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_won = False
        step_count = 0
        max_steps = 50  # Safety limit
        
        while not done and step_count < max_steps:
            try:
                action = model.predict(obs, deterministic=True)[0]
                obs, reward, done, info = env.step(action)
                episode_reward += reward[0] if isinstance(reward, list) else reward
                step_count += 1
                
                # Check for win condition
                if info and isinstance(info, list) and len(info) > 0:
                    if info[0].get('won', False):
                        episode_won = True
                        done = True
                elif info and isinstance(info, dict):
                    if info.get('won', False):
                        episode_won = True
                        done = True
                        
            except Exception as e:
                print(f"⚠️  Error in episode {episode}, step {step_count}: {e}")
                done = True
                break
        
        if step_count >= max_steps:
            print(f"⚠️  Episode {episode} hit max steps ({max_steps})")
            done = True
        
        rewards.append(episode_reward)
        if episode_won:
            wins += 1
        
        if verbose:
            print(f"   Episode {episode + 1}: {'WIN' if episode_won else 'LOSS'} (steps: {step_count})")
    
    mean_reward = sum(rewards) / len(rewards) if rewards else 0
    win_rate = wins / n_eval_episodes
    
    return win_rate, mean_reward, wins, n_eval_episodes


def main():
    print("🧪 Quick Evaluation Test")
    print("=" * 30)
    
    # Train a quick model
    print("🚀 Training quick model...")
    model, win_rate, mean_reward = train_modular(
        board_size=(4, 4),
        max_mines=2,
        total_timesteps=1000,
        device="cpu"
    )
    
    # Create environment for evaluation
    env = DummyVecEnv([make_modular_env((4, 4), 2)])
    
    # Quick evaluation
    print("\n🔍 Running quick evaluation...")
    win_rate_large, mean_reward_large, wins, total_episodes = evaluate_model(
        model, env, n_eval_episodes=10, verbose=True
    )
    
    print(f"\n📊 Quick Results:")
    print(f"   Win Rate: {win_rate_large:.3f} ({wins}/{total_episodes})")
    print(f"   Mean Reward: {float(mean_reward_large):.2f}")
    print(f"✅ Evaluation completed successfully!")


if __name__ == "__main__":
    main() 