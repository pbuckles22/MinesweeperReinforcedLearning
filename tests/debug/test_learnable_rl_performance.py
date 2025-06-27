#!/usr/bin/env python3
"""
Comprehensive RL training test with learnable environment.
Tests DQN performance with learnable-only boards vs random boards.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import time
import json
from datetime import datetime
from core.dqn_agent import DQNAgent
from core.minesweeper_env import MinesweeperEnv
from stable_baselines3.common.vec_env import DummyVecEnv

def create_env(board_size, mines, learnable_only=True):
    """Create environment with specified parameters."""
    return MinesweeperEnv(
        max_board_size=board_size,
        initial_board_size=board_size,
        max_mines=mines,
        initial_mines=mines,
        learnable_only=learnable_only,
        max_learnable_attempts=1000
    )

def create_agent(board_size, action_size):
    """Create DQN agent with optimized parameters."""
    return DQNAgent(
        board_size=board_size,
        action_size=action_size,
        learning_rate=0.0001,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
        replay_buffer_size=50000,
        batch_size=32,
        target_update_freq=500
    )

def train_agent(agent, env, episodes, max_steps_per_episode=50, learnable_only=False):
    """Train agent for specified number of episodes."""
    print(f"   🎯 Training for {episodes} episodes...")
    
    episode_rewards = []
    episode_lengths = []
    wins = 0
    losses = 0
    first_move_losses = 0  # Track first-move losses separately
    
    start_time = time.time()
    
    for episode in range(episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done and steps < max_steps_per_episode:
            action = agent.choose_action(obs, training=True)
            next_obs, reward, done, info = env.step([action])
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
            
            if terminated or truncated:
                break
        
        # Update statistics - check for win
        won = False
        if info and len(info) > 0:
            won = info[0].get('won', False)
        
        # Handle first-move losses differently for learnable environments
        if not won and steps == 1 and learnable_only:
            # First-move loss in learnable environment - don't count as loss
            first_move_losses += 1
            # Don't increment losses, but still count episode
        elif won:
            wins += 1
        else:
            losses += 1
        
        episode_rewards.append(float(episode_reward))  # Convert to Python float
        episode_lengths.append(steps)
        
        # Update epsilon
        agent.update_epsilon()
        
        # Progress update
        if (episode + 1) % 50 == 0:
            current_win_rate = (wins / (episode + 1)) * 100
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"      Episode {episode + 1}: Win rate {current_win_rate:.1f}%, Avg reward {avg_reward:.2f}")
    
    training_time = time.time() - start_time
    
    return {
        'episodes': episodes,
        'wins': wins,
        'losses': losses,
        'first_move_losses': first_move_losses,
        'win_rate': (wins / episodes) * 100,
        'avg_reward': float(np.mean(episode_rewards)),
        'avg_length': float(np.mean(episode_lengths)),
        'training_time': float(training_time),
        'final_epsilon': float(agent.epsilon)
    }

def evaluate_agent(agent, env, episodes=100, learnable_only=False):
    """Evaluate agent performance."""
    print(f"   📊 Evaluating for {episodes} episodes...")
    
    wins = 0
    losses = 0
    first_move_losses = 0
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done and steps < 50:
            action = agent.choose_action(obs, training=False)  # No exploration
            obs, reward, done, info = env.step([action])
            terminated = bool(done[0])
            truncated = False
            
            episode_reward += reward[0]
            steps += 1
            
            # Check for win - only rely on info['won'] flag
            won = False
            if info and len(info) > 0:
                won = info[0].get('won', False)
            
            if won:
                wins += 1
                break
            
            if terminated or truncated:
                break
        
        # Handle first-move losses differently for learnable environments
        if not won and steps == 1 and learnable_only:
            # First-move loss in learnable environment - don't count as loss
            first_move_losses += 1
        elif not won:
            losses += 1
        
        episode_rewards.append(float(episode_reward))  # Convert to Python float
        episode_lengths.append(steps)
    
    return {
        'episodes': episodes,
        'wins': wins,
        'losses': losses,
        'first_move_losses': first_move_losses,
        'win_rate': (wins / episodes) * 100,
        'avg_reward': float(np.mean(episode_rewards)),
        'avg_length': float(np.mean(episode_lengths))
    }

def test_learnable_vs_random():
    """Compare learnable vs random environment performance."""
    print("🧪 Learnable Environment RL Performance Test")
    print("=" * 60)
    
    # Test configuration
    board_size = (4, 4)
    mines = 1
    training_episodes = 500
    eval_episodes = 100
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'board_size': board_size,
        'mines': mines,
        'training_episodes': training_episodes,
        'eval_episodes': eval_episodes,
        'learnable': {},
        'random': {}
    }
    
    # Test 1: Learnable Environment
    print("\n1️⃣ Testing Learnable Environment:")
    print("   📋 Configuration: 4x4 board, 1 mine, learnable-only")
    
    # Create learnable environment
    learnable_env = DummyVecEnv([lambda: create_env(board_size, mines, learnable_only=True)])
    learnable_agent = create_agent(board_size, 16)
    
    # Train learnable agent
    learnable_train_results = train_agent(learnable_agent, learnable_env, training_episodes, learnable_only=True)
    
    # Evaluate learnable agent
    learnable_eval_results = evaluate_agent(learnable_agent, learnable_env, eval_episodes, learnable_only=True)
    
    results['learnable'] = {
        'training': learnable_train_results,
        'evaluation': learnable_eval_results
    }
    
    print(f"   ✅ Learnable Training Complete:")
    print(f"      Training Win Rate: {learnable_train_results['win_rate']:.1f}%")
    print(f"      Evaluation Win Rate: {learnable_eval_results['win_rate']:.1f}%")
    print(f"      Training Time: {learnable_train_results['training_time']:.1f}s")
    if learnable_train_results.get('first_move_losses', 0) > 0:
        print(f"      First-move Losses (excluded): {learnable_train_results['first_move_losses']}")
    
    # Test 2: Random Environment
    print("\n2️⃣ Testing Random Environment:")
    print("   📋 Configuration: 4x4 board, 1 mine, random placement")
    
    # Create random environment
    random_env = DummyVecEnv([lambda: create_env(board_size, mines, learnable_only=False)])
    random_agent = create_agent(board_size, 16)
    
    # Train random agent
    random_train_results = train_agent(random_agent, random_env, training_episodes, learnable_only=False)
    
    # Evaluate random agent
    random_eval_results = evaluate_agent(random_agent, random_env, eval_episodes, learnable_only=False)
    
    results['random'] = {
        'training': random_train_results,
        'evaluation': random_eval_results
    }
    
    print(f"   ✅ Random Training Complete:")
    print(f"      Training Win Rate: {random_train_results['win_rate']:.1f}%")
    print(f"      Evaluation Win Rate: {random_eval_results['win_rate']:.1f}%")
    print(f"      Training Time: {random_train_results['training_time']:.1f}s")
    
    # Test 3: Performance Comparison
    print("\n3️⃣ Performance Comparison:")
    
    learnable_win_rate = learnable_eval_results['win_rate']
    random_win_rate = random_eval_results['win_rate']
    
    print(f"   📊 Learnable Environment: {learnable_win_rate:.1f}% win rate")
    print(f"   📊 Random Environment: {random_win_rate:.1f}% win rate")
    
    if learnable_win_rate > random_win_rate:
        improvement = learnable_win_rate - random_win_rate
        print(f"   🎉 Learnable environment improves performance by {improvement:.1f}%")
    elif random_win_rate > learnable_win_rate:
        difference = random_win_rate - learnable_win_rate
        print(f"   ⚠️  Random environment performs {difference:.1f}% better")
    else:
        print("   ➖ Both environments perform similarly")
    
    # Test 4: Curriculum Progression Test
    print("\n4️⃣ Curriculum Progression Test:")
    
    # Test larger board with learnable environment
    larger_board_size = (5, 5)
    larger_mines = 2
    
    print(f"   📋 Testing 5x5 board with 2 mines (learnable-only)")
    
    larger_env = DummyVecEnv([lambda: create_env(larger_board_size, larger_mines, learnable_only=True)])
    larger_agent = create_agent(larger_board_size, 25)
    
    # Debug: Check agent and environment compatibility
    print(f"   🔍 Debug: Agent board size: {larger_agent.board_size}")
    print(f"   🔍 Debug: Agent action size: {larger_agent.action_size}")
    
    # Test environment reset to see actual state shape
    test_obs = larger_env.reset()
    print(f"   🔍 Debug: Environment obs shape: {test_obs.shape}")
    
    # Quick training on larger board
    larger_results = train_agent(larger_agent, larger_env, 200, learnable_only=True)
    larger_eval = evaluate_agent(larger_agent, larger_env, 50, learnable_only=True)
    
    print(f"   ✅ 5x5 Learnable Training Complete:")
    print(f"      Training Win Rate: {larger_results['win_rate']:.1f}%")
    print(f"      Evaluation Win Rate: {larger_eval['win_rate']:.1f}%")
    
    results['curriculum'] = {
        'board_size': larger_board_size,
        'mines': larger_mines,
        'training': larger_results,
        'evaluation': larger_eval
    }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"learnable_rl_performance_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {filename}")
    
    # Summary
    print("\n🎯 Summary:")
    print(f"   ✅ Learnable Environment: {learnable_win_rate:.1f}% win rate")
    print(f"   ✅ Random Environment: {random_win_rate:.1f}% win rate")
    print(f"   ✅ Curriculum (5x5): {larger_eval['win_rate']:.1f}% win rate")
    
    if learnable_win_rate > random_win_rate:
        print("   🎉 Learnable environment shows improved learning!")
    else:
        print("   📊 Both environments perform similarly - learnable may help with curriculum learning")
    
    return results

if __name__ == "__main__":
    results = test_learnable_vs_random()
    print("\n✅ Learnable Environment RL Performance Test Complete!") 