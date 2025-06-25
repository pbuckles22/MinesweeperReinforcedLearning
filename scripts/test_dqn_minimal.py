#!/usr/bin/env python3
"""
Minimal test script for DQN agent to verify tensor shapes and basic functionality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.core.minesweeper_env import MinesweeperEnv
from src.core.dqn_agent import DQNAgent, train_dqn_agent, evaluate_dqn_agent

def test_dqn_minimal():
    """Test DQN agent with minimal training to verify no tensor shape errors."""
    print("🧪 Testing DQN agent minimal functionality...")
    
    # Create environment
    board_size = (4, 4)
    mine_count = 1
    env = MinesweeperEnv(initial_board_size=board_size, initial_mines=mine_count)
    
    # Create DQN agent
    action_size = board_size[0] * board_size[1]  # 16 actions for 4x4 board
    agent = DQNAgent(
        board_size=board_size,
        action_size=action_size,
        learning_rate=0.001,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        batch_size=32,  # Smaller batch for testing
        device='cpu'
    )
    
    print(f"✅ Agent created successfully")
    print(f"   Board size: {board_size}")
    print(f"   Action size: {action_size}")
    print(f"   Device: {agent.device}")
    
    # Test a few episodes
    episodes = 50
    print(f"\n🎯 Training for {episodes} episodes...")
    
    try:
        stats = train_dqn_agent(env, agent, episodes, mine_count, eval_freq=10)
        print(f"\n✅ Training completed successfully!")
        print(f"   Final win rate: {stats['win_rate']:.3f}")
        print(f"   Final epsilon: {agent.epsilon:.3f}")
        print(f"   Mean loss: {stats['mean_loss']:.4f}")
        
        # Test evaluation
        print(f"\n🔍 Testing evaluation...")
        eval_stats = evaluate_dqn_agent(agent, env, n_episodes=20)
        print(f"✅ Evaluation completed!")
        print(f"   Evaluation win rate: {eval_stats['win_rate']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dqn_minimal()
    if success:
        print("\n🎉 All tests passed! DQN agent is working correctly.")
    else:
        print("\n💥 Tests failed. Please check the error messages above.")
        sys.exit(1) 