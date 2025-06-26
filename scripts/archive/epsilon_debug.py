#!/usr/bin/env python3
"""
Epsilon Decay Debug Script

Quick test to verify epsilon decay is working correctly.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.dqn_agent_enhanced import EnhancedDQNAgent

def test_epsilon_decay():
    """Test epsilon decay over multiple episodes."""
    print("🔍 Testing Epsilon Decay")
    print("=" * 40)
    
    # Create agent with the proven configuration
    agent = EnhancedDQNAgent(
        board_size=(4, 4),
        action_size=16,
        learning_rate=0.0002,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.9995,
        epsilon_min=0.01,
        replay_buffer_size=300000,
        batch_size=128,
        target_update_freq=1000,
        device='cpu'
    )
    
    print(f"Initial epsilon: {agent.epsilon:.6f}")
    print(f"Epsilon decay: {agent.epsilon_decay}")
    print(f"Epsilon min: {agent.epsilon_min}")
    print("-" * 40)
    
    # Test decay over episodes
    for episode in range(2000):
        agent.update_epsilon()
        
        if episode % 100 == 0:
            print(f"Episode {episode:4d}: Epsilon = {agent.epsilon:.6f}")
        
        if episode % 500 == 0 and episode > 0:
            expected = 1.0 * (0.9995 ** episode)
            print(f"Episode {episode:4d}: Actual = {agent.epsilon:.6f}, Expected = {expected:.6f}")
    
    print(f"Final epsilon: {agent.epsilon:.6f}")

if __name__ == "__main__":
    test_epsilon_decay() 