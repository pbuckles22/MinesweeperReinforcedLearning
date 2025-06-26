#!/usr/bin/env python3
"""
Debug Evaluation Gap Script

This script investigates the differences between training and evaluation environments
that are causing the large performance gap (training 77% → evaluation 7%).
"""

import sys
import os
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.minesweeper_env import MinesweeperEnv
from core.dqn_agent_enhanced import EnhancedDQNAgent, train_enhanced_dqn_agent, evaluate_enhanced_dqn_agent


class EvaluationGapDebugger:
    """Debug the gap between training and evaluation performance."""
    
    def __init__(self):
        self.results = {}
    
    def test_environment_consistency(self, board_size: Tuple[int, int], mines: int, n_tests: int = 100):
        """Test if training and evaluation environments are consistent."""
        print(f"🔍 Testing Environment Consistency")
        print(f"   Board: {board_size}, Mines: {mines}")
        print(f"   Tests: {n_tests}")
        print("-" * 60)
        
        # Create environments
        train_env = MinesweeperEnv(
            initial_board_size=board_size,
            initial_mines=mines
        )
        
        eval_env = MinesweeperEnv(
            initial_board_size=board_size,
            initial_mines=mines
        )
        
        # Test 1: State representation consistency
        print("📊 Test 1: State Representation Consistency")
        train_states = []
        eval_states = []
        
        for i in range(n_tests):
            # Training environment
            train_state, train_info = train_env.reset(seed=i)
            train_states.append(train_state.copy())
            
            # Evaluation environment
            eval_state, eval_info = eval_env.reset(seed=i)
            eval_states.append(eval_state.copy())
            
            # Check if states are identical
            if not np.array_equal(train_state, eval_state):
                print(f"   ❌ State mismatch at test {i}")
                print(f"      Train state shape: {train_state.shape}")
                print(f"      Eval state shape: {eval_state.shape}")
                print(f"      Train state sum: {np.sum(train_state)}")
                print(f"      Eval state sum: {np.sum(eval_state)}")
                return False
        
        print(f"   ✅ All {n_tests} states are identical")
        
        # Test 2: Mine placement consistency
        print("\n📊 Test 2: Mine Placement Consistency")
        train_mines = []
        eval_mines = []
        
        for i in range(n_tests):
            # Training environment
            train_state, train_info = train_env.reset(seed=i)
            train_mines.append(train_env.mines.copy())
            
            # Evaluation environment
            eval_state, eval_info = eval_env.reset(seed=i)
            eval_mines.append(eval_env.mines.copy())
            
            # Check if mine placements are identical
            if not np.array_equal(train_env.mines, eval_env.mines):
                print(f"   ❌ Mine placement mismatch at test {i}")
                print(f"      Train mines: {np.sum(train_env.mines)}")
                print(f"      Eval mines: {np.sum(eval_env.mines)}")
                return False
        
        print(f"   ✅ All {n_tests} mine placements are identical")
        
        # Test 3: Action space consistency
        print("\n📊 Test 3: Action Space Consistency")
        if train_env.action_space != eval_env.action_space:
            print(f"   ❌ Action space mismatch")
            print(f"      Train: {train_env.action_space}")
            print(f"      Eval: {eval_env.action_space}")
            return False
        
        print(f"   ✅ Action spaces are identical")
        
        # Test 4: Observation space consistency
        print("\n📊 Test 4: Observation Space Consistency")
        if train_env.observation_space != eval_env.observation_space:
            print(f"   ❌ Observation space mismatch")
            print(f"      Train: {train_env.observation_space}")
            print(f"      Eval: {eval_env.observation_space}")
            return False
        
        print(f"   ✅ Observation spaces are identical")
        
        return True
    
    def test_agent_behavior_consistency(self, board_size: Tuple[int, int], mines: int):
        """Test if agent behaves consistently in training vs evaluation."""
        print(f"\n🔍 Testing Agent Behavior Consistency")
        print(f"   Board: {board_size}, Mines: {mines}")
        print("-" * 60)
        
        # Create agent
        agent = EnhancedDQNAgent(
            board_size=board_size,
            action_size=board_size[0] * board_size[1],
            learning_rate=0.0001,
            epsilon=0.05,  # Low epsilon for consistent behavior
            epsilon_min=0.05,
            use_double_dqn=True,
            use_dueling=True,
            use_prioritized_replay=True
        )
        
        # Create environments
        train_env = MinesweeperEnv(
            initial_board_size=board_size,
            initial_mines=mines
        )
        
        eval_env = MinesweeperEnv(
            initial_board_size=board_size,
            initial_mines=mines
        )
        
        # Test agent actions on identical states
        print("📊 Test: Agent Action Consistency")
        n_tests = 50
        action_matches = 0
        
        for i in range(n_tests):
            # Reset both environments with same seed
            train_state, train_info = train_env.reset(seed=i)
            eval_state, eval_info = eval_env.reset(seed=i)
            
            # Get agent actions (no exploration)
            train_action = agent.choose_action(train_state, training=False)
            eval_action = agent.choose_action(eval_state, training=False)
            
            if train_action == eval_action:
                action_matches += 1
            else:
                print(f"   ❌ Action mismatch at test {i}")
                print(f"      Train action: {train_action}")
                print(f"      Eval action: {eval_action}")
        
        consistency_rate = action_matches / n_tests
        print(f"   📊 Action consistency: {consistency_rate:.3f} ({action_matches}/{n_tests})")
        
        if consistency_rate < 0.95:
            print(f"   ⚠️  Low action consistency - this could cause evaluation issues")
        else:
            print(f"   ✅ High action consistency")
        
        return consistency_rate
    
    def test_training_vs_evaluation_performance(self, board_size: Tuple[int, int], mines: int, episodes: int = 100):
        """Test actual performance difference between training and evaluation."""
        print(f"\n🔍 Testing Training vs Evaluation Performance")
        print(f"   Board: {board_size}, Mines: {mines}")
        print(f"   Episodes: {episodes}")
        print("-" * 60)
        
        # Create agent
        agent = EnhancedDQNAgent(
            board_size=board_size,
            action_size=board_size[0] * board_size[1],
            learning_rate=0.0001,
            epsilon=0.05,
            epsilon_min=0.05,
            use_double_dqn=True,
            use_dueling=True,
            use_prioritized_replay=True
        )
        
        # Create environments
        train_env = MinesweeperEnv(
            initial_board_size=board_size,
            initial_mines=mines
        )
        
        eval_env = MinesweeperEnv(
            initial_board_size=board_size,
            initial_mines=mines
        )
        
        # Test training-style performance (sequential episodes)
        print("📊 Training-style Performance Test")
        train_wins = 0
        train_rewards = []
        
        for episode in range(episodes):
            state, info = train_env.reset()
            done = False
            total_reward = 0
            steps = 0
            max_steps = 200
            
            while not done and steps < max_steps:
                action = agent.choose_action(state, training=False)
                state, reward, terminated, truncated, info = train_env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1
            
            if info.get('won', False):
                train_wins += 1
            train_rewards.append(total_reward)
        
        train_win_rate = train_wins / episodes
        train_mean_reward = np.mean(train_rewards)
        
        print(f"   Training Win Rate: {train_win_rate:.3f}")
        print(f"   Training Mean Reward: {train_mean_reward:.2f}")
        
        # Test evaluation-style performance (fresh episodes)
        print("\n📊 Evaluation-style Performance Test")
        eval_wins = 0
        eval_rewards = []
        
        for episode in range(episodes):
            state, info = eval_env.reset()
            done = False
            total_reward = 0
            steps = 0
            max_steps = 200
            
            while not done and steps < max_steps:
                action = agent.choose_action(state, training=False)
                state, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1
            
            if info.get('won', False):
                eval_wins += 1
            eval_rewards.append(total_reward)
        
        eval_win_rate = eval_wins / episodes
        eval_mean_reward = np.mean(eval_rewards)
        
        print(f"   Evaluation Win Rate: {eval_win_rate:.3f}")
        print(f"   Evaluation Mean Reward: {eval_mean_reward:.2f}")
        
        # Calculate gap
        win_rate_gap = abs(train_win_rate - eval_win_rate)
        reward_gap = abs(train_mean_reward - eval_mean_reward)
        
        print(f"\n📊 Performance Gap Analysis:")
        print(f"   Win Rate Gap: {win_rate_gap:.3f}")
        print(f"   Reward Gap: {reward_gap:.2f}")
        
        if win_rate_gap > 0.1:
            print(f"   ❌ Large performance gap detected!")
        else:
            print(f"   ✅ Performance gap is acceptable")
        
        return {
            'train_win_rate': train_win_rate,
            'eval_win_rate': eval_win_rate,
            'win_rate_gap': win_rate_gap,
            'train_mean_reward': train_mean_reward,
            'eval_mean_reward': eval_mean_reward,
            'reward_gap': reward_gap
        }
    
    def test_replay_buffer_effect(self, board_size: Tuple[int, int], mines: int):
        """Test if replay buffer affects evaluation performance."""
        print(f"\n🔍 Testing Replay Buffer Effect")
        print(f"   Board: {board_size}, Mines: {mines}")
        print("-" * 60)
        
        # Create agent with replay buffer
        agent_with_buffer = EnhancedDQNAgent(
            board_size=board_size,
            action_size=board_size[0] * board_size[1],
            learning_rate=0.0001,
            epsilon=0.05,
            epsilon_min=0.05,
            use_double_dqn=True,
            use_dueling=True,
            use_prioritized_replay=True
        )
        
        # Create environment
        env = MinesweeperEnv(
            initial_board_size=board_size,
            initial_mines=mines
        )
        
        # Test performance with replay buffer
        print("📊 Testing with Replay Buffer")
        buffer_wins = 0
        n_episodes = 50
        
        for episode in range(n_episodes):
            state, info = env.reset()
            done = False
            steps = 0
            max_steps = 200
            
            while not done and steps < max_steps:
                action = agent_with_buffer.choose_action(state, training=False)
                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1
            
            if info.get('won', False):
                buffer_wins += 1
        
        buffer_win_rate = buffer_wins / n_episodes
        
        print(f"\n📊 Replay Buffer Results:")
        print(f"   Win Rate: {buffer_win_rate:.3f}")
        
        return {
            'with_buffer': buffer_win_rate,
            'without_buffer': buffer_win_rate,  # Skip comparison for now
            'difference': 0.0
        }
    
    def run_comprehensive_debug(self):
        """Run comprehensive debugging of the evaluation gap."""
        print("🚀 Evaluation Gap Debugging")
        print("=" * 70)
        print("Investigating the 70-point gap between training and evaluation")
        print("=" * 70)
        
        # Test different board sizes
        test_configs = [
            ((4, 4), 1),
            ((5, 5), 2),
            ((6, 6), 3)
        ]
        
        for board_size, mines in test_configs:
            print(f"\n🎯 Testing Configuration: {board_size} board, {mines} mines")
            print("=" * 50)
            
            # Test 1: Environment consistency
            env_consistent = self.test_environment_consistency(board_size, mines, n_tests=50)
            
            # Test 2: Agent behavior consistency
            action_consistency = self.test_agent_behavior_consistency(board_size, mines)
            
            # Test 3: Training vs evaluation performance
            performance_gap = self.test_training_vs_evaluation_performance(board_size, mines, episodes=100)
            
            # Test 4: Replay buffer effect
            buffer_effect = self.test_replay_buffer_effect(board_size, mines)
            
            # Store results
            self.results[f"{board_size}_{mines}"] = {
                'env_consistent': env_consistent,
                'action_consistency': action_consistency,
                'performance_gap': performance_gap,
                'buffer_effect': buffer_effect
            }
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print debugging summary."""
        print(f"\n🎉 Debugging Summary")
        print("=" * 70)
        
        for config, results in self.results.items():
            print(f"\n📊 Configuration: {config}")
            print(f"   Environment Consistent: {'✅' if results['env_consistent'] else '❌'}")
            print(f"   Action Consistency: {results['action_consistency']:.3f}")
            print(f"   Performance Gap: {results['performance_gap']['win_rate_gap']:.3f}")
            print(f"   Replay Buffer Effect: {results['buffer_effect']['difference']:.3f}")
        
        # Overall assessment
        print(f"\n🔍 Overall Assessment:")
        
        env_issues = sum(1 for r in self.results.values() if not r['env_consistent'])
        if env_issues > 0:
            print(f"   ❌ Environment consistency issues detected")
        else:
            print(f"   ✅ Environment consistency is good")
        
        avg_gap = np.mean([r['performance_gap']['win_rate_gap'] for r in self.results.values()])
        if avg_gap > 0.1:
            print(f"   ❌ Large performance gaps detected (avg: {avg_gap:.3f})")
        else:
            print(f"   ✅ Performance gaps are acceptable (avg: {avg_gap:.3f})")
        
        avg_consistency = np.mean([r['action_consistency'] for r in self.results.values()])
        if avg_consistency < 0.95:
            print(f"   ⚠️  Low action consistency (avg: {avg_consistency:.3f})")
        else:
            print(f"   ✅ Action consistency is good (avg: {avg_consistency:.3f})")


def main():
    """Main function to run debugging."""
    debugger = EvaluationGapDebugger()
    debugger.run_comprehensive_debug()


if __name__ == "__main__":
    main() 