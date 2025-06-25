#!/usr/bin/env python3
"""
Q-Learning Agent for Minesweeper with Experience Replay

Implements Q-learning with experience replay to address catastrophic forgetting
and improve learning across different mine counts and board configurations.
"""

import numpy as np
import random
from collections import deque
from typing import Dict, List, Tuple, Optional, Any
import json
import os
from datetime import datetime
import pickle


class ExperienceReplayBuffer:
    """Experience replay buffer to store and sample past experiences."""
    
    def __init__(self, max_size: int = 10000):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool, mine_count: int):
        """Add experience to buffer."""
        experience = {
            'state': state.copy(),
            'action': action,
            'reward': reward,
            'next_state': next_state.copy(),
            'done': done,
            'mine_count': mine_count
        }
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Dict]:
        """Sample a batch of experiences."""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class QLearningAgent:
    """Q-learning agent with experience replay for Minesweeper."""
    
    def __init__(self, board_size: Tuple[int, int], max_mines: int,
                 learning_rate: float = 0.1, discount_factor: float = 0.99,
                 epsilon: float = 0.1, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, replay_buffer_size: int = 10000,
                 batch_size: int = 32):
        
        self.board_size = board_size
        self.max_mines = max_mines
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # Calculate action space size (all possible cell positions)
        self.action_space_size = board_size[0] * board_size[1]
        
        # Initialize Q-table as nested dictionary for sparse storage
        # Q-table structure: {state_hash: {action: q_value}}
        self.q_table = {}
        
        # Experience replay buffer
        self.replay_buffer = ExperienceReplayBuffer(replay_buffer_size)
        
        # Training statistics
        self.training_stats = {
            'episodes': 0,
            'total_reward': 0,
            'wins': 0,
            'losses': 0,
            'epsilon_history': [],
            'win_rate_history': []
        }
    
    def _state_to_hash(self, state: np.ndarray) -> str:
        """Convert state array to hash string for Q-table lookup."""
        # Use the first channel (game state) for hashing
        game_state = state[0] if len(state.shape) > 2 else state
        return str(game_state.tobytes())
    
    def _get_q_value(self, state: np.ndarray, action: int) -> float:
        """Get Q-value for state-action pair."""
        state_hash = self._state_to_hash(state)
        if state_hash not in self.q_table:
            self.q_table[state_hash] = {}
        if action not in self.q_table[state_hash]:
            self.q_table[state_hash][action] = 0.0
        return self.q_table[state_hash][action]
    
    def _set_q_value(self, state: np.ndarray, action: int, value: float):
        """Set Q-value for state-action pair."""
        state_hash = self._state_to_hash(state)
        if state_hash not in self.q_table:
            self.q_table[state_hash] = {}
        self.q_table[state_hash][action] = value
    
    def _get_valid_actions(self, state: np.ndarray) -> List[int]:
        """Get list of valid actions (unrevealed cells)."""
        # Handle 4-channel state: use channel 0 (game state)
        if len(state.shape) > 2:
            game_state = state[0]  # First channel contains the game state
        else:
            game_state = state
            
        valid_actions = []
        for i in range(self.action_space_size):
            row = i // self.board_size[1]
            col = i % self.board_size[1]
            # Ensure we get a single value and handle the comparison properly
            cell_value = game_state[row, col]
            if isinstance(cell_value, np.ndarray):
                # If it's an array, take the first element or flatten
                if cell_value.size == 1:
                    cell_value = cell_value.item()
                else:
                    # For multi-element arrays, use the first element
                    cell_value = cell_value.flatten()[0]
            if cell_value == -1:  # Unrevealed cell
                valid_actions.append(i)
        return valid_actions
    
    def choose_action(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy."""
        valid_actions = self._get_valid_actions(state)
        
        if not valid_actions:
            return 0  # Fallback action
        
        if training and random.random() < self.epsilon:
            # Exploration: random action
            return random.choice(valid_actions)
        else:
            # Exploitation: best action
            q_values = [self._get_q_value(state, action) for action in valid_actions]
            max_q = max(q_values)
            best_actions = [action for action, q in zip(valid_actions, q_values) if q == max_q]
            return random.choice(best_actions)
    
    def update_q_value(self, state: np.ndarray, action: int, reward: float,
                      next_state: np.ndarray, done: bool):
        """Update Q-value using Q-learning update rule."""
        current_q = self._get_q_value(state, action)
        
        if done:
            # Terminal state
            max_next_q = 0
        else:
            # Non-terminal state
            valid_next_actions = self._get_valid_actions(next_state)
            if valid_next_actions:
                next_q_values = [self._get_q_value(next_state, action) for action in valid_next_actions]
                max_next_q = max(next_q_values)
            else:
                max_next_q = 0
        
        # Q-learning update rule
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self._set_q_value(state, action, new_q)
    
    def train_on_replay(self, mine_count: int):
        """Train on a batch of experiences from replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = self.replay_buffer.sample(self.batch_size)
        
        for experience in batch:
            # Only train on experiences from similar mine counts to prevent forgetting
            if abs(experience['mine_count'] - mine_count) <= 1:
                self.update_q_value(
                    experience['state'],
                    experience['action'],
                    experience['reward'],
                    experience['next_state'],
                    experience['done']
                )
    
    def update_epsilon(self):
        """Decay epsilon for exploration."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filepath: str):
        """Save Q-table and training stats."""
        model_data = {
            'q_table': self.q_table,
            'training_stats': self.training_stats,
            'board_size': self.board_size,
            'max_mines': self.max_mines,
            'epsilon': self.epsilon
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load Q-table and training stats."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.q_table = model_data['q_table']
        self.training_stats = model_data['training_stats']
        self.epsilon = model_data['epsilon']
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        total_episodes = self.training_stats['episodes']
        win_rate = self.training_stats['wins'] / max(total_episodes, 1)
        
        return {
            'episodes': total_episodes,
            'wins': self.training_stats['wins'],
            'losses': self.training_stats['losses'],
            'win_rate': win_rate,
            'epsilon': self.epsilon,
            'q_table_size': len(self.q_table)
        }


def train_q_learning_agent(env, agent: QLearningAgent, episodes: int,
                          mine_count: int, eval_freq: int = 100) -> Dict[str, Any]:
    """Train Q-learning agent on environment."""
    
    print(f"🎯 Training Q-learning agent for {episodes} episodes on {mine_count} mines")
    print(f"   Board size: {agent.board_size}")
    print(f"   Initial epsilon: {agent.epsilon:.3f}")
    print("-" * 60)
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        max_steps = 200
        
        while not done and steps < max_steps:
            # Choose action
            action = agent.choose_action(state, training=True)
            
            # Take action - wrap in list for vectorized environment
            next_state, reward, done, info = env.step([action])
            
            # Store experience in replay buffer
            agent.replay_buffer.add(state, action, reward, next_state, done, mine_count)
            
            # Update Q-value
            agent.update_q_value(state, action, reward, next_state, done)
            
            # Train on replay buffer
            agent.train_on_replay(mine_count)
            
            state = next_state
            total_reward += reward
            steps += 1
        
        # Update statistics
        agent.training_stats['episodes'] += 1
        agent.training_stats['total_reward'] += total_reward
        
        # Check if episode was won
        won = False
        if info and isinstance(info, dict):
            won = info.get('won', False)
        elif info and isinstance(info, list) and len(info) > 0:
            won = info[0].get('won', False)
        
        if won:
            agent.training_stats['wins'] += 1
        else:
            agent.training_stats['losses'] += 1
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Update epsilon
        agent.update_epsilon()
        
        # Evaluation and progress reporting
        if (episode + 1) % eval_freq == 0:
            stats = agent.get_stats()
            recent_win_rate = sum(1 for r in episode_rewards[-eval_freq:] if r > 0) / eval_freq
            
            print(f"Episode {episode + 1:4d}: Win Rate {stats['win_rate']:.3f} "
                  f"(Recent: {recent_win_rate:.3f}), Epsilon: {agent.epsilon:.3f}")
    
    # Final statistics
    final_stats = agent.get_stats()
    final_stats.update({
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'mean_reward': np.mean(episode_rewards),
        'mean_length': np.mean(episode_lengths)
    })
    
    print(f"\n✅ Training completed!")
    print(f"   Final Win Rate: {final_stats['win_rate']:.3f}")
    print(f"   Final Epsilon: {agent.epsilon:.3f}")
    print(f"   Q-table size: {final_stats['q_table_size']}")
    
    return final_stats


def evaluate_q_agent(agent: QLearningAgent, env, n_episodes: int = 100) -> Dict[str, Any]:
    """Evaluate Q-learning agent performance."""
    
    print(f"🔍 Evaluating Q-learning agent on {n_episodes} episodes...")
    
    wins = 0
    rewards = []
    lengths = []
    
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        max_steps = 200
        
        while not done and steps < max_steps:
            action = agent.choose_action(state, training=False)  # No exploration
            state, reward, done, info = env.step([action])  # Wrap in list for vectorized env
            total_reward += reward
            steps += 1
        
        # Check if episode was won
        won = False
        if info and isinstance(info, dict):
            won = info.get('won', False)
        elif info and isinstance(info, list) and len(info) > 0:
            won = info[0].get('won', False)
        
        if won:
            wins += 1
        
        rewards.append(total_reward)
        lengths.append(steps)
    
    win_rate = wins / n_episodes
    mean_reward = np.mean(rewards)
    mean_length = np.mean(lengths)
    
    results = {
        'win_rate': win_rate,
        'mean_reward': mean_reward,
        'mean_length': mean_length,
        'wins': wins,
        'total_episodes': n_episodes,
        'rewards': rewards,
        'lengths': lengths
    }
    
    print(f"📊 Evaluation Results:")
    print(f"   Win Rate: {win_rate:.3f}")
    print(f"   Mean Reward: {mean_reward:.2f}")
    print(f"   Mean Length: {mean_length:.1f} steps")
    
    return results


if __name__ == "__main__":
    # Example usage
    from core.minesweeper_env import MinesweeperEnv
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    # Create environment
    board_size = (4, 4)
    max_mines = 2
    
    env = DummyVecEnv([lambda: MinesweeperEnv(board_size=board_size, max_mines=max_mines)])
    
    # Create Q-learning agent
    agent = QLearningAgent(
        board_size=board_size,
        max_mines=max_mines,
        learning_rate=0.1,
        epsilon=0.3,
        epsilon_decay=0.9995
    )
    
    # Train agent
    stats = train_q_learning_agent(env, agent, episodes=1000, mine_count=max_mines)
    
    # Evaluate agent
    eval_results = evaluate_q_agent(agent, env, n_episodes=100)
    
    # Save model
    agent.save_model("models/q_learning_minesweeper.pkl") 