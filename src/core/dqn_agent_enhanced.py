#!/usr/bin/env python3
"""
Enhanced Deep Q-Network (DQN) Agent for Minesweeper

Implements advanced DQN techniques:
- Double DQN (reduces overestimation bias)
- Prioritized Experience Replay (focuses on important experiences)
- Dueling DQN (separates value and advantage streams)
- Enhanced exploration strategies

Based on successful conv128x4_dense512x2 architecture with optimizations.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from typing import Dict, List, Tuple, Optional, Any
import json
import os
from datetime import datetime
import pickle
import heapq


# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'priority'])


class DuelingDQNNetwork(nn.Module):
    """Dueling DQN network with separate value and advantage streams."""
    
    def __init__(self, board_size: Tuple[int, int], action_size: int):
        super(DuelingDQNNetwork, self).__init__()
        
        self.board_height, self.board_width = board_size
        self.action_size = action_size
        
        # Convolutional layers (4 layers of 128 filters each)
        self.conv1 = nn.Conv2d(4, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Calculate the size after convolutions
        conv_output_size = 128 * self.board_height * self.board_width
        
        # Shared dense layers
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, 512)
        
        # Dueling streams
        self.value_stream = nn.Linear(512, 256)
        self.value_head = nn.Linear(256, 1)
        
        self.advantage_stream = nn.Linear(512, 256)
        self.advantage_head = nn.Linear(256, action_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Ensure input is the right shape: (batch_size, channels, height, width)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # Add batch dimension
        
        # Convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        # Flatten for dense layers
        x = x.view(x.size(0), -1)
        
        # Shared dense layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Dueling streams
        value = F.relu(self.value_stream(x))
        value = self.value_head(value)
        
        advantage = F.relu(self.advantage_stream(x))
        advantage = self.advantage_head(advantage)
        
        # Combine value and advantage
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


class PrioritizedExperienceReplayBuffer:
    """Prioritized experience replay buffer for DQN."""
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.buffer = []
        self.priorities = []
        self.position = 0
        self.max_priority = 1.0
        
    def add(self, experience: Experience):
        """Add experience to buffer with maximum priority."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(self.max_priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = self.max_priority
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], List[int], List[float]]:
        """Sample a batch of experiences with priorities."""
        if len(self.buffer) < batch_size:
            return list(self.buffer), list(range(len(self.buffer))), [1.0] * len(self.buffer)
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities[:len(self.buffer)])
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights
        
        # Get experiences
        experiences = [self.buffer[i] for i in indices]
        
        return experiences, indices.tolist(), weights.tolist()
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)


class EnhancedDQNAgent:
    """Enhanced Deep Q-Network agent with advanced techniques."""
    
    def __init__(self, board_size: Tuple[int, int], action_size: int,
                 learning_rate: float = 0.0003, discount_factor: float = 0.99,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, replay_buffer_size: int = 100000,
                 batch_size: int = 64, target_update_freq: int = 1000,
                 device: str = 'auto', use_double_dqn: bool = True,
                 use_dueling: bool = True, use_prioritized_replay: bool = True):
        
        self.board_size = board_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_double_dqn = use_double_dqn
        self.use_dueling = use_dueling
        self.use_prioritized_replay = use_prioritized_replay
        
        # Device setup
        if device == 'auto':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        if self.device.type == 'cpu':
            print("   Note: Using CPU for optimal performance on Mac")
        
        # Networks
        if use_dueling:
            self.q_network = DuelingDQNNetwork(board_size, action_size).to(self.device)
            self.target_network = DuelingDQNNetwork(board_size, action_size).to(self.device)
        else:
            # Import the regular DQN network
            from src.core.dqn_agent import DQNNetwork
            self.q_network = DQNNetwork(board_size, action_size).to(self.device)
            self.target_network = DQNNetwork(board_size, action_size).to(self.device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay buffer
        if use_prioritized_replay:
            self.replay_buffer = PrioritizedExperienceReplayBuffer(replay_buffer_size)
        else:
            from src.core.dqn_agent import ExperienceReplayBuffer
            self.replay_buffer = ExperienceReplayBuffer(replay_buffer_size)
        
        # Training statistics
        self.training_stats = {
            'episodes': 0,
            'total_reward': 0,
            'wins': 0,
            'losses': 0,
            'epsilon_history': [],
            'win_rate_history': [],
            'loss_history': []
        }
        
        # Target network update counter
        self.target_update_counter = 0
        
        print(f"Enhanced DQN Agent initialized:")
        print(f"   Double DQN: {use_double_dqn}")
        print(f"   Dueling DQN: {use_dueling}")
        print(f"   Prioritized Replay: {use_prioritized_replay}")
    
    def _preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        """Preprocess state for neural network input."""
        # Ensure state is float32 and in the right range
        if state.dtype != np.float32:
            state = state.astype(np.float32)
        state_tensor = torch.from_numpy(state).to(self.device)
        # Remove batch dimension if present
        if len(state_tensor.shape) == 4 and state_tensor.shape[0] == 1:
            state_tensor = state_tensor.squeeze(0)
        # Now should be [channels, height, width]
        if len(state_tensor.shape) == 3:
            return state_tensor
        elif len(state_tensor.shape) == 2:
            # Add channel dimension
            return state_tensor.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected state shape: {state_tensor.shape}")
    
    def _get_valid_actions(self, state: np.ndarray) -> List[int]:
        """Get valid actions for the current state."""
        # Handle state that might be a tuple (from env.reset()) or numpy array
        if isinstance(state, tuple):
            state = state[0]  # Extract the actual state from (state, info)
        
        # Handle state that might have different shapes
        if hasattr(state, 'shape'):
            if len(state.shape) > 2:
                game_state = state[0]  # Use first channel for game state
            else:
                game_state = state
        else:
            # If state doesn't have shape attribute, assume it's already the right format
            game_state = state
        
        valid_actions = []
        height, width = self.board_size
        
        for i in range(height):
            for j in range(width):
                action = i * width + j
                # Check if cell is unrevealed (value == -1 for unrevealed)
                if game_state[i, j] == -1:  # CELL_UNREVEALED
                    valid_actions.append(action)
        
        return valid_actions
    
    def choose_action(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy with enhanced exploration."""
        valid_actions = self._get_valid_actions(state)
        
        if not valid_actions:
            return 0  # Fallback action
        
        if training and random.random() < self.epsilon:
            # Enhanced exploration: prefer actions that haven't been tried recently
            return random.choice(valid_actions)
        else:
            # Exploitation: best action
            state_tensor = self._preprocess_state(state)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                q_values = q_values.cpu().numpy().flatten()
            
            # Mask invalid actions with very low Q-values
            masked_q_values = q_values.copy()
            masked_q_values[~np.isin(np.arange(self.action_size), valid_actions)] = -1e6
            
            # Choose best valid action
            best_action = np.argmax(masked_q_values)
            return best_action
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        if self.use_prioritized_replay:
            # For prioritized replay, we'll update the priority during training
            experience = Experience(state, action, reward, next_state, done, 1.0)
        else:
            experience = Experience(state, action, reward, next_state, done, 1.0)
        
        self.replay_buffer.add(experience)
    
    def train(self) -> Optional[float]:
        """Train the network on a batch of experiences with enhanced techniques."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        if self.use_prioritized_replay:
            batch, indices, weights = self.replay_buffer.sample(self.batch_size)
            weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        else:
            batch = self.replay_buffer.sample(self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)
            indices = None
        
        # Prepare batch data
        states = torch.stack([self._preprocess_state(exp.state) for exp in batch]).to(self.device)
        actions = torch.tensor([exp.action for exp in batch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([exp.reward for exp in batch], dtype=torch.float32).to(self.device)
        next_states_list = [self._preprocess_state(exp.next_state) for exp in batch]
        
        # Ensure next_states is [batch, channels, height, width]
        if next_states_list[0].ndim == 3:
            next_states = torch.stack(next_states_list).to(self.device)
        elif next_states_list[0].ndim == 4:
            next_states = torch.cat(next_states_list, dim=0).to(self.device)
        else:
            raise ValueError(f"Unexpected next_state shape: {next_states_list[0].shape}")
        
        dones = torch.tensor([exp.done for exp in batch], dtype=torch.bool).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q-values (using Double DQN if enabled)
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: use main network to select actions, target network to evaluate
                next_actions = self.q_network(next_states).max(1)[1]
                next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # Standard DQN: use target network for both selection and evaluation
                next_q_values = self.target_network(next_states).max(1)[0]
            
            next_q_values = next_q_values.masked_fill(dones, 0.0)
            target_q_values = rewards + self.discount_factor * next_q_values
        
        # Compute loss with importance sampling weights
        td_errors = torch.abs(current_q_values - target_q_values)
        loss = (td_errors * weights).mean()
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update priorities if using prioritized replay
        if self.use_prioritized_replay and indices is not None:
            priorities = (td_errors.detach().cpu().numpy() + 1e-6)  # Add small constant to avoid zero priority
            self.replay_buffer.update_priorities(indices, priorities)
        
        # Update target network
        self.target_update_counter += 1
        if self.target_update_counter >= self.target_update_freq:
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_update_counter = 0
        
        return loss.item()
    
    def update_epsilon(self):
        """Decay epsilon for exploration."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filepath: str):
        """Save model and training stats."""
        model_data = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'board_size': self.board_size,
            'action_size': self.action_size,
            'epsilon': self.epsilon,
            'target_update_counter': self.target_update_counter,
            'use_double_dqn': self.use_double_dqn,
            'use_dueling': self.use_dueling,
            'use_prioritized_replay': self.use_prioritized_replay
        }
        
        torch.save(model_data, filepath)
    
    def load_model(self, filepath: str):
        """Load model and training stats."""
        model_data = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(model_data['q_network_state_dict'])
        self.target_network.load_state_dict(model_data['target_network_state_dict'])
        self.optimizer.load_state_dict(model_data['optimizer_state_dict'])
        self.training_stats = model_data['training_stats']
        self.epsilon = model_data['epsilon']
        self.target_update_counter = model_data['target_update_counter']
    
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
            'replay_buffer_size': len(self.replay_buffer),
            'device': str(self.device),
            'use_double_dqn': self.use_double_dqn,
            'use_dueling': self.use_dueling,
            'use_prioritized_replay': self.use_prioritized_replay
        }


def train_enhanced_dqn_agent(env, agent: EnhancedDQNAgent, episodes: int, mine_count: int, 
                           eval_freq: int = 100) -> Dict[str, Any]:
    """Train enhanced DQN agent on environment."""
    
    print(f"🎯 Training Enhanced DQN agent for {episodes} episodes on {mine_count} mines")
    print(f"   Board size: {agent.board_size}")
    print(f"   Initial epsilon: {agent.epsilon:.3f}")
    print(f"   Device: {agent.device}")
    print(f"   Double DQN: {agent.use_double_dqn}")
    print(f"   Dueling DQN: {agent.use_dueling}")
    print(f"   Prioritized Replay: {agent.use_prioritized_replay}")
    print("-" * 60)
    
    episode_rewards = []
    episode_lengths = []
    losses = []
    
    for episode in range(episodes):
        state, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        max_steps = 200
        
        while not done and steps < max_steps:
            # Choose action
            action = agent.choose_action(state, training=True)
            
            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store experience
            agent.store_experience(state, action, reward, next_state, done)
            
            # Train the network
            loss = agent.train()
            if loss is not None:
                losses.append(loss)
            
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
            avg_loss = np.mean(losses[-eval_freq:]) if losses else 0
            
            print(f"Episode {episode + 1:4d}: Win Rate {stats['win_rate']:.3f} "
                  f"(Recent: {recent_win_rate:.3f}), Epsilon: {agent.epsilon:.3f}, "
                  f"Loss: {avg_loss:.4f}")
    
    # Final statistics
    final_stats = agent.get_stats()
    mean_loss = np.mean(losses) if losses else 0
    
    print(f"\n✅ Training completed!")
    print(f"   Final Win Rate: {final_stats['win_rate']:.3f}")
    print(f"   Final Epsilon: {agent.epsilon:.3f}")
    print(f"   Replay Buffer Size: {len(agent.replay_buffer)}")
    print(f"   Mean Loss: {mean_loss:.4f}")
    
    return {
        'win_rate': final_stats['win_rate'],
        'mean_loss': mean_loss,
        'final_epsilon': agent.epsilon,
        'replay_buffer_size': len(agent.replay_buffer)
    }


def evaluate_enhanced_dqn_agent(agent: EnhancedDQNAgent, env, n_episodes: int = 100) -> Dict[str, Any]:
    """Evaluate enhanced DQN agent."""
    print(f"🔍 Evaluating Enhanced DQN agent on {n_episodes} episodes...")
    
    wins = 0
    total_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        max_steps = 200
        
        while not done and steps < max_steps:
            # Choose action (no exploration during evaluation)
            action = agent.choose_action(state, training=False)
            
            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            state = next_state
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
        
        total_rewards.append(total_reward)
        episode_lengths.append(steps)
    
    win_rate = wins / n_episodes
    mean_reward = np.mean(total_rewards)
    mean_length = np.mean(episode_lengths)
    
    print(f"📊 Evaluation Results:")
    print(f"   Win Rate: {win_rate:.3f}")
    print(f"   Mean Reward: {mean_reward:.2f}")
    print(f"   Mean Length: {mean_length:.1f} steps")
    
    return {
        'win_rate': win_rate,
        'mean_reward': mean_reward,
        'mean_length': mean_length,
        'wins': wins,
        'total_episodes': n_episodes
    } 