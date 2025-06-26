#!/usr/bin/env python3
"""
Multi-Step DQN Agent for Minesweeper

Implements DQN with multi-step returns for better credit assignment:
- N-step returns (configurable, default 3-5 steps)
- Enhanced experience replay with multi-step experiences
- Better temporal credit assignment
- Improved learning efficiency

Based on successful enhanced DQN architecture with multi-step learning.
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


# Multi-step experience tuple
MultiStepExperience = namedtuple('MultiStepExperience', [
    'state', 'action', 'rewards', 'next_state', 'done', 'priority', 'n_steps'
])


class MultiStepExperienceReplayBuffer:
    """Experience replay buffer with multi-step experiences."""
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.buffer = []
        self.priorities = []
        self.position = 0
        self.max_priority = 1.0
        
    def add(self, experience: MultiStepExperience):
        """Add multi-step experience to buffer with maximum priority."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(self.max_priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = self.max_priority
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List[MultiStepExperience], List[int], List[float]]:
        """Sample a batch of multi-step experiences with priorities."""
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


class MultiStepDQNAgent:
    """Multi-step DQN agent with enhanced learning capabilities."""
    
    def __init__(self, board_size: Tuple[int, int], action_size: int,
                 learning_rate: float = 0.0003, discount_factor: float = 0.99,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, replay_buffer_size: int = 100000,
                 batch_size: int = 64, target_update_freq: int = 1000,
                 device: str = 'auto', use_double_dqn: bool = True,
                 use_dueling: bool = True, use_prioritized_replay: bool = True,
                 n_steps: int = 3, max_n_steps: int = 5):
        
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
        self.n_steps = n_steps
        self.max_n_steps = max_n_steps
        
        # Device setup
        if device == 'auto':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        if self.device.type == 'cpu':
            print("   Note: Using CPU for optimal performance on Mac")
        
        # Networks (import from enhanced DQN)
        from src.core.dqn_agent_enhanced import DuelingDQNNetwork
        
        if use_dueling:
            self.q_network = DuelingDQNNetwork(board_size, action_size).to(self.device)
            self.target_network = DuelingDQNNetwork(board_size, action_size).to(self.device)
        else:
            from src.core.dqn_agent import DQNNetwork
            self.q_network = DQNNetwork(board_size, action_size).to(self.device)
            self.target_network = DQNNetwork(board_size, action_size).to(self.device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Multi-step experience replay buffer
        if use_prioritized_replay:
            self.replay_buffer = MultiStepExperienceReplayBuffer(replay_buffer_size)
        else:
            # Fallback to regular replay buffer
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
            'loss_history': [],
            'n_step_returns': []
        }
        
        # Target network update counter
        self.target_update_counter = 0
        
        # Multi-step experience buffer for collecting N-step returns
        self.step_buffer = deque(maxlen=max_n_steps)
        
        print(f"Multi-Step DQN Agent initialized:")
        print(f"   Double DQN: {use_double_dqn}")
        print(f"   Dueling DQN: {use_dueling}")
        print(f"   Prioritized Replay: {use_prioritized_replay}")
        print(f"   N-steps: {n_steps} (max: {max_n_steps})")
    
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
        """Choose action using epsilon-greedy policy."""
        valid_actions = self._get_valid_actions(state)
        
        if not valid_actions:
            return 0  # Fallback action
        
        if training and random.random() < self.epsilon:
            # Exploration: random action
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
    
    def _compute_n_step_return(self, rewards: List[float], n_steps: int) -> float:
        """Compute N-step return for a sequence of rewards."""
        if len(rewards) < n_steps:
            # If we don't have enough steps, use all available rewards
            n_steps = len(rewards)
        
        # Compute discounted sum of rewards
        n_step_return = 0.0
        for i in range(n_steps):
            n_step_return += (self.discount_factor ** i) * rewards[i]
        
        return n_step_return
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store experience and handle multi-step returns."""
        # Add current step to buffer
        step_data = (state, action, reward, next_state, done)
        self.step_buffer.append(step_data)
        
        # If we have enough steps or episode is done, create multi-step experience
        if len(self.step_buffer) >= self.n_steps or done:
            # Determine actual number of steps to use
            actual_n_steps = min(len(self.step_buffer), self.n_steps)
            
            # Extract data for multi-step experience
            initial_state = self.step_buffer[0][0]  # First state
            initial_action = self.step_buffer[0][1]  # First action
            final_next_state = self.step_buffer[-1][3]  # Last next_state
            final_done = self.step_buffer[-1][4]  # Last done flag
            
            # Compute N-step return
            rewards = [step[2] for step in list(self.step_buffer)[:actual_n_steps]]
            n_step_return = self._compute_n_step_return(rewards, actual_n_steps)
            
            # Create multi-step experience
            if self.use_prioritized_replay:
                experience = MultiStepExperience(
                    initial_state, initial_action, n_step_return, 
                    final_next_state, final_done, 1.0, actual_n_steps
                )
            else:
                # For regular replay buffer, use standard experience format
                from src.core.dqn_agent import Experience
                experience = Experience(initial_state, initial_action, n_step_return, 
                                      final_next_state, final_done)
            
            self.replay_buffer.add(experience)
            
            # Track N-step return statistics
            self.training_stats['n_step_returns'].append(n_step_return)
            
            # Clear step buffer if episode is done
            if done:
                self.step_buffer.clear()
    
    def train(self) -> Optional[float]:
        """Train the network on a batch of multi-step experiences."""
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
        
        # Handle different reward formats (multi-step vs single-step)
        if hasattr(batch[0], 'rewards'):
            # Multi-step experience
            rewards = torch.tensor([exp.rewards for exp in batch], dtype=torch.float32).to(self.device)
        else:
            # Single-step experience (fallback)
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
            'use_prioritized_replay': self.use_prioritized_replay,
            'n_steps': self.n_steps,
            'max_n_steps': self.max_n_steps
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
        
        # Calculate N-step return statistics
        n_step_returns = self.training_stats['n_step_returns']
        avg_n_step_return = np.mean(n_step_returns) if n_step_returns else 0.0
        
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
            'use_prioritized_replay': self.use_prioritized_replay,
            'n_steps': self.n_steps,
            'avg_n_step_return': avg_n_step_return,
            'total_n_step_experiences': len(n_step_returns)
        }


def train_multistep_dqn_agent(env, agent: MultiStepDQNAgent, episodes: int, mine_count: int, 
                             eval_freq: int = 100) -> Dict[str, Any]:
    """Train multi-step DQN agent on environment."""
    
    print(f"🎯 Training Multi-Step DQN agent for {episodes} episodes on {mine_count} mines")
    print(f"   Board size: {agent.board_size}")
    print(f"   Initial epsilon: {agent.epsilon:.3f}")
    print(f"   Device: {agent.device}")
    print(f"   N-steps: {agent.n_steps}")
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
            
            # Store experience (handles multi-step returns internally)
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
                  f"Loss: {avg_loss:.4f}, N-step Return: {stats['avg_n_step_return']:.2f}")
    
    # Final statistics
    final_stats = agent.get_stats()
    mean_loss = np.mean(losses) if losses else 0
    
    print(f"\n✅ Training completed!")
    print(f"   Final Win Rate: {final_stats['win_rate']:.3f}")
    print(f"   Final Epsilon: {agent.epsilon:.3f}")
    print(f"   Replay Buffer Size: {len(agent.replay_buffer)}")
    print(f"   Mean Loss: {mean_loss:.4f}")
    print(f"   Avg N-step Return: {final_stats['avg_n_step_return']:.2f}")
    print(f"   Total N-step Experiences: {final_stats['total_n_step_experiences']}")
    
    return {
        'win_rate': final_stats['win_rate'],
        'mean_loss': mean_loss,
        'final_epsilon': agent.epsilon,
        'replay_buffer_size': len(agent.replay_buffer),
        'avg_n_step_return': final_stats['avg_n_step_return']
    }


def evaluate_multistep_dqn_agent(agent: MultiStepDQNAgent, env, n_episodes: int = 100) -> Dict[str, Any]:
    """Evaluate multi-step DQN agent."""
    print(f"🔍 Evaluating Multi-Step DQN agent on {n_episodes} episodes...")
    
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