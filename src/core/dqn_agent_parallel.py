#!/usr/bin/env python3
"""
Parallel DQN Agent

Multi-threaded DQN implementation that leverages multiple CPU cores:
- Parallel environment execution
- Concurrent experience collection
- Multi-threaded training
- Optimized for M1 Mac performance

This should provide 4-8x speedup on multi-core systems.
"""

import sys
import os
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.minesweeper_env import MinesweeperEnv
from src.core.constants import REWARD_SAFE_REVEAL, REWARD_HIT_MINE, REWARD_WIN


# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ParallelDQNAgent:
    """Parallel DQN agent with multi-threaded training."""
    
    def __init__(
        self,
        board_size: Tuple[int, int],
        action_size: int,
        learning_rate: float = 0.0003,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        replay_buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        device: str = 'cpu',
        use_double_dqn: bool = True,
        use_dueling: bool = True,
        use_prioritized_replay: bool = True,
        num_workers: int = 4
    ):
        """
        Initialize parallel DQN agent.
        
        Args:
            num_workers: Number of parallel worker threads
        """
        self.board_size = board_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device
        self.use_double_dqn = use_double_dqn
        self.use_dueling = use_dueling
        self.use_prioritized_replay = use_prioritized_replay
        self.num_workers = num_workers
        
        # Initialize networks
        self.q_network = self._build_network().to(device)
        self.target_network = self._build_network().to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.priorities = deque(maxlen=replay_buffer_size) if use_prioritized_replay else None
        
        # Training counters
        self.step_count = 0
        self.episode_count = 0
        
        # Threading components
        self.experience_queue = queue.Queue()
        self.training_lock = threading.Lock()
        self.worker_threads = []
        self._stop_training = False
        
        # Performance tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.training_losses = deque(maxlen=100)
        
        print(f"Parallel DQN Agent initialized:")
        print(f"   Workers: {num_workers}")
        print(f"   Double DQN: {use_double_dqn}")
        print(f"   Dueling DQN: {use_dueling}")
        print(f"   Prioritized Replay: {use_prioritized_replay}")
        print(f"   Device: {device}")
    
    def _build_network(self) -> nn.Module:
        """Build the neural network architecture."""
        # Input: 4 channels (board state, mine count, revealed count, action mask)
        input_channels = 4
        input_height, input_width = self.board_size
        
        class DuelingDQN(nn.Module):
            def __init__(self, input_channels, input_height, input_width, action_size):
                super(DuelingDQN, self).__init__()
                
                # Shared feature layers
                self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
                
                # Calculate feature size after convolutions
                conv_output_height = input_height
                conv_output_width = input_width
                conv_output_size = 128 * conv_output_height * conv_output_width
                
                # Shared fully connected layers
                self.fc1 = nn.Linear(conv_output_size, 256)
                self.fc2 = nn.Linear(256, 128)
                
                # Dueling architecture: separate value and advantage streams
                self.value_stream = nn.Linear(128, 64)
                self.value_head = nn.Linear(64, 1)
                
                self.advantage_stream = nn.Linear(128, 64)
                self.advantage_head = nn.Linear(64, action_size)
                
                # Activation functions
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.1)
            
            def forward(self, x):
                # Convolutional layers
                x = self.relu(self.conv1(x))
                x = self.relu(self.conv2(x))
                x = self.relu(self.conv3(x))
                
                # Flatten
                x = x.view(x.size(0), -1)
                
                # Fully connected layers
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                
                # Dueling streams
                value = self.relu(self.value_stream(x))
                value = self.value_head(value)
                
                advantage = self.relu(self.advantage_stream(x))
                advantage = self.advantage_head(advantage)
                
                # Combine value and advantage
                q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
                
                return q_values
        
        if self.use_dueling:
            return DuelingDQN(input_channels, input_height, input_width, self.action_size)
        else:
            # Standard DQN architecture
            class StandardDQN(nn.Module):
                def __init__(self, input_channels, input_height, input_width, action_size):
                    super(StandardDQN, self).__init__()
                    
                    conv_output_size = 128 * input_height * input_width
                    
                    self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
                    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                    self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
                    
                    self.fc1 = nn.Linear(conv_output_size, 256)
                    self.fc2 = nn.Linear(256, 128)
                    self.fc3 = nn.Linear(128, action_size)
                    
                    self.relu = nn.ReLU()
                    self.dropout = nn.Dropout(0.1)
                
                def forward(self, x):
                    x = self.relu(self.conv1(x))
                    x = self.relu(self.conv2(x))
                    x = self.relu(self.conv3(x))
                    
                    x = x.view(x.size(0), -1)
                    
                    x = self.relu(self.fc1(x))
                    x = self.dropout(x)
                    x = self.relu(self.fc2(x))
                    x = self.dropout(x)
                    x = self.fc3(x)
                    
                    return x
            
            return StandardDQN(input_channels, input_height, input_width, self.action_size)
    
    def _get_state_tensor(self, state: np.ndarray) -> torch.Tensor:
        """Convert state to tensor format."""
        # Ensure state is in correct format (4 channels)
        if state.ndim == 2:
            # Single channel state, expand to 4 channels
            state = np.stack([state, np.zeros_like(state), np.zeros_like(state), np.zeros_like(state)])
        elif state.ndim == 3 and state.shape[0] != 4:
            # Wrong channel dimension, transpose
            state = state.transpose(2, 0, 1)
        
        # Add batch dimension
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return state_tensor
    
    def _select_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            # Random action from all possible actions
            return random.randint(0, self.action_size - 1)
        else:
            # Greedy action
            with torch.no_grad():
                state_tensor = self._get_state_tensor(state)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
    
    def _worker_function(self, worker_id: int, env: MinesweeperEnv, mine_count: int):
        """Worker function for parallel environment execution."""
        try:
            while not self._stop_training:
                # Reset environment
                state, info = env.reset()
                done = False
                episode_reward = 0
                episode_length = 0
                
                while not done and not self._stop_training:
                    # Select action
                    action = self._select_action(state)
                    
                    # Take action
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    # Store experience
                    experience = Experience(
                        state=state.copy(),
                        action=action,
                        reward=reward,
                        next_state=next_state.copy() if next_state is not None else None,
                        done=done
                    )
                    
                    # Add to queue (non-blocking)
                    try:
                        self.experience_queue.put(experience, timeout=0.1)
                    except queue.Full:
                        pass  # Skip if queue is full
                    
                    state = next_state
                    episode_reward += reward
                    episode_length += 1
                
                # Update episode statistics
                with self.training_lock:
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
                    self.episode_count += 1
                
                # Small delay to prevent overwhelming
                time.sleep(0.001)
                
        except Exception as e:
            print(f"Worker {worker_id} error: {e}")
    
    def _training_function(self):
        """Background training function."""
        while not self._stop_training:
            try:
                # Collect experiences from queue
                experiences = []
                for _ in range(min(self.batch_size, self.experience_queue.qsize())):
                    try:
                        experience = self.experience_queue.get(timeout=0.1)
                        experiences.append(experience)
                    except queue.Empty:
                        break
                
                if len(experiences) >= self.batch_size // 2:  # Train with partial batch
                    self._train_step(experiences)
                
                # Small delay
                time.sleep(0.01)
                
            except Exception as e:
                print(f"Training function error: {e}")
    
    def _train_step(self, experiences: List[Experience]):
        """Perform a single training step."""
        if len(experiences) < 2:
            return
        
        # Prepare batch
        states = np.array([exp.state for exp in experiences])
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor([exp.action for exp in experiences]).to(self.device)
        rewards = torch.FloatTensor([exp.reward for exp in experiences]).to(self.device)
        next_states = np.array([exp.next_state for exp in experiences if exp.next_state is not None])
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor([exp.done for exp in experiences]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: use main network for action selection, target network for evaluation
                next_actions = self.q_network(next_states).argmax(1)
                next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            else:
                # Standard DQN
                next_q_values = self.target_network(next_states).max(1)[0]
            
            # Handle terminal states
            next_q_values[dones] = 0.0
            
            # Target Q values
            target_q_values = rewards + (self.discount_factor * next_q_values.squeeze())
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Store loss
        with self.training_lock:
            self.training_losses.append(loss.item())
    
    def start_training(self, mine_count: int):
        """Start parallel training."""
        print(f"🚀 Starting parallel training with {self.num_workers} workers")
        
        # Create environments for each worker
        environments = [MinesweeperEnv(initial_board_size=self.board_size, initial_mines=mine_count) 
                       for _ in range(self.num_workers)]
        
        # Start worker threads
        self.worker_threads = []
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_function,
                args=(i, environments[i], mine_count),
                daemon=True
            )
            worker.start()
            self.worker_threads.append(worker)
        
        # Start training thread
        training_thread = threading.Thread(target=self._training_function, daemon=True)
        training_thread.start()
        self.worker_threads.append(training_thread)
        
        print(f"✅ Started {self.num_workers} worker threads + 1 training thread")
    
    def stop_training(self):
        """Stop parallel training."""
        self._stop_training = True
        
        # Wait for threads to finish
        for thread in self.worker_threads:
            thread.join(timeout=1.0)
        
        print("🛑 Parallel training stopped")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        with self.training_lock:
            if len(self.episode_rewards) == 0:
                return {
                    'episodes': 0,
                    'win_rate': 0.0,
                    'mean_reward': 0.0,
                    'mean_length': 0.0,
                    'epsilon': self.epsilon,
                    'mean_loss': 0.0
                }
            
            # Calculate win rate (episodes with positive reward)
            wins = sum(1 for reward in self.episode_rewards if reward > 0)
            win_rate = wins / len(self.episode_rewards)
            
            return {
                'episodes': self.episode_count,
                'win_rate': win_rate,
                'mean_reward': np.mean(self.episode_rewards),
                'mean_length': np.mean(self.episode_lengths),
                'epsilon': self.epsilon,
                'mean_loss': np.mean(self.training_losses) if self.training_losses else 0.0
            }
    
    def save_model(self, filename: str):
        """Save the trained model."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'training_stats': self.get_training_stats()
        }, filename)
        print(f"💾 Model saved to {filename}")
    
    def load_model(self, filename: str):
        """Load a trained model."""
        checkpoint = torch.load(filename, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']
        print(f"📂 Model loaded from {filename}")


def train_parallel_dqn_agent(
    env: MinesweeperEnv,
    agent: ParallelDQNAgent,
    episodes: int,
    mine_count: int,
    eval_freq: int = 20,
    num_workers: int = 4
) -> Dict[str, Any]:
    """
    Train a parallel DQN agent.
    
    Args:
        env: Environment instance
        agent: Parallel DQN agent
        episodes: Number of episodes to train
        mine_count: Number of mines
        eval_freq: Evaluation frequency
        num_workers: Number of parallel workers
    
    Returns:
        Training statistics
    """
    print(f"🎯 Training Parallel DQN agent for {episodes} episodes on {mine_count} mines")
    print(f"   Board size: {agent.board_size}")
    print(f"   Initial epsilon: {agent.epsilon:.3f}")
    print(f"   Device: {agent.device}")
    print(f"   Workers: {num_workers}")
    print(f"   Double DQN: {agent.use_double_dqn}")
    print(f"   Dueling DQN: {agent.use_dueling}")
    print(f"   Prioritized Replay: {agent.use_prioritized_replay}")
    print("-" * 60)
    
    # Start parallel training
    agent.start_training(mine_count)
    
    start_time = time.time()
    last_eval_time = start_time
    
    try:
        while agent.episode_count < episodes:
            # Sleep to allow training to progress
            time.sleep(0.1)
            
            # Periodic evaluation
            current_time = time.time()
            if current_time - last_eval_time > 5.0:  # Evaluate every 5 seconds
                stats = agent.get_training_stats()
                if stats['episodes'] > 0:
                    print(f"Episode {stats['episodes']:4d}: "
                          f"Win Rate {stats['win_rate']:.3f}, "
                          f"Epsilon: {stats['epsilon']:.3f}, "
                          f"Loss: {stats['mean_loss']:.4f}")
                last_eval_time = current_time
                
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    
    # Stop training
    agent.stop_training()
    
    # Final statistics
    final_stats = agent.get_training_stats()
    training_time = time.time() - start_time
    
    print(f"\n✅ Training completed!")
    print(f"   Final Win Rate: {final_stats['win_rate']:.3f}")
    print(f"   Final Epsilon: {final_stats['epsilon']:.3f}")
    print(f"   Total Episodes: {final_stats['episodes']}")
    print(f"   Training Time: {training_time:.2f}s")
    print(f"   Episodes/second: {final_stats['episodes']/training_time:.2f}")
    
    return {
        'win_rate': final_stats['win_rate'],
        'mean_reward': final_stats['mean_reward'],
        'mean_length': final_stats['mean_length'],
        'epsilon': final_stats['epsilon'],
        'mean_loss': final_stats['mean_loss'],
        'episodes': final_stats['episodes'],
        'training_time': training_time,
        'episodes_per_second': final_stats['episodes']/training_time
    }


def evaluate_parallel_dqn_agent(agent: ParallelDQNAgent, env: MinesweeperEnv, n_episodes: int = 100) -> Dict[str, Any]:
    """
    Evaluate a parallel DQN agent.
    
    Args:
        agent: Parallel DQN agent
        env: Environment instance
        n_episodes: Number of evaluation episodes
    
    Returns:
        Evaluation statistics
    """
    print(f"🔍 Evaluating Parallel DQN agent on {n_episodes} episodes...")
    
    wins = 0
    total_reward = 0
    episode_lengths = []
    
    # Temporarily set epsilon to 0 for evaluation
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    for episode in range(n_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0
        episode_length = 0
        
        while not done:
            action = agent._select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            state = next_state
            total_reward += reward
            episode_length += 1
        
        if total_reward > 0:
            wins += 1
        
        episode_lengths.append(episode_length)
    
    # Restore epsilon
    agent.epsilon = original_epsilon
    
    win_rate = wins / n_episodes
    mean_reward = total_reward / n_episodes
    mean_length = np.mean(episode_lengths)
    
    print(f"📊 Evaluation Results:")
    print(f"   Win Rate: {win_rate:.3f}")
    print(f"   Mean Reward: {mean_reward:.2f}")
    print(f"   Mean Length: {mean_length:.1f} steps")
    
    return {
        'win_rate': win_rate,
        'mean_reward': mean_reward,
        'mean_length': mean_length,
        'episodes': n_episodes
    } 