"""
Minesweeper RL Training Agent with Curriculum Learning

This module implements a complete training pipeline for the Minesweeper environment
using Stable Baselines3 PPO with curriculum learning, experiment tracking, and
cross-platform compatibility.

Recent Updates (2024-12-19):
- M1 GPU optimization with automatic MPS detection and performance benchmarking
- Cross-platform script compatibility (Mac/Windows/Linux)
- Enhanced state representation (4-channel) with smart action masking
- Platform-specific requirements and import path resolution
- Training performance insights and monitoring tools

Key Features:
- Automatic device detection (M1 GPU, CUDA, CPU)
- Curriculum learning with 7 difficulty stages
- MLflow experiment tracking
- Custom evaluation callbacks for vectorized environments
- Graceful shutdown handling
- Performance benchmarking and monitoring
"""

import os
import numpy as np
import time
import json
import signal
import sys
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import safe_mean
from src.core.minesweeper_env import MinesweeperEnv
import argparse
import torch
from src.core.constants import REWARD_INVALID_ACTION, REWARD_HIT_MINE, REWARD_SAFE_REVEAL, REWARD_WIN
import mlflow
import mlflow.pytorch

# Global variables for graceful shutdown
training_model = None
training_env = None
eval_env = None
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global shutdown_requested
    print(f"\n🛑 Received signal {signum}. Initiating graceful shutdown...")
    shutdown_requested = True
    
    # Give a second chance to force quit
    print("Press Ctrl+C again to force quit, or wait for graceful shutdown...")
    signal.signal(signal.SIGINT, signal.SIG_DFL)  # Reset to default handler for second Ctrl+C

def detect_optimal_device():
    """
    Detect the optimal device for training, prioritizing M1 GPU (MPS) over CUDA and CPU.
    Returns the device string and a description of what was detected.
    """
    device_info = {
        'device': 'cpu',
        'description': 'CPU (fallback)',
        'performance_notes': 'Slowest option, suitable for testing only'
    }
    
    # Check for M1 GPU (MPS) - highest priority for M1 MacBooks
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device_info = {
            'device': 'mps',
            'description': 'Apple M1 GPU (MPS)',
            'performance_notes': '2-4x faster than CPU, optimal for M1 MacBooks'
        }
        print(f"✅ Detected M1 GPU (MPS) - Using device: {device_info['device']}")
        print(f"   Performance: {device_info['performance_notes']}")
        
    # Check for CUDA GPU - second priority
    elif torch.cuda.is_available():
        device_info = {
            'device': 'cuda',
            'description': f'NVIDIA GPU ({torch.cuda.get_device_name(0)})',
            'performance_notes': 'Fastest option for NVIDIA GPUs'
        }
        print(f"✅ Detected NVIDIA GPU - Using device: {device_info['device']}")
        print(f"   GPU: {device_info['description']}")
        print(f"   Performance: {device_info['performance_notes']}")
        
    else:
        print(f"⚠️  No GPU detected - Using device: {device_info['device']}")
        print(f"   Performance: {device_info['performance_notes']}")
    
    return device_info

def get_optimal_hyperparameters(device_info):
    """
    Get optimal hyperparameters based on the detected device.
    M1 GPU can handle larger batches and more complex training.
    """
    base_params = {
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5
    }
    
    if device_info['device'] == 'mps':
        # M1 GPU optimizations
        optimized_params = base_params.copy()
        optimized_params.update({
            'batch_size': 128,  # M1 can handle larger batches
            'n_steps': 2048,    # Keep standard PPO steps
            'n_epochs': 12,     # Slightly more epochs for better learning
        })
        print("🔧 Applied M1 GPU optimizations:")
        print(f"   - Increased batch size to {optimized_params['batch_size']}")
        print(f"   - Increased epochs to {optimized_params['n_epochs']}")
        return optimized_params
        
    elif device_info['device'] == 'cuda':
        # CUDA GPU optimizations
        optimized_params = base_params.copy()
        optimized_params.update({
            'batch_size': 256,  # CUDA can handle very large batches
            'n_steps': 2048,
            'n_epochs': 10,
        })
        print("🔧 Applied CUDA GPU optimizations:")
        print(f"   - Increased batch size to {optimized_params['batch_size']}")
        return optimized_params
        
    else:
        # CPU optimizations
        optimized_params = base_params.copy()
        optimized_params.update({
            'batch_size': 32,   # Smaller batches for CPU
            'n_steps': 1024,    # Fewer steps to reduce memory usage
            'n_epochs': 8,      # Fewer epochs for faster iteration
        })
        print("🔧 Applied CPU optimizations:")
        print(f"   - Reduced batch size to {optimized_params['batch_size']}")
        print(f"   - Reduced steps to {optimized_params['n_steps']}")
        return optimized_params

def benchmark_device_performance(device_info):
    """
    Test the performance of the detected device with a simple benchmark.
    """
    print(f"\n🔍 Testing {device_info['description']} performance...")
    
    device = torch.device(device_info['device'])
    
    # Create test tensors
    size = 1000
    x = torch.randn(size, size, device=device)
    y = torch.randn(size, size, device=device)
    
    # Warm up
    for _ in range(3):
        _ = torch.mm(x, y)
    
    # Benchmark
    start_time = time.time()
    for _ in range(10):
        z = torch.mm(x, y)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 10
    print(f"✅ Matrix multiplication benchmark: {avg_time:.3f}s per operation")
    
    if device_info['device'] == 'mps':
        if avg_time < 0.1:
            print("🚀 Excellent M1 GPU performance detected!")
        elif avg_time < 0.2:
            print("✅ Good M1 GPU performance detected!")
        else:
            print("⚠️  M1 GPU performance slower than expected")
    elif device_info['device'] == 'cuda':
        if avg_time < 0.05:
            print("🚀 Excellent CUDA GPU performance detected!")
        elif avg_time < 0.1:
            print("✅ Good CUDA GPU performance detected!")
        else:
            print("⚠️  CUDA GPU performance slower than expected")
    else:
        print("ℹ️  CPU performance as expected")
    
    return avg_time

class ExperimentTracker:
    def __init__(self, experiment_dir="experiments"):
        self.experiment_dir = experiment_dir
        self.current_run = None
        self.metrics = {
            "training": [],
            "validation": [],
            "hyperparameters": {},
            "metadata": {}
        }
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)

    def start_new_run(self, hyperparameters):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_run = os.path.join(self.experiment_dir, f"run_{timestamp}")
        counter = 1
        original_run = self.current_run
        while os.path.exists(self.current_run):
            self.current_run = f"{original_run}_{counter}"
            counter += 1
        os.makedirs(self.current_run)
        self.metrics["hyperparameters"] = hyperparameters
        self.metrics["metadata"] = {
            "start_time": timestamp,
            "random_seed": hyperparameters.get("random_seed", None)
        }
        self.metrics["run_id"] = f"run_{timestamp}"
        self.metrics["start_time"] = timestamp  # for test compatibility
        # Do not reset other keys (like test_metric) if present
        if "training" not in self.metrics:
            self.metrics["training"] = []
        if "validation" not in self.metrics:
            self.metrics["validation"] = []
        self._save_metrics()

    def add_training_metric(self, metric_name, value, step):
        metric = {
            "step": step,
            "metric": metric_name,
            "value": value
        }
        self.metrics["training"].append(metric)
        # Only add 'training_metrics' if needed for compatibility
        if "training_metrics" not in self.metrics:
            self.metrics["training_metrics"] = []
        self.metrics["training_metrics"].append(metric)
        self._save_metrics()

    def add_validation_metric(self, metric_name, value, confidence_interval=None):
        if hasattr(value, 'item'):
            value = value.item()
        metric_data = {
            "metric": metric_name,
            "value": value,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        if confidence_interval is not None:
            if hasattr(confidence_interval, 'item'):
                confidence_interval = confidence_interval.item()
            elif isinstance(confidence_interval, (list, tuple)):
                confidence_interval = [ci.item() if hasattr(ci, 'item') else ci for ci in confidence_interval]
            metric_data["confidence_interval"] = confidence_interval
        self.metrics["validation"].append(metric_data)
        # Only add 'validation_metrics' if needed for compatibility
        if "validation_metrics" not in self.metrics:
            self.metrics["validation_metrics"] = []
        self.metrics["validation_metrics"].append(metric_data)
        self._save_metrics()

    def _save_metrics(self):
        if self.current_run:
            metrics_file = os.path.join(self.current_run, "metrics.json")
            # Also save to experiment_dir for compatibility with tests
            experiment_metrics_file = os.path.join(self.experiment_dir, "metrics.json")
        else:
            metrics_file = os.path.join(self.experiment_dir, "metrics.json")
            experiment_metrics_file = None
        
        # Create backup if file exists
        if os.path.exists(metrics_file):
            backup_file = metrics_file.replace(".json", "_backup.json")
            try:
                import shutil
                shutil.copy2(metrics_file, backup_file)
            except Exception as e:
                print(f"Warning: Could not create backup: {e}")
        
        # Determine what to save - empty dict if ALL keys are empty
        metrics_to_save = self.metrics
        if all((not v if isinstance(v, list) else v == {} for k, v in self.metrics.items())):
            metrics_to_save = {}
        
        # Save to primary location
        with open(metrics_file, "w") as f:
            json.dump(metrics_to_save, f, indent=2)
        
        # Save to experiment_dir for compatibility if different from primary location
        if experiment_metrics_file and experiment_metrics_file != metrics_file:
            with open(experiment_metrics_file, "w") as f:
                json.dump(metrics_to_save, f, indent=2)

class CustomEvalCallback(BaseCallback):
    """Custom evaluation callback that works with our Minesweeper environment"""
    def __init__(self, eval_env, eval_freq=1000, n_eval_episodes=5, verbose=1, 
                 best_model_save_path=None, log_path=None):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        
        # Create directories if needed
        if self.best_model_save_path and not os.path.exists(self.best_model_save_path):
            os.makedirs(self.best_model_save_path)
        if self.log_path and not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
    
    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            if self.verbose > 0:
                print(f"Evaluating at step {self.n_calls}...")
            
            # Run evaluation episodes
            rewards = []
            wins = 0
            
            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done = False
                episode_reward = 0
                episode_won = False
                
                while not done:
                    action = self.model.predict(obs, deterministic=True)[0]
                    step_result = self.eval_env.step(action)
                    
                    # Handle both old gym API (4 values) and new gymnasium API (5 values)
                    if len(step_result) == 4:
                        obs, reward, terminated, truncated = step_result
                        info = {}
                    else:
                        obs, reward, terminated, truncated, info = step_result
                    
                    done = terminated or truncated
                    episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
                    
                    # Check if the episode was won from the info dictionary
                    if info and isinstance(info, list) and len(info) > 0:
                        if info[0].get('won', False):
                            episode_won = True
                
                rewards.append(episode_reward)
                if episode_won:
                    wins += 1
            
            # Calculate metrics
            mean_reward = np.mean(rewards)
            win_rate = (wins / self.n_eval_episodes) * 100
            
            if self.verbose > 0:
                print(f"Evaluation completed. Mean reward: {mean_reward:.2f}, Win rate: {win_rate:.1f}%")
            
            # Save best model if reward improved
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.best_model_save_path:
                    best_model_path = os.path.join(self.best_model_save_path, "best_model")
                    self.model.save(best_model_path)
                    if self.verbose > 0:
                        print(f"New best model saved with mean reward: {self.best_mean_reward:.2f}")
            
            # Log metrics
            if self.log_path:
                log_file = os.path.join(self.log_path, "eval_log.txt")
                with open(log_file, "a") as f:
                    f.write(f"Step {self.n_calls}: mean_reward={mean_reward:.2f}, win_rate={win_rate:.1f}%\n")
        
        return True

class IterationCallback(BaseCallback):
    """Custom callback for logging iteration information"""
    def __init__(self, verbose=0, debug_level=2, experiment_tracker=None, stats_file="training_stats.txt", timestamped_stats=False, enable_file_logging=True):
        super().__init__(verbose)
        self.start_time = time.time()
        self.last_iteration_time = self.start_time
        self.iterations = 0
        self.debug_level = debug_level
        self.timestamped_stats = timestamped_stats
        self.enable_file_logging = enable_file_logging
        
        # Handle timestamped stats files
        if self.timestamped_stats:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.stats_file = f"training_stats_{timestamp}.txt"
        else:
            self.stats_file = stats_file
        
        # Track metrics for improvement calculation
        self.last_avg_reward = 0
        self.last_win_rate = 0
        self.last_avg_length = 0
        self.best_reward = float('-inf')
        self.best_win_rate = 0
        
        # Track learning phases
        self.learning_phase = "Initial Random"
        self.phase_start_iteration = 0
        self.phase_start_reward = 0
        self.phase_start_win_rate = 0
        
        # Curriculum tracking
        self.curriculum_stage = 1
        self.stage_start_iteration = 0
        self.stage_wins = 0
        self.stage_games = 0
        
        # Progress monitoring for early termination
        self.no_improvement_count = 0
        self.last_improvement_iteration = 0
        self.stage_start_time = time.time()
        
        # Debug information
        self.last_step_time = time.time()
        self.step_times = []
        self.episode_lengths = []
        self.rewards = []
        self.wins = []
        
        # Debug level descriptions
        self.debug_levels = {
            0: "ERROR",    # Critical issues
            1: "WARNING",  # Potential problems
            2: "INFO",     # Basic progress (default)
            3: "DEBUG",    # Detailed metrics
            4: "VERBOSE"   # Everything
        }
        
        self.experiment_tracker = experiment_tracker
        
        # Initialize stats file only if file logging is enabled
        if self.enable_file_logging:
            try:
                with open(self.stats_file, 'w') as f:
                    f.write("timestamp,iteration,timesteps,win_rate,avg_reward,avg_length,stage,phase,stage_time,no_improvement\n")
            except (IOError, OSError) as e:
                # If file creation fails, disable file logging
                self.enable_file_logging = False
                if self.verbose > 0:
                    print(f"Warning: Could not create stats file {self.stats_file}: {e}")
                    print("File logging disabled for this callback.")

    def log(self, message, level=2, force=False):
        """Log message if debug level is high enough"""
        if force or level <= self.debug_level:
            print(f"[{self.debug_levels[level]}] {message}")

    def _update_learning_phase(self, avg_reward, win_rate):
        """Update the current learning phase based on performance"""
        if self.iterations == 0:
            return
        
        # Calculate improvements
        reward_improvement = ((avg_reward - self.phase_start_reward) / 
                            (abs(self.phase_start_reward) + 1e-6)) * 100
        win_rate_improvement = win_rate - self.phase_start_win_rate
        
        # Determine learning phase
        if self.iterations < 5:
            self.learning_phase = "Initial Random"
        elif win_rate < 10:
            self.learning_phase = "Early Learning"
        elif win_rate < 30:
            self.learning_phase = "Basic Strategy"
        elif win_rate < 50:
            self.learning_phase = "Intermediate"
        elif win_rate < 70:
            self.learning_phase = "Advanced"
        else:
            self.learning_phase = "Expert"
            
        # Check if we should start a new phase
        if (self.learning_phase != self.learning_phase or 
            self.iterations - self.phase_start_iteration >= 10):
            self.phase_start_iteration = self.iterations
            self.phase_start_reward = avg_reward
            self.phase_start_win_rate = win_rate

    def get_env_attr(self, env, attr):
        # Recursively unwrap until the attribute is found or no more wrappers
        # Add safety check to prevent infinite loops
        visited = set()
        while hasattr(env, 'env') and id(env) not in visited:
            visited.add(id(env))
            env = env.env
        return getattr(env, attr, None)

    def _on_step(self):
        # Check for shutdown request
        global shutdown_requested
        if shutdown_requested:
            self.log("🛑 Shutdown requested in callback. Stopping training...", level=0)
            return False  # Return False to stop training
        
        # Track step timing
        current_time = time.time()
        step_time = current_time - self.last_step_time
        self.step_times.append(step_time)
        self.last_step_time = current_time
        
        # Log every 100 steps
        if self.num_timesteps % 100 == 0:
            current_time = time.time()
            iteration_time = current_time - self.last_iteration_time
            total_time = current_time - self.start_time
            self.iterations += 1
            
            # Calculate metrics from episode info
            if len(self.model.ep_info_buffer) > 0:
                # Get current episode buffer
                current_episodes = self.model.ep_info_buffer
                
                # Calculate metrics for current episodes only
                avg_reward = np.mean([ep_info["r"] for ep_info in current_episodes])
                self.rewards.append(avg_reward)
                
                # Win rate based on actual wins (not just positive rewards)
                wins = sum(1 for ep_info in current_episodes if ep_info.get("won", False))
                total_episodes = len(current_episodes)
                win_rate = (wins / total_episodes) * 100 if total_episodes > 0 else 0
                self.wins.append(win_rate)
                
                # Average game length
                avg_length = np.mean([ep_info["l"] for ep_info in current_episodes])
                self.episode_lengths.append(avg_length)
                
                # Update curriculum stage tracking
                self.stage_games += total_episodes
                self.stage_wins += wins
                
                # Calculate improvements
                reward_improvement = ((avg_reward - self.last_avg_reward) / 
                                    (abs(self.last_avg_reward) + 1e-6)) * 100
                win_rate_improvement = win_rate - self.last_win_rate
                
                # Update best metrics
                self.best_reward = max(self.best_reward, avg_reward)
                self.best_win_rate = max(self.best_win_rate, win_rate)
                
                # Update learning phase
                self._update_learning_phase(avg_reward, win_rate)
                
                # Log to MLflow
                try:
                    mlflow.log_metric("win_rate", win_rate, step=self.iterations)
                    mlflow.log_metric("avg_episode_length", avg_length, step=self.iterations)
                    mlflow.log_metric("avg_reward", avg_reward, step=self.iterations)
                    mlflow.log_metric("best_win_rate", self.best_win_rate, step=self.iterations)
                    mlflow.log_metric("best_reward", self.best_reward, step=self.iterations)
                    mlflow.log_metric("curriculum_stage", self.curriculum_stage, step=self.iterations)
                    mlflow.log_metric("stage_win_rate", (self.stage_wins/self.stage_games*100) if self.stage_games > 0 else 0, step=self.iterations)
                except Exception as e:
                    # MLflow not initialized, skip logging
                    pass
                
                # Log progress
                self.log(f"\nIteration {self.iterations}")
                self.log(f"Time: {total_time:.1f}s (Iteration: {iteration_time:.1f}s)")
                self.log(f"Win Rate: {win_rate:.1f}% (Best: {self.best_win_rate:.1f}%)")
                self.log(f"Average Reward: {avg_reward:.2f} (Best: {self.best_reward:.2f})")
                self.log(f"Average Length: {avg_length:.1f}")
                self.log(f"Learning Phase: {self.learning_phase}")
                self.log(f"Curriculum Stage: {self.curriculum_stage}")
                self.log(f"Stage Win Rate: {(self.stage_wins/self.stage_games*100):.1f}%")
                
                # Store current metrics for next iteration
                self.last_avg_reward = avg_reward
                self.last_win_rate = win_rate
                self.last_avg_length = avg_length
                
                # Update experiment tracker
                if self.experiment_tracker:
                    self.experiment_tracker.add_training_metric("win_rate", win_rate, self.iterations)
                    self.experiment_tracker.add_training_metric("avg_reward", avg_reward, self.iterations)
                    self.experiment_tracker.add_training_metric("avg_length", avg_length, self.iterations)
                
                # Write progress to stats file for monitoring
                stage_time = time.time() - self.stage_start_time
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                # Check for improvement - track multiple types of progress
                improvement = False
                recent_rewards = []  # Initialize to avoid UnboundLocalError
                
                # 1. Check for new bests (traditional improvement)
                if win_rate > self.best_win_rate or avg_reward > self.best_reward:
                    improvement = True
                    self.last_improvement_iteration = self.iterations
                    self.no_improvement_count = 0
                
                # 2. Check for consistent positive learning (new metric)
                # Agent is learning if it's getting positive rewards consistently
                elif avg_reward > 0 and self.iterations > 10:
                    # Check if we're maintaining positive rewards (learning is happening)
                    recent_rewards = self.rewards[-min(10, len(self.rewards)):]
                    if len(recent_rewards) >= 5 and all(r > 0 for r in recent_rewards[-5:]):
                        # Agent is consistently getting positive rewards - this is learning!
                        improvement = True
                        self.last_improvement_iteration = self.iterations
                        self.no_improvement_count = 0
                
                # 3. Check for learning phase progression
                elif self.learning_phase != getattr(self, '_last_learning_phase', 'Initial Random'):
                    improvement = True
                    self.last_improvement_iteration = self.iterations
                    self.no_improvement_count = 0
                    self._last_learning_phase = self.learning_phase
                
                # 4. Check for curriculum stage progression
                elif self.curriculum_stage != getattr(self, '_last_curriculum_stage', 1):
                    improvement = True
                    self.last_improvement_iteration = self.iterations
                    self.no_improvement_count = 0
                    self._last_curriculum_stage = self.curriculum_stage
                
                else:
                    self.no_improvement_count += 1
                
                # Log improvement status
                if improvement:
                    if win_rate > self.best_win_rate:
                        self.log(f"🎉 NEW BEST WIN RATE: {win_rate:.1f}% (was {self.best_win_rate:.1f}%)", level=1)
                    elif avg_reward > self.best_reward:
                        self.log(f"🎉 NEW BEST REWARD: {avg_reward:.2f} (was {self.best_reward:.2f})", level=1)
                    elif avg_reward > 0 and len(recent_rewards) >= 5 and all(r > 0 for r in recent_rewards[-5:]):
                        self.log(f"✅ Consistent positive learning: {len(recent_rewards)} iterations with positive rewards", level=2)
                    elif self.learning_phase != getattr(self, '_last_learning_phase', 'Initial Random'):
                        self.log(f"📈 Learning phase progression: {self.learning_phase}", level=2)
                    elif self.curriculum_stage != getattr(self, '_last_curriculum_stage', 1):
                        self.log(f"🎯 Curriculum stage progression: Stage {self.curriculum_stage}", level=2)
                else:
                    # Show learning status even when not improving
                    if avg_reward > 0:
                        self.log(f"📊 Learning in progress: Avg reward {avg_reward:.2f}, Win rate {win_rate:.1f}%", level=3)
                
                # Write one-line stats only if file logging is enabled
                if self.enable_file_logging:
                    try:
                        stats_line = f"{timestamp},{self.iterations},{self.num_timesteps},{win_rate:.1f},{avg_reward:.2f},{avg_length:.1f},{self.curriculum_stage},{self.learning_phase},{stage_time:.0f},{self.no_improvement_count}\n"
                        with open(self.stats_file, 'a') as f:
                            f.write(stats_line)
                    except (IOError, OSError) as e:
                        # If file writing fails, disable file logging
                        self.enable_file_logging = False
                        if self.verbose > 0:
                            print(f"Warning: Could not write to stats file {self.stats_file}: {e}")
                            print("File logging disabled for this callback.")
                
                # Early termination check (every 50 iterations = ~5000 timesteps)
                if self.iterations % 50 == 0 and self.iterations > 100:
                    # Check if we're stuck for too long - more lenient thresholds
                    if self.no_improvement_count > 50:  # No improvement for 50 iterations (was 20)
                        self.log(f"⚠️  WARNING: No improvement for {self.no_improvement_count} iterations", level=1)
                        if self.no_improvement_count > 100:  # No improvement for 100 iterations (was 50)
                            self.log(f"🚨 CRITICAL: No improvement for {self.no_improvement_count} iterations - Consider stopping training", level=0)
                    
                    # Check if win rate is too low for too long - more realistic thresholds
                    if win_rate < 2 and self.iterations > 500:  # Less than 2% win rate after 500 iterations (was 5% after 200)
                        self.log(f"🚨 CRITICAL: Win rate too low ({win_rate:.1f}%) after {self.iterations} iterations - Consider stopping training", level=0)
                    
                    # Check if rewards are consistently negative (real problem)
                    recent_rewards = self.rewards[-min(20, len(self.rewards)):]
                    if len(recent_rewards) >= 10 and all(r < 0 for r in recent_rewards[-10:]):
                        self.log(f"🚨 CRITICAL: Consistently negative rewards for {len(recent_rewards)} iterations - Agent may be stuck", level=0)
            
            self.last_iteration_time = current_time
            
        return True

def make_env(max_board_size, max_mines):
    """Create a wrapped environment"""
    def _init():
        env = MinesweeperEnv(
            max_board_size=max_board_size,
            max_mines=max_mines,
            render_mode=None,
            early_learning_mode=False,  # Disable early learning mode for proper training
            early_learning_threshold=200,
            early_learning_corner_safe=False,  # Disable corner safety
            early_learning_edge_safe=False,    # Disable edge safety
            mine_spacing=1,
            initial_board_size=max_board_size,  # Use curriculum board size
            initial_mines=max_mines,            # Use curriculum mine count
            invalid_action_penalty=REWARD_INVALID_ACTION,
            mine_penalty=REWARD_HIT_MINE,
            safe_reveal_base=REWARD_SAFE_REVEAL,
            win_reward=REWARD_WIN
        )
        # Configure Monitor to track the 'won' field from environment info
        env = Monitor(env, info_keywords=("won",))
        return env
    return _init

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Minesweeper agent')
    parser.add_argument('--total_timesteps', type=int, default=1000000,
                      help='Total number of timesteps to train for')
    parser.add_argument('--eval_freq', type=int, default=10000,
                      help='Frequency of evaluation')
    parser.add_argument('--n_eval_episodes', type=int, default=100,
                      help='Number of episodes to evaluate on')
    parser.add_argument('--save_freq', type=int, default=50000,
                      help='Frequency of saving the model')
    parser.add_argument('--learning_rate', type=float, default=0.0003,
                      help='Learning rate for the agent')
    parser.add_argument('--n_steps', type=int, default=2048,
                      help='Number of steps to run for each environment per update')
    parser.add_argument('--batch_size', type=int, default=64,
                      help='Batch size for training')
    parser.add_argument('--n_epochs', type=int, default=10,
                      help='Number of epochs when optimizing the surrogate loss')
    parser.add_argument('--gamma', type=float, default=0.99,
                      help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                      help='Factor for trade-off of bias vs variance for GAE')
    parser.add_argument('--clip_range', type=float, default=0.2,
                      help='Clipping parameter for PPO')
    parser.add_argument('--clip_range_vf', type=float, default=None,
                      help='Clipping parameter for value function')
    parser.add_argument('--ent_coef', type=float, default=0.01,
                      help='Entropy coefficient for the loss calculation')
    parser.add_argument('--vf_coef', type=float, default=0.5,
                      help='Value function coefficient for the loss calculation')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                      help='Maximum norm for the gradient clipping')
    parser.add_argument('--use_sde', type=bool, default=False,
                      help='Whether to use generalized State Dependent Exploration')
    parser.add_argument('--sde_sample_freq', type=int, default=-1,
                      help='Sample a new noise matrix every n steps')
    parser.add_argument('--target_kl', type=float, default=None,
                      help='Limit the KL divergence between updates')
    parser.add_argument('--policy', type=str, default="MlpPolicy",
                      help='Policy architecture')
    parser.add_argument('--verbose', type=int, default=1,
                      help='Verbosity level')
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed')
    parser.add_argument('--device', type=str, default="auto",
                      help='Device to use for training')
    parser.add_argument('--_init_setup_model', type=bool, default=True,
                      help='Whether to initialize the model')
    parser.add_argument('--strict_progression', type=bool, default=False,
                      help='Require target win rate achievement before stage progression')
    parser.add_argument('--timestamped_stats', type=bool, default=False,
                      help='Use timestamped stats files to preserve training history')
    return parser.parse_args()

def main():
    global training_model, training_env, eval_env, shutdown_requested
    
    # Set up signal handling for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    args = parse_args()
    
    print("🚀 Starting Minesweeper RL Training")
    print("=" * 50)
    print("💡 Press Ctrl+C to stop training gracefully")
    print("")
    
    # Initialize variables
    training_model = None
    training_env = None
    eval_env = None
    
    try:
        # Set up MLflow experiment with error handling
        try:
            mlflow.set_experiment("minesweeper_rl")
            mlflow_run = mlflow.start_run()
        except Exception as e:
            print(f"⚠️  MLflow initialization failed: {e}")
            print("   Continuing without MLflow tracking...")
            mlflow_run = None
        
        # Log hyperparameters if MLflow is available
        if mlflow_run is not None:
            try:
                mlflow.log_params({
                    "total_timesteps": args.total_timesteps,
                    "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size,
                    "n_steps": args.n_steps,
                    "n_epochs": args.n_epochs,
                    "gamma": args.gamma,
                    "gae_lambda": args.gae_lambda,
                    "clip_range": args.clip_range,
                    "ent_coef": args.ent_coef,
                    "vf_coef": args.vf_coef,
                    "max_grad_norm": args.max_grad_norm,
                    "device": args.device
                })
            except Exception as e:
                print(f"⚠️  MLflow parameter logging failed: {e}")
        
        # Detect optimal device and test performance
        device_info = detect_optimal_device()
        benchmark_device_performance(device_info)
    
        # Get optimal hyperparameters for the detected device
        optimal_params = get_optimal_hyperparameters(device_info)
        
        # Override args with optimal parameters if device is auto
        if args.device == "auto":
            args.device = device_info['device']
            print(f"🔧 Auto-detected device: {args.device}")
        
        # Update hyperparameters with optimal values
        args.batch_size = optimal_params['batch_size']
        args.n_steps = optimal_params['n_steps']
        args.n_epochs = optimal_params['n_epochs']
        args.learning_rate = optimal_params['learning_rate']
        args.gamma = optimal_params['gamma']
        args.gae_lambda = optimal_params['gae_lambda']
        args.clip_range = optimal_params['clip_range']
        args.ent_coef = optimal_params['ent_coef']
        args.vf_coef = optimal_params['vf_coef']
        args.max_grad_norm = optimal_params['max_grad_norm']
        
        print(f"\n📊 Training Configuration:")
        print(f"   Device: {args.device} ({device_info['description']})")
        print(f"   Batch Size: {args.batch_size}")
        print(f"   Steps per Update: {args.n_steps}")
        print(f"   Epochs: {args.n_epochs}")
        print(f"   Learning Rate: {args.learning_rate}")
        print(f"   Total Timesteps: {args.total_timesteps:,}")
        
        # Create experiment tracker
        experiment_tracker = ExperimentTracker()
        
        # Save device and performance information
        experiment_tracker.metrics["device_info"] = {
            "device": device_info['device'],
            "description": device_info['description'],
            "performance_notes": device_info['performance_notes'],
            "hyperparameters": optimal_params
        }
        
        # Define curriculum stages with human performance targets
        # ENHANCED CURRICULUM - Human Performance Focus
        curriculum_stages = [
            {
                'name': 'Beginner',
                'size': 4,
                'mines': 2,
                'win_rate_threshold': 0.80,  # 80% - Human expert level
                'min_wins_required': 8,  # 8 out of 10 games
                'learning_based_progression': False,  # Require actual wins
                'description': '4x4 board with 2 mines - Achieve human expert level (80%)',
                'training_multiplier': 3.0,  # 3x extended training
                'eval_episodes': 20  # More evaluation episodes
            },
            {
                'name': 'Intermediate',
                'size': 6,
                'mines': 4,
                'win_rate_threshold': 0.70,  # 70% - Human expert level
                'min_wins_required': 7,  # 7 out of 10 games
                'learning_based_progression': False,  # Require actual wins
                'description': '6x6 board with 4 mines - Achieve human expert level (70%)',
                'training_multiplier': 3.0,  # 3x extended training
                'eval_episodes': 20
            },
            {
                'name': 'Easy',
                'size': 9,
                'mines': 10,
                'win_rate_threshold': 0.60,  # 60% - Human expert level
                'min_wins_required': 6,  # 6 out of 10 games
                'learning_based_progression': False,  # Require actual wins
                'description': '9x9 board with 10 mines - Achieve human expert level (60%)',
                'training_multiplier': 3.0,  # 3x extended training
                'eval_episodes': 20
            },
            {
                'name': 'Normal',
                'size': 16,
                'mines': 40,
                'win_rate_threshold': 0.50,  # 50% - Human expert level
                'min_wins_required': 5,  # 5 out of 10 games
                'learning_based_progression': False,  # Require actual wins
                'description': '16x16 board with 40 mines - Achieve human expert level (50%)',
                'training_multiplier': 4.0,  # 4x extended training
                'eval_episodes': 25
            },
            {
                'name': 'Hard',
                'size': 30,
                'mines': 99,
                'win_rate_threshold': 0.40,  # 40% - Human expert level
                'min_wins_required': 4,  # 4 out of 10 games
                'learning_based_progression': False,  # Require actual wins
                'description': '16x30 board with 99 mines - Achieve human expert level (40%)',
                'training_multiplier': 4.0,  # 4x extended training
                'eval_episodes': 25
            },
            {
                'name': 'Expert',
                'size': 24,
                'mines': 115,
                'win_rate_threshold': 0.30,  # 30% - Human expert level
                'min_wins_required': 3,  # 3 out of 10 games
                'learning_based_progression': False,  # Require actual wins
                'description': '18x24 board with 115 mines - Achieve human expert level (30%)',
                'training_multiplier': 5.0,  # 5x extended training
                'eval_episodes': 30
            },
            {
                'name': 'Chaotic',
                'size': 35,
                'mines': 130,
                'win_rate_threshold': 0.20,  # 20% - Human expert level
                'min_wins_required': 2,  # 2 out of 10 games
                'learning_based_progression': False,  # Require actual wins
                'description': '20x35 board with 130 mines - Achieve human expert level (20%)',
                'training_multiplier': 5.0,  # 5x extended training
                'eval_episodes': 30
            }
        ]
        
        # BACKUP: Original curriculum stages (preserved for reference)
        original_curriculum_stages = [
            {
                'name': 'Beginner',
                'size': 4,
                'mines': 2,
                'win_rate_threshold': 0.15,  # 15% - Realistic for 4x4 with 2 mines
                'description': '4x4 board with 2 mines - Learning basic movement and safe cell identification'
            },
            {
                'name': 'Intermediate',
                'size': 6,
                'mines': 4,
                'win_rate_threshold': 0.12,  # 12% - Slightly harder but achievable
                'description': '6x6 board with 4 mines - Developing pattern recognition and basic strategy'
            },
            {
                'name': 'Easy',
                'size': 9,
                'mines': 10,
                'win_rate_threshold': 0.10,  # 10% - Standard easy difficulty
                'description': '9x9 board with 10 mines - Standard easy difficulty, mastering basic gameplay'
            },
            {
                'name': 'Normal',
                'size': 16,
                'mines': 40,
                'win_rate_threshold': 0.08,  # 8% - Standard normal difficulty
                'description': '16x16 board with 40 mines - Standard normal difficulty, developing advanced strategies'
            },
            {
                'name': 'Hard',
                'size': 30,
                'mines': 99,
                'win_rate_threshold': 0.05,  # 5% - Standard hard difficulty
                'description': '16x30 board with 99 mines - Standard hard difficulty, mastering complex patterns'
            },
            {
                'name': 'Expert',
                'size': 24,
                'mines': 115,
                'win_rate_threshold': 0.03,  # 3% - Expert level
                'description': '18x24 board with 115 mines - Expert level, handling high mine density'
            },
            {
                'name': 'Chaotic',
                'size': 35,
                'mines': 130,
                'win_rate_threshold': 0.02,  # 2% - Ultimate challenge
                'description': '20x35 board with 130 mines - Ultimate challenge, maximum complexity'
            }
        ]
        
        # Use enhanced curriculum by default, but allow fallback to original
        if args.strict_progression:
            # Use original curriculum for strict progression
            curriculum_stages = original_curriculum_stages
            print("📚 Using original curriculum (strict progression mode)")
        else:
            print("📚 Using enhanced curriculum (human performance targets)")
        
        # Define stage timesteps based on curriculum
        stage_timesteps = [
            int(args.total_timesteps * 0.15),  # 15% for beginner
            int(args.total_timesteps * 0.15),  # 15% for intermediate
            int(args.total_timesteps * 0.20),  # 20% for easy
            int(args.total_timesteps * 0.20),  # 20% for normal
            int(args.total_timesteps * 0.15),  # 15% for hard
            int(args.total_timesteps * 0.10),  # 10% for expert
            int(args.total_timesteps * 0.05)   # 5% for chaotic
        ]
        
        print(f"📈 Curriculum Learning: {len(curriculum_stages)} stages")
        print(f"   Expected performance: {device_info['performance_notes']}")
        
        # Curriculum learning loop
        for stage in range(len(curriculum_stages)):
            if shutdown_requested:
                print("\n🛑 Shutdown requested. Stopping training gracefully...")
                break
                
            # Get current stage information
            current_stage_info = curriculum_stages[stage]
            current_stage_name = current_stage_info['name']
            current_stage_size = current_stage_info['size']
            current_stage_mines = current_stage_info['mines']
            
            # Enhanced curriculum parameters
            training_multiplier = current_stage_info.get('training_multiplier', 1.0)
            eval_episodes = current_stage_info.get('eval_episodes', args.n_eval_episodes)
            
            print(f"\n🎯 Stage {stage + 1}: {current_stage_name}")
            print(f"Board: {current_stage_size}x{current_stage_size} with {current_stage_mines} mines")
            print(f"Target win rate: {current_stage_info['win_rate_threshold']*100:.0f}%")
            print(f"Min wins required: {current_stage_info.get('min_wins_required', 1)} out of {eval_episodes} games")
            print(f"Training multiplier: {training_multiplier}x (extended training)")
            print(f"Evaluation episodes: {eval_episodes}")
            
            # Create new environment for current stage
            training_env = DummyVecEnv([make_env(
                max_board_size=current_stage_size,
                max_mines=current_stage_mines
            )])
            
            # Create new model for current stage (observation space changes between stages)
            print(f"🏗️  Creating new PPO model for Stage {stage + 1}...")
            training_model = PPO(
                policy=args.policy,
                env=training_env,
                learning_rate=args.learning_rate,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                clip_range=args.clip_range,
                clip_range_vf=args.clip_range_vf,
                ent_coef=args.ent_coef,
                vf_coef=args.vf_coef,
                max_grad_norm=args.max_grad_norm,
                use_sde=args.use_sde,
                sde_sample_freq=args.sde_sample_freq,
                target_kl=args.target_kl,
                verbose=args.verbose,
                seed=args.seed,
                device=args.device,
                _init_setup_model=args._init_setup_model
            )
            print("✅ New model created successfully!")
            
            # Create evaluation environment for current stage
            eval_env = DummyVecEnv([make_env(
                max_board_size=current_stage_size,
                max_mines=current_stage_mines
            )])
            
            # Calculate enhanced training timesteps
            base_timesteps = stage_timesteps[stage]
            enhanced_timesteps = int(base_timesteps * training_multiplier)
            
            print(f"⏱️  Training timesteps: {base_timesteps} -> {enhanced_timesteps} ({training_multiplier}x extended)")
            
            # Create evaluation callback with enhanced parameters
            eval_callback = CustomEvalCallback(
                eval_env,
                eval_freq=args.eval_freq,
                n_eval_episodes=eval_episodes,
                verbose=args.verbose,
                best_model_save_path=f"./best_model/stage_{stage + 1}",
                log_path="./logs/"
            )
            
            # Create iteration callback
            iteration_callback = IterationCallback(
                verbose=args.verbose,
                debug_level=2,
                experiment_tracker=experiment_tracker,
                stats_file="training_stats.txt",
                timestamped_stats=args.timestamped_stats
            )
            
            # Train for this stage with enhanced timesteps
            print(f"🚀 Starting enhanced training for Stage {stage + 1}...")
            training_model.learn(
                total_timesteps=enhanced_timesteps,
                callback=[eval_callback, iteration_callback],
                progress_bar=True
            )
            
            # Check for shutdown request after training
            if shutdown_requested:
                print("\n🛑 Shutdown requested. Stopping training gracefully...")
                break
            
            # Evaluate current stage with enhanced episodes
            evaluation_results = evaluate_model(training_model, training_env, n_episodes=eval_episodes)
            win_rate = evaluation_results["win_rate"] / 100  # Convert percentage to decimal
            mean_reward = evaluation_results["avg_reward"]
            reward_std = evaluation_results["reward_ci"]
            
            print(f"\nStage {stage + 1} Results:")
            print(f"Mean reward: {mean_reward:.2f} +/- {reward_std:.2f}")
            print(f"Win rate: {win_rate*100:.2f}%")
            print(f"Target win rate: {current_stage_info['win_rate_threshold']*100:.0f}%")
            print(f"Min wins required: {current_stage_info.get('min_wins_required', 1)} out of {eval_episodes} games")
            
            # Enhanced progression logic for curriculum
            target_win_rate = current_stage_info['win_rate_threshold']
            min_wins_required = current_stage_info.get('min_wins_required', 1)
            learning_based_progression = current_stage_info.get('learning_based_progression', True)
            min_positive_reward = 5.0  # Minimum positive reward to show learning
            min_learning_progress = 0.1  # Minimum improvement in rewards over time
            
            # Calculate actual wins from evaluation
            actual_wins = int(win_rate * eval_episodes)
            
            # Check if we should progress to next stage
            should_progress = False
            progression_reason = ""
            
            # 1. Target achieved - always progress
            if win_rate >= target_win_rate and actual_wins >= min_wins_required:
                should_progress = True
                progression_reason = f"🎯 Target achieved: {win_rate*100:.1f}% >= {target_win_rate*100:.0f}% with {actual_wins} wins"
            
            # 2. Learning-based progression (if allowed)
            elif (mean_reward >= min_positive_reward and 
                  learning_based_progression and 
                  not args.strict_progression):
                # Check if this is the last stage
                if stage == len(curriculum_stages) - 1:
                    should_progress = False
                    progression_reason = "🏁 Final stage reached"
                else:
                    # For intermediate stages, allow progression with learning
                    should_progress = True
                    progression_reason = f"📈 Learning progress: {mean_reward:.2f} mean reward (target: {min_positive_reward}) - Learning-based progression allowed"
            
            # 3. No learning progress or strict progression required - don't progress
            else:
                should_progress = False
                if args.strict_progression and win_rate < target_win_rate:
                    progression_reason = f"🔒 Strict progression: Win rate {win_rate*100:.1f}% < {target_win_rate*100:.0f}% required"
                elif actual_wins < min_wins_required:
                    progression_reason = f"🔒 Minimum wins not met: {actual_wins} wins < {min_wins_required} required"
                elif not learning_based_progression and win_rate < target_win_rate:
                    progression_reason = f"🔒 Stage requires actual wins: {win_rate*100:.1f}% < {target_win_rate*100:.0f}% (no learning-based progression)"
                else:
                    progression_reason = f"⚠️ Insufficient learning: {mean_reward:.2f} mean reward < {min_positive_reward}"
            
            # Log progression decision
            if should_progress:
                print(f"{progression_reason}")
                if win_rate < target_win_rate:
                    print(f"⚠️  Stage {stage + 1} target not achieved. Win rate: {win_rate*100:.1f}% < {target_win_rate*100:.0f}%")
                    print(f"📈 But allowing progression due to learning progress")
                print(f"Stage {stage + 1} completed. Moving to next stage...")
            else:
                print(f"{progression_reason}")
                print(f"🔄 Stage {stage + 1} not completed. Consider extending training time.")
                # Could implement retraining logic here
                break  # Stop progression
            
            # Save stage results
            experiment_tracker.add_validation_metric(
                f"stage_{stage + 1}_mean_reward",
                mean_reward,
                confidence_interval=reward_std
            )
            experiment_tracker.add_validation_metric(
                f"stage_{stage + 1}_win_rate",
                win_rate
            )
            
            # Save model for this stage with error handling
            try:
                training_model.save(f"models/stage_{stage + 1}")
            except Exception as e:
                print(f"⚠️  Failed to save stage {stage + 1} model: {e}")
            
            # Add stage completion to metrics
            experiment_tracker.metrics["completed_stages"] = experiment_tracker.metrics.get("completed_stages", [])
            experiment_tracker.metrics["completed_stages"].append({
                "stage": stage + 1,
                "name": current_stage_info['name'],
                "win_rate": win_rate,
                "mean_reward": mean_reward,
                "target_win_rate": current_stage_info['win_rate_threshold'],
                "actual_wins": actual_wins,
                "min_wins_required": min_wins_required,
                "training_multiplier": training_multiplier,
                "eval_episodes": eval_episodes
            })
            experiment_tracker._save_metrics()
        
        # Save final model if training completed normally
        if not shutdown_requested and training_model is not None:
            try:
                training_model.save("models/final_model")
            except Exception as e:
                print(f"⚠️  Failed to save final model: {e}")
            
            # Log final model to MLflow (SB3 models need to be logged as artifacts)
            if mlflow_run is not None:
                try:
                    # Ensure models directory exists
                    os.makedirs("models", exist_ok=True)
                    
                    # Check if the model file exists before logging
                    model_path = "models/final_model.zip"
                    if os.path.exists(model_path):
                        mlflow.log_artifact(model_path, "final_model")
                        print("   ✅ Final model logged to MLflow successfully")
                    else:
                        print(f"   ⚠️  Model file not found at {model_path}")
                except Exception as e:
                    print(f"   ⚠️  Model logging failed: {e}")
                
                # Log final metrics
                try:
                    mlflow.log_metric("final_win_rate", win_rate)
                    mlflow.log_metric("final_mean_reward", mean_reward)
                    mlflow.log_metric("final_reward_std", reward_std)
                except Exception as e:
                    print(f"   ⚠️  Final metrics logging failed: {e}")
            
            print("\n✅ Training completed successfully!")
            print("\nFinal Stage Progression:")
            for stage in range(len(curriculum_stages)):
                if stage < len(experiment_tracker.metrics.get("completed_stages", [])):
                    stage_info = experiment_tracker.metrics["completed_stages"][stage]
                    print(f"\nStage {stage + 1}: {stage_info['name']}")
                    print(f"Final Win Rate: {stage_info['win_rate']:.2%}")
                    print(f"Mean Reward: {stage_info['mean_reward']:.2f} +/- {stage_info['target_win_rate']*100:.2f}%")
            
            if mlflow_run is not None:
                print(f"\n📊 MLflow experiment tracking enabled!")
                print(f"   Run 'mlflow ui' to view training progress")
                print(f"   Then open http://localhost:5000 in your browser")
        else:
            print("\n🛑 Training stopped by user request.")
            if training_model is not None:
                # Save the current model as a checkpoint
                try:
                    checkpoint_path = f"models/checkpoint_stage_{stage + 1}_interrupted"
                    training_model.save(checkpoint_path)
                    print(f"💾 Checkpoint saved to: {checkpoint_path}")
                except Exception as e:
                    print(f"⚠️  Failed to save checkpoint: {e}")
    
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user (Ctrl+C)")
        if training_model is not None:
            # Save the current model as a checkpoint
            try:
                checkpoint_path = "models/checkpoint_interrupted"
                training_model.save(checkpoint_path)
                print(f"💾 Checkpoint saved to: {checkpoint_path}")
            except Exception as e:
                print(f"⚠️  Failed to save checkpoint: {e}")
    
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        if training_model is not None:
            # Save the current model as a checkpoint
            try:
                checkpoint_path = "models/checkpoint_error"
                training_model.save(checkpoint_path)
                print(f"💾 Checkpoint saved to: {checkpoint_path}")
            except Exception as save_error:
                print(f"⚠️  Failed to save error checkpoint: {save_error}")
        raise
    
    finally:
        # Clean up resources
        print("\n🧹 Cleaning up resources...")
        if training_env is not None:
            try:
                training_env.close()
            except Exception as e:
                print(f"⚠️  Failed to close training environment: {e}")
        if eval_env is not None:
            try:
                eval_env.close()
            except Exception as e:
                print(f"⚠️  Failed to close evaluation environment: {e}")
        print("✅ Cleanup completed")

def evaluate_model(model, env, n_episodes=100, raise_errors=False):
    """Evaluate model with proper statistical analysis, supporting both vectorized and non-vectorized environments."""
    if n_episodes <= 0:
        return {
            "win_rate": 0.0,
            "avg_reward": 0.0,
            "avg_length": 0.0,
            "reward_ci": 0.0,
            "length_ci": 0.0,
            "n_episodes": 0
        }
    rewards = []
    lengths = []
    wins = 0
    # More robust vectorized environment detection
    # Check if num_envs is actually a meaningful integer value, not just a MagicMock
    is_vectorized = (hasattr(env, 'num_envs') and 
                    hasattr(env, 'step') and 
                    callable(getattr(env, 'step', None)) and
                    isinstance(getattr(env, 'num_envs', None), int) and
                    getattr(env, 'num_envs', 0) > 0)
    
    for episode in range(n_episodes):
        try:
            if is_vectorized:
                obs = env.reset()
            else:
                obs, _ = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            episode_won = False
            while not done:
                action, _ = model.predict(obs)
                step_result = env.step(action)
                if not step_result or len(step_result) == 0:
                    break
                if len(step_result) == 4:
                    obs, reward, terminated, truncated = step_result
                    info = {}
                elif len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                else:
                    break
                if is_vectorized:
                    # Unwrap arrays/lists
                    if isinstance(terminated, (list, np.ndarray)) and len(terminated) > 0:
                        # Episode ends if ANY environment is done
                        done = bool(any(terminated) or any(truncated))
                    else:
                        done = bool(terminated or truncated)
                    # For vectorized envs, take the mean of all rewards
                    if isinstance(reward, (np.ndarray, list)):
                        r = float(np.mean(reward))
                    else:
                        r = reward
                    episode_reward += r
                    # Win detection for vectorized envs
                    if len(step_result) == 4:
                        # Old gym API: win if any reward >= 500
                        if (isinstance(reward, (np.ndarray, list)) and any(x >= 500 for x in reward)) or (not isinstance(reward, (np.ndarray, list)) and reward >= 500):
                            episode_won = True
                    else:
                        # New API: info is list of dicts
                        if isinstance(info, (list, tuple)) and any(d.get('won', False) for d in info if isinstance(d, dict)):
                            episode_won = True
                else:
                    done = bool(terminated or truncated)
                    episode_reward += reward
                    if len(step_result) == 4:
                        if reward >= 500:
                            episode_won = True
                    else:
                        if info.get('won', False):
                            episode_won = True
                episode_length += 1
                if episode_length > 1000:
                    break
        except Exception as e:
            if raise_errors:
                raise
            print(f"Warning: Episode {episode} failed with error: {e}")
            episode_reward = 0
            episode_length = 0
            episode_won = False
        rewards.append(episode_reward)
        lengths.append(episode_length)
        if episode_won:
            wins += 1
    win_rate = (wins / n_episodes) * 100
    avg_reward = float(np.mean(rewards))
    avg_length = float(np.mean(lengths))
    reward_std = float(np.std(rewards) / np.sqrt(len(rewards)) if len(rewards) > 1 else 0.0)
    length_std = float(np.std(lengths) / np.sqrt(len(lengths)) if len(lengths) > 1 else 0.0)
    return {
        "win_rate": win_rate,
        "avg_reward": avg_reward,
        "avg_length": avg_length,
        "reward_ci": reward_std,
        "length_ci": length_std,
        "n_episodes": n_episodes
    }

if __name__ == "__main__":
    main() 