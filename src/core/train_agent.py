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
from datetime import datetime, timedelta
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper
from stable_baselines3.common.utils import safe_mean
import argparse
import torch
import mlflow
import mlflow.pytorch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import tempfile
import shutil
import copy
import pickle

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.minesweeper_env import MinesweeperEnv
from src.core.constants import REWARD_INVALID_ACTION, REWARD_HIT_MINE, REWARD_SAFE_REVEAL, REWARD_WIN

# Global variables for graceful shutdown
training_model = None
training_env = None
eval_env = None
shutdown_requested = False

# Global flag for deterministic training mode
DETERMINISTIC_TRAINING_MODE = False

class TrainingStatsManager:
    """Manages training stats history with automatic cleanup and organization."""
    
    def __init__(self, history_dir="training_stats/history", max_age_days=14, max_files=None):
        """
        Initialize the training stats manager.
        
        Args:
            history_dir: Directory to store training stats history
            max_age_days: Maximum age of files to keep (default: 14 days)
            max_files: Maximum number of files to keep (default: None - no limit)
        """
        self.history_dir = Path(history_dir)
        self.max_age_days = max_age_days
        self.max_files = max_files
        
        # Create history directory if it doesn't exist
        self.history_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean up old files on initialization
        self.cleanup_old_files()
    
    def get_stats_file_path(self, filename: str) -> Path:
        """Get the full path for a stats file, moving it to history if it's timestamped."""
        file_path = Path(filename)
        
        # If it's a timestamped file, put it directly in history
        if file_path.name.startswith("training_stats_") and file_path.name != "training_stats.txt":
            return self.history_dir / file_path.name
        else:
            # For non-timestamped files, use the main training_stats directory
            return Path("training_stats") / file_path.name
    
    def move_to_history(self, file_path: str) -> Optional[Path]:
        """Move a completed training stats file to the history directory."""
        source_path = Path(file_path)
        
        if not source_path.exists():
            return None
        
        # Don't move the current training_stats.txt file
        if source_path.name == "training_stats.txt":
            return source_path
        
        # Move timestamped files to history
        if source_path.name.startswith("training_stats_"):
            dest_path = self.history_dir / source_path.name
            try:
                source_path.rename(dest_path)
                return dest_path
            except Exception as e:
                print(f"Warning: Could not move {source_path} to history: {e}")
                return source_path
        
        return source_path
    
    def cleanup_old_files(self):
        """Remove old training stats files and experiment results based on age."""
        # Clean up training stats files
        self._cleanup_training_stats_files()
        
        # Clean up experiment result files
        self._cleanup_experiment_files()
        
        # Clean up log directories
        self._cleanup_log_directories()
    
    def _cleanup_training_stats_files(self):
        """Remove old training stats files based on age and count limits."""
        if not self.history_dir.exists():
            return
        
        # Get all training stats files in history
        stats_files = list(self.history_dir.glob("training_stats_*.txt"))
        
        if not stats_files:
            return
        
        # Sort by modification time (newest first)
        stats_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        # Calculate cutoff time
        cutoff_time = time.time() - (self.max_age_days * 24 * 60 * 60)
        
        # Files to remove
        files_to_remove = []
        
        # Remove files older than max_age_days
        for file_path in stats_files:
            if file_path.stat().st_mtime < cutoff_time:
                files_to_remove.append(file_path)
        
        # If we have a max_files limit, remove the oldest ones
        if self.max_files and len(stats_files) > self.max_files:
            files_to_remove.extend(stats_files[self.max_files:])
        
        # Remove duplicate files to remove
        files_to_remove = list(set(files_to_remove))
        
        # Remove the files
        for file_path in files_to_remove:
            try:
                file_path.unlink()
                print(f"Cleaned up old training stats: {file_path.name}")
            except Exception as e:
                print(f"Warning: Could not remove {file_path}: {e}")
    
    def _cleanup_experiment_files(self):
        """Remove old experiment result files based on age."""
        # Patterns for timestamped experiment files
        experiment_patterns = [
            "experiments/modular_results_*.json",
            "experiments/simple_results_*.json",
            "logs/coverage_results_summary.json",
            "logs/cross_platform_test_report.json"
        ]
        
        # Calculate cutoff time
        cutoff_time = time.time() - (self.max_age_days * 24 * 60 * 60)
        
        for pattern in experiment_patterns:
            for file_path in Path(".").glob(pattern):
                try:
                    if file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
                        print(f"Cleaned up old experiment file: {file_path.name}")
                except Exception as e:
                    print(f"Warning: Could not remove {file_path}: {e}")
    
    def _cleanup_log_directories(self):
        """Remove old log directories based on age."""
        # Log directories to clean up
        log_dirs = [
            "logs/benchmark_results",
            "logs/minimal_training_logs", 
            "logs/progressive_logs",
            "logs/simple_logs",
            "logs/conservative_training_logs",
            "logs/training_logs"
        ]
        
        # Calculate cutoff time
        cutoff_time = time.time() - (self.max_age_days * 24 * 60 * 60)
        
        for dir_name in log_dirs:
            dir_path = Path(dir_name)
            if dir_path.exists() and dir_path.is_dir():
                try:
                    # Check if directory is older than max_age_days
                    if dir_path.stat().st_mtime < cutoff_time:
                        import shutil
                        shutil.rmtree(dir_path)
                        print(f"Cleaned up old log directory: {dir_name}")
                except Exception as e:
                    print(f"Warning: Could not remove {dir_name}: {e}")
    
    def get_recent_stats(self, count: int = 5) -> List[Path]:
        """Get the most recent training stats files."""
        if not self.history_dir.exists():
            return []
        
        stats_files = list(self.history_dir.glob("training_stats_*.txt"))
        stats_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        return stats_files[:count]
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """Get a summary of training stats history."""
        if not self.history_dir.exists():
            return {"total_files": 0, "recent_files": []}
        
        stats_files = list(self.history_dir.glob("training_stats_*.txt"))
        
        # Get recent files with basic info
        recent_files = []
        for file_path in sorted(stats_files, key=lambda f: f.stat().st_mtime, reverse=True)[:5]:
            try:
                stat = file_path.stat()
                recent_files.append({
                    "name": file_path.name,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "age_days": (time.time() - stat.st_mtime) / (24 * 60 * 60)
                })
            except Exception:
                continue
        
        return {
            "total_files": len(stats_files),
            "history_dir": str(self.history_dir),
            "recent_files": recent_files
        }

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global shutdown_requested
    print(f"\n🛑 Received signal {signum}. Initiating graceful shutdown...")
    shutdown_requested = True
    
    # Give a second chance to force quit
    print("Press Ctrl+C again to force quit, or wait for graceful shutdown...")
    signal.signal(signal.SIGINT, signal.SIG_DFL)  # Reset to default handler for second Ctrl+C

def detect_optimal_device(board_size=None, policy_type="MlpPolicy"):
    """
    Detect the optimal device for training based on board size and policy type.
    CPU is faster for small boards and MlpPolicy, GPU for large boards and CNN policies.
    """
    # For MlpPolicy and small boards, CPU is much faster (10x faster in our benchmarks)
    if policy_type == "MlpPolicy" and board_size:
        board_area = board_size[0] * board_size[1]
        if board_area <= 144:  # 12x12 or smaller
            device_info = {
                'device': 'cpu',
                'description': 'CPU (optimized for small boards + MlpPolicy)',
                'performance_notes': '10x faster than M1 GPU for small boards with MlpPolicy'
            }
            print(f"🎯 Using CPU for {board_size[0]}x{board_size[1]} board with MlpPolicy")
            print(f"   Performance: {device_info['performance_notes']}")
            return device_info
    
    # For larger boards or CNN policies, check for GPU
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device_info = {
            'device': 'mps',
            'description': 'Apple M1 GPU (MPS)',
            'performance_notes': 'Optimal for large boards or CNN policies'
        }
        print(f"✅ Detected M1 GPU (MPS) - Using device: {device_info['device']}")
        print(f"   Performance: {device_info['performance_notes']}")
        
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
        device_info = {
            'device': 'cpu',
            'description': 'CPU (fallback)',
            'performance_notes': 'No GPU available'
        }
        print(f"⚠️  No GPU detected - Using device: {device_info['device']}")
        print(f"   Performance: {device_info['performance_notes']}")
    
    return device_info

def get_optimal_hyperparameters(device_info, curriculum_mode="current"):
    """
    Get optimal hyperparameters based on device and curriculum mode.
    Using conservative hyperparameters that achieved 30% win rate in testing.
    """
    device = device_info['device']
    
    # Conservative hyperparameters that worked well in testing
    base_params = {
        'learning_rate': 1e-4,      # Much lower learning rate (was 3e-4)
        'n_steps': 1024,            # Smaller steps for more frequent updates
        'batch_size': 32,           # Smaller batch size (was 128)
        'n_epochs': 15,             # More epochs for better learning (was 12)
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.1,          # Tighter clipping
        'ent_coef': 0.005,          # Lower entropy for more focused learning (was 0.01)
        'vf_coef': 0.5,
        'max_grad_norm': 0.3,       # Lower gradient norm
    }
    
    # Device-specific optimizations
    if device == "mps":
        # M1 GPU optimizations
        optimized_params = base_params.copy()
        optimized_params.update({
            'batch_size': 64,       # M1 can handle larger batches
            'n_steps': 2048,        # More steps for M1 efficiency
            'n_epochs': 12,         # Optimized for M1
        })
        print("🔧 Applied M1 GPU optimizations")
    elif device == "cuda":
        # CUDA GPU optimizations
        optimized_params = base_params.copy()
        optimized_params.update({
            'batch_size': 128,      # CUDA can handle very large batches
            'n_steps': 2048,        # More steps for CUDA efficiency
            'n_epochs': 10,         # Optimized for CUDA
        })
        print("🔧 Applied CUDA GPU optimizations")
    else:
        # CPU optimizations (use base conservative params)
        optimized_params = base_params.copy()
        print("🔧 Using CPU-optimized conservative parameters")
    
    # Curriculum mode adjustments
    if curriculum_mode in ["human_performance", "superhuman"]:
        # For extended training modes, use even more conservative settings
        optimized_params.update({
            'learning_rate': 5e-5,  # Even lower for extended training
            'n_epochs': 20,         # More epochs for extended training
            'ent_coef': 0.003,      # Lower entropy for focused learning
        })
        print("🔧 Applied extended training optimizations")
    
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

def make_env(max_board_size, max_mines, apply_epsilon_greedy=True):
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
    
    # Return the environment factory function
    # The epsilon-greedy wrapper will be applied later in the training loop
    # This allows us to control whether training vs evaluation environments have the wrapper
    return _init

def make_simple_env(max_board_size, max_mines):
    """Create a simplified environment without complex wrappers"""
    def _init():
        env = MinesweeperEnv(
            max_board_size=max_board_size,
            max_mines=max_mines,
            render_mode=None,
            early_learning_mode=False,
            early_learning_threshold=200,
            early_learning_corner_safe=False,
            early_learning_edge_safe=False,
            mine_spacing=1,
            initial_board_size=max_board_size,
            initial_mines=max_mines,
            invalid_action_penalty=REWARD_INVALID_ACTION,
            mine_penalty=REWARD_HIT_MINE,
            safe_reveal_base=REWARD_SAFE_REVEAL,
            win_reward=REWARD_WIN
        )
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
                      help='Device to use for training, e.g. "cpu", "cuda", "mps"')
    parser.add_argument('--_init_setup_model', type=bool, default=True, help='Whether to build the model or not')
    parser.add_argument('--strict_progression', action="store_true", help="Enable strict win rate progression")
    parser.add_argument('--timestamped_stats', action="store_true", help="Enable timestamped stats file")
    parser.add_argument('--curriculum_mode', type=str, default="human_performance", 
                      choices=["current", "human_performance", "superhuman", "debug"],
                      help="Curriculum mode: current (15%%/12%%/10%%), human_performance (80%%/70%%/60%%), or superhuman (95%%/80%%/60%%), or debug (4x4 board with 1 mine - Debug test)")
    
    parser.add_argument('--simple_training_mode', action="store_true", 
                      help="Use simplified training approach (no complex wrappers, conservative hyperparameters)")
    
    # Advanced PPO options
    return parser.parse_args()

def get_curriculum_config(mode="current"):
    """
    Get curriculum configuration based on mode.
    Simplified curriculum that focuses on mastering one board size at a time.
    """
    if mode == "current":
        return [
            {
                'name': 'Beginner',
                'size': 4,
                'mines': 2,
                'win_rate_threshold': 0.15,  # 15% target (like successful test)
                'min_wins_required': 2,
                'learning_based_progression': True,
                'description': '4x4 board with 2 mines - Master this first',
                'training_multiplier': 1.0,  # No extension - focus on learning
                'eval_episodes': 20
            },
            {
                'name': 'Intermediate', 
                'size': 4,
                'mines': 3,
                'win_rate_threshold': 0.12,
                'min_wins_required': 2,
                'learning_based_progression': True,
                'description': '4x4 board with 3 mines - Increase difficulty',
                'training_multiplier': 1.0,
                'eval_episodes': 20
            },
            {
                'name': 'Easy',
                'size': 6,
                'mines': 4,
                'win_rate_threshold': 0.10,
                'min_wins_required': 2,
                'learning_based_progression': True,
                'description': '6x6 board with 4 mines - Larger board',
                'training_multiplier': 1.0,
                'eval_episodes': 20
            },
            {
                'name': 'Normal',
                'size': 6,
                'mines': 6,
                'win_rate_threshold': 0.08,
                'min_wins_required': 2,
                'learning_based_progression': True,
                'description': '6x6 board with 6 mines - Standard difficulty',
                'training_multiplier': 1.0,
                'eval_episodes': 20
            },
            {
                'name': 'Hard',
                'size': 8,
                'mines': 8,
                'win_rate_threshold': 0.05,
                'min_wins_required': 1,
                'learning_based_progression': True,
                'description': '8x8 board with 8 mines - Expert level',
                'training_multiplier': 1.0,
                'eval_episodes': 20
            }
        ]
    else:
        raise ValueError(f"Unknown curriculum mode: {mode}")

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
    enhanced_timesteps = None  # Defensive: ensure always defined
    
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
        
        # Get curriculum configuration first to determine board size
        curriculum_stages = get_curriculum_config(args.curriculum_mode)
        initial_board_size = curriculum_stages[0]['size']  # Get board size from first stage
        
        # Detect optimal device and test performance
        device_info = detect_optimal_device(board_size=(initial_board_size, initial_board_size), policy_type="MlpPolicy")
        benchmark_device_performance(device_info)
    
        # Get optimal hyperparameters for the detected device
        optimal_params = get_optimal_hyperparameters(device_info, args.curriculum_mode)
        
        # Override args with optimal parameters if device is auto
        if args.device == "auto":
            args.device = device_info['device']
            print(f"🔧 Auto-detected device: {args.device}")
        
        # Force device to detected device (in case it wasn't auto)
        args.device = device_info['device']
        print(f"🔧 Using device: {args.device}")
        
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
        
        # Reduce verbose output to prevent excessive iteration printing
        if args.verbose > 0:
            args.verbose = 0
            print("🔇 Reduced verbose output to prevent excessive iteration printing")
        
        print(f"\n📊 Training Configuration:")
        print(f"   Device: {args.device} ({device_info['description']})")
        print(f"   Batch Size: {args.batch_size}")
        print(f"   Steps per Update: {args.n_steps}")
        print(f"   Epochs: {args.n_epochs}")
        print(f"   Learning Rate: {args.learning_rate}")
        print(f"   Total Timesteps: {args.total_timesteps:,}")
        
        # Create experiment tracker
        experiment_tracker = ExperimentTracker()
        
        # Create training stats manager for history management
        stats_manager = TrainingStatsManager(
            history_dir="training_stats/history",
            max_age_days=14,  # Keep files for 2 weeks
            max_files=None      # No limit on number of files
        )
        
        # Save device and performance information
        experiment_tracker.metrics["device_info"] = {
            "device": device_info['device'],
            "description": device_info['description'],
            "performance_notes": device_info['performance_notes'],
            "hyperparameters": optimal_params
        }
        
        # Log curriculum configuration (using curriculum_stages from earlier)
        print(f"\n📚 Curriculum Mode: {args.curriculum_mode.upper()}")
        print(f"   Stages: {len(curriculum_stages)}")
        print(f"   First Stage Target: {curriculum_stages[0]['win_rate_threshold']*100:.0f}%")
        print(f"   Last Stage Target: {curriculum_stages[-1]['win_rate_threshold']*100:.0f}%")
        print(f"   Training Multiplier: {curriculum_stages[0]['training_multiplier']}x")
        print(f"   Evaluation Episodes: {curriculum_stages[0]['eval_episodes']}")
        
        # Define stage timesteps based on curriculum - INCREASED for better learning
        stage_timesteps = [
            int(args.total_timesteps * 0.25),  # 25% for beginner (was 15%)
            int(args.total_timesteps * 0.20),  # 20% for intermediate (was 15%)
            int(args.total_timesteps * 0.20),  # 20% for easy (was 20%)
            int(args.total_timesteps * 0.15),  # 15% for normal (was 20%)
            int(args.total_timesteps * 0.10),  # 10% for hard (was 15%)
            int(args.total_timesteps * 0.07),  # 7% for expert (was 10%)
            int(args.total_timesteps * 0.03)   # 3% for chaotic (was 5%)
        ]
        
        # Calculate extended training timesteps for human performance modes
        if args.curriculum_mode in ["human_performance", "superhuman"]:
            total_extended_timesteps = sum(
                stage['training_multiplier'] * stage_timesteps[i] 
                for i, stage in enumerate(curriculum_stages)
            )
            
            print("🚀 Extended Training Configuration:")
            print(f"   Original Total: {args.total_timesteps} timesteps")
            print(f"   Extended Total: {total_extended_timesteps} timesteps")
            
            # Guard against division by zero
            if args.total_timesteps > 0:
                print(f"   Extension Factor: {total_extended_timesteps/args.total_timesteps:.1f}x")
            else:
                print("   Extension Factor: N/A (zero timesteps)")
            
            # Update the total timesteps for the extended training
            args.total_timesteps = total_extended_timesteps
        
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
            if args.simple_training_mode:
                # Use simplified environment setup (no complex wrappers)
                training_env = DummyVecEnv([make_simple_env(
                    max_board_size=current_stage_size,
                    max_mines=current_stage_mines
                )])
                print(f"🔧 Using simplified environment setup (simple training mode)")
            else:
                # Use standard environment setup
                training_env = DummyVecEnv([make_env(
                    max_board_size=current_stage_size,
                    max_mines=current_stage_mines
                )])
                print(f"🔧 Using standard environment setup")
            
            # Apply action masking wrapper to prevent invalid actions
            training_env = ActionMaskingWrapper(training_env)
            print(f"🔒 Applied action masking wrapper to training environment")
            
            # REMOVED: MultiBoardTrainingWrapper - was causing learning interference
            # REMOVED: DeterministicTrainingCallback - was adding unnecessary complexity
            if args.simple_training_mode:
                print(f"🔧 Simplified training pipeline (no complex wrappers)")
            else:
                print(f"🔧 Standard training pipeline")
            
            # Calculate enhanced training timesteps BEFORE applying wrapper
            base_timesteps = stage_timesteps[stage]
            enhanced_timesteps = int(base_timesteps * training_multiplier)
            
            # REMOVED: DeterministicTrainingCallback - was adding unnecessary complexity
            print(f"🔍 Using simple training approach (no deterministic periods)")
            
            # Apply epsilon-greedy exploration wrapper for better exploration
            # TEMPORARILY DISABLED: Let's see if the agent can learn without exploration bias
            # training_env = EpsilonGreedyExploration(
            #     training_env,
            #     initial_epsilon=0.3,    # Start with 30% exploration
            #     final_epsilon=0.05,     # End with 5% exploration
            #     decay_steps=enhanced_timesteps // 2  # Decay over half the training period
            # )
            # print(f"🔍 Applied epsilon-greedy exploration: 30% -> 5% over {enhanced_timesteps // 2} steps")
            print(f"🔍 Epsilon-greedy exploration DISABLED - training deterministically")
            
            # ALTERNATIVE EXPLORATION STRATEGIES (if needed):
            # 1. Use PPO's built-in entropy coefficient (ent_coef) for exploration
            # 2. Implement action masking with random valid actions
            # 3. Use curriculum-based exploration (more exploration in early stages)
            # 4. Implement Thompson sampling or UCB exploration
            
            # Create new model for current stage (observation space changes between stages)
            print(f"🏗️  Creating new PPO model for Stage {stage + 1}...")
            
            training_model = PPO(  # REMOVED: DeterministicPPO - using standard PPO
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
                ent_coef=args.ent_coef,  # Use standard entropy coefficient
                vf_coef=args.vf_coef,
                max_grad_norm=args.max_grad_norm,
                use_sde=args.use_sde,
                sde_sample_freq=args.sde_sample_freq,
                target_kl=args.target_kl,
                verbose=args.verbose,
                seed=args.seed,
                device=args.device,
                _init_setup_model=args._init_setup_model
                # REMOVED: deterministic_callback parameter
            )
            print("✅ New model created successfully!")
            print(f"🔍 Using simple PPO training (no deterministic periods)")
            
            # Create evaluation environment for current stage
            # Use IDENTICAL configuration as training environment for consistency
            # BUT without epsilon-greedy wrapper (evaluation should be deterministic)
            if args.simple_training_mode:
                eval_env = DummyVecEnv([make_simple_env(
                    max_board_size=current_stage_size,
                    max_mines=current_stage_mines
                )])
                print(f"🔧 Using simplified evaluation environment (simple training mode)")
            else:
                eval_env = DummyVecEnv([make_env(
                    max_board_size=current_stage_size,
                    max_mines=current_stage_mines,
                    apply_epsilon_greedy=False  # No exploration wrapper for evaluation
                )])
                print(f"🔧 Using standard evaluation environment")
            
            # Apply action masking wrapper to evaluation environment as well
            eval_env = ActionMaskingWrapper(eval_env)
            print(f"🔒 Applied action masking wrapper to evaluation environment")
            
            # Ensure evaluation environment is identical to training environment
            # This follows best practices: separate but identical environments
            print(f"🔧 Ensuring evaluation environment matches training environment...")
            print(f"   Training: With action masking (no invalid actions)")
            print(f"   Evaluation: With action masking (no invalid actions)")
            
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
            
            # REMOVED: DeterministicEvalCallback - was adding unnecessary complexity
            print(f"🔍 Using simple evaluation approach")
            
            # ENHANCED: Optimize evaluation frequency for extended training
            if args.curriculum_mode in ["human_performance", "superhuman"]:
                # For extended training, use more frequent evaluation
                # This provides better monitoring and early stopping opportunities
                enhanced_eval_freq = max(100, args.eval_freq // 2)  # More frequent evaluation
                eval_callback.eval_freq = enhanced_eval_freq
                
                print(f"🔍 Enhanced evaluation frequency: {enhanced_eval_freq} steps")
                print(f"   (More frequent evaluation for extended training)")
                
                # Also enhance the iteration callback for better monitoring
                iteration_callback = IterationCallback(
                    verbose=args.verbose,
                    debug_level=2,
                    experiment_tracker=experiment_tracker,
                    stats_file="training_stats/training_stats.txt",
                    timestamped_stats=args.timestamped_stats,
                    stats_manager=stats_manager
                )
                # Add curriculum config for board size display
                iteration_callback.curriculum_config = curriculum_stages
                iteration_callback.current_stage = stage + 1  # Add current stage number
            else:
                # Standard evaluation for current mode
                iteration_callback = IterationCallback(
                    verbose=args.verbose,
                    debug_level=2,
                    experiment_tracker=experiment_tracker,
                    stats_file="training_stats/training_stats.txt",
                    timestamped_stats=args.timestamped_stats,
                    stats_manager=stats_manager
                )
                # Add curriculum config for board size display
                iteration_callback.curriculum_config = curriculum_stages
                iteration_callback.current_stage = stage + 1  # Add current stage number
            
            # Train for this stage with enhanced timesteps
            print(f"🚀 Starting enhanced training for Stage {stage + 1}...")
            training_model.learn(
                total_timesteps=enhanced_timesteps,
                callback=[eval_callback, iteration_callback],  # REMOVED: deterministic_callback
                progress_bar=True
            )
            
            # Check for shutdown request after training
            if shutdown_requested:
                print("\n🛑 Shutdown requested. Stopping training gracefully...")
                break
            
            # Evaluate current stage with enhanced episodes
            # Use the EVALUATION environment (following best practices)
            print(f"🔍 Running final evaluation with {eval_episodes} episodes...")
            
            # Use the EVALUATION environment for final assessment (best practice)
            # This ensures unbiased evaluation while maintaining consistency
            rewards = []
            wins = 0
            
            for episode in range(eval_episodes):
                obs = eval_env.reset()
                done = False
                episode_reward = 0
                episode_won = False
                
                while not done:
                    action = training_model.predict(obs, deterministic=True)[0]
                    step_result = eval_env.step(action)
                    
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
                    elif info and isinstance(info, dict):
                        if info.get('won', False):
                            episode_won = True
                
                rewards.append(episode_reward)
                if episode_won:
                    wins += 1
            
            # Calculate metrics (same as CustomEvalCallback)
            mean_reward = np.mean(rewards) if len(rewards) > 0 else 0.0
            win_rate = (wins / eval_episodes) * 100 if eval_episodes > 0 else 0.0
            reward_std = np.std(rewards) / np.sqrt(len(rewards)) if len(rewards) > 1 else 0.0
            
            print(f"Final evaluation results:")
            print(f"  Episodes: {eval_episodes}")
            print(f"  Wins: {wins}")
            print(f"  Win rate: {win_rate:.1f}%")
            print(f"  Mean reward: {mean_reward:.2f} +/- {reward_std:.2f}")
            
            # Convert win rate to decimal for consistency
            win_rate = win_rate / 100
            
            # Compare with training buffer metrics for debugging
            if len(training_model.ep_info_buffer) > 0:
                buffer_episodes = training_model.ep_info_buffer
                buffer_wins = sum(1 for ep_info in buffer_episodes if ep_info.get("won", False))
                buffer_total = len(buffer_episodes)
                buffer_win_rate = (buffer_wins / buffer_total) * 100 if buffer_total > 0 else 0
                buffer_avg_reward = np.mean([ep_info["r"] for ep_info in buffer_episodes])
                
                print(f"\n📊 Training Buffer vs Final Evaluation Comparison:")
                print(f"  Training Buffer: {buffer_win_rate:.1f}% ({buffer_wins}/{buffer_total} wins)")
                print(f"  Final Evaluation: {win_rate*100:.1f}% ({wins}/{eval_episodes} wins)")
                print(f"  Buffer Avg Reward: {buffer_avg_reward:.2f}")
                print(f"  Final Avg Reward: {mean_reward:.2f}")
                
                if abs(buffer_win_rate - win_rate*100) > 10:
                    print(f"  ⚠️  Large discrepancy detected: {abs(buffer_win_rate - win_rate*100):.1f}% difference")
                    print(f"  💡 This suggests evaluation environment differences")
                    print(f"  🔧 Best practice: Separate but identical environments for unbiased evaluation")
                else:
                    print(f"  ✅ Evaluation consistency: {abs(buffer_win_rate - win_rate*100):.1f}% difference")
                    print(f"  ✅ Best practice followed: Separate environments with consistent results")
            
            print(f"\nStage {stage + 1} Results:")
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
                save_model_safely(training_model, "models/final_model")
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
                    save_model_safely(training_model, checkpoint_path)
                    print(f"💾 Checkpoint saved to: {checkpoint_path}")
                except Exception as e:
                    print(f"⚠️  Failed to save checkpoint: {e}")
    
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user (Ctrl+C)")
        if training_model is not None:
            # Save the current model as a checkpoint
            try:
                checkpoint_path = "models/checkpoint_interrupted"
                save_model_safely(training_model, checkpoint_path)
                print(f"💾 Checkpoint saved to: {checkpoint_path}")
                print(f"Last enhanced_timesteps: {enhanced_timesteps if enhanced_timesteps is not None else 'N/A'}")
            except Exception as e:
                print(f"⚠️  Failed to save checkpoint: {e}")
    
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        print(f"Last enhanced_timesteps: {enhanced_timesteps if enhanced_timesteps is not None else 'N/A'}")
                
        if training_model is not None:
            # Save the current model as a checkpoint
            try:
                checkpoint_path = "models/checkpoint_error"
                save_model_safely(training_model, checkpoint_path)
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

class IterationCallback(BaseCallback):
    """Custom callback for logging iteration information"""
    def __init__(self, verbose=0, debug_level=2, experiment_tracker=None, stats_file="training_stats/training_stats.txt", timestamped_stats=False, enable_file_logging=True, stats_manager=None):
        super().__init__(verbose)
        self.start_time = time.time()
        self.last_iteration_time = self.start_time
        self.iterations = 0
        self.debug_level = debug_level
        self.timestamped_stats = timestamped_stats
        self.enable_file_logging = enable_file_logging
        
        # Initialize stats manager if not provided
        if stats_manager is None:
            self.stats_manager = TrainingStatsManager()
        else:
            self.stats_manager = stats_manager
        
        # Handle timestamped stats files
        if self.timestamped_stats:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_stats_{timestamp}.txt"
            self.stats_file = str(self.stats_manager.get_stats_file_path(filename))
        elif stats_file != "training_stats/training_stats.txt":
            # Use the custom stats_file if provided
            self.stats_file = stats_file
        else:
            # Use default stats file
            self.stats_file = "training_stats/training_stats.txt"
        
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
        
        # Track if we've written any data to the file
        self.file_initialized = False
        self.has_written_data = False

    def log(self, message, level=2, force=False):
        """Log message if debug level is high enough"""
        if force or level <= self.debug_level:
            print(f"[{self.debug_levels[level]}] {message}")

    def _initialize_stats_file(self):
        """Initialize the stats file with headers only when we're about to write data"""
        if not self.enable_file_logging or self.file_initialized:
            return
            
        try:
            # Ensure directory exists for timestamped files
            file_path = Path(self.stats_file)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.stats_file, 'w') as f:
                f.write("timestamp,iteration,timesteps,win_rate,avg_reward,avg_length,stage,phase,stage_time,no_improvement\n")
            self.file_initialized = True
        except (IOError, OSError) as e:
            # If file creation fails, disable file logging
            self.enable_file_logging = False
            if self.verbose > 0:
                print(f"Warning: Could not create stats file {self.stats_file}: {e}")
                print("File logging disabled for this callback.")

    def _cleanup_empty_file(self):
        """Remove the stats file if it's empty (no data written)"""
        if self.enable_file_logging and self.file_initialized and not self.has_written_data:
            try:
                import os
                if os.path.exists(self.stats_file):
                    os.remove(self.stats_file)
                    if self.verbose > 0:
                        print(f"Cleaned up empty stats file: {self.stats_file}")
            except (IOError, OSError) as e:
                if self.verbose > 0:
                    print(f"Warning: Could not remove empty stats file {self.stats_file}: {e}")

    def on_training_end(self):
        """Called when training ends - move completed files to history and cleanup empty files"""
        if self.has_written_data and self.timestamped_stats:
            # Move completed timestamped files to history
            self.stats_manager.move_to_history(self.stats_file)
        else:
            # Clean up empty files
            self._cleanup_empty_file()

    def on_training_error(self):
        """Called when training errors - cleanup empty files"""
        self._cleanup_empty_file()

    def __getstate__(self):
        """Exclude non-pickleable attributes when saving the callback"""
        state = self.__dict__.copy()
        # Remove non-pickleable attributes
        state.pop('experiment_tracker', None)
        state.pop('stats_manager', None)
        # Remove any file handles or other non-pickleable objects
        state.pop('_last_learning_phase', None)
        state.pop('_last_curriculum_stage', None)
        return state

    def __setstate__(self, state):
        """Restore state when loading the callback"""
        self.__dict__.update(state)
        # Reinitialize non-pickleable attributes
        self.experiment_tracker = None
        self.stats_manager = None
        self._last_learning_phase = None
        self._last_curriculum_stage = None

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
                if hasattr(self, 'curriculum_config') and self.curriculum_stage <= len(self.curriculum_config):
                    # Update the curriculum stage based on current training
                    self.curriculum_stage = stage + 1  # stage is 0-indexed, so add 1
                
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
                    # Fix division by zero error
                    stage_win_rate = (self.stage_wins/self.stage_games*100) if self.stage_games > 0 else 0.0
                    mlflow.log_metric("stage_win_rate", stage_win_rate, step=self.iterations)
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
                # Fix division by zero error
                stage_win_rate = (self.stage_wins/self.stage_games*100) if self.stage_games > 0 else 0.0
                self.log(f"Stage Win Rate: {stage_win_rate:.1f}%")
                
                # Add board size information if available
                try:
                    # Try to get board size from environment
                    if hasattr(self.model.env, 'get_board_variation_info'):
                        board_info = self.model.env.get_board_variation_info()
                        if 'current_variation' in board_info:
                            self.log(f"Board Variation: {board_info['current_variation']}/{board_info['total_variations']}")
                    elif hasattr(self.model.env, 'venv') and hasattr(self.model.env.venv, 'get_board_variation_info'):
                        board_info = self.model.env.venv.get_board_variation_info()
                        if 'current_variation' in board_info:
                            self.log(f"Board Variation: {board_info['current_variation']}/{board_info['total_variations']}")
                    
                    # Try to get board size from curriculum config
                    if hasattr(self, 'curriculum_config') and hasattr(self, 'current_stage') and self.current_stage <= len(self.curriculum_config):
                        stage_config = self.curriculum_config[self.current_stage - 1]
                        board_size = stage_config.get('size', '?')
                        mines = stage_config.get('mines', '?')
                        board_info = f"Board={board_size}x{board_size}x{mines}"
                    else:
                        board_info = ""
                except Exception as e:
                    # Board info not available, skip logging
                    board_info = ""
                
                # Display compact single-line format like simple training script
                episode_num = self.iterations * 100  # Approximate episode number
                compact_line = f"Episode {episode_num}: Avg Reward={avg_reward:.2f}, Win Rate={win_rate:.3f}, Avg Length={avg_length:.1f}"
                if board_info:
                    compact_line += f", {board_info}"
                self.log(compact_line)
                
                # Track metrics for improvement calculation
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
                        # Initialize file with headers on first write
                        self._initialize_stats_file()
                        
                        stats_line = f"{timestamp},{self.iterations},{self.num_timesteps},{win_rate:.1f},{avg_reward:.2f},{avg_length:.1f},{self.curriculum_stage},{self.learning_phase},{stage_time:.0f},{self.no_improvement_count}\n"
                        with open(self.stats_file, 'a') as f:
                            f.write(stats_line)
                        
                        # Mark that we've written data
                        self.has_written_data = True
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
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            """Recursively convert numpy types to native Python types for JSON serialization."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Convert metrics to JSON-serializable format
        metrics_to_save = convert_numpy_types(metrics_to_save)
        
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
                    elif info and isinstance(info, dict):
                        if info.get('won', False):
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
                    try:
                        save_model_safely(self.model, best_model_path)
                        if self.verbose > 0:
                            print(f"New best model achieved with mean reward: {self.best_mean_reward:.2f}")
                    except Exception as e:
                        if self.verbose > 0:
                            print(f"Warning: Could not save best model: {e}")
            
            # Log metrics
            if self.log_path:
                log_file = os.path.join(self.log_path, "eval_log.txt")
                with open(log_file, "a") as f:
                    f.write(f"Step {self.n_calls}: mean_reward={mean_reward:.2f}, win_rate={win_rate:.1f}%\n")
        
        return True

    def __getstate__(self):
        """Exclude non-pickleable attributes when saving the callback"""
        state = self.__dict__.copy()
        # Remove eval_env as it may contain non-pickleable objects
        state.pop('eval_env', None)
        return state

    def __setstate__(self, state):
        """Restore state when loading the callback"""
        self.__dict__.update(state)
        # Reinitialize non-pickleable attributes
        self.eval_env = None

def save_model_safely(model, path):
    """
    Save model safely by creating a clean copy without non-pickleable attributes.
    """
    import copy
    import pickle
    
    # Store original MLflow state
    original_mlflow_active = False
    original_mlflow_run = None
    
    try:
        import mlflow
        # Check if MLflow is active
        original_mlflow_active = mlflow.active_run() is not None
        if original_mlflow_active:
            # Store the current run
            original_mlflow_run = mlflow.active_run()
            # End the current run temporarily
            mlflow.end_run()
    except Exception:
        # MLflow not available or already disabled
        pass
    
    try:
        # Create a deep copy of the model to avoid modifying the original
        model_copy = copy.deepcopy(model)
        
        # Remove any attributes that might contain threading locks or non-pickleable objects
        problematic_attrs = []
        for attr_name in list(model_copy.__dict__.keys()):
            attr_value = getattr(model_copy, attr_name)
            # Check if attribute contains threading locks or other non-pickleable objects
            if (hasattr(attr_value, '__class__') and 
                ('lock' in attr_name.lower() or 
                 'thread' in attr_name.lower() or
                 'mlflow' in attr_name.lower() or
                 'logger' in attr_name.lower() or
                 'file' in attr_name.lower() or
                 'mappingproxy' in str(type(attr_value)).lower() or
                 'weakref' in str(type(attr_value)).lower())):
                problematic_attrs.append(attr_name)
                delattr(model_copy, attr_name)
        
        # Also check for any callbacks or other objects that might contain locks
        if hasattr(model_copy, '_callbacks'):
            model_copy._callbacks = None
        
        # Remove any mappingproxy objects from the model's attributes
        for attr_name in list(model_copy.__dict__.keys()):
            attr_value = getattr(model_copy, attr_name)
            if hasattr(attr_value, '__class__') and 'mappingproxy' in str(type(attr_value)):
                delattr(model_copy, attr_name)
        
        # Save the cleaned model copy
        model_copy.save(path)
        print(f"✅ Model saved successfully to: {path}")
        
        # Clean up the copy
        del model_copy
        
    except Exception as e:
        print(f"⚠️  Failed to save model: {e}")
        # Try alternative approach: save just the policy
        try:
            print("🔄 Trying alternative save method (policy only)...")
            # Save just the policy network
            policy_path = path + "_policy"
            model.policy.save(policy_path)
            print(f"✅ Policy saved successfully to: {policy_path}")
        except Exception as e2:
            print(f"⚠️  Alternative save also failed: {e2}")
            # Try the most basic save method
            try:
                print("🔄 Trying basic save method...")
                basic_path = path + "_basic"
                # Save only the essential components
                torch.save({
                    'policy_state_dict': model.policy.state_dict(),
                    'policy_optimizer_state_dict': model.policy.optimizer.state_dict() if hasattr(model.policy, 'optimizer') else None,
                }, basic_path)
                print(f"✅ Basic model saved successfully to: {basic_path}")
            except Exception as e3:
                print(f"❌ All save methods failed: {e3}")
                raise e  # Re-raise the original error
    finally:
        # Restore MLflow logging if it was active
        if original_mlflow_active and original_mlflow_run is not None:
            try:
                import mlflow
                # Restart the run
                mlflow.start_run(run_id=original_mlflow_run.info.run_id)
            except Exception:
                # MLflow restoration failed, but model was saved
                pass

class ActionMaskingWrapper(VecEnvWrapper):
    """Wrapper that enforces action masking to prevent invalid actions."""
    
    def __init__(self, venv):
        super().__init__(venv)
        self.action_space = venv.action_space
    
    def reset(self):
        return self.venv.reset()
    
    def step(self, actions):
        # Get action masks from the environment
        action_masks = []
        for env in self.venv.envs:
            if hasattr(env, 'get_action_mask'):
                mask = env.get_action_mask()
            elif hasattr(env, 'action_masks'):
                mask = env.action_masks
            else:
                # If no action mask available, allow all actions
                mask = np.ones(self.action_space.n, dtype=bool)
            action_masks.append(mask)
        
        # Convert to numpy array
        action_masks = np.array(action_masks)
        
        # Mask out invalid actions by setting their probability to 0
        # For now, we'll replace invalid actions with random valid actions
        masked_actions = actions.copy()
        
        for i, (action, mask) in enumerate(zip(actions, action_masks)):
            if not mask[action]:
                # Find a valid action
                valid_actions = np.where(mask)[0]
                if len(valid_actions) > 0:
                    masked_actions[i] = np.random.choice(valid_actions)
                else:
                    # If no valid actions, keep the original (will be penalized)
                    pass
        
        # Take the step with masked actions
        return self.venv.step(masked_actions)
    
    def step_wait(self):
        """Required by VecEnvWrapper, but not used in this wrapper."""
        return self.venv.step_wait()
    
    def get_action_mask(self):
        """Get action mask for the current state."""
        masks = []
        for env in self.venv.envs:
            if hasattr(env, 'get_action_mask'):
                mask = env.get_action_mask()
            elif hasattr(env, 'action_masks'):
                mask = env.action_masks
            else:
                mask = np.ones(self.action_space.n, dtype=bool)
            masks.append(mask)
        return np.array(masks)

class MultiBoardTrainingWrapper(VecEnvWrapper):
    """Wrapper that generates different board layouts during training for better generalization."""
    
    def __init__(self, venv, board_variations=10):
        super().__init__(venv)
        self.board_variations = board_variations
        self.current_variation = 0
        self.action_space = venv.action_space
    
    def reset(self):
        # Generate a new board variation for each reset
        self.current_variation = (self.current_variation + 1) % self.board_variations
        
        # Set a different seed for each variation to ensure different boards
        variation_seed = 42 + self.current_variation
        
        # Reset all environments with the new seed
        for env in self.venv.envs:
            if hasattr(env, 'seed'):
                env.seed(variation_seed)
        
        return self.venv.reset()
    
    def step(self, actions):
        return self.venv.step(actions)
    
    def step_wait(self):
        """Required by VecEnvWrapper, but not used in this wrapper."""
        return self.venv.step_wait()
    
    def get_board_variation_info(self):
        """Get information about the current board variation."""
        return {
            'current_variation': self.current_variation,
            'total_variations': self.board_variations,
            'variation_seed': 42 + self.current_variation
        }

if __name__ == "__main__":
    main() 