import os
import numpy as np
import time
import json
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from minesweeper_env import MinesweeperEnv
import argparse
import torch
from scipy import stats

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
        
        # Create experiment directory if it doesn't exist
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
    
    def start_new_run(self, hyperparameters):
        """Start a new experiment run with given hyperparameters"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_run = os.path.join(self.experiment_dir, f"run_{timestamp}")
        os.makedirs(self.current_run)
        
        # Save hyperparameters
        self.metrics["hyperparameters"] = hyperparameters
        self.metrics["metadata"] = {
            "start_time": timestamp,
            "random_seed": hyperparameters.get("random_seed", None)
        }
        
        # Save initial configuration
        self._save_metrics()
    
    def add_training_metric(self, metric_name, value, step):
        """Add a training metric"""
        self.metrics["training"].append({
            "step": step,
            "metric": metric_name,
            "value": value
        })
        self._save_metrics()
    
    def add_validation_metric(self, metric_name, value, confidence_interval=None):
        """Add a validation metric with optional confidence interval"""
        self.metrics["validation"].append({
            "metric": metric_name,
            "value": value,
            "confidence_interval": confidence_interval
        })
        self._save_metrics()
    
    def _save_metrics(self):
        """Save current metrics to file"""
        if self.current_run:
            with open(os.path.join(self.current_run, "metrics.json"), "w") as f:
                json.dump(self.metrics, f, indent=2)

class IterationCallback(BaseCallback):
    """Custom callback for logging iteration information"""
    def __init__(self, verbose=0, debug_level=2, experiment_tracker=None):
        super().__init__(verbose)
        self.start_time = time.time()
        self.last_iteration_time = self.start_time
        self.iterations = 0
        self.debug_level = debug_level
        
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
        
        # Debug information
        self.last_step_time = time.time()
        self.step_times = []
        self.episode_lengths = []
        self.rewards = []
        
        # Debug level descriptions
        self.debug_levels = {
            0: "ERROR",    # Critical issues
            1: "WARNING",  # Potential problems
            2: "INFO",     # Basic progress (default)
            3: "DEBUG",    # Detailed metrics
            4: "VERBOSE"   # Everything
        }
        
        self.experiment_tracker = experiment_tracker

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
        while hasattr(env, 'env'):
            env = env.env
        return getattr(env, attr, None)

    def _on_step(self):
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
                
                # Win rate (episodes with positive reward)
                wins = sum(1 for ep_info in current_episodes if ep_info["r"] > 0)
                total_episodes = len(current_episodes)
                win_rate = (wins / total_episodes) * 100 if total_episodes > 0 else 0
                
                # Average game length
                avg_length = np.mean([ep_info["l"] for ep_info in current_episodes])
                self.episode_lengths.append(avg_length)
                
                # Calculate improvements
                reward_improvement = ((avg_reward - self.last_avg_reward) / 
                                    (abs(self.last_avg_reward) + 1e-6)) * 100
                win_rate_improvement = win_rate - self.last_win_rate
                length_improvement = ((self.last_avg_length - avg_length) / 
                                    (abs(self.last_avg_length) + 1e-6)) * 100
                
                # Update best metrics
                self.best_reward = max(self.best_reward, avg_reward)
                self.best_win_rate = max(self.best_win_rate, win_rate)
                
                # Update learning phase
                self._update_learning_phase(avg_reward, win_rate)
                
                # Store current metrics for next iteration
                self.last_avg_reward = avg_reward
                self.last_win_rate = win_rate
                self.last_avg_length = avg_length
                
                # Print detailed metrics
                print("\n" + "="*70)
                print(f"Training Progress at {time.strftime('%H:%M:%S')}")
                print("="*70)
                print("\nGame Configuration:")
                env = self.training_env.envs[0]
                board_size = self.get_env_attr(env, 'current_board_size')
                mines = self.get_env_attr(env, 'current_mines')
                max_steps = self.get_env_attr(env, 'max_steps')
                print(f"Board Size: {board_size}x{board_size}")
                print(f"Mines: {mines} (Density: {(mines/(board_size**2))*100:.1f}%)")
                print(f"Max Steps: {max_steps}")
                
                print("\nPerformance Metrics:")
                print(f"Win Rate: {win_rate:.1f}%")
                print(f"Average Reward: {avg_reward:.2f}")
                print(f"Average Game Length: {avg_length:.1f} steps")
                
                print("\nGame Statistics:")
                print(f"Total Games: {total_episodes}")
                print(f"Total Wins: {wins}")
                print(f"Games at Current Size: {total_episodes}")
                
                print("\nRecent Performance:")
                recent_rewards = [f"{r:.1f}" for r in self.rewards[-10:]]
                recent_lengths = [f"{l:.0f}" for l in self.episode_lengths[-10:]]
                print(f"Last 10 Rewards: {recent_rewards}")
                print(f"Last 10 Game Lengths: {recent_lengths}")
                print("="*70 + "\n")
                
                # Level 2 (INFO) - Basic progress updates
                self.log(f"\nIteration {self.iterations}", level=2)
                self.log(f"Timesteps: {self.num_timesteps}", level=2)
                self.log(f"Progress: {self.num_timesteps/self.locals['total_timesteps']*100:.1f}%", level=2)
                self.log(f"Win Rate: {win_rate:.1f}% (Best: {self.best_win_rate:.1f}%)", level=2)
                
                # Level 3 (DEBUG) - Detailed metrics
                self.log(f"\nLearning Phase: {self.learning_phase}", level=3)
                self.log(f"Learning Metrics:", level=3)
                self.log(f"- Average Reward: {avg_reward:.2f} (Best: {self.best_reward:.2f})", level=3)
                self.log(f"- Average Game Length: {avg_length:.1f} steps", level=3)
                self.log(f"- Total Games Played: {total_episodes}", level=3)
                
                self.log(f"\nImprovements (since last iteration):", level=3)
                self.log(f"- Reward: {reward_improvement:+.1f}%", level=3)
                self.log(f"- Win Rate: {win_rate_improvement:+.1f}%", level=3)
                self.log(f"- Game Length: {length_improvement:+.1f}%", level=3)
                
                # Log to experiment tracker
                if self.experiment_tracker:
                    self.experiment_tracker.add_training_metric("avg_reward", avg_reward, self.num_timesteps)
                    self.experiment_tracker.add_training_metric("win_rate", win_rate, self.num_timesteps)
                    self.experiment_tracker.add_training_metric("avg_length", avg_length, self.num_timesteps)
            else:
                avg_reward = 0
                win_rate = 0
                avg_length = 0
                reward_improvement = 0
                win_rate_improvement = 0
                length_improvement = 0
            
            self.last_iteration_time = current_time
            
        return True

def make_env(board_size=8, num_mines=12):
    """Create and wrap the Minesweeper environment"""
    env = MinesweeperEnv(board_size=board_size, num_mines=num_mines)
    # Add debug prints for environment creation
    print("\nEnvironment created with:")
    print(f"- Board size: {env.current_board_size}x{env.current_board_size}")
    print(f"- Number of mines: {env.current_mines}")
    print(f"- Action space: {env.action_space}")
    print(f"- Observation space: {env.observation_space}")
    return env

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--timesteps', type=int, default=100000, help='Number of timesteps to train for')
    parser.add_argument('--test-mode', action='store_true', help='Run in test mode with reduced parameters')
    parser.add_argument('--quick-test', action='store_true', help='Run a very quick test with minimal parameters')
    parser.add_argument('--curriculum', action='store_true', help='Enable curriculum learning')
    parser.add_argument('--board-size', type=int, default=8, help='Board size for training')
    parser.add_argument('--max-mines', type=int, default=12, help='Maximum number of mines')
    parser.add_argument('--debug-level', type=int, default=2, choices=[0,1,2,3,4], help='Debug level (0=ERROR, 1=WARNING, 2=INFO, 3=DEBUG, 4=VERBOSE)')
    parser.add_argument('--load-model', type=str, help='Path to previous model to continue training from')
    parser.add_argument('--random-seed', type=int, help='Random seed for reproducibility')
    # PPO hyperparameters
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate for PPO')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for PPO')
    parser.add_argument('--n-steps', type=int, default=2048, help='Number of steps to run for each environment per update')
    parser.add_argument('--n-epochs', type=int, default=10, help='Number of epochs to run when optimizing the surrogate loss')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='Factor for trade-off of bias vs variance for Generalized Advantage Estimator')
    parser.add_argument('--clip-range', type=float, default=0.2, help='Clipping parameter for PPO')
    parser.add_argument('--ent-coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--vf-coef', type=float, default=0.5, help='Value function coefficient')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='Maximum gradient norm')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seeds for reproducibility
    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
    
    # Create experiment tracker
    tracker = ExperimentTracker()
    
    # Save hyperparameters
    hyperparameters = {
        "timesteps": args.timesteps,
        "board_size": args.board_size,
        "max_mines": args.max_mines,
        "use_gpu": args.use_gpu,
        "random_seed": args.random_seed,
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm
    }
    
    tracker.start_new_run(hyperparameters)
    
    # Set up logging
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create and wrap the environment
    env = make_env(board_size=args.board_size, num_mines=args.max_mines)
    env = Monitor(env, log_dir)
    
    # Create or load the agent
    if args.load_model and os.path.exists(args.load_model):
        print(f"\nLoading previous model from: {args.load_model}")
        model = PPO.load(
            args.load_model,
            env=env,
            tensorboard_log=log_dir,
            device="cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"
        )
        print("Model loaded successfully")
    else:
        print("\nCreating new model")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            policy_kwargs=dict(
                net_arch=[dict(pi=[128, 128], vf=[128, 128])]
            ),
            verbose=1,
            tensorboard_log=log_dir,
            device="cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"
        )
    
    # Train the agent
    total_timesteps = args.timesteps
    if args.test_mode:
        total_timesteps = 10000
    elif args.quick_test:
        total_timesteps = 1000
    
    # Add callback for monitoring with debug level
    callback = IterationCallback(debug_level=args.debug_level, experiment_tracker=tracker)
    
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        callback=callback
    )
    
    # Evaluate the trained agent
    print("\nEvaluating trained agent...")
    eval_results = evaluate_model(model, env)
    
    # Save evaluation results
    tracker.add_validation_metric("win_rate", eval_results["win_rate"])
    tracker.add_validation_metric("avg_reward", eval_results["avg_reward"], 
                                eval_results["reward_ci"])
    tracker.add_validation_metric("avg_length", eval_results["avg_length"],
                                eval_results["length_ci"])
    
    # Print evaluation results
    print("\nEvaluation Results:")
    print(f"Win Rate: {eval_results['win_rate']:.1f}%")
    print(f"Average Reward: {eval_results['avg_reward']:.2f}")
    print(f"95% CI for Reward: [{eval_results['reward_ci'][0]:.2f}, {eval_results['reward_ci'][1]:.2f}]")
    print(f"Average Length: {eval_results['avg_length']:.1f}")
    print(f"95% CI for Length: [{eval_results['length_ci'][0]:.1f}, {eval_results['length_ci'][1]:.1f}]")
    
    # Save the trained model
    model_path = os.path.join(tracker.current_run, "model.zip")
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")

def evaluate_model(model, env, n_episodes=100):
    """Evaluate model with proper statistical analysis"""
    rewards = []
    lengths = []
    wins = 0
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
        
        rewards.append(episode_reward)
        lengths.append(episode_length)
        if episode_reward > 0:
            wins += 1
    
    # Calculate statistics
    win_rate = (wins / n_episodes) * 100
    avg_reward = np.mean(rewards)
    avg_length = np.mean(lengths)
    
    # Calculate confidence intervals (95%)
    reward_ci = stats.t.interval(0.95, len(rewards)-1, 
                               loc=np.mean(rewards), 
                               scale=stats.sem(rewards))
    length_ci = stats.t.interval(0.95, len(lengths)-1, 
                               loc=np.mean(lengths), 
                               scale=stats.sem(lengths))
    
    return {
        "win_rate": win_rate,
        "avg_reward": avg_reward,
        "avg_length": avg_length,
        "reward_ci": reward_ci,
        "length_ci": length_ci,
        "n_episodes": n_episodes
    }

if __name__ == "__main__":
    main() 