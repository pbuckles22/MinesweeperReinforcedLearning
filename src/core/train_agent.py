import os
import numpy as np
import time
import json
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import safe_mean
from src.core.minesweeper_env import MinesweeperEnv
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
        # Convert numpy types to Python native types for JSON serialization
        if hasattr(value, 'item'):
            value = value.item()
        if confidence_interval is not None:
            if hasattr(confidence_interval, 'item'):
                confidence_interval = confidence_interval.item()
            elif isinstance(confidence_interval, (list, tuple)):
                confidence_interval = [ci.item() if hasattr(ci, 'item') else ci for ci in confidence_interval]
        
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
        
        # Curriculum tracking
        self.curriculum_stage = 1
        self.stage_start_iteration = 0
        self.stage_wins = 0
        self.stage_games = 0
        
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
            
            self.last_iteration_time = current_time
            
        return True

def make_env(max_board_size, max_mines):
    """Create a wrapped environment"""
    def _init():
        env = MinesweeperEnv(
            max_board_size=max_board_size,
            max_mines=max_mines,
            render_mode=None,
            early_learning_mode=True,
            early_learning_threshold=200,
            early_learning_corner_safe=True,
            early_learning_edge_safe=True,
            mine_spacing=1,
            initial_board_size=4,  # Start with 4x4
            initial_mines=2,       # Start with 2 mines
            invalid_action_penalty=-0.1,
            mine_penalty=-10.0,
            safe_reveal_base=5.0,
            win_reward=100.0
        )
        env = Monitor(env)
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
    parser.add_argument('--tensorboard_log', type=str, default="./tensorboard/",
                      help='Tensorboard log directory')
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
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create experiment tracker
    experiment_tracker = ExperimentTracker()
    
    # Define curriculum stages with detailed information
    curriculum_stages = [
        {
            'name': 'Beginner',
            'size': 4,
            'mines': 2,
            'win_rate_threshold': 0.7,
            'description': '4x4 board with 2 mines - Learning basic movement and safe cell identification'
        },
        {
            'name': 'Intermediate',
            'size': 6,
            'mines': 4,
            'win_rate_threshold': 0.6,
            'description': '6x6 board with 4 mines - Developing pattern recognition and basic strategy'
        },
        {
            'name': 'Easy',
            'size': 9,
            'mines': 10,
            'win_rate_threshold': 0.5,
            'description': '9x9 board with 10 mines - Standard easy difficulty, mastering basic gameplay'
        },
        {
            'name': 'Normal',
            'size': 16,
            'mines': 40,
            'win_rate_threshold': 0.4,
            'description': '16x16 board with 40 mines - Standard normal difficulty, developing advanced strategies'
        },
        {
            'name': 'Hard',
            'size': (16, 30),
            'mines': 99,
            'win_rate_threshold': 0.3,
            'description': '16x30 board with 99 mines - Standard hard difficulty, mastering complex patterns'
        },
        {
            'name': 'Expert',
            'size': (18, 24),
            'mines': 115,
            'win_rate_threshold': 0.2,
            'description': '18x24 board with 115 mines - Expert level, handling high mine density'
        },
        {
            'name': 'Chaotic',
            'size': (20, 35),
            'mines': 130,
            'win_rate_threshold': 0.1,
            'description': '20x35 board with 130 mines - Ultimate challenge, maximum complexity'
        }
    ]
    
    # Save curriculum information
    experiment_tracker.metrics["curriculum"] = {
        "stages": curriculum_stages,
        "total_stages": len(curriculum_stages),
        "expected_progression": "Beginner -> Intermediate -> Easy -> Normal -> Hard -> Expert -> Chaotic"
    }
    experiment_tracker._save_metrics()
    
    # Initialize environment with first stage
    current_stage = 0
    env = DummyVecEnv([make_env(
        max_board_size=curriculum_stages[current_stage]['size'],
        max_mines=curriculum_stages[current_stage]['mines']
    )])
    
    # Create model
    model = PPO(
        policy=args.policy,
        env=env,
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
        tensorboard_log=args.tensorboard_log,
        verbose=args.verbose,
        seed=args.seed,
        device=args.device,
        _init_setup_model=args._init_setup_model
    )
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(
        max_board_size=curriculum_stages[current_stage]['size'],
        max_mines=curriculum_stages[current_stage]['mines']
    )])
    
    # Create evaluation callback
    eval_callback = CustomEvalCallback(
        eval_env,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        verbose=args.verbose,
        best_model_save_path="./best_model",
        log_path="./logs/"
    )
    
    # Create iteration callback
    iteration_callback = IterationCallback(
        verbose=args.verbose,
        debug_level=2,
        experiment_tracker=experiment_tracker
    )
    
    # Start training
    total_timesteps = args.total_timesteps
    timesteps_per_stage = total_timesteps // len(curriculum_stages)
    
    print("\n=== Minesweeper Training Curriculum ===")
    print("Expected progression through difficulty levels:")
    for i, stage in enumerate(curriculum_stages):
        print(f"\nStage {i + 1}: {stage['name']}")
        print(f"Board: {stage['size'] if isinstance(stage['size'], int) else f'{stage['size'][0]}x{stage['size'][1]}'}")
        print(f"Mines: {stage['mines']}")
        print(f"Target Win Rate: {stage['win_rate_threshold']*100:.0f}%")
        print(f"Description: {stage['description']}")
    print("\n=====================================")
    
    for stage in range(len(curriculum_stages)):
        current_stage_info = curriculum_stages[stage]
        print(f"\n{'='*50}")
        print(f"Starting Stage {stage + 1}/{len(curriculum_stages)}: {current_stage_info['name']}")
        print(f"Board: {current_stage_info['size'] if isinstance(current_stage_info['size'], int) else f'{current_stage_info['size'][0]}x{current_stage_info['size'][1]}'}")
        print(f"Mines: {current_stage_info['mines']}")
        print(f"Target Win Rate: {current_stage_info['win_rate_threshold']*100:.0f}%")
        print(f"Description: {current_stage_info['description']}")
        print(f"{'='*50}\n")
        
        # Update environment for current stage
        env = DummyVecEnv([make_env(
            max_board_size=current_stage_info['size'],
            max_mines=current_stage_info['mines']
        )])
        model.set_env(env)
        
        # Update evaluation environment for current stage
        eval_env = DummyVecEnv([make_env(
            max_board_size=current_stage_info['size'],
            max_mines=current_stage_info['mines']
        )])
        eval_callback.eval_env = eval_env
        
        # Train for this stage
        model.learn(
            total_timesteps=timesteps_per_stage,
            callback=[eval_callback, iteration_callback],
            progress_bar=True
        )
        
        # Evaluate current stage
        evaluation_results = evaluate_model(model, env, n_episodes=args.n_eval_episodes)
        win_rate = evaluation_results["win_rate"] / 100  # Convert percentage to decimal
        mean_reward = evaluation_results["avg_reward"]
        reward_std = evaluation_results["reward_ci"]
        
        print(f"\nStage {stage + 1} Results:")
        print(f"Mean reward: {mean_reward:.2f} +/- {reward_std:.2f}")
        print(f"Win rate: {win_rate:.2%}")
        print(f"Target win rate: {current_stage_info['win_rate_threshold']*100:.0f}%")
        
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
        
        # Save model for this stage
        model.save(f"models/stage_{stage + 1}")
        
        # Add stage completion to metrics
        if "stage_completion" not in experiment_tracker.metrics:
            experiment_tracker.metrics["stage_completion"] = {}
        experiment_tracker.metrics["stage_completion"][f"stage_{stage + 1}"] = {
            "name": current_stage_info['name'],
            "win_rate": win_rate,
            "mean_reward": mean_reward,
            "std_reward": reward_std,
            "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        experiment_tracker._save_metrics()
    
    # Save final model
    model.save("models/final_model")
    print("\nTraining completed!")
    print("\nFinal Stage Progression:")
    for stage in range(len(curriculum_stages)):
        stage_info = experiment_tracker.metrics["stage_completion"][f"stage_{stage + 1}"]
        print(f"\nStage {stage + 1}: {stage_info['name']}")
        print(f"Final Win Rate: {stage_info['win_rate']:.2%}")
        print(f"Mean Reward: {stage_info['mean_reward']:.2f} +/- {stage_info['std_reward']:.2f}")
        print(f"Completed at: {stage_info['completed_at']}")

def evaluate_model(model, env, n_episodes=100):
    """Evaluate model with proper statistical analysis, supporting both vectorized and non-vectorized environments."""
    rewards = []
    lengths = []
    wins = 0

    # Check if this is a vectorized environment
    # More robust detection: check for num_envs attribute AND that it's actually a vectorized env
    is_vectorized = hasattr(env, 'num_envs') and hasattr(env, 'step') and callable(getattr(env, 'step', None))

    for episode in range(n_episodes):
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

            # Handle both old gym API (4 values) and new gymnasium API (5 values)
            if len(step_result) == 4:
                obs, reward, terminated, truncated = step_result
                info = {}  # No info in old API
            else:
                obs, reward, terminated, truncated, info = step_result

            # For vectorized envs, unwrap arrays/lists
            if is_vectorized:
                # DummyVecEnv returns arrays/lists of length num_envs
                # Check if terminated/truncated are arrays/lists before indexing
                if isinstance(terminated, (list, np.ndarray)) and len(terminated) > 0:
                    done = bool(terminated[0] or truncated[0])
                else:
                    done = bool(terminated or truncated)
                
                r = reward[0] if isinstance(reward, (np.ndarray, list)) else reward
                episode_reward += r
                # Info can be a list of dicts
                info_dict = info[0] if isinstance(info, (list, tuple)) and len(info) > 0 else info
                if info_dict.get('won', False):
                    episode_won = True
            else:
                done = bool(terminated or truncated)
                episode_reward += reward
                if info.get('won', False):
                    episode_won = True

            episode_length += 1

        rewards.append(episode_reward)
        lengths.append(episode_length)
        if episode_won:
            wins += 1

    # Calculate statistics
    win_rate = (wins / n_episodes) * 100
    avg_reward = float(np.mean(rewards))
    avg_length = float(np.mean(lengths))

    # Calculate standard error for confidence intervals
    reward_std = float(np.std(rewards) / np.sqrt(len(rewards)) if len(rewards) > 1 else 0.0)
    length_std = float(np.std(lengths) / np.sqrt(len(lengths)) if len(lengths) > 1 else 0.0)

    # Return dictionary with all metrics
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