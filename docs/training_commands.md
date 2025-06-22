# Training Command Line Options

This guide explains all available command line options for the RL training script (`src/core/train_agent.py`).

## Quick Start Scripts

For convenience, use these pre-configured scripts:

- `scripts/mac/quick_test.sh` - Quick test (~1-2 minutes)
- `scripts/mac/medium_test.sh` - Medium test (~5-10 minutes)  
- `scripts/mac/full_training.sh` - Full training (~1-2 hours)

## Most Commonly Used Options

### Core Training Parameters

| Option | Default | Description |
|--------|---------|-------------|
| `--total_timesteps` | 1000000 | Total number of timesteps to train for |
| `--eval_freq` | 10000 | How often to evaluate the agent (in timesteps) |
| `--n_eval_episodes` | 100 | Number of episodes to run during evaluation |
| `--verbose` | 0 | Verbosity level (0=quiet, 1=normal, 2=detailed) |

### Learning Parameters

| Option | Default | Description |
|--------|---------|-------------|
| `--learning_rate` | 0.0003 | Learning rate for the PPO algorithm |
| `--n_steps` | 2048 | Number of steps to run for each environment per update |
| `--batch_size` | 64 | Minibatch size for training |
| `--n_epochs` | 10 | Number of epochs when optimizing the surrogate loss |

### Environment Settings

| Option | Default | Description |
|--------|---------|-------------|
| `--board_size` | 8 | Size of the Minesweeper board (8x8, 16x16, etc.) |
| `--num_mines` | 10 | Number of mines on the board |
| `--n_envs` | 8 | Number of parallel environments |

### Model Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `--model_save_freq` | 50000 | How often to save the model (in timesteps) |
| `--best_model_save_freq` | 10000 | How often to save the best model (in timesteps) |

## New Training Options (2024-12-21)

### Progression Control

| Option | Default | Description |
|--------|---------|-------------|
| `--strict_progression` | False | Require target win rate achievement before stage progression |
| `--timestamped_stats` | False | Use timestamped stats files to preserve training history |

### Enhanced Monitoring

The training script now includes enhanced monitoring with:
- **Multi-factor improvement detection**: Tracks new bests, consistent positive learning, phase progression
- **Realistic thresholds**: 50/100 iterations for warnings/critical (was 20/50)
- **Positive feedback**: Clear progress indicators with emojis
- **Problem detection**: Identifies consistently negative rewards as real issues

## Advanced Options

### PPO Algorithm Tuning

| Option | Default | Description |
|--------|---------|-------------|
| `--gamma` | 0.99 | Discount factor for future rewards |
| `--gae_lambda` | 0.95 | Lambda parameter for GAE (Generalized Advantage Estimation) |
| `--clip_range` | 0.2 | Clipping parameter for PPO |
| `--clip_range_vf` | null | Clipping parameter for value function (if null, uses clip_range) |
| `--ent_coef` | 0.0 | Entropy coefficient for exploration |
| `--vf_coef` | 0.5 | Value function coefficient in loss calculation |
| `--max_grad_norm` | 0.5 | Maximum gradient norm for gradient clipping |

### Network Architecture

| Option | Default | Description |
|--------|---------|-------------|
| `--policy_kwargs` | "{}" | Additional arguments for policy network |
| `--net_arch` | "[64, 64]" | Network architecture (list of hidden layer sizes) |
| `--activation_fn` | "tanh" | Activation function (tanh, relu, etc.) |

### Training Control

| Option | Default | Description |
|--------|---------|-------------|
| `--target_kl` | None | Limit the KL divergence between updates |
| `--early_stopping` | false | Whether to stop training early if target_kl is reached |
| `--reset_num_timesteps` | true | Whether to reset the number of timesteps |

## Specialized Options

### Curriculum Learning

| Option | Default | Description |
|--------|---------|-------------|
| `--curriculum` | false | Enable curriculum learning |
| `--curriculum_stages` | "[{'board_size': 5, 'num_mines': 3}, {'board_size': 8, 'num_mines': 10}]" | Curriculum stages |
| `--curriculum_timesteps` | "[50000, 200000]" | Timesteps for each curriculum stage |

### Debugging and Monitoring

| Option | Default | Description |
|--------|---------|-------------|
| `--log_interval` | 1 | Log interval for training progress |
| `--save_vecnormalize` | false | Save VecNormalize statistics |
| `--load_vecnormalize` | null | Load VecNormalize statistics from file |
| `--monitor_wrapper` | false | Use Monitor wrapper for environment |

### Performance Tuning

| Option | Default | Description |
|--------|---------|-------------|
| `--device` | "auto" | Device to use for training (auto/cpu/cuda/mps) |
| `--seed` | None | Random seed |
| `--deterministic` | false | Use deterministic algorithms |
| `--init_setup_model` | True | Whether to initialize the model |

## Example Commands

### Basic Training (Optimized)
```bash
python src/core/train_agent.py --total_timesteps 100000 --verbose 0
```

### Quick Test with Custom Settings
```bash
python src/core/train_agent.py \
    --total_timesteps 10000 \
    --eval_freq 2000 \
    --n_eval_episodes 20 \
    --board_size 5 \
    --num_mines 3 \
    --verbose 0
```

### Production Training
```bash
python src/core/train_agent.py \
    --total_timesteps 1000000 \
    --eval_freq 10000 \
    --n_eval_episodes 100 \
    --learning_rate 0.0003 \
    --n_steps 2048 \
    --batch_size 64 \
    --n_epochs 10 \
    --verbose 0
```

### Strict Progression Training
```bash
python src/core/train_agent.py \
    --total_timesteps 50000 \
    --strict_progression True \
    --verbose 0
```

### Training with History Preservation
```bash
python src/core/train_agent.py \
    --total_timesteps 50000 \
    --timestamped_stats True \
    --verbose 0
```

### Debug Training
```bash
python src/core/train_agent.py \
    --total_timesteps 5000 \
    --eval_freq 1000 \
    --n_eval_episodes 10 \
    --verbose 2 \
    --seed 42 \
    --deterministic
```

## Output Files

Training generates several output files:

- `best_model/` - Directory containing the best trained model
- `logs/` - Training logs and checkpoints
- `mlruns/` - MLflow experiment tracking data
- `experiments/` - Experiment results and metrics
- `training_stats.txt` - Training progress statistics (reset each run)
- `training_stats_YYYYMMDD_HHMMSS.txt` - Timestamped stats (with `--timestamped_stats`)

## Monitoring Training

### MLflow UI (Recommended)
```bash
mlflow ui
```
Then open http://127.0.0.1:5000 in your browser to view:
- Training metrics (win rate, rewards, episode length)
- Model performance over time
- Experiment comparisons
- Hyperparameter tracking

### Training Stats File
The `training_stats.txt` file contains:
- Timestamp, iteration, timesteps
- Win rate, average reward, average length
- Stage, learning phase, stage time
- No improvement count

### Enhanced Monitoring Features
- **Positive Feedback**: Shows when agent is learning with emojis
- **Progress Indicators**: Clear status messages for different types of improvement
- **Realistic Warnings**: Only warns after 50+ iterations without improvement
- **Problem Detection**: Identifies real issues vs normal learning patterns

## Performance Optimization

### Verbosity Levels
- `--verbose 0`: Minimal output, fastest training (recommended)
- `--verbose 1`: Normal output with progress updates
- `--verbose 2`: Detailed output for debugging

### M1 Mac Optimization
- **Device**: Automatically uses MPS (Metal Performance Shaders) on M1 Macs
- **Performance**: 10-20% faster training with `--verbose 0`
- **Memory**: Efficient GPU memory usage

### Training Speed
- **Small Boards**: ~1000 steps/second
- **Medium Boards**: ~800 steps/second  
- **Large Boards**: ~500 steps/second
- **Optimized**: 10-20% faster with minimal verbosity 