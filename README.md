# Minesweeper Reinforcement Learning

A reinforcement learning project that trains an agent to play Minesweeper using PPO (Proximal Policy Optimization).

## Installation

### Windows
1. Run the PowerShell installation script:
```powershell
.\install_and_run.ps1
```

Options:
- `-Force`: Automatically delete existing virtual environment if present
- `-NoCache`: Install dependencies without using pip cache
- `--use-gpu`: Install PyTorch with CUDA support

Example:
```powershell
.\install_and_run.ps1 -Force -NoCache
```

### Linux/Mac
1. Run the shell installation script:
```bash
./install_and_run.sh
```

Options:
- `--force`: Automatically delete existing virtual environment if present
- `--no-cache`: Install dependencies without using pip cache
- `--use-gpu`: Install PyTorch with CUDA support

Example:
```bash
./install_and_run.sh --force --no-cache
```

## Testing

### Basic Tests
Run the basic test suite to verify environment setup and dependencies:
```powershell
python test_train_agent.py
```

This will run:
- Environment creation test
- Environment interaction test
- Model initialization test

### Deep Tests
To run additional tests that verify model functionality:
```powershell
python test_train_agent.py --deep-tests
```

This includes all basic tests plus:
- Model prediction test
- Model saving/loading test
- Training step test

## Training

Run the training script:
```powershell
python train_agent.py
```

Options:
- `--use-gpu`: Use GPU for training (not recommended for MLP policies)
- `--timesteps`: Number of timesteps to train for (default: 1000)

Example:
```powershell
python train_agent.py --timesteps 10000
```

## Visualization

### GUI Board Display
To watch the agent play Minesweeper in real-time:

1. Install the required visualization package:
```powershell
pip install pygame
```

2. Run the visualization script:
```powershell
python visualize_agent.py
```

Options:
- `--model-path`: Path to a trained model (default: uses a new model)
- `--board-size`: Size of the Minesweeper board (default: 5)
- `--num-mines`: Number of mines on the board (default: 4)
- `--speed`: Game speed in frames per second (default: 2)
- `--episodes`: Number of episodes to play (default: 1)

Example:
```powershell
python visualize_agent.py --board-size 8 --num-mines 10 --speed 1
```

The visualization window shows:
- The current state of the board
- Revealed numbers and mines
- Agent's actions in real-time
- Current score and episode number
- Game statistics

Controls:
- `Space`: Pause/Resume
- `R`: Reset current episode
- `N`: Next episode
- `Q`: Quit visualization

### TensorBoard Visualization
To view training metrics and performance:

1. Start TensorBoard:
```powershell
tensorboard --logdir=logs/
```

2. Open your browser and navigate to:
```
http://localhost:6006
```

Available visualizations:
- Training rewards
- Episode lengths
- Loss values
- Policy entropy
- Value function estimates

## TODO: Future Testing Improvements

### Environment Tests
- [ ] Test different board sizes (3x3, 5x5, 8x8, 16x16)
- [ ] Test different mine densities (10%, 20%, 30%)
- [ ] Test edge cases (0 mines, maximum mines)
- [ ] Test invalid actions handling
- [ ] Test game state transitions
- [ ] Test win/loss conditions

### RL Parameter Tests
- [ ] Test different reward structures
  - [ ] Time-based rewards
  - [ ] Mine proximity rewards
  - [ ] Safe cell discovery rewards
- [ ] Test different learning rates
- [ ] Test different batch sizes
- [ ] Test different network architectures
- [ ] Test different optimization algorithms

### Performance Tests
- [ ] Measure training time vs. performance
- [ ] Track win rate over time
- [ ] Measure average game length
- [ ] Track fastest win times
- [ ] Compare performance across different board sizes
- [ ] Benchmark against random play
- [ ] Benchmark against simple rule-based strategies

### Statistical Analysis
- [ ] Track success rate distribution
- [ ] Analyze common failure patterns
- [ ] Measure exploration vs. exploitation
- [ ] Track policy entropy
- [ ] Analyze action distribution
- [ ] Measure state value estimates

### Visualization Tests
- [ ] Test TensorBoard integration
- [ ] Test progress bar functionality
- [ ] Test environment rendering
- [ ] Test action visualization
- [ ] Test heatmap generation
- [ ] Add color themes for the GUI
- [ ] Add replay functionality
- [ ] Add action history display
- [ ] Add confidence visualization
- [ ] Add mine probability heatmap

### Integration Tests
- [ ] Test model saving/loading across sessions
- [ ] Test environment serialization
- [ ] Test multi-process training
- [ ] Test GPU/CPU switching
- [ ] Test memory management
- [ ] Test error handling and recovery

### Documentation
- [ ] Add docstrings to all functions
- [ ] Create API documentation
- [ ] Add usage examples
- [ ] Document best practices
- [ ] Add troubleshooting guide
- [ ] Create contribution guidelines 