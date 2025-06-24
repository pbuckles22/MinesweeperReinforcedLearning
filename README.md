# Minesweeper Reinforcement Learning

A Reinforcement Learning environment for Minesweeper using Stable Baselines3 (PPO) with curriculum learning and MLflow tracking.

## 🎯 Current Status

✅ **Environment**: Fully functional 4-channel Minesweeper RL environment  
✅ **Test Coverage**: 639 tests passing (100% success rate)  
✅ **Training System**: Complete RL training pipeline with modular approach  
✅ **Project Structure**: Clean, organized directory structure  
✅ **State Representation**: Enhanced 4-channel state (game state, safety hints, revealed cell count, game progress indicators)  
✅ **Action Masking**: Intelligent action masking for revealed cells  
✅ **Reward System**: Comprehensive reward system for RL training  
✅ **Experiment Tracking**: Full experiment tracking and metrics collection  
✅ **Integration Tests**: Comprehensive tests to prevent RL system issues  
✅ **Debug Tools**: Complete debugging toolkit for development and troubleshooting  
✅ **M1 GPU Support**: Optimized for Apple Silicon with Metal Performance Shaders (MPS)  
✅ **Cross-Platform**: Full compatibility across Mac, Windows, and Linux  

## 🚀 Recent Updates

### Project Structure Cleanup (2024-12-24)
- **Organized Directories**: Clean root directory with proper file organization
- **Results Management**: All experiment outputs saved to `experiments/`
- **Log Management**: All logs and reports saved to `logs/`
- **Script Organization**: All utility scripts moved to `scripts/`
- **Test Organization**: All test files properly organized in `tests/`
- **Cleanup Automation**: Enhanced cleanup script with 14-day retention

### Modular Training System (2024-12-24)
- **Modular Script**: `src/core/train_agent_modular.py` - Clean, flexible training
- **Simple Mode**: Easy-to-use training with conservative hyperparameters
- **Proven Performance**: 21% win rate achieved in 1000 timesteps
- **Clean Output**: Compact, readable progress display
- **Flexible Parameters**: Easy parameter overrides and customization

### Test Suite Completion (2024-12-24)
- **Achieved**: 639/639 tests passing (100% success rate)
- **Fixed**: All reward consistency and callback issues
- **Enhanced**: Modular training tests and performance comparisons
- **Cleaned**: Removed legacy curriculum and MLflow tests
- **Validated**: Complete end-to-end training pipeline verification

### M1 GPU Optimization & Cross-Platform Compatibility (2024-12-19)
- **M1 GPU Support**: Automatic MPS detection with 2-4x performance improvement
- **Performance Benchmarking**: Built-in matrix multiplication tests (~0.012s benchmark)
- **Cross-Platform Scripts**: Organized scripts for Mac/Windows/Linux
- **Platform-Specific Requirements**: Compatible NumPy versions for Python 3.10
- **Test Compatibility**: Dynamic state shape detection and platform-agnostic validation
- **Import Path Resolution**: Fixed module path issues across platforms

## 📁 Project Structure

```
MinesweeperReinforcedLearning/
├── src/                          # Core source code
│   └── core/
│       ├── minesweeper_env.py    # Main RL environment
│       ├── train_agent.py        # Legacy training script
│       ├── train_agent_modular.py # Modular training script (recommended)
│       └── constants.py          # Environment constants
├── tests/                        # Comprehensive test suite
│   ├── unit/                     # Unit tests
│   ├── functional/               # Functional tests
│   ├── integration/              # Integration tests
│   └── e2e/                      # End-to-end tests
├── scripts/                      # Utility and training scripts
│   ├── mac/                      # Mac-specific scripts
│   ├── windows/                  # Windows-specific scripts
│   ├── linux/                    # Linux-specific scripts
│   └── manage_training_stats.py  # Cleanup and stats management
├── experiments/                  # Experiment results and outputs
│   ├── modular_results_*.json    # Modular training results
│   ├── simple_results_*.json     # Simple training results
│   └── metrics.json              # Experiment metrics
├── logs/                         # Logs and reports
│   ├── benchmark_results/        # Performance benchmarks
│   ├── training_logs/            # Training logs
│   └── coverage_results_summary.json # Test coverage reports
├── models/                       # Trained model checkpoints
├── training_stats/               # Training statistics
├── docs/                         # Documentation
└── requirements.txt              # Dependencies
```

## 🚀 Quick Start

### 1. Installation
```bash
# Clone and setup
git clone <repository>
cd MinesweeperReinforcedLearning
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Quick Training (Recommended)
```bash
# Use the modular training script for best results
python -m src.core.train_agent_modular --board_size 4 4 --max_mines 2 --timesteps 1000
```

### 3. Platform-Specific Scripts
```bash
# Mac (with M1 optimization)
./scripts/mac/install_and_run.sh

# Windows
./scripts/windows/install_and_run.ps1

# Linux
./scripts/linux/install_and_run.sh
```

## 🏗️ Architecture

### Environment Features
- **4-Channel State**: Game state, safety hints, revealed cell count, and game progress indicators for enhanced learning
- **Cascade Revelation**: Automatic neighbor revelation for empty cells
- **Action Masking**: Intelligent masking of revealed cells
- **Modular Training**: Clean, flexible training system
- **Rectangular Boards**: Support for non-square board configurations
- **Invalid Action Handling**: Proper termination after consecutive invalid actions
- **M1 GPU Support**: Optimized for Apple Silicon with Metal Performance Shaders (MPS)
- **Cross-Platform**: Full compatibility across Mac, Windows, and Linux

### State Representation
- **Channel 0**: Game state (-1: unrevealed, 0-8: revealed numbers, -4: mine hit)
- **Channel 1**: Safety hints (adjacent mine counts for unrevealed cells)
- **Channel 2**: Revealed cell count (total revealed cells across board)
- **Channel 3**: Game progress indicators (safe bet flags for obvious safe cells)

### Reward System
- `REWARD_SAFE_REVEAL = 15`: Regular safe reveal (immediate positive feedback)
- `REWARD_WIN = 500`: Game win (massive reward to encourage winning)
- `REWARD_HIT_MINE = -20`: Mine hit penalty (immediate negative feedback)
- `REWARD_INVALID_ACTION = -25`: Invalid action penalty
- `REWARD_REPEATED_CLICK = -35`: Repeated click penalty

## 🧪 Testing

### Test Coverage
- **Total Tests**: 639 tests covering all components
- **Unit Tests**: Core mechanics, state management, rewards, RL components
- **Functional Tests**: End-to-end scenarios, curriculum progression
- **Integration Tests**: Cross-component behavior, performance, RL system
- **Script Tests**: Infrastructure and utility scripts

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/unit/ -v
python -m pytest tests/functional/ -v
python -m pytest tests/integration/ -v
```

## 🧹 Maintenance

### Cleanup and Organization
```bash
# Clean up old files (14-day retention)
python scripts/manage_training_stats.py --action cleanup

# View cleanup summary
python scripts/manage_training_stats.py --action summary

# List recent files
python scripts/manage_training_stats.py --action list --count 10
```

### File Locations
- **Experiment Results**: `experiments/` (modular_results_*.json, simple_results_*.json)
- **Training Logs**: `logs/` (benchmark_results/, training_logs/, etc.)
- **Model Checkpoints**: `models/`
- **Training Stats**: `training_stats/`
- **Test Reports**: `logs/` (coverage_results_summary.json, etc.)

## 🔧 Cross-Platform Compatibility

### Platform-Specific Scripts
- **Mac**: `scripts/mac/` with shell scripts and M1 optimization
- **Windows**: `scripts/windows/` with PowerShell scripts
- **Linux**: `scripts/linux/` with shell scripts

### Platform-Specific Requirements
- **Mac**: M1-optimized versions in `requirements.txt`
- **Windows**: Compatible NumPy versions for Python 3.10
- **Linux**: Standard requirements with platform detection

## 🚀 M1 GPU Performance

### M1 MacBook Optimization
- **Automatic Detection**: MPS (Metal Performance Shaders) detection and optimization
- **Performance**: 2-4x faster than CPU training on M1 MacBooks
- **Benchmark**: ~0.012s matrix multiplication (excellent performance)
- **Training Speed**: ~179 iterations/second (normal for early training)
- **Memory Efficiency**: Optimized batch sizes for M1 GPU memory

### M1 Setup Verification
```bash
# Test M1 GPU performance
python -c "
import torch
import time
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)
start = time.time()
for _ in range(10):
    z = torch.mm(x, y)
print(f'M1 GPU benchmark: {(time.time() - start)/10:.3f}s')
"
```