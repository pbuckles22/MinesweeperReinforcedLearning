# Minesweeper RL Project Context

## 🎯 **Project Overview**
This is a Reinforcement Learning environment for Minesweeper using Stable Baselines3 (PPO) with modular training approach and comprehensive testing. Optimized for M1 MacBook performance with GPU acceleration.

## 🏗️ **Key Design Decisions**

### **Modular Training System** ⚡ **CURRENT**
- **Modular Script**: `src/core/train_agent_modular.py` - Clean, flexible training
- **Simple Mode**: Easy-to-use training with conservative hyperparameters
- **Proven Performance**: 21% win rate achieved in 1000 timesteps
- **Clean Output**: Compact, readable progress display
- **Flexible Parameters**: Easy parameter overrides and customization

### **Simplified Reward System**
- **Immediate Rewards**: Every safe reveal gets +15, every mine hit gets -20, wins get +500
- **No Special First-Move Logic**: Removed confusing pre-cascade neutral rewards
- **Clear Learning Signals**: Agent gets immediate feedback for all actions
- **Purpose**: Provides consistent learning signals without artificial distinctions

### **Enhanced State Representation**
- **4-Channel State**: Game state + safety hints + revealed count + progress indicators
- **Channel 0**: Game state (revealed cells with numbers, unrevealed as -1, mine hits as -4)
- **Channel 1**: Safety hints (adjacent mine count for unrevealed cells)
- **Channel 2**: Revealed cell count (total revealed cells across board)
- **Channel 3**: Game progress indicators (safe bet flags for obvious safe cells)
- **Purpose**: Makes patterns more obvious to the agent for better learning

### **Smart Action Masking**
- **Basic Masking**: Prevents revealing already revealed cells
- **Smart Masking**: Avoids cells that are guaranteed to be mines based on revealed information
- **Pattern Recognition**: Uses revealed cell numbers to identify guaranteed mines
- **Purpose**: Prevents obviously bad moves and guides agent toward better decisions

### **Board Size Convention**
- **All board sizes use (height, width) format** throughout the codebase
- **Example**: `board_size=(4, 4)` means height=4, width=4
- **This matches numpy/Gym conventions**

### **Reward System**
```python
REWARD_SAFE_REVEAL = 15           # Every safe reveal (immediate)
REWARD_WIN = 500                  # Win reward (always given)
REWARD_HIT_MINE = -20             # Every mine hit (immediate)
REWARD_INVALID_ACTION = -25       # Invalid action penalty
REWARD_REPEATED_CLICK = -35       # Repeated click penalty
```

### **Environment Features**
- **4-channel state representation**: Game state + safety hints + revealed count + progress indicators
- **Smart action masking**: Prevents obviously bad moves
- **Modular training system**: Clean, flexible training approach
- **M1 GPU Support**: Optimized for Apple Silicon with Metal Performance Shaders (MPS)
- **Cross-platform compatibility**: Works on Mac, Windows, and Linux

## 🔧 **Key Files**
- `src/core/minesweeper_env.py` - Main environment (simplified rewards)
- `src/core/train_agent_modular.py` - Modular training script (recommended)
- `src/core/train_agent.py` - Legacy training script (complex, advanced features)
- `src/core/q_learning_agent.py` - Q-learning agent with experience replay
- `src/core/constants.py` - Reward constants and configuration
- `scripts/train_q_learning.py` - Q-learning curriculum training script
- `tests/` - Comprehensive test suite (639 tests)
- `scripts/` - Utility and platform-specific scripts

## 🚀 **Quick Start**
```bash
# Install dependencies
pip install -r requirements.txt

# Run modular training (recommended)
python -m src.core.train_agent_modular --board_size 4 4 --max_mines 2 --timesteps 1000

# Run tests
python -m pytest tests/ -v
```

## 📊 **Current Status** ⚡ **UPDATED**
- ✅ Environment fully functional with correct game logic
- ✅ Modular training pipeline complete and proven
- ✅ Test suite comprehensive (639 tests passing)
- ✅ Board size standardization complete
- ✅ **Simplified reward system implemented**
- ✅ **Cross-platform scripts organized**
- ✅ **M1 GPU support implemented**
- ✅ **Enhanced state representation (4 channels)**
- ✅ **Smart action masking implemented**
- ✅ **Project structure cleaned and organized**
- ✅ **Automatic cleanup with 14-day retention**
- ✅ **Modular training achieves 21% win rate**
- ✅ Variable mine and mixed mine curriculum scripts implemented and tested
- ✅ Catastrophic forgetting identified and mitigated
- ✅ Q-learning with experience replay implemented and tested
- ✅ Catastrophic forgetting problem solved
- ✅ Q-learning outperforms PPO in curriculum learning

## 🎯 **Critical Learning Insights**
- **Game Logic is Perfect**: Environment randomization and win conditions work correctly
- **Reward System Matters**: Immediate rewards (not sparse) are essential for learning
- **Modular Approach Works**: Simple, clean training achieves better results than complex curriculum
- **Training Complexity**: Even simple 4x4 boards are challenging for RL agents
- **Performance**: M1 Mac with GPU acceleration significantly faster for training
- **State Representation**: 4-channel state makes patterns more obvious to agent
- **Action Masking**: Smart masking prevents obviously bad moves
- **Project Organization**: Clean structure makes development and maintenance easier

## 🎯 **Next Priorities**
1. **Test Enhanced Features**: Run training with new 4-channel state and smart masking
2. **Visualization Tools**: Watch agent play in real-time with new state representation
3. **Hyperparameter Tuning**: Optimize for the enhanced environment
4. **Longer Training Runs**: Use M1 Mac for extended training with new features
5. **Win Rate Analysis**: Monitor if enhanced features improve win rates

## 🚀 **Training Examples**

### **Quick Training (Recommended)**
```bash
# Basic training (4x4 board, 2 mines)
python -m src.core.train_agent_modular --board_size 4 4 --max_mines 2 --timesteps 1000

# Custom board size
python -m src.core.train_agent_modular --board_size 6 6 --max_mines 4 --timesteps 2000

# Parameter overrides
python -m src.core.train_agent_modular --learning_rate 0.0002 --batch_size 64 --timesteps 1000

# GPU training (M1 Mac)
python -m src.core.train_agent_modular --device mps --timesteps 1000
```

### **Platform-Specific Scripts**
```bash
# Mac (with M1 optimization)
./scripts/mac/install_and_run.sh

# Windows
./scripts/windows/install_and_run.ps1

# Linux
./scripts/linux/install_and_run.sh
```

### **Q-Learning Training**
```bash
# Quick Q-learning test
python -m scripts.train_q_learning --quick

# Q-learning curriculum training
python -m scripts.train_q_learning --curriculum

# Q-learning with custom parameters
python -c "
from src.core.q_learning_agent import QLearningAgent
agent = QLearningAgent(board_size=(4,4), max_mines=2, learning_rate=0.1)
"
```

## 📁 **Project Structure**

```
MinesweeperReinforcedLearning/
├── src/                          # Core source code
│   └── core/
│       ├── minesweeper_env.py    # Main RL environment
│       ├── train_agent.py        # Legacy training script
│       ├── train_agent_modular.py # Modular training script (recommended)
│       └── constants.py          # Environment constants
├── tests/                        # Comprehensive test suite
├── scripts/                      # Utility and training scripts
├── experiments/                  # Experiment results and outputs
├── logs/                         # Logs and reports
├── models/                       # Trained model checkpoints
├── training_stats/               # Training statistics
└── docs/                         # Documentation
```

## 🧹 **Maintenance**

### **Cleanup and Organization**
```bash
# Clean up old files (14-day retention)
python scripts/manage_training_stats.py --action cleanup

# View cleanup summary
python scripts/manage_training_stats.py --action summary

# List recent files
python scripts/manage_training_stats.py --action list --count 10
```

### **File Locations**
- **Experiment Results**: `experiments/` (modular_results_*.json, simple_results_*.json)
- **Training Logs**: `logs/` (benchmark_results/, training_logs/, etc.)
- **Model Checkpoints**: `models/`
- **Training Stats**: `training_stats/`
- **Test Reports**: `logs/` (coverage_results_summary.json, etc.)

## 🧪 **Testing**

### **Test Coverage**
- **Total Tests**: 639 tests covering all components
- **Unit Tests**: Core mechanics, state management, rewards, RL components
- **Functional Tests**: End-to-end scenarios, curriculum progression
- **Integration Tests**: Cross-component behavior, performance, RL system
- **Script Tests**: Infrastructure and utility scripts

### **Running Tests**
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/unit/ -v
python -m pytest tests/functional/ -v
python -m pytest tests/integration/ -v
```

## 🚀 **M1 GPU Performance**

### **M1 MacBook Optimization**
- **Automatic Detection**: MPS (Metal Performance Shaders) detection and optimization
- **Performance**: 2-4x faster than CPU training on M1 MacBooks
- **Benchmark**: ~0.012s matrix multiplication (excellent performance)
- **Training Speed**: ~179 iterations/second (normal for early training)
- **Memory Efficiency**: Optimized batch sizes for M1 GPU memory

### **M1 Setup Verification**
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

## 🔧 **Cross-Platform Compatibility**

### **Platform-Specific Scripts**
- **Mac**: `scripts/mac/` with shell scripts and M1 optimization
- **Windows**: `scripts/windows/` with PowerShell scripts
- **Linux**: `scripts/linux/` with shell scripts

### **Platform-Specific Requirements**
- **Mac**: M1-optimized versions in `requirements.txt`
- **Windows**: Compatible NumPy versions for Python 3.10
- **Linux**: Standard requirements with platform detection

## 📈 **Performance Comparison**

| Script | Win Rate | Code Complexity | Parameter Flexibility | Recommended |
|--------|----------|-----------------|---------------------|-------------|
| **Modular** | **21%+** | **Low** | **High** | **✅ Yes** |
| Legacy (Simple Mode) | 15-22% | Medium | Medium | ⚠️ Backup |
| Legacy (Full) | 0-5% | High | High | ❌ No |

## 🎉 **Success Story**

The modular approach solved the complexity problem:
- **Before**: 2,300 lines, 0% win rates, complex debugging
- **After**: 300 lines, 21%+ win rates, simple and flexible

This demonstrates the power of **simplicity over complexity** in reinforcement learning! 🚀

### **Curriculum Learning Approaches**
- **Variable Mine Training**: Trains the agent on a range of mine counts (e.g., 1-5) to encourage generalization and prevent overfitting to a single configuration.
- **Mixed Mine Training with Experience Replay**: Trains on multiple mine counts simultaneously and leverages experience replay to prevent catastrophic forgetting, ensuring the agent retains skills across all difficulties.
- **Catastrophic Forgetting Mitigation**: Identified and addressed the issue where agents forget easier tasks when exposed to harder ones by using mixed training and replay.

### **Experience Replay**
- **Catastrophic Forgetting**: Training only on the hardest scenario can cause the agent to lose performance on easier ones. Mixed training and experience replay are effective solutions.
- **Experience Replay**: Incorporating replay buffers or mixed environments helps maintain performance across all curriculum stages.

### **Q-Learning with Experience Replay** ⚡ **BREAKTHROUGH**
- **Q-Learning Agent**: `src/core/q_learning_agent.py` - Tabular Q-learning with experience replay
- **Experience Replay**: Prevents catastrophic forgetting across curriculum stages
- **Proven Performance**: 15% → 20% → 25% win rate progression (vs PPO regression)
- **Knowledge Transfer**: Successfully maintains skills across difficulty levels
- **Curriculum Success**: Solved the catastrophic forgetting problem identified with PPO

### **Algorithm Comparison Results**
- **PPO Curriculum**: 56% → 14% → 12% (catastrophic forgetting)
- **Q-Learning Curriculum**: 15% → 20% → 25% (progressive improvement)
- **Experience Replay**: Key to preventing skill loss across stages
- **Q-Learning Advantage**: Better suited for discrete action spaces and curriculum learning

### **Simplified Reward System**

---

**Last Updated**: 2024-12-24  
**Status**: Production ready with modular training system and organized project structure 