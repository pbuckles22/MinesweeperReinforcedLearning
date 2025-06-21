# Minesweeper RL Project Context

## 🎯 **Project Overview**
This is a Reinforcement Learning environment for Minesweeper using Stable Baselines3 (PPO) with curriculum learning, MLflow tracking, and comprehensive testing. Optimized for M1 MacBook performance with GPU acceleration.

## 🏗️ **Key Design Decisions**

### **Simplified Reward System** ⚡ **UPDATED**
- **Immediate Rewards**: Every safe reveal gets +15, every mine hit gets -20, wins get +500
- **No Special First-Move Logic**: Removed confusing pre-cascade neutral rewards
- **Clear Learning Signals**: Agent gets immediate feedback for all actions
- **Purpose**: Provides consistent learning signals without artificial distinctions

### **Enhanced State Representation** ⚡ **NEW**
- **4-Channel State**: Game state + safety hints + revealed count + progress indicators
- **Channel 0**: Game state (revealed cells with numbers, unrevealed as -1, mine hits as -4)
- **Channel 1**: Safety hints (adjacent mine count for unrevealed cells)
- **Channel 2**: Revealed cell count (total revealed cells across board)
- **Channel 3**: Game progress indicators (safe bet flags for obvious safe cells)
- **Purpose**: Makes patterns more obvious to the agent for better learning

### **Smart Action Masking** ⚡ **NEW**
- **Basic Masking**: Prevents revealing already revealed cells
- **Smart Masking**: Avoids cells that are guaranteed to be mines based on revealed information
- **Pattern Recognition**: Uses revealed cell numbers to identify guaranteed mines
- **Purpose**: Prevents obviously bad moves and guides agent toward better decisions

### **Curriculum Learning** ⚡ **UPDATED**
- **8 Stages**: Tiny (2x2) → Beginner (4x4) → Intermediate (6x6) → Easy (9x9) → Normal (16x16) → Hard (16x30) → Expert (18x24) → Chaotic (20x35)
- **Realistic Thresholds**: 25%, 15%, 12%, 10%, 8%, 5%, 3%, 2% win rates
- **Adaptive Training**: More time for simpler stages (2x, 1.5x, 1.2x multipliers)
- **Purpose**: Progressive difficulty with appropriate learning time allocation

### **Board Size Convention**
- **All board sizes use (height, width) format** throughout the codebase
- **Example**: `initial_board_size=(4, 3)` means height=4, width=3
- **This matches numpy/Gym conventions**

### **Reward System** ⚡ **UPDATED**
```python
REWARD_SAFE_REVEAL = 15           # Every safe reveal (immediate)
REWARD_WIN = 500                  # Win reward (always given)
REWARD_HIT_MINE = -20             # Every mine hit (immediate)
REWARD_INVALID_ACTION = -10       # Invalid action penalty
```

### **Environment Features**
- **4-channel state representation**: Game state + safety hints + revealed count + progress indicators
- **Smart action masking**: Prevents obviously bad moves
- **Curriculum learning**: 8 stages with adaptive training times
- **MLflow integration**: Experiment tracking and model logging
- **M1 GPU Support**: Optimized for Apple Silicon with Metal Performance Shaders (MPS)

## 🔧 **Key Files**
- `src/core/minesweeper_env.py` - Main environment (simplified rewards)
- `src/core/train_agent.py` - Training script with curriculum learning
- `src/core/constants.py` - Reward constants and configuration
- `tests/` - Comprehensive test suite (516 tests)
- `scripts/mac/` - Mac-specific training scripts

## 🚀 **Quick Start**
```bash
# Install dependencies
pip install -r requirements.txt

# Run training (use Mac for GPU acceleration)
python src/core/train_agent.py --total_timesteps 10000

# Start MLflow UI
mlflow ui

# Run tests
pytest
```

## 📊 **Current Status** ⚡ **UPDATED**
- ✅ Environment fully functional with correct game logic
- ✅ Training pipeline complete with curriculum learning
- ✅ MLflow integration working
- ✅ Test suite comprehensive (516 tests)
- ✅ Board size standardization complete
- ✅ **Simplified reward system implemented**
- ✅ **Realistic curriculum thresholds (25%, 15%, 12%, 10%, 8%, 5%, 3%, 2%)**
- ✅ **Cross-platform scripts organized**
- ✅ **M1 GPU support implemented**
- ✅ **Enhanced state representation (4 channels)**
- ✅ **Smart action masking implemented**
- ✅ **Tiny stage (2x2) added for simplest learning**
- ✅ **Adaptive training times implemented**

## 🎯 **Critical Learning Insights** ⚡ **UPDATED**
- **Game Logic is Perfect**: Environment randomization and win conditions work correctly
- **Reward System Matters**: Immediate rewards (not sparse) are essential for learning
- **Training Complexity**: Even simple 4x4 boards are challenging for RL agents
- **Performance**: M1 Mac with GPU acceleration significantly faster for training
- **Agent Learning**: Getting positive rewards (8-15 range) but not winning complete games yet
- **State Representation**: 4-channel state makes patterns more obvious to agent
- **Action Masking**: Smart masking prevents obviously bad moves

## 🎯 **Next Priorities** ⚡ **UPDATED**
1. **Test Enhanced Features**: Run training with new 4-channel state and smart masking
2. **Visualization Tools**: Watch agent play in real-time with new state representation
3. **Hyperparameter Tuning**: Optimize for the enhanced environment
4. **Longer Training Runs**: Use M1 Mac for extended training with new features
5. **Win Rate Analysis**: Monitor if enhanced features improve win rates

## 💡 **Important Notes**
- **Use M1 Mac for intensive training** (GPU acceleration)
- **Environment randomizes properly** between episodes
- **Agent is learning** (positive rewards) but not winning yet
- **Win rates are expectedly low** - even humans struggle with Minesweeper
- **Focus on learning improvements**, not game logic bugs
- **Cross-platform scripts** available in `scripts/windows/`, `scripts/linux/`, `scripts/mac/`

## 🔍 **Recent Findings** ⚡ **NEW**
- **Reward System Audit**: Simplified from confusing first-move logic to immediate rewards
- **Curriculum Improvement**: Lowered win rate thresholds to realistic levels
- **Game Logic Verification**: Confirmed environment works correctly
- **Performance Optimization**: M1 GPU provides significant speedup
- **Learning Progress**: Agent achieving positive rewards but not complete wins

---
**Last Updated**: 2024-12-19  
**Status**: Production ready with optimized training pipeline and M1 GPU support 