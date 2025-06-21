# Minesweeper RL Project Context

## 🎯 **Project Overview**
This is a Reinforcement Learning environment for Minesweeper using Stable Baselines3 (PPO) with curriculum learning, MLflow tracking, and comprehensive testing.

## 🏗️ **Key Design Decisions**

### **Pre-Cascade Safety System**
- **Mines are placed BEFORE the first move** (not after)
- **Pre-cascade period**: No rewards (neutral = 0) until a cascade happens
- **Purpose**: Prevents punishing the RL agent for random guessing before it has information
- **If mine hit during pre-cascade**: Game ends, neutral reward (0), start fresh
- **If cascade happens**: Normal reward system begins

### **Board Size Convention**
- **All board sizes use (height, width) format** throughout the codebase
- **Example**: `initial_board_size=(4, 3)` means height=4, width=3
- **This matches numpy/Gym conventions**

### **Reward System**
```python
REWARD_FIRST_CASCADE_SAFE = 0     # Pre-cascade safe reveal (neutral)
REWARD_FIRST_CASCADE_HIT_MINE = 0  # Pre-cascade hit mine (neutral)
REWARD_SAFE_REVEAL = 15           # Post-cascade safe reveal
REWARD_WIN = 500                  # Win reward
REWARD_HIT_MINE = -20             # Post-cascade mine hit penalty
```

### **Environment Features**
- **2-channel state representation**: Game state + safety hints
- **Action masking**: Prevents revealing already-revealed cells
- **Curriculum learning**: 7 stages from 4x4 to 20x35 boards
- **MLflow integration**: Experiment tracking and model logging
- **Early learning mode**: Optional safety features for beginners

## 🔧 **Key Files**
- `src/core/minesweeper_env.py` - Main environment
- `src/core/train_agent.py` - Training script with curriculum
- `src/core/constants.py` - Reward constants and configuration
- `tests/` - Comprehensive test suite (500+ tests)

## 🚀 **Quick Start**
```bash
# Install dependencies
pip install -r requirements.txt

# Run training
python src/core/train_agent.py --total_timesteps 10000

# Start MLflow UI
mlflow ui

# Run tests
pytest
```

## 📊 **Current Status**
- ✅ Environment fully functional
- ✅ Training pipeline complete
- ✅ MLflow integration working
- ✅ Test suite comprehensive (500+ tests)
- ✅ Board size standardization complete
- ✅ Pre-cascade neutral rewards implemented

## 🎯 **Next Priorities**
1. Improve RL agent win rates
2. Optimize training hyperparameters
3. Add visualization tools
4. Enhance curriculum learning

## 💡 **Important Notes**
- The environment intentionally allows first moves to hit mines (realistic)
- Pre-cascade period gives neutral rewards to prevent punishing random guessing
- All board dimensions use (height, width) format
- MLflow experiments are stored in `mlruns/` directory
- Tests are organized by unit/integration/functional categories

---
**Last Updated**: 2024-12-19  
**Status**: Production ready with complete training pipeline 