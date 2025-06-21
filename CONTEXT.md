# Minesweeper RL Project Context

## 🎯 **Project Overview**
This is a Reinforcement Learning environment for Minesweeper using Stable Baselines3 (PPO) with curriculum learning, MLflow tracking, and comprehensive testing. Optimized for M1 MacBook performance with GPU acceleration.

## 🚨 **CRITICAL DEVELOPMENT RULE** ⚡ **NEW**
**When making ANY changes to the environment, reward system, or core functionality, IMMEDIATELY update the corresponding tests to match the new behavior. This prevents cascading test failures and ensures the test suite remains a reliable validation tool.**

**Examples of changes that require test updates:**
- Environment state representation changes (e.g., 2-channel → 4-channel)
- Reward system modifications (e.g., neutral → immediate rewards)
- Action space or masking changes
- Board size conventions or defaults
- Training configuration updates

**This rule was learned from the painful experience of updating from 2-channel to 4-channel state representation and neutral to immediate rewards, which caused widespread test failures across the entire test suite.**

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
- ✅ **Cross-platform test compatibility (Mac/Windows/Linux)**

## 🔧 **Cross-Platform Test Compatibility** ⚡ **NEW**

### **Test Suite Improvements**
The test suite has been enhanced to work seamlessly across all platforms (Mac, Windows, Linux) with the following improvements:

#### **Script Testing Flexibility**
- **Platform Detection**: Tests automatically detect the operating system
- **PowerShell Handling**: Tests check for PowerShell availability before using it
- **Content Validation**: When platform-specific tools aren't available, tests fall back to content-based validation
- **Permission Handling**: Different permission requirements per platform (executable vs readable)

#### **Cross-Platform Script Validation**
- **Output Handling**: Accepts various output methods (`echo`, `write-host`, `python`, `source`)
- **Error Handling**: Flexible validation for both simple and complex scripts
- **Environment Checks**: Validates both training and visualization scripts
- **Syntax Validation**: Platform-appropriate syntax checking

#### **Test Files Updated**
- `tests/scripts/test_run_script.py` - Cross-platform script validation
- `tests/unit/infrastructure/test_infra_run_scripts_unit.py` - Infrastructure test compatibility
- `tests/unit/infrastructure/test_infra_scripts_unit.py` - Script infrastructure compatibility

#### **Benefits for Development**
- **Consistent Testing**: Same test suite runs on all platforms
- **No Platform-Specific Failures**: Tests adapt to platform capabilities
- **Future-Proof**: New platforms will work without test modifications
- **Development Workflow**: Developers can switch between platforms seamlessly

### **Platform-Specific Considerations**

#### **Mac (macOS)**
- Uses shell scripts in `scripts/mac/`
- M1 GPU acceleration with Metal Performance Shaders (MPS)
- PowerShell not available (tests use content validation)

#### **Windows**
- Uses PowerShell scripts in `scripts/windows/`
- PowerShell syntax validation available
- Different permission model for scripts

#### **Linux**
- Uses shell scripts in `scripts/linux/`
- Similar to Mac but with different system paths
- PowerShell not available (tests use content validation)

### **Running Tests Across Platforms**
```bash
# All platforms use the same command
python -m pytest tests/ -v

# Expected result: 506 tests passed, 0 failed
# All tests work regardless of platform
```

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

## 🚀 **Next Training Steps** ⚡ **NEW**

### **Immediate Next Steps (Recommended Order)**

#### **1. Quick Training Test (5-10 minutes)**
```bash
# On Mac (recommended for GPU acceleration)
./scripts/mac/quick_test.sh

# On Windows
.\scripts\windows\quick_test.ps1

# On Linux
./scripts/linux/quick_test.sh
```
**Purpose**: Verify the new 4-channel state and immediate rewards work correctly
**Expected**: Agent should achieve positive rewards (8-15 range) and show learning progress

#### **2. Medium Training Test (15-30 minutes)**
```bash
# On Mac (recommended)
./scripts/mac/medium_test.sh

# On Windows
.\scripts\windows\medium_test.ps1

# On Linux
./scripts/linux/medium_test.sh
```
**Purpose**: Test curriculum learning progression through multiple stages
**Expected**: Agent should progress through Tiny (2x2) → Beginner (4x4) → Intermediate (6x6) stages

#### **3. Full Training Run (1-2 hours)**
```bash
# On Mac (recommended for M1 GPU acceleration)
./scripts/mac/full_training.sh

# On Windows
.\scripts\windows\full_training.ps1

# On Linux
./scripts/linux/full_training.sh
```
**Purpose**: Complete curriculum learning through all 8 stages
**Expected**: Agent should progress through all stages with realistic win rate thresholds

### **Training Command Reference**

#### **Quick Test (10k timesteps, ~5-10 minutes)**
```bash
python src/core/train_agent.py \
    --total_timesteps 10000 \
    --eval_freq 2000 \
    --n_eval_episodes 20 \
    --verbose 1
```

#### **Medium Test (50k timesteps, ~15-30 minutes)**
```bash
python src/core/train_agent.py \
    --total_timesteps 50000 \
    --eval_freq 5000 \
    --n_eval_episodes 50 \
    --verbose 1
```

#### **Full Training (1M timesteps, ~1-2 hours)**
```bash
python src/core/train_agent.py \
    --total_timesteps 1000000 \
    --eval_freq 10000 \
    --n_eval_episodes 100 \
    --verbose 1
```

### **Curriculum Learning Stages**

| Stage | Board Size | Mines | Win Rate Target | Expected Time |
|-------|------------|-------|-----------------|---------------|
| **Tiny** | 2x2 | 1 | 25% | 2-5 minutes |
| **Beginner** | 4x4 | 2 | 15% | 5-10 minutes |
| **Intermediate** | 6x6 | 4 | 12% | 10-15 minutes |
| **Easy** | 9x9 | 10 | 10% | 15-25 minutes |
| **Normal** | 16x16 | 40 | 8% | 25-40 minutes |
| **Hard** | 16x30 | 99 | 5% | 40-60 minutes |
| **Expert** | 18x24 | 115 | 3% | 60-90 minutes |
| **Chaotic** | 20x35 | 130 | 2% | 90-120 minutes |

### **Monitoring Training Progress**

#### **Real-time Monitoring**
```bash
# Start MLflow UI to monitor training
mlflow ui

# Open in browser: http://localhost:5000
```

#### **Key Metrics to Watch**
- **Win Rate**: Should increase as agent learns each stage
- **Average Reward**: Should be positive (8-15 range) for safe reveals
- **Stage Progression**: Agent should advance through curriculum stages
- **Episode Length**: Should increase as agent learns to avoid mines

#### **Success Indicators**
- ✅ Agent achieves positive rewards consistently
- ✅ Win rates meet stage thresholds (25%, 15%, 12%, etc.)
- ✅ Smooth progression through curriculum stages
- ✅ No training crashes or hanging issues

### **Troubleshooting Training Issues**

#### **If Training Hangs**
```bash
# Use the debug training script
.\scripts\debug_training.ps1
```

#### **If Win Rates Are Too Low**
- Check if agent is getting positive rewards for safe reveals
- Verify curriculum thresholds are realistic (they are!)
- Consider running longer on simpler stages

#### **If Agent Isn't Learning**
- Verify 4-channel state is working correctly
- Check that immediate rewards are being given
- Ensure action masking is preventing obviously bad moves

### **Advanced Training Options**

#### **Custom Hyperparameters**
```bash
python src/core/train_agent.py \
    --total_timesteps 1000000 \
    --learning_rate 0.0001 \
    --batch_size 128 \
    --n_steps 2048 \
    --verbose 1
```

#### **Device-Specific Optimizations**
- **M1 Mac**: Automatically uses MPS with optimized batch sizes
- **NVIDIA GPU**: Automatically uses CUDA with larger batches
- **CPU**: Automatically optimizes for CPU training

#### **Experiment Tracking**
```bash
# All training runs are automatically logged to MLflow
# View experiments at: http://localhost:5000
```

### **Post-Training Analysis**

#### **Model Evaluation**
```bash
# Evaluate trained model
python src/core/train_agent.py --evaluate_only --model_path best_model/
```

#### **Agent Visualization**
```bash
# Watch trained agent play
python src/visualization/visualize_agent.py --model-path best_model/
```

#### **Performance Analysis**
- Review MLflow metrics and charts
- Analyze win rates across different stages
- Compare with previous training runs

### **Next Research Directions**

#### **Short Term (Next 1-2 weeks)**
1. **Hyperparameter Optimization**: Tune learning rates, batch sizes, etc.
2. **Architecture Improvements**: Try different neural network architectures
3. **Reward Function Tuning**: Experiment with different reward values

#### **Medium Term (Next 1-2 months)**
1. **Advanced Curriculum**: Dynamic difficulty adjustment
2. **Multi-Agent Training**: Competitive scenarios
3. **Transfer Learning**: Pre-trained model utilization

#### **Long Term (Next 3-6 months)**
1. **Novel Architectures**: Transformer-based models
2. **Hierarchical Learning**: Multi-level decision making
3. **Meta-Learning**: Learning to learn new board configurations

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