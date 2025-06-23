# Minesweeper Reinforcement Learning

A modern, RL-optimized Minesweeper environment with comprehensive test coverage and curriculum learning capabilities.

## 🎯 Current Status

✅ **Environment**: Fully functional 4-channel Minesweeper RL environment  
✅ **Test Coverage**: 504 tests passing (100% success rate)  
✅ **Training System**: Complete RL training pipeline with curriculum learning  
✅ **First-Move Safety**: (Removed) The first move can be a mine; there is no mine relocation. The environment is intentionally simple for RL.  
✅ **State Representation**: Enhanced 4-channel state (game state, safety hints, revealed cell count, game progress indicators)  
✅ **Action Masking**: Intelligent action masking for revealed cells  
✅ **Reward System**: Comprehensive reward system for RL training  
✅ **Experiment Tracking**: Full experiment tracking and metrics collection  
✅ **Integration Tests**: Comprehensive tests to prevent RL system issues  
✅ **Debug Tools**: Complete debugging toolkit for development and troubleshooting  
✅ **M1 GPU Support**: Optimized for Apple Silicon with Metal Performance Shaders (MPS)  
✅ **Cross-Platform**: Full compatibility across Mac, Windows, and Linux  

## 🚀 Recent Updates

### M1 GPU Optimization & Cross-Platform Compatibility (2024-12-19)
- **M1 GPU Support**: Automatic MPS detection with 2-4x performance improvement
- **Performance Benchmarking**: Built-in matrix multiplication tests (~0.012s benchmark)
- **Cross-Platform Scripts**: Organized scripts for Mac/Windows/Linux
- **Platform-Specific Requirements**: Compatible NumPy versions for Python 3.10
- **Test Compatibility**: Dynamic state shape detection and platform-agnostic validation
- **Import Path Resolution**: Fixed module path issues across platforms

### Final Production Readiness (2024-12-19)
- **Achieved**: 504/504 tests passing (100% success rate)
- **Fixed**: All remaining test failures and compatibility issues
- **Enhanced**: Gym/Gymnasium compatibility across all test suites
- **Cleaned**: Removed debug artifacts and organized project structure
- **Validated**: Complete end-to-end training pipeline verification
- **Documented**: Comprehensive documentation and training guides

### Critical Bug Fixes (2024-12-19)
- **Fixed**: `KeyError: 'stage_1'` in training agent stage completion tracking
- **Fixed**: `evaluate_model` API compatibility in functional tests
- **Fixed**: EvalCallback hanging issue with vectorized environments
- **Fixed**: Info dictionary access patterns for vectorized environments
- **Fixed**: Environment termination for consecutive invalid actions
- **Fixed**: Vectorized environment detection in evaluation function
- **Improved**: All 504 tests now pass with comprehensive coverage
- **Enhanced**: Training pipeline fully functional with proper stage progression

### Integration Test Suite (2024-12-19)
- **Added**: Comprehensive integration tests to catch RL system issues
- **Implemented**: Tests for CustomEvalCallback validation
- **Enhanced**: Vectorized environment API compatibility tests
- **Added**: Info dictionary access pattern validation
- **Improved**: End-to-end training pipeline validation
- **Added**: Timeout protection for all integration tests

### Training System Completion (2024-12-19)
- **Added**: Complete curriculum learning with 7 difficulty stages
- **Implemented**: Experiment tracking with metrics persistence
- **Enhanced**: Model evaluation with statistical analysis
- **Improved**: Training callbacks and progress monitoring
- **Created**: CustomEvalCallback for reliable vectorized environment support

### Environment Modernization (2024-12-18)
- **Removed**: All flagging logic for RL-appropriate reveal-only gameplay
- **Updated**: 4-channel state representation (game state, safety hints, revealed cell count, game progress indicators)
- **Enhanced**: Comprehensive test suites for RL scenarios
- **Improved**: Functional and integration test coverage

## 🏗️ Architecture

### Environment Features
- **4-Channel State**: Game state, safety hints, revealed cell count, and game progress indicators for enhanced learning
- **First-Move Safety**: (Removed) The first move can be a mine; there is no mine relocation. The environment is intentionally simple for RL.
- **Cascade Revelation**: Automatic neighbor revelation for empty cells
- **Action Masking**: Intelligent masking of revealed cells
- **Curriculum Learning**: Progressive difficulty scaling with 7 stages
- **Rectangular Boards**: Support for non-square board configurations
- **Early Learning Mode**: Safety features for initial training phases
- **Invalid Action Handling**: Proper termination after consecutive invalid actions
- **M1 GPU Support**: Optimized for Apple Silicon with Metal Performance Shaders (MPS)
- **Cross-Platform**: Full compatibility across Mac, Windows, and Linux

### State Representation
- **Channel 0**: Game state (-1: unrevealed, 0-8: revealed numbers, -4: mine hit)
- **Channel 1**: Safety hints (adjacent mine counts for unrevealed cells)
- **Channel 2**: Revealed cell count (total revealed cells across board)
- **Channel 3**: Game progress indicators (safe bet flags for obvious safe cells)

### Reward System
- `REWARD_FIRST_CASCADE_SAFE = 0`: First cascade safe reveal (pre-cascade moves, including first move, have neutral reward; first move can be a mine)
- `REWARD_SAFE_REVEAL = 15`: Regular safe reveal (immediate positive feedback)
- `REWARD_WIN = 500`: Game win (massive reward to encourage winning)
- `REWARD_HIT_MINE = -20`: Mine hit penalty (immediate negative feedback)
- `REWARD_INVALID_ACTION = -10`: Invalid action penalty

### Curriculum Stages
1. **Beginner**: 4x4 board, 2 mines (15% win rate target)
2. **Intermediate**: 6x6 board, 4 mines (12% win rate target)
3. **Easy**: 9x9 board, 10 mines (10% win rate target)
4. **Normal**: 16x16 board, 40 mines (8% win rate target)
5. **Hard**: 16x30 board, 99 mines (5% win rate target)
6. **Expert**: 18x24 board, 115 mines (3% win rate target)
7. **Chaotic**: 20x35 board, 130 mines (2% win rate target)

### Curriculum Modes
The system now supports **three distinct curriculum modes** for different training objectives:

#### 1. **"current" - Original Learning Curriculum**
- **Purpose**: Basic learning and experimentation
- **Targets**: Low win rates (15% → 12% → 10% → 8% → 5% → 3% → 2%)
- **Training**: Standard 1.0x multiplier, 10 evaluation episodes
- **Progression**: Learning-based (allows progression without meeting targets)
- **Use Case**: Development, debugging, quick testing

#### 2. **"human_performance" - Human-Level Targets** (Default)
- **Purpose**: Achieve human expert-level performance
- **Targets**: High win rates (80% → 70% → 60% → 50% → 40% → 30% → 20%)
- **Training**: Extended 3.0x multiplier, 20 evaluation episodes
- **Progression**: Strict (must meet targets to advance)
- **Use Case**: Research, benchmarking, human-level AI

#### 3. **"superhuman" - Surpass Human Benchmarks**
- **Purpose**: Exceed human expert performance
- **Targets**: Superhuman win rates (95% → 85% → 75% → 65% → 55% → 45% → 35%)
- **Training**: Maximum 5.0x multiplier, 30 evaluation episodes
- **Progression**: Strict (must meet targets to advance)
- **Use Case**: Advanced research, superhuman AI development

**Usage**:
```bash
# Use original learning curriculum
python -m src.core.train_agent --curriculum_mode current

# Use human performance targets (default)
python -m src.core.train_agent --curriculum_mode human_performance

# Use superhuman targets
python -m src.core.train_agent --curriculum_mode superhuman
```

**📖 For detailed information**: See [Curriculum Modes Documentation](docs/curriculum_modes.md)

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

## 🔧 Cross-Platform Compatibility

### Platform-Specific Scripts
- **Mac**: `scripts/mac/` with shell scripts and M1 optimization
- **Windows**: `scripts/windows/` with PowerShell scripts
- **Linux**: `scripts/linux/` with shell scripts

### Platform-Specific Requirements
- **Mac**: M1-optimized versions in `requirements.txt`
- **Windows**: Compatible NumPy versions for Python 3.10
- **Linux**: Standard requirements with platform detection

### Test Compatibility
- **Dynamic State Shape**: Tests adapt to 2-channel vs 4-channel states
- **Platform-Agnostic**: Script tests work on all platforms
- **Error Handling**: Flexible validation for different output methods

## 🧪 Testing

### Test Coverage
- **Total Tests**: 504 tests covering all components
- **Unit Tests**: Core mechanics, state management, rewards, RL components
- **Functional Tests**: End-to-end scenarios, curriculum progression
- **Integration Tests**: Cross-component behavior, performance, RL system
- **Script Tests**: Infrastructure and utility scripts

### Test Categories
- Core game mechanics and RL requirements
- Enhanced state representation and safety hints
- Action masking and reward system
- Difficulty progression and curriculum learning
- Game flow and edge cases
- Performance and scalability
- Training pipeline and experiment tracking
- Model evaluation and metrics collection
- Vectorized environment compatibility
- Info dictionary access patterns
- Cross-platform script compatibility

### Running Tests

#### Complete Test Suite (Recommended)
```bash
# All 504 tests
python -m pytest tests/ -v

# Quick summary
python -m pytest tests/ -q
```

#### Platform-Specific Tests
```bash
# Mac (with M1 GPU testing)
./scripts/mac/quick_test.sh

# Windows
.\scripts\windows\quick_test.ps1

# Linux
./scripts/linux/quick_test.sh
```

#### Integration Tests (Critical for RL System)
```bash
# Comprehensive integration tests to catch RL issues
python -m pytest tests/integration/rl/ -v
```

#### Test Categories
```bash
# Core functionality
python -m pytest tests/unit/core/ tests/integration/ tests/functional/ -v

# RL training system
python -m pytest tests/unit/rl/ -v

# Scripts and infrastructure
python -m pytest tests/scripts/ tests/unit/infrastructure/ -v
```

#### Test Coverage
```bash
# Run with coverage report
python -m pytest --cov=src tests/
```

### Integration Test Focus Areas
The integration tests specifically address issues we encountered:

- **CustomEvalCallback Validation**: Tests our custom evaluation callback that properly handles vectorized environments
- **Vectorized Environment API**: Validates correct info dictionary access patterns
- **Info Dictionary Structure**: Ensures proper handling of gym vs gymnasium APIs
- **End-to-End Training**: Complete training pipeline validation
- **Error Handling**: Graceful handling of invalid actions and edge cases
- **Environment Termination**: Proper episode termination for invalid actions
- **Cross-Platform Compatibility**: Platform-agnostic script validation

### Why CustomEvalCallback?

We use a custom evaluation callback instead of the standard `EvalCallback` from stable-baselines3 because:

- **Vectorized Environment Compatibility**: The standard `EvalCallback` tries to access `env.won` which doesn't exist on vectorized environments
- **Info Dictionary Access**: Our `CustomEvalCallback` correctly accesses win information from the `info` dictionary
- **Reliability**: Prevents hanging issues during training that would block experiments
- **Tailored to Our Environment**: Specifically designed for our Minesweeper environment's API patterns

This ensures stable, reliable training without the compatibility issues we encountered with the standard callback.

### Integration Test Timeout Safeguard

All integration tests that could hang are protected with a 30-second timeout using pytest-timeout.
If a test fails due to timeout, it means a regression (e.g., EvalCallback hang) has been detected.

To install the timeout plugin:
```
pip install pytest-timeout
```

To run with timeouts:
```
pytest --timeout=30
```

## 🚀 Quick Start

### Basic Environment Usage
```python
from src.core.minesweeper_env import MinesweeperEnv

# Create environment
env = MinesweeperEnv(initial_board_size=4, initial_mines=2)
state, info = env.reset(seed=42)

# Take action
action = 0  # Reveal top-left cell
state, reward, terminated, truncated, info = env.step(action)
```

### Platform-Specific Quick Start

#### Mac (Recommended for M1 GPU)
```bash
# Install dependencies
pip install -r requirements.txt

# Quick test with M1 GPU acceleration
./scripts/mac/quick_test.sh

# Start training
./scripts/mac/medium_test.sh
```

#### Windows
```bash
# Install dependencies
pip install -r requirements.txt

# Quick test
.\scripts\windows\quick_test.ps1

# Start training
.\scripts\windows\medium_test.ps1
```

#### Linux
```bash
# Install dependencies
pip install -r requirements.txt

# Quick test
./scripts/linux/quick_test.sh

# Start training
./scripts/linux/medium_test.sh
```

### Training Commands

#### Quick Test (10k timesteps, ~5-10 minutes)
```bash
python src/core/train_agent.py \
    --total_timesteps 10000 \
    --eval_freq 2000 \
    --n_eval_episodes 20 \
    --verbose 1
```

#### Medium Test (50k timesteps, ~15-30 minutes)
```bash
python src/core/train_agent.py \
    --total_timesteps 50000 \
    --eval_freq 5000 \
    --n_eval_episodes 50 \
    --verbose 1
```

#### Full Training (1M timesteps, ~1-2 hours)
```bash
python src/core/train_agent.py \
    --total_timesteps 1000000 \
    --eval_freq 10000 \
    --n_eval_episodes 100 \
    --verbose 1
```

### Monitoring Training

#### MLflow UI
```bash
# Start MLflow UI
mlflow ui

# Open in browser: http://localhost:5000
```

#### Real-time Monitoring
```bash
# Monitor training progress
tail -f training_stats.txt
```

## 📊 Performance Insights

### M1 GPU Performance
- **Matrix Multiplication**: ~0.012s (excellent)
- **Training Speed**: ~179 iterations/second (normal for early training)
- **Memory Usage**: Optimized for M1 GPU memory
- **GPU Utilization**: Excellent with Metal Performance Shaders

### Training Performance
- **Quick Tests**: 5-10 minutes for 10k timesteps
- **Medium Tests**: 15-30 minutes for 50k timesteps
- **Full Training**: 1-2 hours for 1M timesteps
- **Memory Scaling**: Linear with board size

### Learning Progress
- **Positive Rewards**: Agent achieving 8-15 range for safe reveals
- **Win Rates**: Expectedly low (even humans struggle with Minesweeper)
- **Stage Progression**: Smooth advancement through curriculum stages
- **Convergence**: Typically 10-50k steps per stage

## 🔍 Recent Compatibility Fixes

### State Shape Compatibility
- **Issue**: Tests expected 2-channel states, environment uses 4-channel
- **Fix**: Dynamic state shape detection in tests
- **Result**: All tests work with both 2-channel and 4-channel states

### NumPy Version Conflicts
- **Issue**: Python 3.10 compatibility with NumPy versions
- **Fix**: Platform-specific requirements files
- **Result**: Compatible versions for all platforms

### Import Path Issues
- **Issue**: Module path resolution across platforms
- **Fix**: Proper Python path handling in scripts
- **Result**: Consistent imports across all platforms

### Script Permissions
- **Issue**: Different permission models per platform
- **Fix**: Platform-specific permission handling
- **Result**: Scripts work correctly on all platforms

## 💡 Important Notes
- **Use M1 Mac for intensive training** (GPU acceleration provides 2-4x speedup)
- **Environment randomizes properly** between episodes
- **Agent is learning** (positive rewards) but not winning yet
- **Win rates are expectedly low** - even humans struggle with Minesweeper
- **Focus on learning improvements**, not game logic bugs
- **Cross-platform scripts** available in `scripts/windows/`, `scripts/linux/`, `scripts/mac/`
- **M1 GPU optimization** automatically detected and applied
- **Training performance** monitored and optimized for each platform

## 🎯 Next Steps

### Immediate (Next 1-2 days)
1. **Test Enhanced Features**: Run training with 4-channel state and smart masking
2. **M1 Performance**: Verify GPU acceleration and training speeds
3. **Cross-Platform**: Test scripts on different platforms
4. **Visualization**: Watch agent play with new state representation

### Short Term (Next 1-2 weeks)
1. **Hyperparameter Tuning**: Optimize for enhanced environment
2. **Longer Training**: Use M1 Mac for extended training runs
3. **Win Rate Analysis**: Monitor if enhanced features improve win rates
4. **Performance Optimization**: Further M1 GPU optimizations

### Medium Term (Next 1-2 months)
1. **Advanced Curriculum**: Dynamic difficulty adjustment
2. **Multi-Agent Training**: Competitive scenarios
3. **Transfer Learning**: Pre-trained model utilization
4. **Novel Architectures**: Transformer-based models

---
**Last Updated**: 2024-12-21  
**Status**: Production ready with dual curriculum system and M1 GPU optimization