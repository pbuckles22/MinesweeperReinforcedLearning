# Minesweeper Reinforcement Learning

A modern, RL-optimized Minesweeper environment with comprehensive test coverage and curriculum learning capabilities.

## 🎯 Current Status

✅ **Environment**: Fully functional 2-channel Minesweeper RL environment  
✅ **Test Coverage**: 504 tests passing (100% success rate)  
✅ **Training System**: Complete RL training pipeline with curriculum learning  
✅ **First-Move Safety**: (Removed) The first move can be a mine; there is no mine relocation. The environment is intentionally simple for RL.  
✅ **State Representation**: Enhanced 2-channel state with safety hints  
✅ **Action Masking**: Intelligent action masking for revealed cells  
✅ **Reward System**: Comprehensive reward system for RL training  
✅ **Experiment Tracking**: Full experiment tracking and metrics collection  
✅ **Integration Tests**: Comprehensive tests to prevent RL system issues  
✅ **Debug Tools**: Complete debugging toolkit for development and troubleshooting  

## 🚀 Recent Updates

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
- **Updated**: 2-channel state representation with safety hints
- **Enhanced**: Comprehensive test suites for RL scenarios
- **Improved**: Functional and integration test coverage

## 🏗️ Architecture

### Environment Features
- **2-Channel State**: Game state + safety hints for enhanced learning
- **First-Move Safety**: (Removed) The first move can be a mine; there is no mine relocation. The environment is intentionally simple for RL.
- **Cascade Revelation**: Automatic neighbor revelation for empty cells
- **Action Masking**: Intelligent masking of revealed cells
- **Curriculum Learning**: Progressive difficulty scaling with 7 stages
- **Rectangular Boards**: Support for non-square board configurations
- **Early Learning Mode**: Safety features for initial training phases
- **Invalid Action Handling**: Proper termination after consecutive invalid actions

### State Representation
- **Channel 0**: Game state (-1: unrevealed, 0-8: revealed numbers, -4: mine hit)
- **Channel 1**: Safety hints (adjacent mine counts for unrevealed cells)

### Reward System
- `REWARD_FIRST_CASCADE_SAFE = 0`: First cascade safe reveal (pre-cascade moves, including first move, have neutral reward; first move can be a mine)
- `REWARD_SAFE_REVEAL = 5`: Regular safe reveal
- `REWARD_WIN = 100`: Game win
- `REWARD_HIT_MINE = -50`: Mine hit penalty
- `REWARD_INVALID_ACTION = -10`: Invalid action penalty

### Curriculum Stages
1. **Beginner**: 4x4 board, 2 mines (70% win rate target)
2. **Intermediate**: 6x6 board, 4 mines (60% win rate target)
3. **Easy**: 9x9 board, 10 mines (50% win rate target)
4. **Normal**: 16x16 board, 40 mines (40% win rate target)
5. **Hard**: 16x30 board, 99 mines (30% win rate target)
6. **Expert**: 18x24 board, 115 mines (20% win rate target)
7. **Chaotic**: 20x35 board, 130 mines (10% win rate target)

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

### Running Tests

#### Complete Test Suite (Recommended)
```bash
# All 504 tests
python -m pytest tests/ -v

# Quick summary
python -m pytest tests/ -q
```

#### Integration Tests (Critical for RL System)
```bash
# Comprehensive integration tests to catch RL issues
.\scripts\test_integration.ps1

# Or run manually
python -m pytest tests/integration/rl/test_rl_integration.py -v
python -m pytest tests/unit/rl/test_evaluation_unit.py -v
python -m pytest tests/unit/rl/test_rl_comprehensive_unit.py -v
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

print(f"Reward: {reward}")
print(f"Game state:\n{state[0]}")
print(f"Safety hints:\n{state[1]}")
```

### Training with Curriculum Learning
```python
from src.core.train_agent import main

# Run complete training pipeline
# This will train through all 7 curriculum stages
main()
```

## 🎮 Training Scripts

For convenience, we provide pre-configured training scripts:

### Quick Start Scripts
```bash
# Quick test (~1-2 minutes)
.\scripts\quick_test.ps1

# Medium test (~5-10 minutes)
.\scripts\medium_test.ps1

# Full training (~1-2 hours)
.\scripts\full_training.ps1
```

### Manual Training Commands
```bash
# Activate virtual environment first
.\venv\Scripts\Activate.ps1

# Quick test
python src/core/train_agent.py --total_timesteps 10000 --eval_freq 2000 --n_eval_episodes 20 --verbose 1

# Medium test
python src/core/train_agent.py --total_timesteps 50000 --eval_freq 5000 --n_eval_episodes 50 --verbose 1

# Full training
python src/core/train_agent.py --total_timesteps 1000000 --eval_freq 10000 --n_eval_episodes 100 --verbose 1
```

### Command Line Options
For detailed information about all available command line options, see [Training Commands Guide](docs/training_commands.md).

## 🛠️ Debug Tools

The project includes comprehensive debugging tools in the `/scripts/` directory:

### Debug Scripts
- `debug_env.ps1` - Environment debugging and validation
- `debug_simple.ps1` - Simple environment testing
- `debug_training.ps1` - Training pipeline debugging
- `debug_evaluation.ps1` - Model evaluation debugging
- `debug_custom_eval.ps1` - Custom evaluation callback testing
- `debug_minimal_step.ps1` - Minimal environment step testing
- `debug_eval_callback.ps1` - EvalCallback compatibility testing
- `debug_training_loop.ps1` - Training loop debugging
- `debug_episode_completion.ps1` - Episode completion testing

### Usage
```bash
# Debug environment issues
.\scripts\debug_env.ps1

# Debug training problems
.\scripts\debug_training.ps1

# Test evaluation system
.\scripts\debug_evaluation.ps1
```

These scripts create temporary debug files, run tests, and clean up automatically.

## 📁 Project Structure

```
MinesweeperReinforcedLearning/
├── src/
│   └── core/
│       ├── minesweeper_env.py    # Main environment
│       ├── constants.py          # Environment constants
│       └── train_agent.py        # Training pipeline
├── tests/
│   ├── functional/               # End-to-end tests
│   ├── unit/                     # Component tests
│   │   ├── core/                 # Environment tests
│   │   ├── rl/                   # Training tests
│   │   └── infrastructure/       # Script tests
│   └── integration/              # Cross-component tests
│       └── rl/                   # RL integration tests
├── docs/                         # Documentation
├── scripts/                      # Utility and debug scripts
├── experiments/                  # Training outputs
└── models/                       # Saved models
```

## 🎓 Curriculum Learning

The training system automatically progresses through difficulty levels:

```python
from src.core.train_agent import make_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Create environment with curriculum support
env = DummyVecEnv([make_env(max_board_size=4, max_mines=2)])

# Train with automatic progression
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000000)
```

## 🔧 Development

### Running Tests
```bash
# All tests
python -m pytest tests/ -v

# Specific categories
python -m pytest tests/unit/core/ -v
python -m pytest tests/unit/rl/ -v
python -m pytest tests/functional/ -v
```

### Environment Validation
```bash
# Quick validation
python -c "from src.core.minesweeper_env import MinesweeperEnv; env = MinesweeperEnv(); env.reset(); print('✅ Environment ready')"

# Training validation
python -c "from src.core.train_agent import make_env; env = make_env()(); print('✅ Training ready')"
```

### Training Pipeline
```bash
# Run complete training
python src/core/train_agent.py

# Monitor training progress with MLflow
mlflow ui
# Then open http://127.0.0.1:5000 in your browser
```

## 📊 Performance

- **Small Boards (4x4)**: <1ms per step
- **Medium Boards (8x8)**: ~2ms per step
- **Large Boards (16x16)**: ~5ms per step
- **Memory Usage**: Linear with board size
- **Scalability**: Supports boards up to 20x35
- **Training Speed**: ~1000 steps/second on modern hardware

## 🤝 Contributing

1. **Test Coverage**: All changes must maintain 100% test pass rate (504 tests)
2. **RL Principles**: Maintain strict RL environment contracts
3. **Documentation**: Update docs for significant changes
4. **Validation**: Run full test suite before committing
5. **Training**: Ensure training pipeline remains functional
6. **Debug Tools**: Use provided debug scripts for troubleshooting

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Status**: ✅ Production ready with complete training pipeline  
**Last Updated**: 2024-12-19  
**Test Status**: 504/504 tests passing (100%)  
**Training Status**: ✅ Complete curriculum learning system operational  
**Debug Tools**: ✅ Comprehensive debugging toolkit available 