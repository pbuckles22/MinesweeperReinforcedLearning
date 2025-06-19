# Minesweeper Reinforcement Learning

A modern, RL-optimized Minesweeper environment with comprehensive test coverage and curriculum learning capabilities.

## 🎯 Current Status

✅ **Environment**: Fully functional 2-channel Minesweeper RL environment  
✅ **Test Coverage**: 53 functional tests + 116 unit tests (100% passing)  
✅ **First-Move Safety**: Guaranteed safe first move with proper RL contract  
✅ **State Representation**: Enhanced 2-channel state with safety hints  
✅ **Action Masking**: Intelligent action masking for revealed cells  
✅ **Reward System**: Comprehensive reward system for RL training  

## 🚀 Recent Updates

### Critical Bug Fix (2024-12-19)
- **Fixed**: First-move mine hit handling that was breaking RL contract
- **Improved**: Environment now properly relocates mines instead of resetting
- **Enhanced**: All tests now pass with correct behavior validation
- **Maintained**: First-move safety guarantee without compromising RL principles

### Environment Modernization (2024-12-18)
- **Removed**: All flagging logic for RL-appropriate reveal-only gameplay
- **Updated**: 2-channel state representation with safety hints
- **Enhanced**: Comprehensive test suites for RL scenarios
- **Improved**: Functional and integration test coverage

## 🏗️ Architecture

### Environment Features
- **2-Channel State**: Game state + safety hints for enhanced learning
- **First-Move Safety**: Guaranteed safe first move with mine relocation
- **Cascade Revelation**: Automatic neighbor revelation for empty cells
- **Action Masking**: Intelligent masking of revealed cells
- **Curriculum Learning**: Progressive difficulty scaling
- **Rectangular Boards**: Support for non-square board configurations

### State Representation
- **Channel 0**: Game state (-1: unrevealed, 0-8: revealed numbers, -4: mine hit)
- **Channel 1**: Safety hints (adjacent mine counts for unrevealed cells)

### Reward System
- `REWARD_FIRST_MOVE_SAFE = 0`: First move safe reveal
- `REWARD_SAFE_REVEAL = 5`: Regular safe reveal
- `REWARD_WIN = 100`: Game win
- `REWARD_HIT_MINE = -50`: Mine hit penalty
- `REWARD_INVALID_ACTION = -10`: Invalid action penalty

## 🧪 Testing

### Test Coverage
- **Functional Tests**: 53 tests covering end-to-end scenarios
- **Unit Tests**: 116 tests covering individual components
- **Integration Tests**: Cross-component behavior validation
- **Performance Tests**: Large board and high-density scenarios

### Test Categories
- Core game mechanics and RL requirements
- Enhanced state representation and safety hints
- Action masking and reward system
- Difficulty progression and curriculum learning
- Game flow and edge cases
- Performance and scalability

## 🚀 Quick Start

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

## 📁 Project Structure

```
MinesweeperReinforcedLearning/
├── src/
│   └── core/
│       ├── minesweeper_env.py    # Main environment
│       ├── constants.py          # Environment constants
│       └── train_agent.py        # Training utilities
├── tests/
│   ├── functional/               # End-to-end tests
│   ├── unit/                     # Component tests
│   └── integration/              # Cross-component tests
├── docs/                         # Documentation
└── scripts/                      # Utility scripts
```

## 🎓 Curriculum Learning

The environment supports progressive difficulty scaling:

```python
env = MinesweeperEnv(
    max_board_size=(20, 35),
    max_mines=130,
    initial_board_size=4,
    initial_mines=2
)

# Progressive difficulty
env.current_board_width = 6
env.current_board_height = 6
env.current_mines = 4
env.reset(seed=42)
```

## 🔧 Development

### Running Tests
```bash
# All tests
python -m pytest tests/ -v

# Functional tests only
python -m pytest tests/functional/ -v

# Unit tests only
python -m pytest tests/unit/ -v
```

### Environment Validation
```bash
# Quick validation
python -c "from src.core.minesweeper_env import MinesweeperEnv; env = MinesweeperEnv(); env.reset(); print('✅ Environment ready')"
```

## 📊 Performance

- **Small Boards (4x4)**: <1ms per step
- **Medium Boards (8x8)**: ~2ms per step
- **Large Boards (16x16)**: ~5ms per step
- **Memory Usage**: Linear with board size
- **Scalability**: Supports boards up to 20x35

## 🤝 Contributing

1. **Test Coverage**: All changes must maintain 100% test pass rate
2. **RL Principles**: Maintain strict RL environment contracts
3. **Documentation**: Update docs for significant changes
4. **Validation**: Run full test suite before committing

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Status**: ✅ Production Ready  
**Last Updated**: 2024-12-19  
**Test Status**: 169/169 tests passing 