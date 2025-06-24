# Project Structure

This document explains the organized directory structure of the Minesweeper Reinforcement Learning project.

## 📁 Directory Organization

### Root Directory
The root directory contains only essential files and organized folders:

```
MinesweeperReinforcedLearning/
├── README.md                    # Project documentation
├── requirements.txt             # Core dependencies
├── requirements_full.txt        # Full dependency list
├── pytest.ini                  # Test configuration
├── .gitignore                  # Git ignore rules
├── .cursorrules                # Cursor IDE rules
├── CONTEXT.md                  # Project context
├── CONTEXT_SUMMARY_FOR_RESTART.md # Quick context summary
└── [organized folders]         # See below
```

### Core Source Code (`src/`)
Contains the main application code:

```
src/
├── core/
│   ├── minesweeper_env.py      # Main RL environment
│   ├── train_agent.py          # Legacy training script
│   ├── train_agent_modular.py  # Modular training script (recommended)
│   ├── constants.py            # Environment constants
│   └── gym_compatibility.py    # Gym/Gymnasium compatibility
└── visualization/
    └── visualize_agent.py      # Agent visualization tools
```

### Tests (`tests/`)
Comprehensive test suite with 639 tests:

```
tests/
├── unit/                       # Unit tests
│   ├── core/                   # Core environment tests
│   ├── rl/                     # RL component tests
│   └── infrastructure/         # Infrastructure tests
├── functional/                 # Functional tests
├── integration/                # Integration tests
├── e2e/                        # End-to-end tests
└── scripts/                    # Script tests
```

### Scripts (`scripts/`)
Utility and platform-specific scripts:

```
scripts/
├── mac/                        # Mac-specific scripts
│   ├── install_and_run.sh      # Mac installation and training
│   └── quick_training.sh       # Quick Mac training
├── windows/                    # Windows-specific scripts
│   ├── install_and_run.ps1     # Windows installation and training
│   └── quick_training.ps1      # Quick Windows training
├── linux/                      # Linux-specific scripts
│   └── install_and_run.sh      # Linux installation and training
├── manage_training_stats.py    # Cleanup and stats management
├── coverage_analysis.py        # Test coverage analysis
├── automated_testing_workflow.py # Automated testing
└── [other utility scripts]     # Various utility scripts
```

### Experiments (`experiments/`)
Experiment results and outputs:

```
experiments/
├── modular_results_*.json      # Modular training results
├── simple_results_*.json       # Simple training results
├── metrics.json                # Experiment metrics
└── metrics_backup.json         # Backup metrics
```

### Logs (`logs/`)
All logs, reports, and temporary outputs:

```
logs/
├── benchmark_results/          # Performance benchmarks
├── minimal_training_logs/      # Minimal training logs
├── progressive_logs/           # Progressive training logs
├── simple_logs/                # Simple training logs
├── conservative_training_logs/ # Conservative training logs
├── training_logs/              # General training logs
├── coverage_results_summary.json # Test coverage reports
└── cross_platform_test_report.json # Cross-platform test reports
```

### Models (`models/`)
Trained model checkpoints and saved models.

### Training Stats (`training_stats/`)
Training statistics and history:

```
training_stats/
├── training_stats.txt          # Current training stats
└── history/                    # Historical training stats
    └── training_stats_*.txt    # Timestamped training stats
```

### Documentation (`docs/`)
Project documentation and guides.

## 🧹 File Management

### Automatic Cleanup
The project includes automatic cleanup with 14-day retention:

```bash
# Clean up old files
python scripts/manage_training_stats.py --action cleanup

# View cleanup summary
python scripts/manage_training_stats.py --action summary
```

### File Locations by Type

| File Type | Location | Retention |
|-----------|----------|-----------|
| Experiment Results | `experiments/` | 14 days |
| Training Logs | `logs/` | 14 days |
| Model Checkpoints | `models/` | Manual |
| Training Stats | `training_stats/` | 14 days |
| Test Reports | `logs/` | 14 days |

### Git Ignore Rules
The `.gitignore` file is configured to ignore:
- All experiment results (`experiments/*.json`)
- All log directories (`logs/*/`)
- All test reports (`logs/*.json`)
- Training stats history (`training_stats/history/`)

## 🚀 Benefits of Organization

1. **Clean Root**: Only essential files in the root directory
2. **Logical Grouping**: Related files are grouped together
3. **Easy Navigation**: Clear directory structure for new contributors
4. **Automatic Cleanup**: Built-in cleanup prevents accumulation of old files
5. **Platform Support**: Platform-specific scripts are clearly organized
6. **Test Organization**: Comprehensive test suite with clear categories

## 📝 Usage Examples

### Running Training
```bash
# Modular training (recommended)
python -m src.core.train_agent_modular --board_size 4 4 --max_mines 2

# Results saved to: experiments/modular_results_*.json
```

### Running Tests
```bash
# All tests
python -m pytest tests/ -v

# Specific categories
python -m pytest tests/unit/ -v
python -m pytest tests/functional/ -v
```

### Platform-Specific Scripts
```bash
# Mac
./scripts/mac/install_and_run.sh

# Windows
./scripts/windows/install_and_run.ps1

# Linux
./scripts/linux/install_and_run.sh
```

### Cleanup
```bash
# Clean old files
python scripts/manage_training_stats.py --action cleanup

# View what will be cleaned
python scripts/manage_training_stats.py --action summary
```

This organized structure makes the project easy to navigate, maintain, and contribute to while keeping the root directory clean and professional. 