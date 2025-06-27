# Scripts Organization

## 📁 New Folder Structure

### **Scripts Directory** (`scripts/`)
```
scripts/
├── audit/                    # Audit scripts (empty - moved to tests/)
├── debug/                    # Debug scripts (empty - moved to tests/)
├── training/                 # Training scripts
├── analysis/                 # Analysis scripts
├── platform/                 # Platform-specific scripts
└── archive/                  # Archived scripts
```

### **Tests Directory** (`tests/`)
```
tests/
├── unit/                     # Unit tests (existing)
├── integration/              # Integration tests (existing)
├── functional/               # Functional tests (existing)
├── audit/                    # Audit tests (moved from scripts/)
├── debug/                    # Debug tests (moved from scripts/)
├── scripts/                  # Script tests (existing)
└── e2e/                      # End-to-end tests (existing)
```

## 🏷️ Naming Convention

### **Audit Scripts** (`tests/audit/`)
```
test_<scope>_<purpose>.py

Examples:
- test_learnable_filtering_comprehensive.py     # (was comprehensive_audit_3000.py)
- test_first_move_statistics.py                 # (was audit_first_move_mine_hits.py)
- test_learnable_environment.py                 # (was audit_learnable_environment.py)
```

### **Debug Scripts** (`tests/debug/`)
```
test_<component>_<issue>.py

Examples:
- test_cascade_simulation.py                    # (was debug_cascade_simulation.py)
- test_first_move_issue.py                      # (was debug_first_move_issue.py)
- test_learnable_filtering.py                   # (was debug_learnable_filtering.py)
- test_learnable_behavior.py                    # (was test_learnable_behavior.py)
```

### **Training Scripts** (`scripts/training/`)
```
train_<method>_<purpose>.py

Examples:
- train_dqn_curriculum.py                       # (was curriculum_learning_extended.py)
- train_dqn_focused.py                          # (was focused_dqn_training.py)
- train_dqn_enhanced.py                         # (was enhanced_multistep_training.py)
- train_next_level.py                           # (was focused_next_level_training.py)
```

### **Analysis Scripts** (`scripts/analysis/`)
```
analyze_<subject>_<purpose>.py

Examples:
- analyze_5x5_difficulty.py                     # (was analyze_5x5_difficulty.py)
- analyze_one_move_wins.py                      # (was analyze_one_move_wins.py)
- analyze_curriculum_planning.py                # (was curriculum_analysis_and_planning.py)
- analyze_dqn_hyperparameters.py                # (was dqn_hyperparameter_optimization.py)
```

### **Platform Scripts** (`scripts/platform/`)
```
Platform-specific scripts organized by OS:
├── mac/                       # macOS scripts
├── linux/                     # Linux scripts
├── windows/                   # Windows scripts
├── *.sh                       # Shell scripts
└── *.ps1                      # PowerShell scripts
```

## 🎯 Key Changes

### **Consolidated Testing**
- **All testing scripts** moved to `tests/` directory
- **Audit scripts** → `tests/audit/`
- **Debug scripts** → `tests/debug/`
- **Single test command**: `pytest tests/` runs everything

### **Clean Scripts Directory**
- **Training scripts** → `scripts/training/`
- **Analysis scripts** → `scripts/analysis/`
- **Platform scripts** → `scripts/platform/`
- **No testing scripts** in `scripts/` directory

### **Consistent Naming**
- **Audit scripts**: `test_<scope>_<purpose>.py`
- **Debug scripts**: `test_<component>_<issue>.py`
- **Training scripts**: `train_<method>_<purpose>.py`
- **Analysis scripts**: `analyze_<subject>_<purpose>.py`

## 🚀 Usage

### **Running Audits**
```bash
# Run all audits
pytest tests/audit/

# Run specific audit
pytest tests/audit/test_learnable_filtering_comprehensive.py

# Run comprehensive audit (overnight)
python tests/audit/test_learnable_filtering_comprehensive.py
```

### **Running Debug Tests**
```bash
# Run all debug tests
pytest tests/debug/

# Run specific debug test
pytest tests/debug/test_learnable_behavior.py
```

### **Running Training**
```bash
# Run training scripts
python scripts/training/train_dqn_curriculum.py
python scripts/training/train_dqn_focused.py
```

### **Running Analysis**
```bash
# Run analysis scripts
python scripts/analysis/analyze_5x5_difficulty.py
python scripts/analysis/analyze_curriculum_planning.py
```

## ✅ Benefits

1. **Single test command**: `pytest tests/` runs all tests
2. **Clear organization**: Scripts vs tests clearly separated
3. **Consistent naming**: Easy to find and understand
4. **CI/CD ready**: All tests can be automated
5. **No redundancy**: Eliminated duplicate testing approaches 