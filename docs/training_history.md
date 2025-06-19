# Training History Log

This document tracks successful training runs, their configurations, and results. Use this as a reference to reproduce successful training runs or understand what configurations work best.

## Format
Each entry follows this structure:
```
### Run [DATE]_[TIME]
- **Board Config**: [size]x[size] with [mines] mines ([density]% density)
- **Win Rate**: [X]%
- **Training Steps**: [N]
- **Key Parameters**:
  - Learning Rate: [X]
  - Batch Size: [X]
  - Steps per Update: [X]
  - Network Architecture: [X]
  - Entropy Coefficient: [X]
- **Environment Settings**:
  - Curriculum Mode: [Yes/No]
  - Early Learning Assistance: [Yes/No]
  - Reward Structure: [Standard/Custom]
- **Notes**: [Any relevant observations or special conditions]
```

## Successful Runs

### Run 20250603_164515
- **Board Config**: 8x8 with 12 mines (18.75% density)
- **Win Rate**: 52%
- **Training Steps**: 150,000
- **Key Parameters**:
  - Learning Rate: 0.0001
  - Batch Size: 64
  - Steps per Update: 2048
  - Network Architecture: [256, 256] for both policy and value
  - Entropy Coefficient: 0.01
- **Environment Settings**:
  - Curriculum Mode: Yes
  - Early Learning Assistance: Yes (first 100 games)
  - Reward Structure: Standard
- **Notes**: 
  - Started with 4x4 board and 2 mines
  - Smooth curriculum progression
  - Stable learning curve
  - First move always safe

## Configuration Templates

### High Performance Template
Use this configuration for maximum performance:
```python
board_size = 8
max_mines = 12
learning_rate = 0.0001
batch_size = 64
n_steps = 2048
n_epochs = 10
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
ent_coef = 0.01
vf_coef = 0.5
max_grad_norm = 0.5
policy_kwargs = {
    "net_arch": [
        {
            "pi": [256, 256],
            "vf": [256, 256]
        }
    ]
}
```

### Quick Learning Template
Use this configuration for faster initial learning:
```python
board_size = 4
max_mines = 2
learning_rate = 0.0003
batch_size = 32
n_steps = 1024
n_epochs = 10
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
ent_coef = 0.02
vf_coef = 0.5
max_grad_norm = 0.5
policy_kwargs = {
    "net_arch": [
        {
            "pi": [128, 128],
            "vf": [128, 128]
        }
    ]
}
```

## Failed Runs Analysis

### Common Failure Patterns
1. **Early Collapse**
   - Symptoms: Win rate drops to 0% and stays there
   - Likely Causes: 
     - Learning rate too high
     - Network too small
     - Insufficient exploration

2. **Unstable Learning**
   - Symptoms: Win rate fluctuates wildly
   - Likely Causes:
     - Batch size too small
     - Entropy coefficient too high
     - Reward structure too extreme

3. **Plateau**
   - Symptoms: Win rate stops improving
   - Likely Causes:
     - Learning rate too low
     - Network capacity insufficient
     - Curriculum progression too fast

## Best Practices

1. **Starting New Training**
   - Begin with the Quick Learning Template
   - Use curriculum learning
   - Enable early learning assistance
   - Start with smaller board size

2. **Monitoring Progress**
   - Check win rate every 10k steps
   - Look for steady improvement
   - Watch for signs of early collapse

3. **Adjusting Parameters**
   - If learning is too slow: increase learning rate
   - If unstable: increase batch size
   - If plateauing: increase network size
   - If overfitting: increase entropy coefficient

## Notes
- Always document any changes from the templates
- Record both successful and failed runs
- Note any special conditions or modifications
- Update this document after each significant training run 

# Training History and Development Log

## 2024-12-19: Critical First-Move Mine Hit Bug Fix

### Issue Identified
- **Problem**: Environment was resetting after first-move mine hits, breaking the RL contract
- **Impact**: 5/53 functional tests failing, state/mask/reward inconsistencies
- **Root Cause**: `step()` method called `self.reset()` on first-move mine hits, returning fresh board instead of action result

### Solution Implemented
- **Environment Fix**: Added `_relocate_mine_from_position()` method to handle first-move safety
- **Behavior Change**: First-move mine hits now relocate the mine and reveal the intended cell
- **RL Contract**: Every `step()` now returns the actual result of the action, not a side effect

### Test Suite Improvements
- **Unit Tests**: Updated 4 failing unit tests to expect correct behavior
- **Functional Tests**: Fixed 2 failing functional tests with proper test setup
- **Coverage**: All 53 functional tests and 116 unit tests now passing

### Key Changes
1. **Environment** (`src/core/minesweeper_env.py`):
   - Added `_relocate_mine_from_position()` method
   - Modified `step()` to handle first-move mine hits correctly
   - Maintained first-move safety guarantee without breaking RL contract

2. **Unit Tests**:
   - `test_first_move_mine_hit`: Expects mine relocation, not reset
   - `test_first_move_mine_hit_reward`: Handles win scenarios properly
   - `test_mine_hit_after_first_move`: Handles first-move wins
   - `test_action_masking_consistency`: Fixed cascade handling

3. **Functional Tests**:
   - `test_mine_placement_avoids_first_cell`: Handles immediate wins correctly
   - `test_safety_hints_accuracy`: Fixed test setup with proper state updates

### Why Unit Tests Didn't Catch This
- **Incorrect Design**: Some unit tests were testing buggy behavior as correct
- **Missing Contract Testing**: No verification of fundamental RL contract
- **Isolated Testing**: Focused on components rather than complete workflows
- **Wrong Assumptions**: Tests assumed reset behavior was correct

### Impact
- ✅ **RL Contract Maintained**: Agents can now learn properly from action results
- ✅ **First-Move Safety**: Guarantee preserved without breaking environment consistency
- ✅ **Test Coverage**: Comprehensive test suite now validates correct behavior
- ✅ **Documentation**: Clear understanding of why functional tests caught what unit tests missed

### Lessons Learned
1. **Functional tests are crucial** for catching integration issues that unit tests miss
2. **RL environments must maintain strict contracts** - `step()` should always return action results
3. **Test design matters** - tests should validate correct behavior, not current implementation
4. **End-to-end testing** reveals issues that isolated component testing cannot

---

## Previous Entries

### 2024-12-18: Environment Modernization and Flagging Removal
- Removed all flagging logic from environment
- Updated to 2-channel state representation
- Modernized test suites for RL-appropriate scenarios
- Enhanced functional and integration tests 