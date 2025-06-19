# Test Checklist - Minesweeper Reinforcement Learning

## Overview
- **Total Tests**: 181
- **Current Coverage**: 43% (580 statements, 331 missing)
- **Passing**: 101 tests (core) + 11 tests (RL) = 112 tests
- **Failing**: 10 tests (core deterministic) + 0 tests (RL) = 10 tests
- **Target Coverage**: 60% by end of audit

## Testing Philosophy
- **Explicit Board Setup**: Use deterministic board configurations instead of random seeds
- **Public API Only**: Use only `step()`, `reset()`, and public properties - no direct state manipulation
- **Deterministic Tests**: Each test should have predictable outcomes
- **Environment API Compliance**: Tests must match actual environment behavior and gymnasium error types
- **Comprehensive Coverage**: Target 60% code coverage with focus on critical paths
- **No Flagging/Unflagging**: The environment and tests are now reveal-only. All flagging/unflagging logic and tests have been removed.
- **RL Test Separation**: RL/agent/early-learning tests are now in `tests/unit/rl/` and allow non-determinism for realistic training scenarios.

## Priority Order (Mechanics-First Approach)

### Priority 1: Fix Edge Case Tests ✅ **COMPLETE**
**Goal**: Get all edge case and cascade logic working correctly before RL training
- [x] Fix edge cases tests (10/10 passing)
- [x] Fix mine hits tests (5/5 passing)
- [x] Fix error handling edge cases (26/26 passing)
- [ ] Fix reward system tests (3/16 failing)

### Priority 2: Deterministic Scenarios ✅ **COMPLETED**
- [x] All deterministic scenario tests passing (10/10)
- [x] Environment consistency test added and working

### Priority 3: Core Mechanics & Action Masking ✅ **COMPLETED**
- [x] All core mechanics tests passing (7/7)
- [x] All action masking tests passing (6/6)

### Priority 4: RL Test Suite ✅ **COMPLETED**
- [x] RL tests moved to `tests/unit/rl/` directory
- [x] Non-deterministic philosophy documented
- [x] All RL tests passing (11/11)
- [x] Early learning tests moved to RL suite

### Priority 5: Integration & Functional (Integration Layer) 📋 **TODO**
- [ ] Test early learning progression (NEEDS NEW PHILOSOPHY)
- [ ] Test win rate tracking (NEEDS NEW PHILOSOPHY)
- [ ] Test memory usage (NEEDS NEW PHILOSOPHY)

### Priority 6: Script Tests 📋 **TODO**
- [ ] Test script syntax (NEEDS SCRIPT FIX)
- [ ] Test script parameters (NEEDS SCRIPT FIX)
- [ ] Test script environment check (NEEDS SCRIPT FIX)
- [ ] Test script output handling (NEEDS SCRIPT FIX)
- [ ] Test script error handling (NEEDS SCRIPT FIX)

### Priority 7: Agent Training (New Module) 📋 **TODO**
- [ ] Add comprehensive tests for `train_agent.py` module (0% coverage)
- [ ] Test training loop functionality
- [ ] Test model saving/loading
- [ ] Test hyperparameter handling
- [ ] Test logging and monitoring

## Coverage Targets

### Current Status
- **Overall Coverage**: 43% (580/331 statements)
- **Core Environment**: 74% (312/81 statements)
- **Training Agent**: 0% (250/250 statements) - **CRITICAL GAP**

### Target Milestones
- **Phase 1**: 50% coverage (fix existing tests)
- **Phase 2**: 60% coverage (add missing tests)
- **Phase 3**: 70% coverage (comprehensive coverage)

### Coverage by Module
- [x] `src/__init__.py`: 100% ✅
- [x] `src/core/__init__.py`: 100% ✅
- [x] `src/core/constants.py`: 100% ✅
- [x] `src/core/vec_env.py`: 100% ✅
- [ ] `src/core/minesweeper_env.py`: 74% → 85% (target)
- [ ] `src/core/train_agent.py`: 0% → 60% (target)

## Test Status Summary

### ✅ Completed Areas
- **Deterministic Scenarios**: All 10 tests passing
- **RL Test Suite**: All 11 tests passing (moved to `tests/unit/rl/`)
- **Core Mechanics**: All 7 tests passing
- **Action Masking**: All 6 tests passing

### 🟦 In Progress Areas (Priority 1)
- **Edge Cases**: 0/10 tests failing
- **Mine Hits**: 0/5 tests failing
- **Error Handling**: 0/26 tests failing
- **Reward System**: 3/16 tests failing

### 📋 TODO Areas
- **Integration Tests**: 3 tests need new philosophy
- **Script Tests**: 5 tests need PowerShell fixes
- **Training Agent**: 0% coverage - needs comprehensive tests

## Next Steps
1. **Fix all failing edge case, mine hit, error handling, and reward system tests** (Priority 1 - 10 tests)
2. **Ensure all core mechanics and edge cases are working correctly**
3. **Move to RL training tests** only after mechanics are clean
4. **Target 60% coverage** by end of audit

## Test Inventory
See `tests/TEST_INVENTORY.md` for detailed test breakdown and status. 