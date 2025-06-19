# Test Checklist - Minesweeper Reinforcement Learning

## Overview
- **Total Tests**: 181
- **Current Coverage**: 43% (580 statements, 331 missing)
- **Passing**: 148 tests
- **Failing**: 33 tests
- **Target Coverage**: 60% by end of audit

## Testing Philosophy
- **Explicit Board Setup**: Use deterministic board configurations instead of random seeds
- **Public API Only**: Use only `step()`, `reset()`, and public properties - no direct state manipulation
- **Deterministic Tests**: Each test should have predictable outcomes
- **Environment API Compliance**: Tests must match actual environment behavior and gymnasium error types
- **Comprehensive Coverage**: Target 60% code coverage with focus on critical paths

## Priority Order (Dependency-Based)

### Priority 1: Action Handling ✅ **COMPLETED**
- [x] All action masking tests passing (8/8)
- [x] All action space tests passing (6/6)
- [x] All action consistency tests passing

### Priority 2: Core State Management ✅ **COMPLETED**
- [x] All state management tests passing (20/20)
- [x] All state transition tests passing
- [x] All state persistence tests passing

### Priority 3: Game Logic & Win/Loss ✅ **COMPLETED**
- [x] All game flow tests passing (4/4)
- [x] All win/loss condition tests passing
- [x] All game completion tests passing

### Priority 4: Error Handling & API Consistency ✅ **COMPLETED**
- [x] All error handling tests refactored to match actual environment and gymnasium error behavior (26/26 passing)
- [x] All API consistency tests passing

### Priority 5: Reward System ✅ **COMPLETED**
- [x] All reward system tests passing (16/16)

### Priority 6: Flag Placement ✅ **COMPLETED**
- [x] Test flag removal (refactored to new philosophy)
- [x] Test flag mine hit (refactored to new philosophy)

### Priority 7: Early Learning ✅ **COMPLETED**
- [x] Test corner safety (refactored to new philosophy)
- [x] Test edge safety (refactored to new philosophy)
- [x] Test parameter updates (refactored to new philosophy)
- [x] Test state preservation (refactored to new philosophy)
- [x] Test transition out of early learning (refactored to new philosophy)
- [x] Test early learning mine spacing (refactored to new philosophy)
- [x] Test early learning win rate tracking (refactored to new philosophy)

### Priority 8: Core Mechanics 🔄 **IN PROGRESS**
- [ ] Test safe cell reveal (NEEDS NEW PHILOSOPHY)

### Priority 9: Mine Hits 🔄 **IN PROGRESS**
- [ ] Test first move behavior (NEEDS NEW PHILOSOPHY)

### Priority 10: Environment API Fixes 🔄 **IN PROGRESS**
- [ ] Test invalid board size (NEEDS API FIX)
- [ ] Test invalid mine spacing (NEEDS API FIX)
- [ ] Test invalid initial parameters (NEEDS API FIX)
- [ ] Test invalid reward parameters (NEEDS API FIX)
- [ ] Test board boundary actions (NEEDS API FIX)
- [ ] Test initialization (NEEDS API FIX)
- [ ] Test reset (NEEDS API FIX)
- [ ] Test adjacent mines initialization (NEEDS NEW PHILOSOPHY)
- [ ] Test mine placement (NEEDS NEW PHILOSOPHY)
- [ ] Test safe cell reveal (NEEDS NEW PHILOSOPHY)
- [ ] Test rectangular board actions (NEEDS NEW PHILOSOPHY)
- [ ] Test difficulty levels (NEEDS NEW PHILOSOPHY)
- [ ] Test flag action (NEEDS NEW PHILOSOPHY)
- [ ] Test unflag action (NEEDS NEW PHILOSOPHY)
- [ ] Test invalid actions (NEEDS NEW PHILOSOPHY)
- [ ] Test win condition (NEEDS NEW PHILOSOPHY)
- [ ] Test state transitions (NEEDS NEW PHILOSOPHY)
- [ ] Test state representation (NEEDS NEW PHILOSOPHY)

### Priority 11: Integration & Functional (Integration Layer) 🔄 **IN PROGRESS**
- [ ] Test early learning progression (NEEDS NEW PHILOSOPHY)
- [ ] Test win rate tracking (NEEDS NEW PHILOSOPHY)
- [ ] Test memory usage (NEEDS NEW PHILOSOPHY)

### Priority 12: Script Tests 🔄 **IN PROGRESS**
- [ ] Test script syntax (NEEDS SCRIPT FIX)
- [ ] Test script parameters (NEEDS SCRIPT FIX)
- [ ] Test script environment check (NEEDS SCRIPT FIX)
- [ ] Test script output handling (NEEDS SCRIPT FIX)
- [ ] Test script error handling (NEEDS SCRIPT FIX)

### Priority 13: Agent Training (New Module) 📋 **TODO**
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

### ✅ Completed Areas (Summary Style)
- **Action Handling**: All 12 tests passing
- **Core State Management**: All 20 tests passing (refactored to new philosophy)
- **Game Logic & Win/Loss**: All 4 tests passing (refactored to new philosophy)
- **Error Handling**: All 26 tests passing (refactored to match environment)
- **Reward System**: All 16 tests passing (refactored to new philosophy)
- **Flag Placement**: All 6 tests passing (refactored to new philosophy)
- **Early Learning**: All 11 tests passing (refactored to new philosophy)

### 🔄 In Progress Areas
- **Core Mechanics**: 1 test needs new philosophy
- **Mine Hits**: 1 test needs new philosophy
- **Environment API**: 15 tests need API fixes
- **Integration**: 3 tests need new philosophy
- **Script Tests**: 5 tests need PowerShell fixes

### ❌ Needs Fixes
- **API Consistency**: 15 tests need environment API fixes
- **Script Tests**: 5 tests need PowerShell script fixes
- **Training Agent**: 0% coverage - needs comprehensive tests

## Next Steps
1. **Apply new philosophy** to remaining 5 tests (Priorities 8-9)
2. **Fix environment API inconsistencies** (15 tests)
3. **Fix PowerShell script tests** (5 tests)
4. **Add tests for `train_agent.py`** module (0% coverage)
5. **Target 60% coverage** by end of audit

## Test Inventory
See `tests/TEST_INVENTORY.md` for detailed test breakdown and status. 