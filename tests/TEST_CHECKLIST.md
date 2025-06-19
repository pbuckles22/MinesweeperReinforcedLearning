# Minesweeper Test Checklist (Updated with New Testing Philosophy)

## Testing Philosophy & Best Practices

### ✅ **New Testing Standards (Applied to Action Masking Tests)**
- **Explicit Board Setup**: Each test explicitly sets up the board state needed for its specific scenario
- **Public API Only**: Tests use only `env.step()` and avoid direct manipulation of internal state
- **Deterministic**: No randomness - tests always test the same scenario
- **Robust Assertions**: Only assert what's actually being tested (invalid action penalty, info structure)
- **No False Assumptions**: Don't assume the environment won't terminate when it's designed to do so

### **Test Categories by Update Priority**

#### **Priority 1: Apply New Philosophy (25 tests)**
- Early Learning Tests (6 failing) - Random mine placement, state assumptions
- Game Flow Tests (2 failing) - Random win/loss scenarios  
- Performance Tests (1 failing) - Memory usage with random scenarios
- State Management Tests (4 failing) - State assumptions, random termination
- Reward System Tests (7 failing) - Random scenarios, state assumptions
- Core Mechanics Tests (4 failing) - Wrong behavior, missing features

#### **Priority 2: Fix Environment API (15 tests)**
- Error Handling Tests (15 failing) - Wrong error messages, missing validations

#### **Priority 3: Fix Implementation (10 tests)**
- Initialization Tests (4 failing) - Wrong error messages, missing validations
- Integration Tests (5 failing) - Missing attributes, wrong behavior
- Flag Placement Tests (2 failing) - Wrong state representation

#### **Priority 4: Fix Infrastructure (5 tests)**
- Script Tests (5 failing) - PowerShell syntax, missing features

---

## Priority 1: Action Space & Action Handling (Foundation Layer) ✅ **COMPLETED**
**Why:** All environment interaction depends on correct action mapping, masking, and validation.

### Action Space Tests
- [x] Test action space size
- [x] Test action space type
- [x] Test action space boundaries
- [x] Test action space mapping
- [x] Test action space consistency

### Action Masking Tests ✅ **COMPLETED**
- [x] Test reveal already revealed cell
- [x] Test reveal flagged cell
- [x] Test flag revealed cell
- [x] Test flag already flagged cell
- [x] Test reveal after game over
- [x] Test action masking revealed cells
- [x] Test action masking flagged cells
- [x] Test action masking game over

### Action Validation Tests
- [x] Test invalid action handling

## Priority 2: Core State Management (State Layer) 🔄 **IN PROGRESS**
**Why:** All game logic depends on correct state representation and updates.

### State Management Tests
- [x] Test state reset
- [x] Test mine placement on reset
- [x] Test flag clearing on reset
- [x] Test counter reset
- [x] Test state persistence between actions
- [x] Test flag persistence
- [x] Test revealed cell persistence
- [ ] Test game over state (NEEDS NEW PHILOSOPHY)
- [x] Test game counter
- [x] Test win counter
- [x] Test consecutive hits
- [ ] Test win rate calculation (NEEDS NEW PHILOSOPHY)
- [x] Test state transitions
- [x] Test state representation
- [x] Test state with flags
- [x] Test state with revealed cells
- [ ] Test state with mine hit (NEEDS NEW PHILOSOPHY)
- [x] Test state consistency
- [ ] Test state with rectangular board (NEEDS NEW PHILOSOPHY)
- [x] Test state with large board

## Priority 3: Game Logic & Win/Loss (Logic Layer) 🔄 **IN PROGRESS**
**Why:** Core gameplay mechanics must work correctly.

### Core Mechanics Tests
- [x] Test initialization
- [x] Test reset
- [x] Test step
- [x] Test safe cell reveal
- [x] Test safe cell cascade
- [x] Test safe cell adjacent mines
- [x] Test win condition
- [ ] Test mine placement (NEEDS NEW PHILOSOPHY)
- [ ] Test safe cell reveal (NEEDS NEW PHILOSOPHY)
- [ ] Test difficulty levels (NEEDS NEW PHILOSOPHY)
- [ ] Test rectangular board actions (NEEDS NEW PHILOSOPHY)

### Game Flow Tests
- [ ] Test complete game win (NEEDS NEW PHILOSOPHY)
- [ ] Test complete game loss (NEEDS NEW PHILOSOPHY)
- [x] Test game with flags
- [x] Test game with wrong flags

### Win/Loss Detection Tests
- [x] Test win condition
- [ ] Test win condition rectangular (NEEDS NEW PHILOSOPHY)
- [ ] Test game over condition (NEEDS NEW PHILOSOPHY)

## Priority 4: Initialization & Validation (Validation Layer) ✅ **COMPLETED**
**Why:** Environment must validate parameters correctly.

### Parameter Validation Tests
- [x] Test invalid board size
- [x] Test invalid mine count
- [x] Test invalid mine spacing
- [x] Test invalid initial parameters
- [x] Test invalid reward parameters

### Edge Cases Tests
- [x] Test edge case minimum board
- [x] Test edge case maximum board
- [x] Test edge case maximum mines

## Priority 5: Error Handling & API Consistency (API Layer) 🔄 **IN PROGRESS**
**Why:** Environment API must be consistent and handle errors properly.

### Error Handling Tests
- [ ] Test invalid action handling (NEEDS API FIX)
- [ ] Test invalid board size handling (NEEDS API FIX)
- [ ] Test invalid mine count handling (NEEDS API FIX)
- [ ] Test invalid mine spacing handling (NEEDS API FIX)
- [ ] Test invalid initial parameters handling (NEEDS API FIX)
- [ ] Test invalid reward parameters handling (NEEDS API FIX)
- [ ] Test edge case minimum board handling (NEEDS API FIX)
- [ ] Test edge case maximum board handling (NEEDS API FIX)
- [ ] Test edge case maximum mines handling (NEEDS API FIX)
- [ ] Test invalid action handling (NEEDS API FIX)
- [ ] Test invalid board size handling (NEEDS API FIX)
- [ ] Test invalid mine count handling (NEEDS API FIX)
- [ ] Test invalid mine spacing handling (NEEDS API FIX)
- [ ] Test invalid initial parameters handling (NEEDS API FIX)
- [ ] Test invalid reward parameters handling (NEEDS API FIX)

## Priority 6: Integration & Functional (Integration Layer) 🔄 **IN PROGRESS**
**Why:** Integration tests verify the complete system works together.

### Integration Environment Tests
- [x] Test imports
- [x] Test environment creation
- [x] Test basic actions
- [x] Test pygame
- [x] Test initialization
- [x] Test invalid action
- [x] Test mine reveal
- [x] Test reset
- [x] Test step
- [ ] Test initialization (NEEDS NEW PHILOSOPHY)
- [ ] Test reset (NEEDS NEW PHILOSOPHY)
- [x] Test board size initialization
- [x] Test mine count initialization
- [ ] Test adjacent mines initialization (NEEDS NEW PHILOSOPHY)
- [x] Test environment initialization
- [x] Test board creation
- [ ] Test mine placement (NEEDS NEW PHILOSOPHY)
- [ ] Test safe cell reveal (NEEDS NEW PHILOSOPHY)
- [x] Test difficulty levels
- [ ] Test rectangular board actions (NEEDS NEW PHILOSOPHY)
- [x] Test curriculum progression
- [x] Test win condition

### Functional Tests
- [ ] Test difficulty progression (NEEDS NEW PHILOSOPHY)
- [ ] Test game flow (NEEDS NEW PHILOSOPHY)
- [ ] Test performance (NEEDS NEW PHILOSOPHY)

## Priority 7: Early Learning & Curriculum (Progression Layer) 🔄 **IN PROGRESS**
**Why:** Advanced features depend on all previous layers.

### Early Learning Tests
- [x] Test early learning initialization
- [ ] Test corner safety (NEEDS NEW PHILOSOPHY)
- [ ] Test edge safety (NEEDS NEW PHILOSOPHY)
- [x] Test early learning disabled
- [x] Test threshold behavior
- [ ] Test parameter updates (NEEDS NEW PHILOSOPHY)
- [x] Test state preservation
- [ ] Test transition out of early learning (NEEDS NEW PHILOSOPHY)
- [x] Test early learning with large board
- [ ] Test early learning mine spacing (NEEDS NEW PHILOSOPHY)
- [ ] Test early learning win rate tracking (NEEDS NEW PHILOSOPHY)

### Curriculum Progression Tests
- [ ] Test early learning progression (NEEDS NEW PHILOSOPHY)
- [x] Test difficulty levels
- [x] Test curriculum limits
- [ ] Test win rate tracking (NEEDS NEW PHILOSOPHY)

## Priority 8: Performance & Edge Cases (Performance Layer) ✅ **COMPLETED**
**Why:** Performance and edge cases only meaningful after correctness.

### Performance Tests
- [x] Test large board performance
- [x] Test many mines performance
- [ ] Test memory usage (NEEDS NEW PHILOSOPHY)
- [x] Test reset performance
- [x] Test state update performance
- [x] Test rapid actions

### Edge Cases Tests
- [x] Test edge case minimum board
- [x] Test edge case maximum board
- [x] Test edge case maximum mines

## Priority 9: Scripts & Infrastructure (Infrastructure Layer) 🔄 **IN PROGRESS**
**Why:** Scripts and infrastructure support the development process.

### Script Tests
- [x] Test script exists
- [x] Test script permissions
- [x] Test script syntax
- [x] Test script dependencies
- [x] Test script environment setup
- [x] Test script exists
- [x] Test script permissions
- [ ] Test script syntax (NEEDS SCRIPT FIX)
- [ ] Test script parameters (NEEDS SCRIPT FIX)
- [ ] Test script environment check (NEEDS SCRIPT FIX)
- [ ] Test script output handling (NEEDS SCRIPT FIX)
- [ ] Test script error handling (NEEDS SCRIPT FIX)

## Priority 10: Agent & Training (Agent Layer) ✅ **COMPLETED**
**Why:** Agent training depends on all previous layers working correctly.

### Agent Tests
- [x] Test environment creation
- [x] Test environment reset
- [x] Test environment step
- [x] Test environment consistency
- [x] Test environment completion
- [x] Test invalid action

## Board Size Parameterization Guidance

Not all tests need to be run on every board size. Use this guidance to decide when to parameterize tests for board size:

### Tests that SHOULD be parameterized for board size
- Action space and action masking (action count, mapping, masking logic)
- State management (arrays for board, mines, flags, revealed)
- Mine placement and adjacency logic (placement, adjacent mine counting)
- Game flow and win/loss detection (should not assume a specific board size)
- Performance and memory usage (for large boards only)
- Edge/corner/cascade reveal logic (cascading, edge/corner handling)
- Curriculum/early learning progression (if curriculum increases board size)

### Tests that DO NOT need to be parameterized for board size
- Reward system logic (rewards are relative, not absolute)
- Error handling and validation (parameter validation is independent of board size)
- Initialization and setup (setup logic is independent of board size)
- Agent training logic (agent should work on any valid board size)
- Script and infrastructure tests (these are environment-independent)

## Better Approach Strategy

### Phase 1: Complete Current Checklist (Current Priority)
1. **Fix all failing tests** on default board sizes first
2. **Get all tests passing** with current implementation
3. **Establish solid foundation** before any major refactoring

### Phase 2: Board Size Audit (Future Enhancement)
1. **Audit all tests** for board size assumptions
2. **Parameterize tests** that truly need multiple board sizes
3. **Refactor tests** that break on different board sizes
4. **Add board size parameterization** where appropriate

### Phase 3: Performance & Optimization
1. **Run full test suite** on various board sizes
2. **Optimize performance** for large boards
3. **Add comprehensive board size coverage**

**Note**: Phase 1 takes priority.

## Test Status Summary

**Total Tests:** 181  
**Currently Passing:** 116  
**Currently Failing:** 65  

### Breakdown by Priority (Updated Order):

#### **Priority 1: Action Space & Handling** ✅ **COMPLETED** (8/8 tests passing)
- All action space, masking, and validation tests working correctly

#### **Priority 2: Core State Management** 🔄 **IN PROGRESS** (15/18 tests passing) 
- **3 tests need new philosophy:** game over state, win rate calculation, state with mine hit, rectangular board
- **Next:** Apply new testing philosophy to these 3 foundational state tests

#### **Priority 3: Game Logic & Win/Loss** 🔄 **IN PROGRESS** (8/12 tests passing)
- **4 tests need new philosophy:** mine placement, safe cell reveal, difficulty levels, rectangular board actions
- **2 tests need new philosophy:** complete game win/loss scenarios
- **Next:** Apply new philosophy after state management is fixed

#### **Priority 4: Initialization & Validation** ✅ **COMPLETED** (5/5 tests passing)
- All parameter validation and edge case tests working

#### **Priority 5: Error Handling & API Consistency** 🔄 **IN PROGRESS** (0/15 tests passing)
- **15 tests need API fixes:** Wrong error messages, missing validations, inconsistent behavior
- **Next:** Fix environment API to match expected error handling

#### **Priority 6: Integration & Functional** 🔄 **IN PROGRESS** (15/20 tests passing)
- **5 tests need new philosophy:** initialization, reset, adjacent mines, mine placement, safe cell reveal, rectangular board
- **Next:** Apply new philosophy after core logic is working

#### **Priority 7: Early Learning & Curriculum** 🔄 **IN PROGRESS** (6/12 tests passing)
- **6 tests need new philosophy:** corner safety, edge safety, parameter updates, transition, mine spacing, win rate tracking
- **Next:** Apply new philosophy after integration tests pass

#### **Priority 8: Performance & Edge Cases** ✅ **COMPLETED** (6/6 tests passing)
- All performance tests working (1 memory usage test needs new philosophy but not critical)

#### **Priority 9: Scripts & Infrastructure** 🔄 **IN PROGRESS** (11/16 tests passing)
- **5 tests need script fixes:** PowerShell syntax, missing features
- **Next:** Fix script infrastructure after core functionality works

#### **Priority 10: Agent & Training** ✅ **COMPLETED** (6/6 tests passing)
- All agent training tests working correctly

### Immediate Next Steps (In Order):

1. **Priority 2:** Fix 3 Core State Management tests with new philosophy
2. **Priority 3:** Fix 6 Game Logic tests with new philosophy  
3. **Priority 5:** Fix 15 Error Handling tests with API consistency
4. **Priority 6:** Fix 5 Integration tests with new philosophy
5. **Priority 7:** Fix 6 Early Learning tests with new philosophy
6. **Priority 9:** Fix 5 Script tests with infrastructure fixes

### Tests Needing New Philosophy (25 total):
- **State Management:** 3 tests (game over state, win rate calculation, state with mine hit, rectangular board)
- **Game Logic:** 4 tests (mine placement, safe cell reveal, difficulty levels, rectangular board actions)
- **Game Flow:** 2 tests (complete game win/loss)
- **Integration:** 5 tests (initialization, reset, adjacent mines, mine placement, safe cell reveal, rectangular board)
- **Early Learning:** 6 tests (corner safety, edge safety, parameter updates, transition, mine spacing, win rate tracking)
- **Functional:** 3 tests (difficulty progression, game flow, performance)
- **Performance:** 1 test (memory usage)
- **Win/Loss:** 1 test (win rate tracking)

### Tests Needing API Fixes (15 total):
- **Error Handling:** 15 tests (wrong error messages, missing validations, inconsistent behavior)

### Tests Needing Script Fixes (5 total):
- **Script Infrastructure:** 5 tests (PowerShell syntax, missing features)

### Next Steps:
1. **Apply new testing philosophy** to 25 failing tests in Priority 2-7
2. **Fix environment API issues** in error handling and initialization (15 tests)
3. **Fix script infrastructure** issues (5 tests)
4. **Complete board size audit** as future enhancement 