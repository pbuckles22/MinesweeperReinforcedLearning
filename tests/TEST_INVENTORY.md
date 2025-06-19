# Test Inventory

This document provides a comprehensive inventory of all tests in the Minesweeper Reinforcement Learning project.

## Test Statistics
- **Total Tests**: 181
- **Current Coverage**: 43% (580 statements, 331 missing)
- **Passing**: 101 tests (core) + 11 tests (RL) = 112 tests
- **Failing**: 10 tests (core deterministic) + 0 tests (RL) = 10 tests

## Test Categories

### 1. Unit Tests (Core) - 101 tests

#### Action Handling (14 tests)
- **Action Masking** (6 tests): `tests/unit/core/test_action_masking.py`
  - 6 passing, 0 failing
- **Action Space** (6 tests): `tests/unit/core/test_action_space.py`
  - 6 passing, 0 failing

#### Core Mechanics (7 tests): `tests/unit/core/test_core_mechanics.py`
- 7 passing, 0 failing

#### Deterministic Scenarios (10 tests): `tests/unit/core/test_deterministic_scenarios.py`
- 10 passing, 0 failing

#### Edge Cases (10 tests): `tests/unit/core/test_edge_cases.py`
- 5 passing, 5 failing

#### Error Handling (26 tests): `tests/unit/core/test_error_handling.py`
- 21 passing, 5 failing

#### Initialization (5 tests): `tests/unit/core/test_initialization.py`
- 5 passing, 0 failing

#### Mine Hits (5 tests): `tests/unit/core/test_mine_hits.py`
- 2 passing, 3 failing

#### Minesweeper Environment (18 tests): `tests/unit/core/test_minesweeper_env.py`
- 18 passing, 0 failing

#### Reward System (16 tests): `tests/unit/core/test_reward_system.py`
- 13 passing, 3 failing

#### State Management (20 tests): `tests/unit/core/test_state_management.py`
- 20 passing, 0 failing

### 2. Unit Tests (RL) - 11 tests
- **Early Learning** (11 tests): `tests/unit/rl/test_early_learning.py`
  - 11 passing, 0 failing
- **Agent Training** (0 tests): `tests/unit/rl/test_train_agent.py`
  - File moved but tests need to be added

### 3. Integration Tests - 25 tests
- All tests now use reveal-only logic. No flagging/unflagging remains.

### 4. Functional Tests - 15 tests
- All tests now use reveal-only logic. No flagging/unflagging remains.

### 5. Script Tests - 12 tests
- All tests now use reveal-only logic. No flagging/unflagging remains.

## Coverage Analysis

### Current Coverage by Module
- `src/__init__.py`: 100% ✅
- `src/core/__init__.py`: 100% ✅
- `src/core/constants.py`: 100% ✅
- `src/core/vec_env.py`: 100% ✅
- `src/core/minesweeper_env.py`: 74% (312/81 statements)
- `src/core/train_agent.py`: 0% (250/250 statements) - **CRITICAL GAP**

## Test Organization

### Core Tests (`tests/unit/core/`)
- **Philosophy**: Deterministic, explicit board setup
- **Purpose**: Verify core mechanics, edge cases, and deterministic behavior
- **Status**: 101 tests, 10 failing (Priority 1 to fix)

### RL Tests (`tests/unit/rl/`)
- **Philosophy**: Non-deterministic, realistic training scenarios
- **Purpose**: Verify RL agent integration, early learning, and training behavior
- **Status**: 11 tests, 0 failing

## Priority 1: Core Test Fixes Needed

### Edge Cases (5 failing)
- `test_cascade_boundary_conditions`
- `test_multiple_disconnected_zero_regions`
- `test_win_on_first_move`
- `test_rectangular_board_cascade`
- `test_cascade_with_mines_at_boundaries`

### Mine Hits (3 failing)
- `test_first_move_mine_hit`
- `test_mine_hit_after_first_move`
- `test_mine_hit_reward_consistency`

### Error Handling (5 failing)
- `test_edge_case_minimum_board`
- `test_edge_case_maximum_board`
- `test_edge_case_maximum_mines`
- `test_boundary_conditions`
- `test_edge_case_rectangular_board`

### Reward System (3 failing)
- `test_safe_reveal_reward`
- `test_mine_hit_reward`
- `test_reward_with_custom_parameters`

## Next Steps
1. **Fix all 10 failing core tests** (Priority 1)
2. **Ensure deterministic behavior and edge/cascade logic is working correctly**
3. **Move to RL training tests** only after mechanics are clean
4. **Target 60% coverage** by end of audit