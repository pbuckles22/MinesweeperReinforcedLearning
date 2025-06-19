# Test Running Guide

## 🎯 **Overview**

This guide provides comprehensive instructions for running tests in the Minesweeper Reinforcement Learning project. The test suite is organized into multiple categories with different purposes and execution strategies.

---

## 📊 **Current Test Structure**

### Test Categories
- **Unit Tests**: 486 tests - Individual component validation
- **Core Tests**: Environment mechanics, state management, rewards
- **RL Tests**: Training agent, experiment tracking, callbacks
- **Functional Tests**: End-to-end scenarios, curriculum progression
- **Integration Tests**: Cross-component behavior, performance
- **Script Tests**: Infrastructure and utility scripts

### Test Status
- **Total Tests**: 486 tests
- **Tests Passing**: 486/486 (100%)
- **Test Categories**: All categories fully operational
- **Coverage**: Comprehensive coverage across all components

---

## 🚀 **Quick Start**

### 1. Run Complete Test Suite (Recommended)
For comprehensive validation, run the entire test suite:

```bash
# All 486 tests with verbose output
python -m pytest tests/ -v

# Quick summary
python -m pytest tests/ -q
```

**Expected Output**: All tests should pass with no failures.

### 2. Run Test Categories
For focused testing, run specific categories:

```bash
# Core functionality
python -m pytest tests/unit/core/ tests/integration/ tests/functional/ -v

# RL training system
python -m pytest tests/unit/rl/ -v

# Scripts and infrastructure
python -m pytest tests/scripts/ tests/unit/infrastructure/ -v
```

---

## 📋 **Detailed Test Commands**

### Core Functionality Tests

#### Unit Tests (Component Level)
```bash
# All core unit tests
python -m pytest tests/unit/core/ -v

# Specific unit test categories
python -m pytest tests/unit/core/test_core_mechanics_unit.py -v
python -m pytest tests/unit/core/test_core_state_unit.py -v
python -m pytest tests/unit/core/test_core_actions_unit.py -v
python -m pytest tests/unit/core/test_core_rewards_unit.py -v
python -m pytest tests/unit/core/test_core_initialization_unit.py -v
python -m pytest tests/unit/core/test_core_edge_cases_unit.py -v
```

#### Integration Tests (Module Level)
```bash
# All integration tests
python -m pytest tests/integration/ -v

# Specific integration test files
python -m pytest tests/integration/core/ -v
python -m pytest tests/integration/test_environment.py -v
```

#### Functional Tests (System Level)
```bash
# All functional tests
python -m pytest tests/functional/ -v

# Specific functional test categories
python -m pytest tests/functional/game_flow/ -v
python -m pytest tests/functional/performance/ -v
python -m pytest tests/functional/curriculum/ -v
```

### RL Training System Tests

#### RL Unit Tests
```bash
# All RL unit tests
python -m pytest tests/unit/rl/ -v

# Specific RL test files
python -m pytest tests/unit/rl/test_rl_training_unit.py -v
python -m pytest tests/unit/rl/test_rl_evaluation_unit.py -v
python -m pytest tests/unit/rl/test_rl_comprehensive_unit.py -v
python -m pytest tests/unit/rl/test_rl_curriculum_unit.py -v
python -m pytest tests/unit/rl/test_rl_learning_unit.py -v
```

#### Training Pipeline Tests
```bash
# Training agent tests
python -m pytest tests/unit/rl/test_train_agent.py -v

# Early learning tests
python -m pytest tests/unit/rl/test_early_learning.py -v

# Comprehensive RL tests
python -m pytest tests/unit/rl/test_comprehensive_rl.py -v
```

### Infrastructure Tests

#### Script Tests
```bash
# Script tests
python -m pytest tests/scripts/ -v

# Specific script tests
python -m pytest tests/scripts/test_install_script.py -v
python -m pytest tests/scripts/test_run_script.py -v
```

#### Infrastructure Unit Tests
```bash
# Infrastructure tests
python -m pytest tests/unit/infrastructure/ -v

# Specific infrastructure tests
python -m pytest tests/unit/infrastructure/test_infra_scripts_unit.py -v
python -m pytest tests/unit/infrastructure/test_infra_run_scripts_unit.py -v
```

---

## 🔍 **Test Analysis Commands**

### Test Coverage
```bash
# Run with coverage report for all components
python -m pytest --cov=src tests/

# Generate HTML coverage report
python -m pytest --cov=src --cov-report=html tests/
```

### Test Performance
```bash
# Run with timing information
python -m pytest --durations=10 tests/ -v

# Run with memory profiling
python -m pytest --memray tests/ -v
```

### Test Debugging
```bash
# Run single test with maximum verbosity
python -m pytest tests/unit/core/test_core_mechanics_unit.py::test_environment_initialization -vvv

# Run tests with print statement output
python -m pytest tests/ -s

# Run tests and stop on first failure
python -m pytest tests/ -x
```

---

## 📈 **Test Statistics**

### Performance Benchmarks
- **Unit Tests**: ~15 seconds
- **Integration Tests**: ~8 seconds  
- **Functional Tests**: ~12 seconds
- **RL Tests**: ~6 seconds
- **Total Runtime**: ~41 seconds

### Test Distribution
- **Unit Tests**: 486 total
  - Core: Environment mechanics, state management, rewards
  - RL: Training agent, experiment tracking, callbacks
  - Infrastructure: Scripts and utilities
- **Integration Tests**: Cross-component behavior
- **Functional Tests**: End-to-end scenarios, curriculum progression

---

## 🚨 **Troubleshooting**

### Common Issues

#### PowerShell Display Issues
If you encounter PowerShell/PSReadLine errors during test execution:
```bash
# Use Command Prompt instead
cmd
python -m pytest tests/ -v

# Or redirect output to file
python -m pytest tests/ -v > test_output.txt 2>&1
```

#### Import Errors
If you encounter import errors:
```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Verify virtual environment
python -c "import src.core.minesweeper_env; print('✅ Import successful')"
```

#### Test Discovery Issues
```bash
# Check test discovery
python -m pytest --collect-only -q

# Check for specific errors
python -m pytest --collect-only 2>&1 | findstr "ERROR"
```

### Performance Optimization

#### Parallel Execution
```bash
# Run tests in parallel (if pytest-xdist installed)
python -m pytest tests/ -n auto

# Run specific categories in parallel
python -m pytest tests/unit/core/ -n 4
```

#### Selective Testing
```bash
# Run only fast tests
python -m pytest tests/unit/core/ -m "not slow"

# Run only specific markers
python -m pytest tests/ -m "core"
```

---

## ✅ **Success Criteria**

### All Tests Passing
- **Core Tests**: Environment mechanics, state management, rewards
- **RL Tests**: Training agent, experiment tracking, callbacks
- **Functional Tests**: End-to-end scenarios, curriculum progression
- **Integration Tests**: Cross-component behavior, performance
- **Script Tests**: Infrastructure and utility scripts

### Quality Gates
- **Test Coverage**: >90% code coverage
- **Performance**: <60 seconds total runtime
- **Reliability**: 100% test pass rate
- **Documentation**: All test categories documented

---

## 📝 **Recent Updates**

### 2024-12-19: Complete Test Suite Success
- **Fixed**: `KeyError: 'stage_1'` in training agent stage completion tracking
- **Fixed**: `evaluate_model` API compatibility in functional tests
- **Achieved**: 486/486 tests passing (100% success rate)
- **Enhanced**: Comprehensive test coverage across all components

### Test Categories Now Fully Operational
- ✅ **Core Unit Tests**: Environment mechanics, state management, rewards
- ✅ **RL Unit Tests**: Training agent, experiment tracking, callbacks
- ✅ **Functional Tests**: End-to-end scenarios, curriculum progression
- ✅ **Integration Tests**: Cross-component behavior, performance
- ✅ **Script Tests**: Infrastructure and utility scripts

---

**Status**: ✅ All 486 tests passing (100% success rate)  
**Last Updated**: 2024-12-19  
**Test Runtime**: ~41 seconds  
**Coverage**: Comprehensive across all components 