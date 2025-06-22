# Testing Context & Cross-Platform Guidelines

## Overview
This document establishes the testing context, cross-platform requirements, and coverage reporting standards for the Minesweeper RL project.

## Cross-Platform Testing Requirements

### Platform Support
- **macOS**: Primary development platform (M1/M2 optimized)
- **Linux**: Production deployment platform
- **Windows**: User accessibility platform

### Test Script Consistency
All platform-specific test scripts must maintain consistency:

#### Quick Test Scripts
- **Duration**: ~30-60 seconds
- **Test Suite**: `tests/unit/core tests/functional/game_flow`
- **Purpose**: Verify basic functionality and system health

#### Medium Test Scripts  
- **Duration**: ~2-3 minutes
- **Test Suite**: `tests/unit/core tests/functional tests/integration/core`
- **Purpose**: Comprehensive validation of core systems

#### Full Test Scripts
- **Duration**: ~10-15 minutes
- **Test Suite**: All tests (`pytest tests/`)
- **Purpose**: Complete system validation

### Script Validation Requirements
Every test script must:
1. **Activate virtual environment** correctly for the platform
2. **Set PYTHONPATH** to include `src/`
3. **Run pytest** with appropriate test suites
4. **Handle errors gracefully** with proper exit codes
5. **Provide clear output** with progress indicators

## Code Coverage Standards

### Coverage Requirements
- **Minimum Coverage**: 85% overall
- **Critical Modules**: 90%+ (core environment, training logic)
- **Test Coverage**: 95%+ (all tests should be covered by tests)

### Coverage Categories
1. **Unit Tests**: Core functionality, edge cases, error handling
2. **Integration Tests**: Component interactions, data flow
3. **Functional Tests**: End-to-end workflows, user scenarios
4. **Performance Tests**: Benchmarks, timing validation

### Coverage Reporting
- **Format**: HTML + JSON reports
- **Location**: `htmlcov/` directory
- **Frequency**: After every test run, before commits
- **Thresholds**: Fail builds if coverage drops below minimums

## New Test Development Guidelines

### Test Naming Conventions
```
tests/
├── unit/
│   ├── core/
│   │   ├── test_core_[feature]_unit.py
│   │   └── test_[specific]_unit.py
│   ├── rl/
│   │   ├── test_rl_[feature]_unit.py
│   │   └── test_[specific]_unit.py
│   └── infrastructure/
│       └── test_infra_[feature]_unit.py
├── integration/
│   ├── core/
│   │   └── test_core_[feature]_integration.py
│   └── rl/
│       └── test_rl_[feature]_integration.py
└── functional/
    ├── game_flow/
    │   └── test_game_[feature]_functional.py
    ├── curriculum/
    │   └── test_curriculum_[feature]_functional.py
    └── performance/
        └── test_perf_[feature]_functional.py
```

### Test Structure Requirements
```python
import pytest
from src.core.minesweeper_env import MinesweeperEnv

class TestFeatureName:
    """Test suite for [feature description]."""
    
    def setup_method(self):
        """Set up test fixtures."""
        pass
    
    def teardown_method(self):
        """Clean up test fixtures."""
        pass
    
    def test_specific_behavior(self):
        """Test specific behavior description."""
        # Arrange
        # Act  
        # Assert
        pass
    
    @pytest.mark.parametrize("input,expected", [
        (test_case_1, expected_1),
        (test_case_2, expected_2),
    ])
    def test_parameterized_behavior(self, input, expected):
        """Test parameterized behavior."""
        pass
```

### Cross-Platform Considerations
1. **File Paths**: Use `pathlib.Path` for cross-platform compatibility
2. **Line Endings**: Handle both `\n` and `\r\n`
3. **Permissions**: Test both executable and non-executable scenarios
4. **Shell Commands**: Use platform-agnostic subprocess calls
5. **Environment Variables**: Handle different platform conventions

### Test Categories & Priorities

#### High Priority (Must Have)
- Core game mechanics
- Environment initialization
- Action validation
- Reward calculation
- State management
- Error handling

#### Medium Priority (Should Have)
- Training pipeline
- Model evaluation
- Curriculum progression
- Performance benchmarks
- Integration workflows

#### Low Priority (Nice to Have)
- Edge cases
- Performance optimizations
- UI/UX features
- Documentation examples

## Automated Validation Workflows

### Pre-Commit Hooks
1. **Run quick tests** on current platform
2. **Validate cross-platform consistency**
3. **Check coverage thresholds**
4. **Lint code quality**

### CI/CD Pipeline
1. **Cross-platform test execution**
2. **Coverage analysis and reporting**
3. **Performance benchmarking**
4. **Documentation generation**

### Periodic Validation
1. **Weekly**: Full cross-platform validation
2. **Monthly**: Coverage trend analysis
3. **Quarterly**: Performance regression testing

## Coverage Analysis Tools

### Memory-Optimized Coverage
- **Script**: `scripts/coverage_analysis.py`
- **Purpose**: Handle large test suites without memory issues
- **Usage**: Chunked analysis for comprehensive coverage

### Quick Coverage Check
- **Script**: `scripts/quick_rl_coverage.py`
- **Purpose**: Fast coverage validation for RL components
- **Usage**: Pre-commit validation

### Cross-Platform Validation
- **Script**: `scripts/validate_cross_platform_tests.py`
- **Purpose**: Ensure test consistency across platforms
- **Usage**: Automated validation and reporting

## Quality Gates

### Test Quality
- **No flaky tests**: Tests must be deterministic
- **Fast execution**: Individual tests < 1 second
- **Clear assertions**: One assertion per test concept
- **Proper isolation**: Tests don't interfere with each other

### Coverage Quality
- **Meaningful coverage**: Not just line count
- **Branch coverage**: Test all code paths
- **Mutation testing**: Verify test effectiveness
- **Coverage trends**: Monitor over time

### Cross-Platform Quality
- **Consistent behavior**: Same results on all platforms
- **Proper error handling**: Graceful degradation
- **Performance parity**: Similar performance across platforms
- **Documentation**: Clear platform-specific instructions

## Monitoring & Reporting

### Coverage Reports
- **Location**: `htmlcov/` directory
- **Format**: HTML + JSON
- **Metrics**: Line, branch, function coverage
- **Trends**: Historical coverage data

### Test Reports
- **Location**: `test_reports/` directory
- **Format**: JUnit XML + HTML
- **Metrics**: Pass/fail rates, execution times
- **Trends**: Test performance over time

### Cross-Platform Reports
- **Location**: `cross_platform_test_report.json`
- **Format**: JSON with detailed validation results
- **Metrics**: Script validation, consistency checks
- **Trends**: Platform compatibility over time

## Best Practices

### Writing Cross-Platform Tests
1. **Use abstractions**: Don't hardcode platform-specific paths
2. **Test assumptions**: Verify platform behavior
3. **Handle differences**: Graceful platform-specific code
4. **Document requirements**: Clear platform dependencies

### Maintaining Coverage
1. **Regular updates**: Run coverage after changes
2. **Threshold enforcement**: Fail builds on coverage drops
3. **Trend monitoring**: Track coverage over time
4. **Quality focus**: Meaningful coverage, not just numbers

### Continuous Improvement
1. **Regular reviews**: Assess test effectiveness
2. **Performance monitoring**: Track test execution times
3. **Feedback loops**: Learn from test failures
4. **Tool updates**: Keep testing tools current

## Troubleshooting

### Common Issues
1. **Platform-specific failures**: Check path handling
2. **Memory issues**: Use chunked coverage analysis
3. **Flaky tests**: Add retry logic or fix race conditions
4. **Slow tests**: Optimize or parallelize

### Debugging Tools
1. **pytest -v**: Verbose output
2. **pytest --tb=short**: Short tracebacks
3. **coverage debug**: Coverage debugging
4. **platform validation**: Cross-platform debugging

This context ensures consistent, high-quality testing across all platforms while maintaining comprehensive coverage reporting. 