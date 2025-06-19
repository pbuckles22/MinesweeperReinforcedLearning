# Test Naming Convention & Organization

## 🎯 **Overview**

This document establishes a clear, hierarchical naming convention for organizing tests by their scope, purpose, and target area. The convention distinguishes between:

1. **Test Types** (Unit, Integration, Functional, E2E)
2. **Target Areas** (Core, RL, Agent, Scripts, Infrastructure)
3. **Scope Levels** (Component, Module, System, End-to-End)

---

## 📁 **Directory Structure Convention**

```
tests/
├── unit/                          # Component-level tests
│   ├── core/                      # Core game mechanics
│   ├── rl/                        # Reinforcement learning components
│   ├── agent/                     # Agent-specific logic
│   └── infrastructure/            # Utilities, configs, helpers
├── integration/                   # Module-level tests
│   ├── core/                      # Core module integration
│   ├── rl/                        # RL module integration
│   └── agent/                     # Agent module integration
├── functional/                    # System-level tests
│   ├── game_flow/                 # Game flow scenarios
│   ├── performance/               # Performance scenarios
│   └── curriculum/                # Learning progression
└── e2e/                          # End-to-end tests
    ├── training/                  # Complete training workflows
    ├── evaluation/                # Complete evaluation workflows
    └── deployment/                # Deployment scenarios
```

---

## 🏷️ **File Naming Convention**

### Pattern: `test_{area}_{scope}_{purpose}.py`

#### Area Prefixes:
- `core_` - Core game mechanics and environment
- `rl_` - Reinforcement learning components
- `agent_` - Agent-specific functionality
- `infra_` - Infrastructure and utilities
- `game_` - Game flow and scenarios
- `perf_` - Performance and benchmarks
- `curriculum_` - Learning progression
- `training_` - Training workflows
- `eval_` - Evaluation workflows

#### Scope Suffixes:
- `unit` - Unit/component tests
- `integration` - Integration tests
- `functional` - Functional/system tests
- `e2e` - End-to-end tests

#### Purpose Suffixes:
- `mechanics` - Core game mechanics
- `state` - State management
- `actions` - Action handling
- `rewards` - Reward systems
- `learning` - Learning algorithms
- `training` - Training processes
- `evaluation` - Model evaluation
- `performance` - Performance testing
- `flow` - Workflow testing

---

## 📊 **Test Classification Matrix**

| Test Type | Scope | Target Area | Purpose | Example Files |
|-----------|-------|-------------|---------|---------------|
| **Unit** | Component | Core | Mechanics | `test_core_mechanics_unit.py` |
| **Unit** | Component | RL | Learning | `test_rl_learning_unit.py` |
| **Unit** | Component | Agent | Training | `test_agent_training_unit.py` |
| **Integration** | Module | Core | State | `test_core_state_integration.py` |
| **Integration** | Module | RL | Training | `test_rl_training_integration.py` |
| **Functional** | System | Game | Flow | `test_game_flow_functional.py` |
| **Functional** | System | Performance | Benchmarks | `test_perf_benchmarks_functional.py` |
| **E2E** | Complete | Training | Workflow | `test_training_workflow_e2e.py` |

---

## 🎯 **Regional Differences**

### **Core Area** (Game Mechanics)
- **Focus**: Minesweeper game logic, environment, state management
- **Tests**: Unit tests for individual game components
- **Integration**: How game components work together
- **Functional**: Complete game scenarios and flows

### **RL Area** (Reinforcement Learning)
- **Focus**: Training algorithms, model management, learning processes
- **Tests**: Unit tests for RL components (ExperimentTracker, callbacks)
- **Integration**: Training pipeline integration
- **Functional**: Learning scenarios and curriculum progression

### **Agent Area** (Intelligent Agents)
- **Focus**: Agent behavior, decision making, performance
- **Tests**: Unit tests for agent-specific logic
- **Integration**: Agent-environment interaction
- **Functional**: Agent performance scenarios

### **Infrastructure Area** (Supporting Systems)
- **Focus**: Utilities, configuration, deployment, monitoring
- **Tests**: Unit tests for helper functions and utilities
- **Integration**: System configuration and setup
- **Functional**: Deployment and operational scenarios

---

## 📈 **Statistics vs. Targeted Testing**

### **General Project Statistics**
- **Purpose**: Overall project health and coverage metrics
- **Scope**: Project-wide metrics and trends
- **Examples**:
  - Overall test coverage percentage
  - Total number of tests by category
  - Pass/fail rates across all areas
  - Performance benchmarks across modules

### **Regional/Targeted Statistics**
- **Purpose**: Specific area performance and health
- **Scope**: Focused on particular modules or functionality
- **Examples**:
  - Core game mechanics test coverage
  - RL training pipeline performance
  - Agent decision-making accuracy
  - Infrastructure reliability metrics

---

## 🔄 **Migration Plan**

### Phase 1: Reorganize Existing Tests
1. Move current tests to new directory structure
2. Rename files according to new convention
3. Update import statements and references

### Phase 2: Update Documentation
1. Update test documentation to reflect new structure
2. Create area-specific test guides
3. Update CI/CD pipelines for new organization

### Phase 3: Establish Metrics
1. Set up area-specific coverage reporting
2. Create regional performance dashboards
3. Implement targeted test execution

---

## 📋 **Implementation Checklist**

- [ ] Create new directory structure
- [ ] Migrate existing test files
- [ ] Update file names to follow convention
- [ ] Update import statements
- [ ] Update documentation
- [ ] Update CI/CD configuration
- [ ] Set up area-specific reporting
- [ ] Validate all tests still pass
- [ ] Update test execution scripts

---

## 🎯 **Benefits of This Convention**

1. **Clarity**: Clear distinction between test types and target areas
2. **Scalability**: Easy to add new areas without confusion
3. **Maintainability**: Logical organization makes tests easier to find
4. **Targeted Execution**: Can run tests for specific areas or purposes
5. **Metrics**: Better tracking of coverage and performance by area
6. **Onboarding**: New developers can quickly understand test organization 