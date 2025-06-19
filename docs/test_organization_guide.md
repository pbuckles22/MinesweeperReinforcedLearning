# Test Organization & Naming Convention Guide

## 🎯 **Overview**

This guide establishes a clear, hierarchical naming convention for organizing tests by their scope, purpose, and target area. The convention distinguishes between:

1. **Test Types** (Unit, Integration, Functional, E2E)
2. **Target Areas** (Core, RL, Agent, Infrastructure)
3. **Scope Levels** (Component, Module, System, End-to-End)

---

## 📁 **Current vs. Proposed Structure**

### **Current Structure** (Before)
```
tests/
├── functional/
│   ├── test_core_functional_requirements.py
│   ├── test_difficulty_progression.py
│   ├── test_game_flow.py
│   └── test_performance.py
├── integration/
│   ├── conftest.py
│   └── test_environment.py
├── unit/
│   ├── core/
│   │   ├── test_action_masking.py
│   │   ├── test_action_space.py
│   │   ├── test_core_mechanics.py
│   │   ├── test_deterministic_scenarios.py
│   │   ├── test_edge_cases.py
│   │   ├── test_error_handling.py
│   │   ├── test_initialization.py
│   │   ├── test_mine_hits.py
│   │   ├── test_minesweeper_env.py
│   │   ├── test_reward_system.py
│   │   └── test_state_management.py
│   ├── rl/
│   │   ├── test_comprehensive_rl.py
│   │   ├── test_early_learning.py
│   │   ├── test_train_agent.py
│   │   ├── test_train_agent_unit.py
│   │   └── test_train_agent_functional.py
│   └── agent/
│       └── conftest.py
└── scripts/
    ├── test_install_script.py
    └── test_run_script.py
```

### **Proposed Structure** (After)
```
tests/
├── unit/                          # Component-level tests
│   ├── core/                      # Core game mechanics
│   │   ├── test_core_mechanics_unit.py
│   │   ├── test_core_state_unit.py
│   │   ├── test_core_actions_unit.py
│   │   ├── test_core_rewards_unit.py
│   │   ├── test_core_initialization_unit.py
│   │   └── test_core_edge_cases_unit.py
│   ├── rl/                        # Reinforcement learning components
│   │   ├── test_rl_training_unit.py
│   │   ├── test_rl_evaluation_unit.py
│   │   ├── test_rl_callbacks_unit.py
│   │   └── test_rl_curriculum_unit.py
│   ├── agent/                     # Agent-specific logic
│   │   ├── test_agent_behavior_unit.py
│   │   ├── test_agent_decision_unit.py
│   │   └── test_agent_performance_unit.py
│   └── infrastructure/            # Utilities, configs, helpers
│       ├── test_infra_scripts_unit.py
│       ├── test_infra_config_unit.py
│       └── test_infra_utils_unit.py
├── integration/                   # Module-level tests
│   ├── core/                      # Core module integration
│   │   ├── test_core_environment_integration.py
│   │   └── test_core_state_integration.py
│   ├── rl/                        # RL module integration
│   │   ├── test_rl_training_integration.py
│   │   └── test_rl_evaluation_integration.py
│   └── agent/                     # Agent module integration
│       └── test_agent_environment_integration.py
├── functional/                    # System-level tests
│   ├── game_flow/                 # Game flow scenarios
│   │   ├── test_game_flow_functional.py
│   │   ├── test_game_requirements_functional.py
│   │   └── test_game_scenarios_functional.py
│   ├── performance/               # Performance scenarios
│   │   ├── test_perf_benchmarks_functional.py
│   │   ├── test_perf_scalability_functional.py
│   │   └── test_perf_stability_functional.py
│   └── curriculum/                # Learning progression
│       ├── test_curriculum_progression_functional.py
│       └── test_curriculum_learning_functional.py
└── e2e/                          # End-to-end tests
    ├── training/                  # Complete training workflows
    │   ├── test_training_workflow_e2e.py
    │   └── test_training_pipeline_e2e.py
    ├── evaluation/                # Complete evaluation workflows
    │   ├── test_evaluation_workflow_e2e.py
    │   └── test_evaluation_pipeline_e2e.py
    └── deployment/                # Deployment scenarios
        └── test_deployment_scenarios_e2e.py
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

## 🚀 **Targeted Test Execution Examples**

### **Run all Core tests:**
```bash
pytest tests/unit/core/ tests/integration/core/ tests/functional/game_flow/
```

### **Run all RL tests:**
```bash
pytest tests/unit/rl/ tests/integration/rl/ tests/functional/curriculum/
```

### **Run all Performance tests:**
```bash
pytest tests/functional/performance/
```

### **Run all Unit tests:**
```bash
pytest tests/unit/
```

### **Run all Integration tests:**
```bash
pytest tests/integration/
```

---

## 📊 **Coverage Reporting by Area**

### **Core Area Coverage**
```bash
pytest --cov=src.core tests/unit/core/ tests/integration/core/ tests/functional/game_flow/
```

### **RL Area Coverage**
```bash
pytest --cov=src.core.train_agent tests/unit/rl/ tests/integration/rl/ tests/functional/curriculum/
```

### **Infrastructure Coverage**
```bash
pytest --cov=src.scripts tests/unit/infrastructure/
```

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
