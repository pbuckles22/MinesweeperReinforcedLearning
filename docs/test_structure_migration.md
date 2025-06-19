# Test Structure Migration Guide

## 🔄 **Current vs. Proposed Structure**

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

## 📋 **File Migration Mapping**

### **Unit Tests Migration**

| Current File | New File | Reason |
|--------------|----------|---------|
| `unit/core/test_action_masking.py` | `unit/core/test_core_actions_unit.py` | Consolidate action-related tests |
| `unit/core/test_action_space.py` | `unit/core/test_core_actions_unit.py` | Consolidate action-related tests |
| `unit/core/test_core_mechanics.py` | `unit/core/test_core_mechanics_unit.py` | Follow naming convention |
| `unit/core/test_state_management.py` | `unit/core/test_core_state_unit.py` | Consolidate state-related tests |
| `unit/core/test_reward_system.py` | `unit/core/test_core_rewards_unit.py` | Follow naming convention |
| `unit/rl/test_train_agent_unit.py` | `unit/rl/test_rl_training_unit.py` | Follow naming convention |
| `unit/rl/test_train_agent_functional.py` | `functional/curriculum/test_curriculum_training_functional.py` | Move to functional (system-level) |
| `scripts/test_install_script.py` | `unit/infrastructure/test_infra_scripts_unit.py` | Move to infrastructure area |

### **Functional Tests Migration**

| Current File | New File | Reason |
|--------------|----------|---------|
| `functional/test_core_functional_requirements.py` | `functional/game_flow/test_game_requirements_functional.py` | Move to game flow area |
| `functional/test_game_flow.py` | `functional/game_flow/test_game_flow_functional.py` | Follow naming convention |
| `functional/test_performance.py` | `functional/performance/test_perf_benchmarks_functional.py` | Move to performance area |
| `functional/test_difficulty_progression.py` | `functional/curriculum/test_curriculum_progression_functional.py` | Move to curriculum area |

### **Integration Tests Migration**

| Current File | New File | Reason |
|--------------|----------|---------|
| `integration/test_environment.py` | `integration/core/test_core_environment_integration.py` | Move to core area |

---

## 🎯 **Benefits of Migration**

### **Before Migration**
- ❌ Inconsistent naming patterns
- ❌ Mixed test types in same directories
- ❌ Difficult to target specific areas
- ❌ Unclear test purposes
- ❌ Hard to track coverage by area

### **After Migration**
- ✅ Clear naming convention
- ✅ Logical organization by area and type
- ✅ Easy targeted test execution
- ✅ Clear test purposes
- ✅ Area-specific coverage tracking

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

## 🔄 **Migration Steps**

1. **Create new directory structure**
2. **Move files to new locations**
3. **Rename files according to convention**
4. **Update import statements**
5. **Update pytest configuration**
6. **Update CI/CD pipelines**
7. **Update documentation**
8. **Validate all tests pass**
9. **Update test execution scripts**
10. **Set up area-specific reporting** 