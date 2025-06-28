# CSP + Probabilistic Hybrid Approach for Minesweeper

## Overview

This document outlines the implementation plan for a Constraint Satisfaction Problem (CSP) + Probabilistic guessing hybrid approach to solve Minesweeper. This approach combines deterministic logical reasoning with probability-based guessing when logical moves are exhausted.

## Current Baseline

- **RL Agent Performance**: 30% win rate on 4x4/2mines boards
- **Training Time**: 4.3 minutes for optimal performance
- **Stability**: Loss explosion issues resolved with learnable filtering

## Phase 1: Setup & Infrastructure (Day 1)

### 1.1 Create CSP Branch
```bash
git checkout -b feature/csp-probabilistic
git push -u origin feature/csp-probabilistic
```

### 1.2 Create CSP Core Files
- [ ] `src/core/csp_solver.py` - Main CSP constraint solver
- [ ] `src/core/probabilistic_guesser.py` - Probability-based guessing
- [ ] `src/core/csp_agent.py` - Hybrid agent combining CSP + probability
- [ ] `tests/unit/csp/test_csp_solver.py` - Unit tests for CSP
- [ ] `tests/unit/csp/test_probabilistic_guesser.py` - Unit tests for probability

### 1.3 Reuse Existing Infrastructure
- [ ] Adapt `minesweeper_env.py` for CSP integration
- [ ] Reuse learnable filtering logic
- [ ] Adapt testing framework for CSP evaluation
- [ ] Reuse performance metrics and evaluation

## Phase 2: CSP Solver Implementation (Day 2-3)

### 2.1 Basic CSP Structure
- [ ] Define variables (cells) and domains (safe/mine)
- [ ] Implement adjacency constraints
- [ ] Implement mine count constraints
- [ ] Add constraint propagation logic

### 2.2 CSP Solver Methods
- [ ] `solve_step()` - Find safe moves using constraint satisfaction
- [ ] `get_constraints()` - Extract constraints from current board state
- [ ] `propagate_constraints()` - Apply constraint propagation
- [ ] `find_safe_cells()` - Identify cells that can be safely revealed

### 2.3 Testing CSP Solver
- [ ] Test on simple 4x4 boards with known solutions
- [ ] Test constraint propagation on edge cases
- [ ] Verify solver finds all logically safe moves
- [ ] Performance testing on various board sizes

## Phase 3: Probabilistic Guessing (Day 4-5)

### 3.1 Probability Models
- [ ] Mine density probability (global mine count)
- [ ] Edge probability (cells near board edges)
- [ ] Corner probability (corner cells)
- [ ] Adjacency probability (based on revealed neighbors)

### 3.2 Guessing Logic
- [ ] `get_guessing_candidates()` - Return cells with lowest mine probability
- [ ] `calculate_cell_probability()` - Compute mine probability for each cell
- [ ] `rank_guessing_candidates()` - Sort cells by safety probability
- [ ] `select_best_guess()` - Choose optimal guessing candidate

### 3.3 Integration with CSP
- [ ] Detect when CSP can't make progress
- [ ] Switch to probabilistic guessing
- [ ] Maintain CSP state during guessing
- [ ] Resume CSP solving after reveals

## Phase 4: Hybrid Agent (Day 6-7)

### 4.1 CSP Agent Implementation
- [ ] `CSPAgent` class combining CSP + probability
- [ ] `choose_action()` method with CSP → probability fallback
- [ ] State management and tracking
- [ ] Performance statistics collection

### 4.2 Environment Integration
- [ ] Adapt environment to work with CSP agent
- [ ] Maintain compatibility with existing RL infrastructure
- [ ] Add CSP-specific evaluation metrics
- [ ] Update action selection logic

### 4.3 Testing Framework
- [ ] Create comprehensive test script for CSP agent
- [ ] Performance comparison with current RL baseline
- [ ] Win rate evaluation on 4x4/2mines boards
- [ ] Statistical analysis of performance

## Phase 5: Evaluation & Comparison (Day 8)

### 5.1 Performance Testing
- [ ] Run CSP + Probabilistic on same test suite as RL
- [ ] Compare win rates: CSP vs RL (30% baseline)
- [ ] Analyze decision patterns and efficiency
- [ ] Document performance characteristics

### 5.2 Analysis & Documentation
- [ ] Performance comparison report
- [ ] Decision pattern analysis
- [ ] Strengths/weaknesses of each approach
- [ ] Recommendations for next steps

## Phase 6: Next Steps Planning (Day 9)

### 6.1 Results Analysis
- [ ] If CSP + Probability > RL: Optimize and extend
- [ ] If CSP + Probability < RL: Plan CSP + RL hybrid
- [ ] If similar performance: Choose based on complexity/interpretability

### 6.2 Future Implementation
- [ ] Plan CSP + RL hybrid if needed
- [ ] Design three-way comparison framework
- [ ] Outline curriculum learning integration
- [ ] Plan for larger board sizes

## Success Criteria

### Minimum Viable Product
- [ ] CSP solver that finds all logical moves
- [ ] Probabilistic guessing that works when CSP can't progress
- [ ] Hybrid agent that achieves >25% win rate on 4x4/2mines
- [ ] Performance comparison with current RL baseline

### Stretch Goals
- [ ] >30% win rate (beat current RL baseline)
- [ ] Efficient CSP solving (<1 second per move)
- [ ] Interpretable decision making
- [ ] Extensible to larger board sizes

## Daily Checkpoints

- **Day 1**: CSP branch created, basic structure in place
- **Day 3**: CSP solver working on simple boards
- **Day 5**: Probabilistic guessing integrated
- **Day 7**: Hybrid agent complete and tested
- **Day 8**: Performance evaluation complete
- **Day 9**: Results analysis and next steps planned

## Technical Details

### CSP Variables and Domains
- **Variables**: Each cell (i, j) on the board
- **Domains**: {safe, mine} for each cell
- **Constraints**: 
  - Adjacency constraints (revealed numbers)
  - Mine count constraints (total mines)
  - Revealed cell constraints (known safe/mine)

### Probability Models
- **Global Mine Density**: P(mine) = remaining_mines / unrevealed_cells
- **Edge Probability**: P(mine) = edge_mine_density * edge_factor
- **Corner Probability**: P(mine) = corner_mine_density * corner_factor
- **Adjacency Probability**: Based on revealed neighbor constraints

### Integration Strategy
1. **CSP First**: Always try to find logical moves
2. **Probability Fallback**: When CSP can't progress, use probability
3. **State Maintenance**: Keep CSP state updated during guessing
4. **Resume CSP**: After reveals, return to CSP solving

## Files to Create

```
src/core/
├── csp_solver.py           # Main CSP constraint solver
├── probabilistic_guesser.py # Probability-based guessing
└── csp_agent.py           # Hybrid agent

tests/unit/csp/
├── test_csp_solver.py     # CSP solver tests
└── test_probabilistic_guesser.py # Probability tests

scripts/
└── test_csp_performance.py # CSP performance testing
```

## Performance Metrics

- **Win Rate**: Primary success metric
- **Move Efficiency**: Average moves per game
- **CSP vs Probability Usage**: How often each method is used
- **Solving Time**: Time per move and per game
- **Decision Interpretability**: Can we explain why moves were chosen

## Comparison Framework

### Against Current RL Baseline
- **Win Rate**: Target >30%
- **Training Time**: CSP requires no training
- **Interpretability**: CSP decisions are explainable
- **Complexity**: CSP may be simpler to maintain

### Future CSP + RL Hybrid
- **CSP for logical moves**
- **RL for complex guessing patterns**
- **Best of both worlds approach**

---

**Status**: Phase 1 - Setup & Infrastructure
**Last Updated**: 2024-06-27
**Next Milestone**: CSP branch creation and basic structure 