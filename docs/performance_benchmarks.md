# Minesweeper RL Agent Performance Benchmarks

This document outlines the performance goals and benchmarks for the Minesweeper Reinforcement Learning agent.

## Human Performance Reference

| Skill Level | Win Rate Range |
|-------------|----------------|
| Beginner    | 10-20%         |
| Intermediate| 30-40%         |
| Expert      | 50-60%         |
| World-class | 70-80%         |

## AI Performance Expectations

| Performance Level | Win Rate Range | Description |
|-------------------|----------------|-------------|
| Learning          | >20%           | Basic learning demonstrated |
| Good             | >30%           | Competent performance |
| Very Good        | >40%           | Strong performance |
| Excellent        | >50%           | Expert-level performance |
| Exceptional      | >60%           | World-class performance |

## Board-Specific Goals

### 4x4 Board (2 mines, 12.5% density)
- Target: 60-70% win rate
- Acceptable: >50% win rate
- Excellent: >70% win rate

### 5x5 Board (4 mines, 16% density)
- Target: 40-50% win rate
- Acceptable: >30% win rate
- Excellent: >60% win rate

### 8x8 Board (12 mines, 18.75% density)
- Target: 30-40% win rate
- Acceptable: >20% win rate
- Excellent: >50% win rate

### 10x10 Board (20 mines, 20% density)
- Target: 20-30% win rate
- Acceptable: >15% win rate
- Excellent: >40% win rate

## Training Progress Indicators

### Early Training (First 10k steps)
- Should show some learning (win rate >5%)
- Should demonstrate basic mine avoidance

### Mid Training (10k-50k steps)
- Should show steady improvement
- Win rate should be approaching target range

### Late Training (50k+ steps)
- Should be close to or at target win rate
- Performance should be stable

## Training Duration Expectations

### Standard Training (350,000 timesteps)
- Excellent: > 50% win rate
- Good: 40-50% win rate
- Acceptable: 30-40% win rate
- Needs Improvement: < 30% win rate

### Short Training (150,000 timesteps)
- Excellent: > 40% win rate
- Good: 30-40% win rate
- Acceptable: 20-30% win rate
- Needs Improvement: < 20% win rate

### Training Duration Impact
- 150k timesteps (43% of standard training):
  - Expected win rate: 30-40% of standard training performance
  - Success criteria adjusted for shorter learning time
  - Early learning phase is more critical
  - Curriculum progression should be more gradual

### Performance Scaling Factors
1. Training Duration
   - Longer training allows for better strategy development
   - More time to learn from mistakes
   - Better exploration of the state space

2. Learning Efficiency
   - Early learning assistance effectiveness
   - Curriculum progression smoothness
   - Reward structure clarity

3. Environment Complexity
   - Board size progression
   - Mine density increases
   - Pattern recognition requirements

### Success Criteria for Different Training Durations

#### 350,000 timesteps (Standard)
- Target win rate: 52%
- Early learning phase: First 100 games
- Curriculum progression: Gradual
- Expected final performance: 45-55% win rate

#### 150,000 timesteps (Short)
- Target win rate: 35%
- Early learning phase: First 200 games
- Curriculum progression: More gradual
- Expected final performance: 30-40% win rate

## Success Criteria

A training run is considered successful if:
1. The agent achieves the target win rate for its board configuration
2. The win rate is stable (not fluctuating wildly)
3. The agent demonstrates consistent performance across multiple evaluation episodes
4. The learning curve shows steady improvement over time

## Notes

- These benchmarks assume standard Minesweeper rules
- Performance may vary based on:
  - Training duration
  - Learning rate
  - Network architecture
  - Reward structure
  - Curriculum learning settings
- Regular evaluation against these benchmarks helps track progress and identify areas for improvement
- Performance expectations should be adjusted based on:
  - Training duration
  - Board size
  - Mine density
  - Learning assistance features
- Early learning phase is more critical in shorter training runs
- Curriculum progression should be more conservative with less training time
- Success metrics should be evaluated relative to training duration 