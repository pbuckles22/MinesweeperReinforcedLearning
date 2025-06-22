# Curriculum Learning System

## Overview

The Minesweeper RL project features a sophisticated curriculum learning system that gradually increases difficulty from simple 4x4 boards to complex 20x35 boards. The system supports two progression modes:

1. **Learning-Based Progression** (Default) - Allows progression based on learning indicators
2. **Realistic Progression** (Strict) - Requires actual win rate achievement

## Curriculum Stages

### Stage 1: Beginner (4x4, 2 mines)
- **Target Win Rate**: 15%
- **Min Wins Required**: 1
- **Learning-Based Progression**: ✅ Allowed
- **Description**: Learning basic movement and safe cell identification

### Stage 2: Intermediate (6x6, 4 mines)
- **Target Win Rate**: 12%
- **Min Wins Required**: 1
- **Learning-Based Progression**: ✅ Allowed
- **Description**: Developing pattern recognition and basic strategy

### Stage 3: Easy (9x9, 10 mines)
- **Target Win Rate**: 10%
- **Min Wins Required**: 2
- **Learning-Based Progression**: ✅ Allowed
- **Description**: Standard easy difficulty, mastering basic gameplay

### Stage 4: Normal (16x16, 40 mines)
- **Target Win Rate**: 8%
- **Min Wins Required**: 3
- **Learning-Based Progression**: ❌ Not allowed
- **Description**: Standard normal difficulty, developing advanced strategies

### Stage 5: Hard (16x30, 99 mines)
- **Target Win Rate**: 5%
- **Min Wins Required**: 3
- **Learning-Based Progression**: ❌ Not allowed
- **Description**: Standard hard difficulty, mastering complex patterns

### Stage 6: Expert (18x24, 115 mines)
- **Target Win Rate**: 3%
- **Min Wins Required**: 2
- **Learning-Based Progression**: ❌ Not allowed
- **Description**: Expert level, handling high mine density

### Stage 7: Chaotic (20x35, 130 mines)
- **Target Win Rate**: 2%
- **Min Wins Required**: 1
- **Learning-Based Progression**: ❌ Not allowed
- **Description**: Ultimate challenge, maximum complexity

## Progression Modes

### Learning-Based Progression (Default)

**Command**: `python src/core/train_agent.py` (no `--strict_progression` flag)

**How it works**:
- Early stages (1-3): Allow progression with learning indicators
- Later stages (4-7): Require actual win rate achievement
- Learning indicators include:
  - Consistent positive rewards (>5.0 average)
  - Learning phase progression
  - Curriculum stage progression

**Advantages**:
- Faster progression through early stages
- Encourages exploration and learning
- Prevents getting stuck on simple boards

**Disadvantages**:
- May progress without mastering fundamentals
- Less realistic for advanced stages

### Realistic Progression (Strict)

**Command**: `python src/core/train_agent.py --strict_progression True`

**How it works**:
- **All stages**: Require actual win rate achievement
- **All stages**: Require minimum number of wins
- No learning-based progression allowed

**Advantages**:
- Ensures mastery before progression
- More realistic training progression
- Better preparation for advanced stages

**Disadvantages**:
- Slower progression
- May get stuck on difficult stages
- Requires more training time

## Progression Logic

### Target Achievement
```python
if win_rate >= target_win_rate and actual_wins >= min_wins_required:
    should_progress = True
```

### Learning-Based Progression (Early Stages Only)
```python
elif (mean_reward >= min_positive_reward and 
      learning_based_progression and 
      not args.strict_progression):
    should_progress = True
```

### No Progression
```python
else:
    should_progress = False
    # Various reasons logged:
    # - Strict progression required but target not met
    # - Minimum wins not achieved
    # - Stage requires actual wins
    # - Insufficient learning
```

## Usage Examples

### Quick Learning-Based Training
```bash
python src/core/train_agent.py \
    --total_timesteps 50000 \
    --verbose 0
```

### Strict Realistic Training
```bash
python src/core/train_agent.py \
    --total_timesteps 100000 \
    --strict_progression True \
    --verbose 0
```

### Production Training with History
```bash
python src/core/train_agent.py \
    --total_timesteps 1000000 \
    --strict_progression True \
    --timestamped_stats True \
    --verbose 0
```

## Monitoring Progression

### Training Output
```
🎯 Using REALISTIC curriculum - requires actual wins for progression
Starting Stage 1/7: Beginner
Board: 4x4
Mines: 2
Target Win Rate: 15%
Description: 4x4 board with 2 mines - Must win 15% of games or show consistent learning

Stage 1 Results:
Mean reward: 12.34 +/- 2.1
Win rate: 18.50%
Target win rate: 15%
✅ Target achieved: 18.5% >= 15% with 18 wins (required: 1)
Stage 1 completed. Moving to next stage...
```

### Progression Reasons

**Successful Progression**:
- `✅ Target achieved: 18.5% >= 15% with 18 wins (required: 1)`
- `📈 Learning progress: 12.34 mean reward (target: 5.0) - Learning-based progression allowed`

**Failed Progression**:
- `🔒 Strict progression: Win rate 12.1% < 15% required`
- `🔒 Minimum wins not met: 0 wins < 1 required`
- `🔒 Stage requires actual wins: 5.2% < 8% (no learning-based progression)`
- `⚠️ Insufficient learning: 2.1 mean reward < 5.0`

## Backward Compatibility

The old curriculum system is fully backed up and available:

```python
# OLD CURRICULUM (BACKED UP) - Learning-based progression
old_curriculum_stages = [
    # ... original curriculum stages
]

# NEW REALISTIC CURRICULUM - Requires actual wins for progression
realistic_curriculum_stages = [
    # ... new curriculum stages with additional requirements
]
```

Both curricula use the same stage definitions but differ in progression requirements.

## Recommendations

### For Beginners
- Start with learning-based progression (default)
- Focus on understanding the learning process
- Use shorter training runs to experiment

### For Production Training
- Use realistic progression (`--strict_progression True`)
- Allow sufficient time for each stage
- Monitor progression carefully

### For Research
- Compare both progression modes
- Use timestamped stats to preserve history
- Analyze learning patterns across stages

## Troubleshooting

### Agent Stuck on Stage
1. **Check win rate**: Is it close to target?
2. **Check minimum wins**: Are enough games being won?
3. **Extend training**: Increase `--total_timesteps`
4. **Adjust difficulty**: Consider modifying stage parameters

### Slow Progression
1. **Use learning-based mode**: Remove `--strict_progression`
2. **Reduce timesteps**: Focus on early stages
3. **Check hyperparameters**: Optimize for your device

### Inconsistent Results
1. **Set random seed**: Use `--seed 42`
2. **Use deterministic mode**: Add `--deterministic`
3. **Increase evaluation episodes**: Higher `--n_eval_episodes` 