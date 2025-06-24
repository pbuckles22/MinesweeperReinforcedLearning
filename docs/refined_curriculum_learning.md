# Refined Curriculum Learning for Minesweeper RL

## Overview

The refined curriculum learning approach implements a more gradual difficulty progression with longer training on intermediate stages to improve knowledge transfer and generalization across different board configurations.

## Key Improvements

### 1. More Gradual Difficulty Progression

**Original Approach:**
- Stage 1: 4x4 with 1 mine (6.25% density)
- Stage 2: 4x4 with 2 mines (12.5% density) 
- Stage 3: 6x6 with 3 mines (8.3% density) - **Sudden jump to larger board**

**Refined Approach:**
- Stage 1: 4x4 with 1 mine (6.25% density) - Master this first
- Stage 2: 4x4 with 2 mines (12.5% density) - Build on Stage 1
- Stage 3: 4x4 with 3 mines (18.75% density) - Same board, more mines
- Stage 4: 6x6 with 4 mines (11.1% density) - Larger board, similar density to Stage 2
- Stage 5: 6x6 with 6 mines (16.7% density) - More mines on larger board
- Stage 6: 6x6 with 8 mines (22.2% density) - Challenge level

### 2. Longer Training on Intermediate Stages

**Training Timesteps:**
- Stage 1: 200,000 timesteps (master the basics)
- Stage 2: 300,000 timesteps (build on knowledge)
- Stage 3: 400,000 timesteps (consolidate on same board)
- Stage 4: 500,000 timesteps (transfer to larger board)
- Stage 5: 600,000 timesteps (increase difficulty)
- Stage 6: 700,000 timesteps (challenge level)

### 3. Realistic Target Win Rates

**Target Progression:**
- Stage 1: 75% win rate (master easy configuration)
- Stage 2: 60% win rate (good performance with more mines)
- Stage 3: 45% win rate (challenging but achievable)
- Stage 4: 40% win rate (transfer to larger board)
- Stage 5: 30% win rate (difficult but possible)
- Stage 6: 20% win rate (expert level)

### 4. Enhanced Evaluation

**Improved Metrics:**
- **Win Rate**: Percentage of successful games
- **Mean Reward**: Average reward per episode
- **Average Steps**: Efficiency measure
- **Success Rate**: Percentage of episodes that complete (don't hit max steps)
- **Mine Density**: Difficulty metric

**Evaluation Episodes:**
- 200 episodes for comprehensive evaluation
- Better statistical significance
- More reliable performance assessment

### 5. Better Knowledge Transfer

**Progressive Learning:**
1. **Master the Basics**: Learn fundamental strategies on 4x4 with 1 mine
2. **Build Complexity**: Gradually increase mine count on same board size
3. **Transfer Knowledge**: Apply learned strategies to larger boards
4. **Scale Up**: Increase difficulty while maintaining transferable skills

## Implementation Details

### Files Created

1. **`scripts/curriculum_training_refined.py`** - Main refined curriculum script
2. **`scripts/mac/refined_curriculum_training.sh`** - Mac-specific execution script
3. **`scripts/test_refined_curriculum_minimal.py`** - Minimal test for validation
4. **`tests/functional/curriculum/test_refined_curriculum_functional.py`** - Functional tests

### Key Functions

#### `evaluate_model(model, board_size, max_mines, n_episodes=200)`
- Comprehensive evaluation with multiple metrics
- 200 episodes for statistical significance
- Tracks success rate and average steps

#### `refined_curriculum_training()`
- Implements the 6-stage curriculum
- Interactive decision making for stage progression
- Detailed results tracking and saving

## Usage

### Quick Test
```bash
python scripts/test_refined_curriculum_minimal.py
```

### Full Curriculum Training
```bash
./scripts/mac/refined_curriculum_training.sh
```

### Manual Execution
```bash
python scripts/curriculum_training_refined.py
```

## Expected Results

### Stage Progression
1. **Stage 1**: Should achieve 75%+ win rate on 4x4 with 1 mine
2. **Stage 2**: Should achieve 60%+ win rate on 4x4 with 2 mines
3. **Stage 3**: Should achieve 45%+ win rate on 4x4 with 3 mines
4. **Stage 4**: Should achieve 40%+ win rate on 6x6 with 4 mines
5. **Stage 5**: Should achieve 30%+ win rate on 6x6 with 6 mines
6. **Stage 6**: Should achieve 20%+ win rate on 6x6 with 8 mines

### Knowledge Transfer Benefits
- **Better Generalization**: Agent learns transferable strategies
- **Improved Performance**: Higher win rates on harder configurations
- **Faster Learning**: Each stage builds on previous knowledge
- **More Robust**: Less overfitting to specific board configurations

## Comparison with Original Approach

| Aspect | Original | Refined |
|--------|----------|---------|
| Stages | 5 stages | 6 stages |
| Progression | Sudden jumps | Gradual increases |
| Training Time | 100K-300K steps | 200K-700K steps |
| Evaluation | 100 episodes | 200 episodes |
| Target Rates | Fixed 15% | Variable (20%-75%) |
| Knowledge Transfer | Limited | Enhanced |
| Board Size Changes | Abrupt | Gradual |

## Best Practices

1. **Start with Minimal Test**: Always run the minimal test first to verify setup
2. **Monitor Progress**: Check results after each stage
3. **Adjust if Needed**: Modify targets or timesteps based on performance
4. **Save Models**: Each stage saves a model for potential reuse
5. **Analyze Results**: Review detailed metrics in the JSON output files

## Troubleshooting

### Common Issues

1. **Low Win Rates**: Increase training timesteps or lower targets
2. **Poor Transfer**: Extend training on intermediate stages
3. **Memory Issues**: Reduce batch size or use CPU instead of GPU
4. **Slow Progress**: Check hyperparameters and learning rate

### Performance Tips

1. **Use M1 Mac**: GPU acceleration for faster training
2. **Monitor Resources**: Check CPU/memory usage during long runs
3. **Save Checkpoints**: Models are saved after each stage
4. **Analyze Trends**: Look for patterns in win rates across stages

## Future Enhancements

1. **Adaptive Curriculum**: Automatically adjust difficulty based on performance
2. **Multi-Board Training**: Train on multiple board sizes simultaneously
3. **Advanced Metrics**: Add efficiency and strategy analysis
4. **Hyperparameter Tuning**: Optimize parameters for each stage
5. **Ensemble Methods**: Combine multiple models for better performance 

### Comparison with Variable Mine and Mixed Mine Approaches

**Variable Mine Training:**
- Trains on a range of mine counts in sequence (e.g., 1-5 mines)
- Improves generalization but can suffer from catastrophic forgetting when the agent is exposed to much harder scenarios

**Mixed Mine Training with Experience Replay:**
- Trains on multiple mine counts simultaneously in a mixed environment
- Uses experience replay (or mixed sampling) to ensure the agent retains skills across all difficulties
- Effectively mitigates catastrophic forgetting, maintaining high performance on easier tasks while learning harder ones

**Key Insights:**
- Catastrophic forgetting is a major challenge in curriculum learning for RL
- Mixed training and experience replay are effective solutions
- The project now includes scripts for both variable mine and mixed mine curriculum, enabling robust comparison and further research

**Recent Results:**
- Variable mine training improved transfer but regressed on easier tasks at higher difficulty
- Mixed mine training maintained performance across all mine counts, showing better generalization and stability 