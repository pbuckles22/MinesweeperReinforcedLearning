# Learnable Configuration Analysis

## Overview

This document explains how we determine which mine configurations are "learnable" (require strategic play) vs. "lucky" (can be won in 1 move via cascade).

## Key Insight

A configuration is **learnable** if it requires 2+ moves to win. A configuration is **lucky** if it can be won in exactly 1 move through a cascade that reveals the entire board except the mine.

## Exact Cascade-Based Logic

### Single-Mine Configurations

For single-mine boards of **any shape** (squares and rectangles), we determine learnability by:
1. Placing a mine at each possible position
2. Simulating a cascade from any non-mine cell
3. Checking if the cascade reveals exactly (total_cells - 1) cells

**Results from regression test (square boards):**

| Board Size | Total Positions | 1-Move Wins | Learnable | 1-Move Win % | Learnable % |
|------------|----------------|-------------|-----------|--------------|-------------|
| 4×4        | 16             | 4           | 12        | 25.0%        | 75.0%       |
| 5×5        | 25             | 9           | 16        | 36.0%        | 64.0%       |
| 6×6        | 36             | 15          | 21        | 41.7%        | 58.3%       |
| 8×8        | 64             | 35          | 29        | 54.7%        | 45.3%       |

**Note:** 
- For 4×4, all 4 corners allow 1-move wins: (0,0), (0,3), (3,0), (3,3)
- For 5×5, 9 positions allow 1-move wins: corners (0,0), (0,4), (4,0), (4,4) and edge centers (0,2), (2,0), (2,2), (2,4), (4,2)
- **The cascade logic works identically for rectangular boards** (e.g., 4×5, 5×6, etc.) - corners and strategic positions are identified the same way

### Multi-Mine Configurations

For multi-mine boards, the analysis is more complex because:
- Mines can block cascades
- Multiple mines create more strategic scenarios
- The interaction between mines affects learnability

**Current assumption:** All multi-mine configurations are considered learnable, as they typically require strategic reasoning about mine interactions.

## Implementation Strategy

### Environment Integration

The `MinesweeperEnv` class now includes:

1. **`learnable_only` parameter**: When `True`, only generates learnable configurations
2. **`max_learnable_attempts` parameter**: Maximum attempts to find a learnable configuration
3. **Cascade simulation methods**: To determine if a configuration is learnable

### Current Implementation

The environment uses a simplified geometric approach:
- **Single mine**: Filters out corner positions (assumes corners = lucky)
- **Multi mine**: Assumes all configurations are learnable

### Target Implementation

We should update the environment to use exact cascade simulation:

```python
def _is_learnable_configuration(self, mine_positions):
    """Check if mine placement creates a learnable scenario using cascade simulation."""
    if len(mine_positions) == 1:
        return self._is_single_mine_learnable_cascade(mine_positions[0])
    else:
        return self._is_multi_mine_learnable_cascade(mine_positions)

def _is_single_mine_learnable_cascade(self, mine_pos):
    """Check if single mine placement requires 2+ moves using cascade simulation."""
    # Create board with mine at position
    # Simulate cascade from non-mine cell
    # Check if cascade reveals (total_cells - 1) cells
    # Return False if 1-move win, True if learnable
```

## Training Implications

### Benefits of Learnable-Only Training

1. **Focused Learning**: Agents only train on scenarios that require strategic thinking
2. **Realistic Performance**: Win rates reflect actual strategic ability, not luck
3. **Better Generalization**: Agents learn patterns that apply to complex scenarios
4. **Efficient Training**: No time wasted on trivial 1-move wins

### Recommended Training Configurations

Based on the analysis:

1. **4×4 (1 mine)**: 75.0% learnable - Good for early learning
2. **5×5 (1 mine)**: 64.0% learnable - Moderate difficulty
3. **5×5 (2 mines)**: 100% learnable - Excellent for strategic training
4. **6×6 (1 mine)**: 58.3% learnable - Higher difficulty
5. **8×8 (1 mine)**: 45.3% learnable - Very high difficulty

### Curriculum Progression

Recommended progression for learnable-only training:

1. **Stage 1**: 4×4 (1 mine) - 75.0% learnable scenarios
2. **Stage 2**: 5×5 (1 mine) - 64.0% learnable scenarios  
3. **Stage 3**: 5×5 (2 mines) - 100% learnable scenarios
4. **Stage 4**: 6×6 (1 mine) - 58.3% learnable scenarios
5. **Stage 5**: 8×8 (1 mine) - 45.3% learnable scenarios

## Testing and Validation

### Test Scripts

- `scripts/analyze_one_move_wins.py`: Calculates exact percentages
- `scripts/test_learnable_configurations.py`: Tests environment filtering
- `scripts/debug_learnable_filtering.py`: Debugs filtering logic

### Expected Results

When `learnable_only=True`:
- 4×4 (1 mine): Should generate 100% learnable configurations
- 5×5 (1 mine): Should generate 100% learnable configurations
- 5×5 (2 mines): Should generate 100% learnable configurations

When `learnable_only=False`:
- 4×4 (1 mine): Should generate ~75.0% learnable configurations
- 5×5 (1 mine): Should generate ~64.0% learnable configurations
- 5×5 (2 mines): Should generate 100% learnable configurations

## Future Enhancements

1. **Multi-mine cascade analysis**: Implement exact cascade simulation for multi-mine boards
2. **Dynamic difficulty**: Adjust mine count based on board size to maintain optimal learnability
3. **Advanced filtering**: Consider mine spacing and interaction patterns
4. **Performance optimization**: Cache cascade results for common configurations

## Conclusion

The cascade-based approach provides the most accurate determination of learnable vs. lucky configurations. By implementing this logic in the environment, we ensure that RL agents train only on scenarios that require strategic thinking, leading to more meaningful learning and better performance on complex scenarios. 