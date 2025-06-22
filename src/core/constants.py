"""
Constants for the Minesweeper environment.

Recent Updates (2024-12-19):
- Simplified reward system: Immediate rewards for all actions (no first-move special cases)
- Enhanced state representation: 4-channel state for better pattern recognition
- Cross-platform compatibility: Platform-specific requirements and test adaptations
- M1 GPU optimization: Automatic device detection and performance benchmarking

Key Changes:
- Removed confusing first-move neutral rewards
- Implemented immediate positive/negative feedback for all actions
- Added 4-channel state representation (game state, safety hints, revealed count, progress)
- Enhanced action masking for better learning guidance
"""

# Cell states
CELL_UNREVEALED = -1
CELL_MINE = -2
CELL_MINE_HIT = -4

# Enhanced state representation constants
UNKNOWN_SAFETY = -1     # Value for unknown safety in safety channel

# Reward values - Optimized for winning
REWARD_FIRST_CASCADE_SAFE = 0     # Pre-cascade safe reveal (neutral - no punishment for guessing)
REWARD_FIRST_CASCADE_HIT_MINE = 0  # Pre-cascade hit mine (neutral - no punishment for bad luck)
REWARD_SAFE_REVEAL = 15           # Regular safe reveal (after cascade - normal learning signal)
REWARD_WIN = 500                  # Win the game (massive reward to encourage winning)
REWARD_HIT_MINE = -20             # Hit a mine (after cascade - strategic mistake penalty)
REWARD_INVALID_ACTION = -3        # Invalid action penalty (increased to discourage)

# Difficulty level constants
DIFFICULTY_LEVELS = {
    'easy': {'size': 9, 'mines': 10},
    'normal': {'size': 16, 'mines': 40},
    'hard': {'size': (16, 30), 'mines': 99},
    'expert': {'size': (18, 24), 'mines': 115},
    'chaotic': {'size': (20, 35), 'mines': 130}
} 