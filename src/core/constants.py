"""
Constants for the Minesweeper environment.
"""

# Cell states
CELL_UNREVEALED = -1
CELL_MINE = -2
CELL_MINE_HIT = -4

# Enhanced state representation constants
MINE_INDICATOR = 1      # Value for mine locations in mine channel
SAFE_INDICATOR = 0      # Value for safe cells in mine channel
UNKNOWN_SAFETY = -1     # Value for unknown safety in safety channel

# Reward values
REWARD_FIRST_MOVE_SAFE = 0    # First move safe reveal (just luck)
REWARD_FIRST_MOVE_HIT_MINE = 0  # First move hit mine (just luck)
REWARD_SAFE_REVEAL = 5        # Regular safe reveal (progress)
REWARD_WIN = 100             # Win the game (achievement)
REWARD_HIT_MINE = -50        # Hit a mine (failure) - significant penalty to encourage caution
REWARD_INVALID_ACTION = -10   # Invalid action penalty

# Difficulty level constants
DIFFICULTY_LEVELS = {
    'easy': {'size': 9, 'mines': 10},
    'normal': {'size': 16, 'mines': 40},
    'hard': {'size': (16, 30), 'mines': 99},
    'expert': {'size': (18, 24), 'mines': 115},
    'chaotic': {'size': (20, 35), 'mines': 130}
} 