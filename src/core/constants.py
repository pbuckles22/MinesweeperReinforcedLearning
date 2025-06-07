"""
Constants for the Minesweeper environment.
"""

# Cell states
CELL_UNREVEALED = -1
CELL_MINE = -2
CELL_FLAGGED = -3
CELL_MINE_HIT = -4

# Reward values
REWARD_FIRST_MOVE_SAFE = 0    # First move safe reveal (just luck)
REWARD_FIRST_MOVE_HIT_MINE = 0  # First move hit mine (just luck)
REWARD_SAFE_REVEAL = 5        # Regular safe reveal (progress)
REWARD_WIN = 100             # Win the game (achievement)
REWARD_HIT_MINE = -50        # Hit a mine (failure) - significant penalty to encourage caution
REWARD_FLAG_MINE = 0         # No reward for flagging a mine
REWARD_FLAG_SAFE = 0         # No reward for flagging a safe cell
REWARD_UNFLAG = 0           # No reward for unflagging
REWARD_INVALID_ACTION = -10   # Invalid action penalty
REWARD_FLAG_PLACED = 0       # No reward for placing flags to prevent reward hacking
REWARD_FLAG_REMOVED = 0      # No reward for removing flags to prevent reward hacking

# Difficulty level constants
DIFFICULTY_LEVELS = {
    'easy': {'size': 9, 'mines': 10},
    'normal': {'size': 16, 'mines': 40},
    'hard': {'size': (16, 30), 'mines': 99},
    'expert': {'size': (18, 24), 'mines': 115},
    'chaotic': {'size': (20, 35), 'mines': 130}
} 