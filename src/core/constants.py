"""
Constants for the Minesweeper environment.
"""

# Cell states
CELL_UNREVEALED = -1
CELL_MINE = 9
CELL_FLAGGED = -3
CELL_MINE_HIT = -2

# Reward values
REWARD_FIRST_MOVE_SAFE = 0    # First move safe reveal (just luck)
REWARD_FIRST_MOVE_HIT_MINE = 0  # First move hit mine (just luck)
REWARD_SAFE_REVEAL = 1        # Regular safe reveal (progress)
REWARD_WIN = 100             # Win the game (achievement)
REWARD_HIT_MINE = -50        # Hit a mine (failure) - significant penalty to encourage caution 