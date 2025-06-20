import pytest
import numpy as np
from src.core.minesweeper_env import MinesweeperEnv
from src.core.constants import (
    CELL_UNREVEALED,
    CELL_MINE,
    CELL_MINE_HIT,
    REWARD_FIRST_CASCADE_SAFE, REWARD_FIRST_CASCADE_HIT_MINE,
    REWARD_SAFE_REVEAL,
    REWARD_WIN,
    REWARD_HIT_MINE
)

@pytest.fixture
def env():
    """Create a test environment."""
    return MinesweeperEnv(
        max_board_size=(10, 10),
        max_mines=10,
        initial_board_size=(4, 4),
        initial_mines=2
    ) 