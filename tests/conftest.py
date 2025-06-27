import warnings
import pytest
import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

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

def pytest_configure(config):
    # Suppress pkg_resources deprecation warning from pygame
    warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*deprecated.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*slated for removal.*")

@pytest.fixture
def env():
    """Create a test environment."""
    return MinesweeperEnv(
        max_board_size=4,
        max_mines=2,
        initial_board_size=4,
        initial_mines=2,
        mine_spacing=1
    ) 