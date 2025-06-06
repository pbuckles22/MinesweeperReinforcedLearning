import pytest
import numpy as np
from stable_baselines3 import PPO
from src.core.minesweeper_env import MinesweeperEnv
from src.core.vec_env import DummyVecEnv

@pytest.fixture
def env():
    """Create a test environment."""
    env_fn = lambda: MinesweeperEnv(
        max_board_size=4,
        max_mines=2,
        early_learning_mode=True
    )
    return DummyVecEnv([env_fn]) 