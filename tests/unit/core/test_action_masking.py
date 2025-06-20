import pytest
import numpy as np
from src.core.minesweeper_env import MinesweeperEnv
from src.core.constants import (
    CELL_UNREVEALED,
    CELL_MINE_HIT,
    REWARD_INVALID_ACTION
)

@pytest.fixture
def env():
    """Create a test environment."""
    return MinesweeperEnv(initial_board_size=3, initial_mines=1)

def test_action_masking_initial(env):
    """Test that all actions are initially valid."""
    env.reset()
    masks = env.action_masks
    assert np.all(masks), "All actions should be valid initially"
    assert np.sum(masks) == env.current_board_width * env.current_board_height

def test_action_masking_after_reveal(env):
    """Test that revealed cells are masked and valid actions match unrevealed cells."""
    env.reset()
    
    # Set up controlled board to avoid cascades
    env.mines.fill(False)
    env.mines[1, 1] = True  # Mine in center
    env._update_adjacent_counts()
    env.mines_placed = True
    env.is_first_cascade = False
    env.first_cascade_done = True
    
    # Reveal a cell (should not cause cascade)
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # Check that the revealed cell is masked
    masks = env.action_masks
    assert not masks[action], "Revealed cell should be masked"
    
    # The number of valid actions should match the number of unrevealed cells
    # (mines are valid actions since they can be revealed, resulting in mine hit)
    unrevealed_cells = np.sum(~env.revealed)
    assert np.sum(masks) == unrevealed_cells, (
        f"Expected {unrevealed_cells} valid actions, got {np.sum(masks)}."
    )

def test_action_masking_after_game_over(env):
    """Test that all actions are masked after game over."""
    env.reset()
    # Place mine at (0,0) and hit it
    env.mines[0, 0] = True
    env._update_adjacent_counts()
    env.mines_placed = True
    env.is_first_cascade = False
    env.first_cascade_done = True
    
    # Hit the mine
    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # All actions should be masked after game over
    masks = env.action_masks
    assert np.all(~masks), "All actions should be masked after game over"

def test_action_masking_after_win(env):
    """Test that all actions are masked after winning."""
    env.reset()
    # Set up a simple win scenario: mine at corner, reveal all others
    env.mines.fill(False)
    env.mines[0, 0] = True  # Mine at corner
    env._update_adjacent_counts()
    env.mines_placed = True
    env.is_first_cascade = False
    env.first_cascade_done = True
    
    # Reveal all safe cells
    for i in range(1, env.current_board_width * env.current_board_height):
        state, reward, terminated, truncated, info = env.step(i)
        if terminated:
            break
    
    # All actions should be masked after win
    masks = env.action_masks
    assert np.all(~masks), "All actions should be masked after win"

def test_action_masking_invalid_actions(env):
    """Test that invalid actions are properly masked."""
    env.reset()
    
    # Test out of bounds actions
    invalid_actions = [-1, env.action_space.n, env.action_space.n + 1]
    for action in invalid_actions:
        # These should not be in the action space
        assert not env.action_space.contains(action)

def test_action_masking_consistency(env):
    """Test that action masking is consistent across multiple actions."""
    env.reset()
    
    # Set up controlled board to avoid large cascades
    env.mines.fill(False)
    env.mines[1, 1] = True  # Mine in center
    env._update_adjacent_counts()
    env.mines_placed = True
    env.is_first_cascade = False
    env.first_cascade_done = True
    
    # Take multiple actions and verify masking consistency
    actions = [0, 1, 2]
    for action in actions:
        # Check if action is still valid before taking it
        if not env.action_masks[action]:
            print(f"Action {action} is already masked, skipping")
            continue
            
        state, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break
        
        masks = env.action_masks
        # All taken actions should be masked
        for taken_action in actions[:actions.index(action) + 1]:
            if taken_action < len(masks):
                assert not masks[taken_action], f"Action {taken_action} should be masked"
        
        # Remaining actions should be valid if they're still unrevealed
        remaining_actions = [i for i in range(env.action_space.n) if i not in actions[:actions.index(action) + 1]]
        for remaining_action in remaining_actions:
            if remaining_action < len(masks):
                # Only check if the cell is still unrevealed
                row = remaining_action // env.current_board_width
                col = remaining_action % env.current_board_width
                if not env.revealed[row, col]:
                    assert masks[remaining_action], f"Action {remaining_action} should be valid if unrevealed" 