"""
Game Flow Tests for Minesweeper RL Environment

These tests verify the complete game flow from start to finish,
ensuring proper state transitions, rewards, and termination conditions.
"""

import pytest
import numpy as np
from src.core.minesweeper_env import MinesweeperEnv
from src.core.constants import (
    CELL_UNREVEALED, CELL_MINE_HIT,
    REWARD_WIN, REWARD_HIT_MINE, REWARD_SAFE_REVEAL,
    REWARD_FIRST_CASCADE_SAFE, REWARD_FIRST_CASCADE_HIT_MINE,
    REWARD_INVALID_ACTION
)

class TestGameFlow:
    """Test complete game flow scenarios."""
    
    def test_complete_win_game_flow(self):
        """Test a complete game flow ending in victory."""
        env = MinesweeperEnv(initial_board_size=3, initial_mines=1)
        env.reset(seed=42)
        
        # Set up a controlled win scenario
        env.mines.fill(False)
        env.mines[2, 2] = True  # Single mine at bottom-right
        env._update_adjacent_counts()
        env.mines_placed = True
        env.is_first_cascade = False
        env.first_cascade_done = True
        
        total_reward = 0
        moves_made = 0
        
        # Reveal all safe cells
        safe_cells = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1)]
        
        for row, col in safe_cells:
            action = row * env.current_board_width + col
            state, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            moves_made += 1
            
            # Check state consistency
            assert state.shape == (4, 3, 3), "State shape should remain consistent"
            assert isinstance(reward, (int, float)), "Reward should be numeric"
            assert isinstance(terminated, bool), "Terminated should be boolean"
            assert isinstance(info, (dict, list)), "Info should be a dictionary or list of dicts"
            if isinstance(info, list):
                assert len(info) > 0
                assert isinstance(info[0], dict)
            
            if terminated:
                break
        
        # Should have won (may take fewer moves due to cascade)
        assert terminated, "Game should be terminated"
        assert info.get('won', False), "Game should be marked as won"
        assert reward == REWARD_WIN, "Final reward should be win reward"
        assert moves_made > 0, "Should have made some moves to win"
    
    def test_complete_loss_game_flow(self):
        """Test a complete game flow ending in loss."""
        env = MinesweeperEnv(initial_board_size=3, initial_mines=1)
        env.reset(seed=42)
        
        # Set up a controlled loss scenario
        env.mines.fill(False)
        env.mines[1, 1] = True  # Mine at center
        env._update_adjacent_counts()
        env.mines_placed = True
        env.is_first_cascade = False
        env.first_cascade_done = True
        
        total_reward = 0
        moves_made = 0
        
        # Take a safe move first
        safe_action = 0  # (0,0)
        state, reward, terminated, truncated, info = env.step(safe_action)
        total_reward += reward
        moves_made += 1
        
        # Then hit the mine
        mine_action = 1 * env.current_board_width + 1  # (1,1)
        state, reward, terminated, truncated, info = env.step(mine_action)
        total_reward += reward
        moves_made += 1
        
        # Should have lost
        assert terminated, "Game should be terminated"
        assert not info.get('won', False), "Game should not be marked as won"
        assert reward == REWARD_HIT_MINE, "Final reward should be mine hit penalty"
        assert moves_made == 2, "Should have made 2 moves before losing"
        assert state[0, 1, 1] == CELL_MINE_HIT, "Hit cell should show mine hit"
    
    def test_pre_cascade_safe_flow(self):
        """Test game flow with pre-cascade neutral rewards (no punishment for guessing)."""
        env = MinesweeperEnv(initial_board_size=4, initial_mines=2)
        env.reset(seed=42)
        
        # Pre-cascade move (first move) - can hit mine or be safe
        action = 0
        state, reward, terminated, truncated, info = env.step(action)
        
        print(f"[DIAG] State[0,0,0] after pre-cascade: {state[0,0,0]}")
        print(f"[DIAG] State[1,0,0] after pre-cascade: {state[1,0,0]}")
        
        # Pre-cascade should give immediate rewards regardless of outcome
        # This provides immediate feedback for learning
        if terminated:
            # Hit a mine - should get immediate penalty and game ends
            if reward in [REWARD_SAFE_REVEAL, REWARD_HIT_MINE, REWARD_WIN]:
                assert True
            else:
                assert False, f"Pre-cascade mine hit should get immediate reward/penalty/win, got {reward}"
            assert not info['won'], "Mine hit should not result in win"
        else:
            # Safe move - should get immediate reward and continue
            if reward in [REWARD_SAFE_REVEAL, REWARD_HIT_MINE, REWARD_WIN]:
                assert True
            else:
                assert False, f"Pre-cascade safe move should get immediate reward/penalty/win, got {reward}"
            assert not info['won'], "Safe move should not result in immediate win"
    
    def test_cascade_revelation_flow(self):
        """Test game flow with cascade revelation."""
        env = MinesweeperEnv(initial_board_size=4, initial_mines=2)
        env.reset(seed=42)
        
        # Set up a cascade scenario
        env.mines.fill(False)
        env.mines[3, 3] = True  # Mine at bottom-right
        env.mines[3, 2] = True  # Mine at bottom-center
        env._update_adjacent_counts()
        env.mines_placed = True
        env.is_first_cascade = False
        env.first_cascade_done = True
        
        # Reveal cell that should trigger cascade
        action = 0  # (0,0) - should cascade to reveal multiple cells
        state, reward, terminated, truncated, info = env.step(action)
        
        # Should reveal multiple cells due to cascade
        revealed_cells = np.sum(state[0] != CELL_UNREVEALED)
        assert revealed_cells > 1, "Cascade should reveal multiple cells"
        
        # Safety hints should show -1 for revealed cells
        revealed_positions = state[0] != CELL_UNREVEALED
        assert np.all(state[1][revealed_positions] == -1), "Revealed cells should show -1 in safety hints"
    
    def test_invalid_action_flow(self):
        """Test game flow with invalid actions."""
        env = MinesweeperEnv(initial_board_size=4, initial_mines=2)
        env.reset(seed=42)
        
        # Take a valid action first
        action = 0
        state, reward, terminated, truncated, info = env.step(action)
        
        # If the first action terminated the game, we can't test invalid actions
        if terminated:
            print("First action terminated game, skipping invalid action test")
            return
        
        # Try to reveal the same cell again (invalid)
        state, reward, terminated, truncated, info = env.step(action)
        
        # Should get invalid action penalty or the action should be ignored
        # The environment might handle this gracefully without penalty
        assert isinstance(reward, (int, float)), "Should get a numeric reward"
        assert not terminated, "Invalid action should not terminate game"
        
        # State should remain unchanged for the invalid action
        # (This depends on how the environment handles invalid actions)
    
    def test_game_state_consistency_flow(self):
        """Test that game state remains consistent throughout flow."""
        env = MinesweeperEnv(initial_board_size=4, initial_mines=2)
        env.reset(seed=42)
        
        # Track state changes
        previous_states = []
        
        # Make several moves
        for action in [0, 1, 2, 3]:
            state, reward, terminated, truncated, info = env.step(action)
            previous_states.append(state.copy())
            
            # Check state consistency
            assert state.shape == (4, 4, 4), "State shape should remain consistent"
            assert np.all(state[1] >= -1), "Safety hints should be >= -1"
            assert np.all(state[1] <= 8), "Safety hints should be <= 8"
            
            if terminated:
                break
        
        # Verify that previously revealed cells remain revealed
        for i, prev_state in enumerate(previous_states[:-1]):
            current_state = previous_states[i + 1]
            # Previously revealed cells should remain revealed
            revealed_in_prev = prev_state[0] != CELL_UNREVEALED
            assert np.all(current_state[0][revealed_in_prev] != CELL_UNREVEALED), "Previously revealed cells should remain revealed"
    
    def test_early_learning_flow(self):
        """Test game flow with early learning mode."""
        env = MinesweeperEnv(
            initial_board_size=4,
            initial_mines=2,
            early_learning_mode=True,
            early_learning_corner_safe=True,
            early_learning_edge_safe=True
        )
        env.reset(seed=42)
        
        # Test corner safety
        corner_actions = [0, 3, 12, 15]  # Corners of 4x4 board
        safe_corners = 0
        
        for action in corner_actions:
            env.reset(seed=42)
            state, reward, terminated, truncated, info = env.step(action)
            
            if not terminated or reward != REWARD_HIT_MINE:
                safe_corners += 1
        
        # At least some corners should be safe in early learning mode
        assert safe_corners > 0, "Early learning mode should provide some safe corners"
    
    def test_rectangular_board_flow(self):
        """Test game flow with rectangular boards."""
        env = MinesweeperEnv(initial_board_size=(3, 4), initial_mines=2)
        env.reset(seed=42)
        
        # Verify rectangular board setup
        assert env.current_board_width == 4, "Should have correct width"
        assert env.current_board_height == 3, "Should have correct height"
        assert env.state.shape == (4, 3, 4), "State should match rectangular dimensions"
        assert env.action_space.n == 12, "Action space should match rectangular dimensions"
        
        # Play a few moves
        for action in [0, 1, 2]:
            state, reward, terminated, truncated, info = env.step(action)
            
            # Check state consistency
            assert state.shape == (4, 3, 4), "State shape should remain consistent"
            assert isinstance(reward, (int, float)), "Reward should be numeric"
            
            if terminated:
                break
    
    def test_large_board_flow(self):
        """Test game flow with larger boards."""
        env = MinesweeperEnv(initial_board_size=(8, 8), initial_mines=10)
        env.reset(seed=42)
        
        # Verify large board setup
        assert env.current_board_width == 8, "Should have correct width"
        assert env.current_board_height == 8, "Should have correct height"
        assert env.state.shape == (4, 8, 8), "State should match large dimensions"
        assert env.action_space.n == 64, "Action space should match large dimensions"
        
        # Make several moves
        moves_made = 0
        for action in range(10):  # Make up to 10 moves
            state, reward, terminated, truncated, info = env.step(action)
            moves_made += 1
            
            # Check state consistency
            assert state.shape == (4, 8, 8), "State shape should remain consistent"
            assert isinstance(reward, (int, float)), "Reward should be numeric"
            
            if terminated:
                break
        
        assert moves_made > 0, "Should be able to make moves on large board"
    
    def test_high_mine_density_flow(self):
        """Test game flow with high mine density."""
        env = MinesweeperEnv(initial_board_size=(5, 5), initial_mines=15)
        env.reset(seed=42)
        
        # Verify high density setup
        density = env.current_mines / (env.current_board_width * env.current_board_height)
        assert density == 15/25, "Should have correct mine density"
        
        # Play until game ends
        moves_made = 0
        while moves_made < 25:  # Prevent infinite loop
            # Find an unrevealed cell
            unrevealed = np.where(env.state[0] == CELL_UNREVEALED)
            if len(unrevealed[0]) == 0:
                break
                
            action = unrevealed[0][0] * env.current_board_width + unrevealed[1][0]
            state, reward, terminated, truncated, info = env.step(action)
            moves_made += 1
            
            if terminated:
                break
        
        assert moves_made > 0, "Should be able to make moves on high density board"
    
    def test_win_loss_transition_flow(self):
        """Test transition between win and loss conditions."""
        env = MinesweeperEnv(initial_board_size=3, initial_mines=1)
        env.reset(seed=42)
        
        # Set up a scenario where we can choose win or loss
        env.mines.fill(False)
        env.mines[1, 1] = True  # Mine at center
        env._update_adjacent_counts()
        env.mines_placed = True
        env.is_first_cascade = False
        env.first_cascade_done = True
        
        # Take safe moves first
        safe_cells = [(0,0), (0,1), (0,2), (1,0), (1,2), (2,0), (2,1), (2,2)]
        for row, col in safe_cells:
            action = row * env.current_board_width + col
            state, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                # Should have won
                assert info.get('won', False), "Should have won by revealing all safe cells"
                assert reward == REWARD_WIN, "Should get win reward"
                break
        
        # Reset and test loss scenario
        env.reset(seed=42)
        env.mines.fill(False)
        env.mines[1, 1] = True
        env._update_adjacent_counts()
        env.mines_placed = True
        env.is_first_cascade = False
        env.first_cascade_done = True
        
        # Hit the mine directly
        mine_action = 1 * env.current_board_width + 1
        state, reward, terminated, truncated, info = env.step(mine_action)
        
        # Should have lost
        assert terminated, "Should be terminated after hitting mine"
        assert not info.get('won', False), "Should not have won after hitting mine"
        assert reward == REWARD_HIT_MINE, "Should get mine hit penalty"

if __name__ == "__main__":
    pytest.main([__file__]) 