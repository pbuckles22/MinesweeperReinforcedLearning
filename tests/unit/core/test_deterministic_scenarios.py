"""
Deterministic Scenario Tests for Minesweeper RL Environment

These tests use explicit board setup to create deterministic scenarios
for testing specific game situations. This allows for predictable,
repeatable tests while keeping the RL agent's gameplay realistic.
"""

import pytest
import numpy as np
from src.core.minesweeper_env import MinesweeperEnv
from src.core.constants import (
    CELL_UNREVEALED, CELL_MINE_HIT,
    REWARD_WIN, REWARD_HIT_MINE, REWARD_SAFE_REVEAL,
    REWARD_FIRST_MOVE_SAFE, REWARD_FIRST_MOVE_HIT_MINE
)

class TestDeterministicScenarios:
    """Test deterministic board scenarios for predictable outcomes."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        self.env = MinesweeperEnv(
            max_board_size=4,
            max_mines=2,
            initial_board_size=4,
            initial_mines=2
        )
    
    def test_deterministic_safe_corner_start(self):
        """Test deterministic scenario: safe corner start with known mine locations."""
        print("🧪 Testing deterministic safe corner start...")
        
        # Set up deterministic board: mines at (1,1) and (2,2), safe start at (0,0)
        self.env.reset()
        self.env.mines.fill(False)
        self.env.mines[1, 1] = True  # Mine at center
        self.env.mines[2, 2] = True  # Mine at bottom-right
        self.env._update_adjacent_counts()
        self.env.mines_placed = True
        self.env.is_first_move = True
        self.env.first_move_done = False
        
        # Take first action at safe corner (0,0)
        action = 0  # (0,0)
        state, reward, terminated, truncated, info = self.env.step(action)
        
        # Debug print for diagnosis
        print(f"Top row after reveal: {state[0, 0, :].tolist()}")
        print(f"Actual value at (0,1): {state[0, 0, 1]}")
        
        # Verify deterministic outcome
        assert not terminated, "Safe corner start should not terminate"
        assert reward == REWARD_FIRST_MOVE_SAFE, f"Should get first move safe reward, got {reward}"
        assert state[0, 0, 0] == 1, "Corner cell should show 1 adjacent mine"
        assert state[0, 0, 1] == -1, "Cell (0,1) should remain unrevealed after revealing (0,0)"
        assert state[0, 1, 0] == -1, "Cell (1,0) should remain unrevealed after revealing (0,0)"
        
        print("✅ Deterministic safe corner start passed")
    
    def test_deterministic_mine_hit_scenario(self):
        """Test deterministic scenario: hitting a mine at known location."""
        print("🧪 Testing deterministic mine hit scenario...")
        
        # Set up deterministic board: mine at (1,1), safe start at (0,0)
        self.env.reset()
        self.env.mines.fill(False)
        self.env.mines[1, 1] = True  # Mine at center
        self.env._update_adjacent_counts()
        self.env.mines_placed = True
        self.env.is_first_move = False
        self.env.first_move_done = True
        
        # First make a safe move
        safe_action = 0  # (0,0)
        state, reward, terminated, truncated, info = self.env.step(safe_action)
        assert not terminated, "Safe move should not terminate"
        
        # Now hit the mine
        mine_action = 1 * 4 + 1  # (1,1)
        state, reward, terminated, truncated, info = self.env.step(mine_action)
        
        # Verify deterministic outcome
        assert terminated, "Hitting mine should terminate game"
        assert reward == REWARD_HIT_MINE, f"Should get mine hit penalty, got {reward}"
        assert state[0, 1, 1] == CELL_MINE_HIT, "Hit cell should show mine hit"
        assert not info.get('won', False), "Game should not be marked as won"
        
        print("✅ Deterministic mine hit scenario passed")
    
    def test_deterministic_win_scenario(self):
        """Test deterministic scenario: winning game with known safe path."""
        print("🧪 Testing deterministic win scenario...")
        
        # Set up deterministic board: mines at corners, safe path in center
        self.env.reset()
        self.env.mines.fill(False)
        self.env.mines[0, 0] = True  # Mine at top-left
        self.env.mines[0, 3] = True  # Mine at top-right
        self.env.mines[3, 0] = True  # Mine at bottom-left
        self.env.mines[3, 3] = True  # Mine at bottom-right
        self.env._update_adjacent_counts()
        self.env.mines_placed = True
        self.env.is_first_move = False
        self.env.first_move_done = True
        
        # Reveal all non-corner cells
        safe_cells = [(i, j) for i in range(4) for j in range(4)
                      if (i, j) not in [(0,0), (0,3), (3,0), (3,3)]]
        for row, col in safe_cells:
            action = row * 4 + col
            state, reward, terminated, truncated, info = self.env.step(action)
            if terminated:
                break
        
        # Verify deterministic outcome
        assert terminated, "Game should terminate after revealing all safe cells"
        assert info.get('won', False), "Game should be marked as won"
        assert reward == REWARD_WIN, f"Should get win reward, got {reward}"
        
        # Verify all safe cells are revealed
        for row, col in safe_cells:
            assert state[0, row, col] >= 0, f"Safe cell ({row},{col}) should be revealed"
        
        print("✅ Deterministic win scenario passed")
    
    def test_deterministic_first_move_mine_hit(self):
        """Test deterministic scenario: hitting mine on first move."""
        print("🧪 Testing deterministic first move mine hit...")
        
        # Set up deterministic board: mine at (0,0)
        self.env.reset()
        self.env.mines.fill(False)
        self.env.mines[0, 0] = True  # Mine at first cell
        self.env._update_adjacent_counts()
        self.env.mines_placed = True
        self.env.is_first_move = True
        self.env.first_move_done = False
        
        # Hit mine on first move
        action = 0  # (0,0)
        state, reward, terminated, truncated, info = self.env.step(action)
        
        # Verify deterministic outcome
        assert not terminated, "First move mine hit should reset, not terminate"
        assert reward == REWARD_FIRST_MOVE_HIT_MINE, f"Should get first move mine hit reward, got {reward}"
        assert np.all(state[0] == CELL_UNREVEALED), "Board should be reset to unrevealed"
        
        print("✅ Deterministic first move mine hit passed")
    
    def test_deterministic_adjacent_mine_counts(self):
        """Test deterministic scenario: verify adjacent mine counts are correct."""
        print("🧪 Testing deterministic adjacent mine counts...")
        
        # Set up deterministic board: mines in specific pattern
        self.env.reset()
        self.env.mines.fill(False)
        self.env.mines[1, 1] = True  # Mine at center
        self.env.mines[1, 2] = True  # Mine at center-right
        self.env._update_adjacent_counts()
        self.env.mines_placed = True
        self.env.is_first_move = False
        self.env.first_move_done = True
        
        # Reveal cells and verify adjacent counts
        test_cells = [
            (0, 0, 1),  # Top-left should have 1 adjacent mine
            (0, 1, 2),  # Top-center should have 2 adjacent mines
            (0, 2, 2),  # Top-right should have 2 adjacent mines
            (1, 0, 1),  # Center-left should have 1 adjacent mine
            (2, 0, 1),  # Bottom-left should have 1 adjacent mine
            (2, 1, 2),  # Bottom-center should have 2 adjacent mines
            (2, 2, 2),  # Bottom-right should have 2 adjacent mines
        ]
        
        for row, col, expected_count in test_cells:
            action = row * 4 + col
            state, reward, terminated, truncated, info = self.env.step(action)
            
            if not terminated:
                actual_count = state[0, row, col]
                assert actual_count == expected_count, f"Cell ({row},{col}) should have {expected_count} adjacent mines, got {actual_count}"
        
        print("✅ Deterministic adjacent mine counts passed")
    
    def test_deterministic_safety_hints(self):
        """Test deterministic scenario: verify safety hints channel works correctly."""
        print("🧪 Testing deterministic safety hints...")
        
        # Set up deterministic board: mine at (1,1)
        self.env.reset()
        self.env.mines.fill(False)
        self.env.mines[1, 1] = True  # Mine at center
        self.env._update_adjacent_counts()
        self.env.mines_placed = True
        self.env.is_first_move = False
        self.env.first_move_done = True
        self.env._update_enhanced_state()  # Ensure state is updated
        
        # Check initial safety hints for unrevealed cells
        initial_state = self.env.state
        safety_hints = initial_state[1]
        
        # Verify safety hints show adjacent mine counts for unrevealed cells
        assert safety_hints[0, 0] == 1, "Corner cell should show 1 adjacent mine in safety hints"
        assert safety_hints[0, 1] == 1, "Edge cell should show 1 adjacent mine in safety hints"
        assert safety_hints[1, 0] == 1, "Edge cell should show 1 adjacent mine in safety hints"
        assert safety_hints[2, 2] == 1, "Corner cell should show 1 adjacent mine in safety hints"
        
        # Reveal a cell and verify safety hints update
        action = 0  # (0,0)
        state, reward, terminated, truncated, info = self.env.step(action)
        
        new_safety_hints = state[1]
        # Revealed cell should show -1 (unknown) in safety hints
        assert new_safety_hints[0, 0] == -1, "Revealed cell should show -1 in safety hints"
        
        print("✅ Deterministic safety hints passed")
    
    def test_deterministic_action_masking(self):
        """Test deterministic scenario: verify action masking works correctly."""
        print("🧪 Testing deterministic action masking...")
        
        # Set up deterministic board: mine at (1,1)
        self.env.reset()
        self.env.mines.fill(False)
        self.env.mines[1, 1] = True  # Mine at center
        self.env._update_adjacent_counts()
        self.env.mines_placed = True
        self.env.is_first_move = False
        self.env.first_move_done = True
        
        # Check initial action masks
        initial_masks = self.env.action_masks
        assert np.sum(initial_masks) == 16, "All 16 cells should be valid actions initially"
        
        # Reveal a cell
        action = 0  # (0,0)
        state, reward, terminated, truncated, info = self.env.step(action)
        
        # Check action masks after reveal
        new_masks = self.env.action_masks
        assert not new_masks[0], "Revealed cell should be masked"
        assert np.sum(new_masks) == 15, "15 cells should remain valid actions"
        
        print("✅ Deterministic action masking passed")
    
    def test_deterministic_state_consistency(self):
        """Test deterministic scenario: verify state consistency across actions."""
        print("🧪 Testing deterministic state consistency...")
        
        # Set up deterministic board: mine at (1,1)
        self.env.reset()
        self.env.mines.fill(False)
        self.env.mines[1, 1] = True  # Mine at center
        self.env._update_adjacent_counts()
        self.env.mines_placed = True
        self.env.is_first_move = False
        self.env.first_move_done = True
        
        # Take multiple actions and verify state consistency
        actions = [0, 1, 2, 3]  # Reveal top row
        previous_state = None
        
        for i, action in enumerate(actions):
            state, reward, terminated, truncated, info = self.env.step(action)
            
            # Verify state shape is consistent
            assert state.shape == (2, 4, 4), f"State shape should be (2, 4, 4), got {state.shape}"
            
            # Verify state changed from previous
            if previous_state is not None:
                assert not np.array_equal(state, previous_state), f"State should change after action {action}"
            
            # Verify observation space bounds
            assert self.env.observation_space.contains(state), f"State should be within observation space bounds"
            
            previous_state = state.copy()
            
            if terminated:
                break
        
        print("✅ Deterministic state consistency passed")
    
    def test_deterministic_reward_consistency(self):
        """Test deterministic scenario: verify reward consistency for same actions."""
        print("🧪 Testing deterministic reward consistency...")
        
        # Set up deterministic board: mine at (1,1)
        self.env.reset()
        self.env.mines.fill(False)
        self.env.mines[1, 1] = True  # Mine at center
        self.env._update_adjacent_counts()
        self.env.mines_placed = True
        self.env.is_first_move = False
        self.env.first_move_done = True
        
        # Take same action multiple times and verify consistent rewards
        action = 0  # (0,0)
        
        # First action should give safe reveal reward
        state1, reward1, terminated1, truncated1, info1 = self.env.step(action)
        assert reward1 == REWARD_SAFE_REVEAL, f"First action should give safe reveal reward, got {reward1}"
        
        # Reset and take same action again
        self.env.reset()
        self.env.mines.fill(False)
        self.env.mines[1, 1] = True
        self.env._update_adjacent_counts()
        self.env.mines_placed = True
        self.env.is_first_move = False
        self.env.first_move_done = True
        
        state2, reward2, terminated2, truncated2, info2 = self.env.step(action)
        assert reward2 == REWARD_SAFE_REVEAL, f"Second action should give same reward, got {reward2}"
        assert reward1 == reward2, "Same action should give same reward"
        
        print("✅ Deterministic reward consistency passed")

    def test_deterministic_environment_consistency(self):
        """Test that the environment behaves consistently across multiple runs."""
        print("🧪 Testing deterministic environment consistency...")
        
        # Test 1: Same seed produces same results
        env1 = MinesweeperEnv(max_board_size=4, max_mines=2)
        env2 = MinesweeperEnv(max_board_size=4, max_mines=2)
        
        state1, info1 = env1.reset(seed=42)
        state2, info2 = env2.reset(seed=42)
        
        assert np.array_equal(state1, state2), "Same seed should produce same state"
        print("✅ Same seed produces same state")
        
        # Test 2: Same action produces same result (only for first action with same seed)
        action = 0
        state1, reward1, terminated1, truncated1, info1 = env1.step(action)
        state2, reward2, terminated2, truncated2, info2 = env2.step(action)

        assert np.array_equal(state1, state2), "Same action should produce same state with same seed"
        assert reward1 == reward2, "Same action should produce same reward with same seed"
        assert terminated1 == terminated2, "Same action should produce same termination with same seed"
        print("✅ Same action produces same result with same seed")
        
        # Test 3: Deterministic board setup produces consistent results
        env3 = MinesweeperEnv(max_board_size=4, max_mines=2)
        env3.reset()
        
        # Set up deterministic board
        env3.mines.fill(False)
        env3.mines[1, 1] = True
        env3._update_adjacent_counts()
        env3.mines_placed = True
        env3.is_first_move = False
        env3.first_move_done = True
        
        # Take same action multiple times
        action = 0
        results = []
        
        for i in range(3):
            env3.reset()
            env3.mines.fill(False)
            env3.mines[1, 1] = True
            env3._update_adjacent_counts()
            env3.mines_placed = True
            env3.is_first_move = False
            env3.first_move_done = True
            
            state, reward, terminated, truncated, info = env3.step(action)
            results.append((reward, terminated))
        
        # All results should be identical
        for i in range(1, len(results)):
            assert results[i] == results[0], f"Result {i} should match result 0"
        
        print("✅ Deterministic board setup produces consistent results")
        print("✅ Environment consistency tests passed!")

if __name__ == "__main__":
    # Run all deterministic tests
    test_suite = TestDeterministicScenarios()
    test_suite.setup_method()
    
    test_methods = [
        test_suite.test_deterministic_safe_corner_start,
        test_suite.test_deterministic_mine_hit_scenario,
        test_suite.test_deterministic_win_scenario,
        test_suite.test_deterministic_first_move_mine_hit,
        test_suite.test_deterministic_adjacent_mine_counts,
        test_suite.test_deterministic_safety_hints,
        test_suite.test_deterministic_action_masking,
        test_suite.test_deterministic_state_consistency,
        test_suite.test_deterministic_reward_consistency,
        test_suite.test_deterministic_environment_consistency,
    ]
    
    print("🧪 Running Deterministic Scenario Tests...")
    for test_method in test_methods:
        try:
            test_method()
            print(f"✅ {test_method.__name__} passed")
        except Exception as e:
            print(f"❌ {test_method.__name__} failed: {e}")
    
    print("🎉 All deterministic scenario tests completed!") 