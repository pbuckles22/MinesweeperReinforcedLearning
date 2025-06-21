import pytest
import numpy as np
from src.core.minesweeper_env import MinesweeperEnv
from src.core.constants import (
    CELL_UNREVEALED,
    CELL_MINE_HIT,
    REWARD_FIRST_CASCADE_SAFE, REWARD_FIRST_CASCADE_HIT_MINE,
    REWARD_SAFE_REVEAL,
    REWARD_WIN,
    REWARD_HIT_MINE
)

@pytest.fixture
def env():
    """Create a test environment."""
    return MinesweeperEnv(initial_board_size=4, initial_mines=2)

class TestEdgeCases:
    """Test edge cases and complex scenarios for Minesweeper RL environment."""

    def setup_method(self):
        """Set up test environment before each test."""
        self.env = MinesweeperEnv(initial_board_size=4, initial_mines=2)

    def test_complex_cascade_scenarios(self):
        """Test complex cascade patterns that reveal large areas."""
        env = MinesweeperEnv(initial_board_size=6, initial_mines=3)
        env.reset()
        
        # Set up a complex board with L-shaped zero region
        env.mines.fill(False)
        # Place mines in a pattern that creates L-shaped zero region
        env.mines[0, 0] = True  # Top-left corner
        env.mines[5, 5] = True  # Bottom-right corner  
        env.mines[2, 2] = True  # Center
        env._update_adjacent_counts()
        env.mines_placed = True
        env.is_first_cascade = False
        env.first_cascade_done = True
        
        # Reveal cell that should trigger cascade (should be a zero)
        # Find a zero cell (should be at (1,1) or similar)
        zero_cell_found = False
        for i in range(env.current_board_height):
            for j in range(env.current_board_width):
                if env.board[i, j] == 0 and not env.mines[i, j]:
                    action = i * env.current_board_width + j
                    state, reward, terminated, truncated, info = env.step(action)
                    zero_cell_found = True
                    break
            if zero_cell_found:
                break
        
        assert zero_cell_found, "Should find a zero cell for cascade test"
        
        # Count revealed cells after cascade
        revealed_count = np.sum(env.revealed)
        assert revealed_count > 1, f"Cascade should reveal multiple cells, got {revealed_count}"

    def test_cascade_boundary_conditions(self):
        """Test cascade behavior at board edges and corners."""
        env = MinesweeperEnv(initial_board_size=4, initial_mines=1)
        env.reset()
        
        # Set up board with mine in center, create zero cells
        env.mines.fill(False)
        env.mines[1, 1] = True  # Center mine
        env._update_adjacent_counts()
        env.mines_placed = True
        env.is_first_cascade = False
        env.first_cascade_done = True
        
        # Find a zero cell to trigger cascade
        zero_cell_found = False
        for i in range(env.current_board_height):
            for j in range(env.current_board_width):
                if env.board[i, j] == 0 and not env.mines[i, j]:
                    action = i * env.current_board_width + j
                    state, reward, terminated, truncated, info = env.step(action)
                    zero_cell_found = True
                    break
            if zero_cell_found:
                break
        
        assert zero_cell_found, "Should find a zero cell for cascade test"
        
        # Should reveal multiple cells including edge cells
        revealed_count = np.sum(env.revealed)
        assert revealed_count > 1, f"Zero cell cascade should reveal multiple cells, got {revealed_count}"
        
        # Check that edge cells are properly revealed
        # Find some revealed cells and verify they're not mines
        revealed_cells = np.where(env.revealed)
        assert len(revealed_cells[0]) > 1, "Should have multiple revealed cells"

    def test_multiple_disconnected_zero_regions(self):
        """Test that revealing one zero region doesn't affect others."""
        env = MinesweeperEnv(initial_board_size=6, initial_mines=4)
        env.reset()
        
        # Set up board with multiple isolated zero regions
        env.mines.fill(False)
        # Create isolated zero regions by placing mines in a pattern that creates zero cells
        env.mines[0, 0] = True  # Corner mine
        env.mines[0, 5] = True  # Corner mine  
        env.mines[5, 0] = True  # Corner mine
        env.mines[5, 5] = True  # Corner mine
        env._update_adjacent_counts()
        env.mines_placed = True
        env.is_first_cascade = False
        env.first_cascade_done = True
        
        # Find and reveal first zero region
        first_zero_found = False
        for i in range(env.current_board_height):
            for j in range(env.current_board_width):
                if env.board[i, j] == 0 and not env.mines[i, j]:
                    action = i * env.current_board_width + j
                    state, reward, terminated, truncated, info = env.step(action)
                    first_zero_found = True
                    break
            if first_zero_found:
                break
        
        assert first_zero_found, "Should find a zero cell"
        
        # Count cells revealed by first cascade
        first_revealed = np.sum(env.revealed)
        
        # Find and reveal second zero region
        second_zero_found = False
        for i in range(env.current_board_height):
            for j in range(env.current_board_width):
                if env.board[i, j] == 0 and not env.mines[i, j] and not env.revealed[i, j]:
                    action = i * env.current_board_width + j
                    state, reward, terminated, truncated, info = env.step(action)
                    second_zero_found = True
                    break
            if second_zero_found:
                break
        
        if second_zero_found:
            # Should reveal additional cells
            second_revealed = np.sum(env.revealed)
            assert second_revealed > first_revealed, "Second cascade should reveal additional cells"

    def test_win_condition_edge_cases(self):
        """Test win conditions in edge cases."""
        env = MinesweeperEnv(initial_board_size=3, initial_mines=1)
        env.reset()
        
        # Set up board where all cells except one mine are safe
        env.mines.fill(False)
        env.mines[0, 0] = True  # Single mine in corner
        env._update_adjacent_counts()
        env.mines_placed = True
        env.is_first_cascade = False
        env.first_cascade_done = True
        
        # Reveal all safe cells
        for i in range(1, env.current_board_width * env.current_board_height):
            state, reward, terminated, truncated, info = env.step(i)
            if terminated:
                break
        
        # Should win by revealing all safe cells
        assert terminated, "Should terminate when all safe cells are revealed"
        assert info.get('won', False), "Should win when all safe cells are revealed"
        assert reward == REWARD_WIN, f"Should get win reward, got {reward}"

    def test_win_on_pre_cascade(self):
        """Test winning on the very pre-cascade (extremely rare but possible)."""
        print("🧪 Testing win on pre-cascade...")
        
        # Set up a board where revealing the first cell wins
        # This requires a very specific mine configuration
        self.env.reset()
        self.env.mines.fill(True)  # Fill with mines
        self.env.mines[0, 0] = False  # Except the first cell
        self.env._update_adjacent_counts()
        self.env.mines_placed = True
        
        # Reveal the only safe cell
        action = 0
        state, reward, terminated, truncated, info = self.env.step(action)
        
        # Should win immediately with full win reward (immediate rewards now)
        assert terminated, "Game should terminate on win"
        assert info.get('won', False), "Game should be marked as won"
        assert reward == REWARD_WIN, "Win should get full win reward"
        
        print("✅ Win on pre-cascade passed")

    def test_state_consistency_during_cascade(self):
        """Test that state remains consistent during complex cascades."""
        env = MinesweeperEnv(initial_board_size=5, initial_mines=2)
        env.reset()
        
        # Set up board with known pattern
        env.mines.fill(False)
        env.mines[0, 0] = True
        env.mines[4, 4] = True
        env._update_adjacent_counts()
        env.mines_placed = True
        env.is_first_cascade = False
        env.first_cascade_done = True
        
        # Find a zero cell to trigger cascade
        zero_cell_found = False
        for i in range(env.current_board_height):
            for j in range(env.current_board_width):
                if env.board[i, j] == 0 and not env.mines[i, j]:
                    action = i * env.current_board_width + j
                    state, reward, terminated, truncated, info = env.step(action)
                    zero_cell_found = True
                    break
            if zero_cell_found:
                break
        
        assert zero_cell_found, "Should find a zero cell"
        
        # Verify state consistency
        for i in range(env.current_board_height):
            for j in range(env.current_board_width):
                if env.revealed[i, j]:
                    # Revealed cells should show correct values
                    if env.mines[i, j]:
                        assert state[0, i, j] == CELL_MINE_HIT, f"Revealed mine at ({i},{j}) should show mine hit"
                    else:
                        assert state[0, i, j] == env.board[i, j], f"Revealed cell at ({i},{j}) should show correct number"
                else:
                    # Unrevealed cells should show unrevealed
                    assert state[0, i, j] == CELL_UNREVEALED, f"Unrevealed cell at ({i},{j}) should show unrevealed"

    def test_large_board_cascade_performance(self):
        """Test cascade performance on large boards."""
        env = MinesweeperEnv(initial_board_size=10, initial_mines=5)
        env.reset()
        
        # Set up large board with sparse mines to create large zero regions
        env.mines.fill(False)
        # Place mines in corners and edges to create large zero regions
        env.mines[0, 0] = True
        env.mines[0, 9] = True
        env.mines[9, 0] = True
        env.mines[9, 9] = True
        env.mines[4, 4] = True
        env._update_adjacent_counts()
        env.mines_placed = True
        env.is_first_cascade = False
        env.first_cascade_done = True
        
        # Find a zero cell to trigger large cascade
        zero_cell_found = False
        for i in range(env.current_board_height):
            for j in range(env.current_board_width):
                if env.board[i, j] == 0 and not env.mines[i, j]:
                    action = i * env.current_board_width + j
                    state, reward, terminated, truncated, info = env.step(action)
                    zero_cell_found = True
                    break
            if zero_cell_found:
                break
        
        assert zero_cell_found, "Should find a zero cell on large board"
        
        # Should reveal many cells (large cascade)
        revealed_count = np.sum(env.revealed)
        assert revealed_count > 10, f"Large board cascade should reveal many cells, got {revealed_count}"

    def test_rectangular_board_cascade(self):
        """Test cascade behavior on rectangular boards."""
        env = MinesweeperEnv(initial_board_size=(3, 5), initial_mines=2)
        env.reset()
        
        # Set up rectangular board (3x5: height=3, width=5)
        env.mines.fill(False)
        env.mines[0, 0] = True
        env.mines[2, 4] = True  # Use correct indices for 3x5 board
        env._update_adjacent_counts()
        env.mines_placed = True
        env.is_first_cascade = False
        env.first_cascade_done = True
        
        # Find a zero cell to trigger cascade
        zero_cell_found = False
        for i in range(env.current_board_height):
            for j in range(env.current_board_width):
                if env.board[i, j] == 0 and not env.mines[i, j]:
                    action = i * env.current_board_width + j
                    state, reward, terminated, truncated, info = env.step(action)
                    zero_cell_found = True
                    break
            if zero_cell_found:
                break
        
        assert zero_cell_found, "Should find a zero cell on rectangular board"
        
        # Should reveal multiple cells in cascade
        revealed_count = np.sum(env.revealed)
        assert revealed_count > 1, f"Rectangular board cascade should reveal multiple cells, got {revealed_count}"

    def test_cascade_with_mines_at_boundaries(self):
        """Test cascade behavior when mines are at boundaries of zero regions."""
        env = MinesweeperEnv(initial_board_size=5, initial_mines=3)
        env.reset()
        
        # Set up board with mines surrounding a zero region
        env.mines.fill(False)
        env.mines[1, 1] = True  # Mine adjacent to zero region
        env.mines[1, 3] = True  # Mine adjacent to zero region
        env.mines[3, 2] = True  # Mine adjacent to zero region
        env._update_adjacent_counts()
        env.mines_placed = True
        env.is_first_cascade = False
        env.first_cascade_done = True
        
        # Find the zero cell (should be at (2,2))
        zero_cell_found = False
        for i in range(env.current_board_height):
            for j in range(env.current_board_width):
                if env.board[i, j] == 0 and not env.mines[i, j]:
                    action = i * env.current_board_width + j
                    state, reward, terminated, truncated, info = env.step(action)
                    zero_cell_found = True
                    break
            if zero_cell_found:
                break
        
        assert zero_cell_found, "Should find a zero cell"
        
        # Should reveal the zero cell and connected safe cells
        revealed_count = np.sum(env.revealed)
        assert revealed_count >= 1, f"Should reveal at least the zero cell, got {revealed_count}"
        
        # Verify that mines are not revealed
        for i in range(env.current_board_height):
            for j in range(env.current_board_width):
                if env.mines[i, j]:
                    assert not env.revealed[i, j], f"Mine at ({i},{j}) should not be revealed"

    def test_action_masking_after_cascade(self):
        """Test that action masks are correctly updated after cascade."""
        env = MinesweeperEnv(initial_board_size=4, initial_mines=1)
        env.reset()
        
        # Set up board with zero region
        env.mines.fill(False)
        env.mines[0, 0] = True
        env._update_adjacent_counts()
        env.mines_placed = True
        env.is_first_cascade = False
        env.first_cascade_done = True
        
        # Find and reveal zero cell
        zero_cell_found = False
        for i in range(env.current_board_height):
            for j in range(env.current_board_width):
                if env.board[i, j] == 0 and not env.mines[i, j]:
                    action = i * env.current_board_width + j
                    state, reward, terminated, truncated, info = env.step(action)
                    zero_cell_found = True
                    break
            if zero_cell_found:
                break
        
        assert zero_cell_found, "Should find a zero cell"
        
        # Check action masks
        masks = env.action_masks
        revealed_count = np.sum(env.revealed)
        masked_count = np.sum(~masks)
        
        # All revealed cells should be masked
        assert masked_count >= revealed_count, f"All revealed cells should be masked, revealed: {revealed_count}, masked: {masked_count}"

    # ===== DIAGNOSTIC TESTS =====

    def test_diagnostic_cascade_boundary_conditions(self):
        """Diagnostic test for cascade boundary conditions - prints board state."""
        print("\n🔍 DIAGNOSTIC: Cascade Boundary Conditions")
        env = MinesweeperEnv(initial_board_size=4, initial_mines=1)
        env.reset()
        
        # Set up board with mine in center
        env.mines.fill(False)
        env.mines[1, 1] = True  # Center mine
        env._update_adjacent_counts()
        env.mines_placed = True
        env.is_first_cascade = False
        env.first_cascade_done = True
        
        print("Mines:")
        print(env.mines)
        print("Adjacent counts:")
        print(env.board)
        
        # Check what's at corner (0,0)
        corner_value = env.board[0, 0]
        print(f"Corner (0,0) value: {corner_value}")
        
        # Reveal corner cell
        action = 0  # (0,0) corner
        state, reward, terminated, truncated, info = env.step(action)
        
        print("State after revealing corner:")
        print(state[0])  # Channel 0 (game state)
        print("Revealed cells:")
        print(env.revealed)
        
        revealed_count = np.sum(env.revealed)
        print(f"Total revealed: {revealed_count}")
        
        # This test doesn't assert - just prints diagnostic info

    def test_diagnostic_multiple_zero_regions(self):
        """Diagnostic test for multiple zero regions - prints board state."""
        print("\n🔍 DIAGNOSTIC: Multiple Zero Regions")
        env = MinesweeperEnv(initial_board_size=6, initial_mines=4)
        env.reset()
        
        # Set up board with mines
        env.mines.fill(False)
        env.mines[1, 1] = True
        env.mines[1, 4] = True
        env.mines[4, 1] = True
        env.mines[4, 4] = True
        env._update_adjacent_counts()
        env.mines_placed = True
        env.is_first_cascade = False
        env.first_cascade_done = True
        
        print("Mines:")
        print(env.mines)
        print("Adjacent counts:")
        print(env.board)
        
        # Find all zero cells
        zero_cells = []
        for i in range(env.current_board_height):
            for j in range(env.current_board_width):
                if env.board[i, j] == 0 and not env.mines[i, j]:
                    zero_cells.append((i, j))
        
        print(f"Zero cells found: {zero_cells}")
        
        if zero_cells:
            # Try revealing first zero cell
            i, j = zero_cells[0]
            action = i * env.current_board_width + j
            state, reward, terminated, truncated, info = env.step(action)
            
            print("State after revealing zero cell:")
            print(state[0])
            print("Revealed cells:")
            print(env.revealed)
            print(f"Total revealed: {np.sum(env.revealed)}")

    def test_diagnostic_win_on_pre_cascade(self):
        """Diagnostic test for win on pre-cascade - prints board state."""
        print("\n🔍 DIAGNOSTIC: Win on pre-cascade")
        env = MinesweeperEnv(initial_board_size=2, initial_mines=1)
        env.reset()
        
        # Set up 2x2 board
        env.mines.fill(False)
        env.mines[0, 0] = True  # Mine in corner
        env._update_adjacent_counts()
        env.mines_placed = True
        env.is_first_cascade = False
        env.first_cascade_done = True
        
        print("Mines:")
        print(env.mines)
        print("Adjacent counts:")
        print(env.board)
        
        # Try revealing safe corner (1,1)
        action = 3  # (1,1)
        state, reward, terminated, truncated, info = env.step(action)
        
        print("State after revealing safe corner:")
        print(state[0])
        print("Revealed cells:")
        print(env.revealed)
        print(f"Terminated: {terminated}")
        print(f"Won: {info.get('won', False)}")
        print(f"Reward: {reward}")
        
        # Check win condition manually
        safe_cells = 0
        total_cells = env.current_board_width * env.current_board_height
        for i in range(env.current_board_height):
            for j in range(env.current_board_width):
                if not env.mines[i, j]:
                    safe_cells += 1
        
        print(f"Total cells: {total_cells}")
        print(f"Safe cells: {safe_cells}")
        print(f"Revealed safe cells: {np.sum(env.revealed)}")

    def test_diagnostic_rectangular_board_dimensions(self):
        """Diagnostic test for rectangular board dimensions."""
        print("\n🔍 DIAGNOSTIC: Rectangular Board Dimensions")
        env = MinesweeperEnv(initial_board_size=(5, 3), initial_mines=2)
        env.reset()
        
        print(f"Board shape: {env.mines.shape}")
        print(f"Expected: (3, 5) - height x width")
        print(f"Actual: {env.mines.shape}")
        
        # Try to access the problematic index
        try:
            env.mines[2, 4] = True
            print("✅ Successfully set mine at (2, 4)")
        except IndexError as e:
            print(f"❌ IndexError: {e}")
            print("This suggests the board dimensions are not as expected")

    def test_diagnostic_cascade_boundary_behavior(self):
        """Diagnostic test for cascade behavior with mines at boundaries."""
        print("\n🔍 DIAGNOSTIC: Cascade with Mines at Boundaries")
        env = MinesweeperEnv(initial_board_size=5, initial_mines=3)
        env.reset()
        
        # Set up board with mines surrounding a zero region
        env.mines.fill(False)
        env.mines[1, 1] = True  # Mine adjacent to zero region
        env.mines[1, 3] = True  # Mine adjacent to zero region
        env.mines[3, 2] = True  # Mine adjacent to zero region
        env._update_adjacent_counts()
        env.mines_placed = True
        env.is_first_cascade = False
        env.first_cascade_done = True
        
        print("Mines:")
        print(env.mines)
        print("Adjacent counts:")
        print(env.board)
        
        # Find the zero cell
        zero_cells = []
        for i in range(env.current_board_height):
            for j in range(env.current_board_width):
                if env.board[i, j] == 0 and not env.mines[i, j]:
                    zero_cells.append((i, j))
        
        print(f"Zero cells found: {zero_cells}")
        
        if zero_cells:
            i, j = zero_cells[0]
            action = i * env.current_board_width + j
            state, reward, terminated, truncated, info = env.step(action)
            
            print("State after revealing zero cell:")
            print(state[0])
            print("Revealed cells:")
            print(env.revealed)
            print(f"Total revealed: {np.sum(env.revealed)}")
            
            # Show which cells were revealed
            revealed_positions = []
            for ri in range(env.current_board_height):
                for rj in range(env.current_board_width):
                    if env.revealed[ri, rj]:
                        revealed_positions.append((ri, rj))
            print(f"Revealed positions: {revealed_positions}")

    def test_diagnostic_zero_cell_finding(self):
        """Diagnostic test to understand zero cell patterns."""
        print("\n🔍 DIAGNOSTIC: Zero Cell Finding")
        
        # Test different board sizes and mine patterns
        test_cases = [
            (4, 1, "Small board, one mine"),
            (6, 4, "Medium board, four mines"),
            (5, 3, "Medium board, three mines")
        ]
        
        for board_size, mine_count, description in test_cases:
            print(f"\n--- {description} ---")
            env = MinesweeperEnv(initial_board_size=board_size, initial_mines=mine_count)
            env.reset()
            
            # Set up specific mine patterns
            env.mines.fill(False)
            if board_size == 4 and mine_count == 1:
                env.mines[1, 1] = True  # Center
            elif board_size == 6 and mine_count == 4:
                env.mines[1, 1] = True
                env.mines[1, 4] = True
                env.mines[4, 1] = True
                env.mines[4, 4] = True
            elif board_size == 5 and mine_count == 3:
                env.mines[1, 1] = True
                env.mines[1, 3] = True
                env.mines[3, 2] = True
            
            env._update_adjacent_counts()
            env.mines_placed = True
            env.is_first_cascade = False
            env.first_cascade_done = True
            
            print(f"Board size: {board_size}")
            print("Adjacent counts:")
            print(env.board)
            
            # Find zero cells
            zero_cells = []
            for i in range(env.current_board_height):
                for j in range(env.current_board_width):
                    if env.board[i, j] == 0 and not env.mines[i, j]:
                        zero_cells.append((i, j))
            
            print(f"Zero cells: {zero_cells}")
            print(f"Count: {len(zero_cells)}") 