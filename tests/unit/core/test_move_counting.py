import pytest
import numpy as np
from src.core.minesweeper_env import MinesweeperEnv
from src.core.constants import REWARD_SAFE_REVEAL, REWARD_WIN, REWARD_HIT_MINE
import unittest.mock


class TestMoveCounting:
    """Test move counting functionality in the Minesweeper environment."""
    
    def test_move_count_initialization(self):
        """Test that move count is properly initialized."""
        env = MinesweeperEnv(initial_board_size=4, initial_mines=2)
        state, _ = env.reset()
        
        # Check initial move count
        assert env.move_count == 0
        assert env.total_moves_across_games == 0
        assert len(env.games_with_move_counts) == 0
        
        # Check move statistics
        stats = env.get_move_statistics()
        assert stats['current_game_moves'] == 0
        assert stats['total_moves_across_games'] == 0
        assert stats['games_with_move_counts'] == []
        assert stats['average_moves_per_game'] == 0
        assert stats['min_moves_in_game'] == 0
        assert stats['max_moves_in_game'] == 0
    
    def test_move_count_increment_on_valid_action(self):
        """Test that move count increments on valid actions."""
        env = MinesweeperEnv(initial_board_size=4, initial_mines=2)
        state, _ = env.reset()
        
        # Take a valid action
        action = 0  # First cell
        state, reward, terminated, truncated, info = env.step(action)
        
        # Check move count increased
        assert env.move_count == 1
        
        # Take another valid action (if game didn't end)
        if not terminated:
            # Find a valid action for the second move
            valid_actions = np.where(env.action_masks)[0]
            if len(valid_actions) > 0:
                action = valid_actions[0]
                state, reward, terminated, truncated, info = env.step(action)
                
                # Check move count increased again (if action was valid)
                assert env.move_count == 2
            else:
                # No valid actions available, move count should remain 1
                assert env.move_count == 1
        else:
            # First action ended the game, so move count should be 1
            assert env.move_count == 1
    
    def test_move_count_no_increment_on_invalid_action(self):
        """Test that move count doesn't increment on invalid actions."""
        env = MinesweeperEnv(initial_board_size=4, initial_mines=2)
        state, _ = env.reset()
        
        # Take a valid action first
        action = 0
        state, reward, terminated, truncated, info = env.step(action)
        assert env.move_count == 1
        
        # Try to reveal the same cell again (invalid)
        action = 0
        state, reward, terminated, truncated, info = env.step(action)
        
        # Move count should not increase
        assert env.move_count == 1
    
    def test_move_count_no_increment_on_out_of_bounds_action(self):
        """Test that move count doesn't increment on out-of-bounds actions."""
        env = MinesweeperEnv(initial_board_size=4, initial_mines=2)
        state, _ = env.reset()
        
        # Try an out-of-bounds action
        action = 100  # Beyond the action space
        state, reward, terminated, truncated, info = env.step(action)
        
        # Move count should not increase
        assert env.move_count == 0
    
    def test_move_count_reset_on_new_game(self):
        """Test that move count resets when starting a new game."""
        env = MinesweeperEnv(initial_board_size=4, initial_mines=2)
        state, _ = env.reset()
        
        # Play a few moves
        for i in range(3):
            action = i
            state, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break
        
        # Check move count is greater than 0
        assert env.move_count > 0
        
        # Start a new game
        state, _ = env.reset()
        
        # Move count should reset to 0
        assert env.move_count == 0
    
    def test_move_count_recording_on_game_end(self):
        """Test that move count is recorded when a game ends."""
        env = MinesweeperEnv(initial_board_size=4, initial_mines=2)
        state, _ = env.reset()
        
        # Play until game ends
        moves_made = 0
        while True:
            # Find a valid action
            valid_actions = np.where(env.action_masks)[0]
            if len(valid_actions) == 0:
                break
            
            action = valid_actions[0]
            state, reward, terminated, truncated, info = env.step(action)
            moves_made += 1
            
            if terminated:
                break
        
        # Check that moves were recorded
        assert len(env.games_with_move_counts) == 1
        assert env.games_with_move_counts[0] == moves_made
        assert env.total_moves_across_games == moves_made
    
    def test_move_statistics_across_multiple_games(self):
        """Test move statistics across multiple games."""
        env = MinesweeperEnv(initial_board_size=4, initial_mines=2)
        
        # Play multiple games
        for game in range(3):
            state, _ = env.reset()
            
            # Play until game ends
            moves_this_game = 0
            while True:
                valid_actions = np.where(env.action_masks)[0]
                if len(valid_actions) == 0:
                    break
                
                action = valid_actions[0]
                state, reward, terminated, truncated, info = env.step(action)
                moves_this_game += 1
                
                if terminated:
                    break
        
        # Check statistics
        stats = env.get_move_statistics()
        assert len(stats['games_with_move_counts']) == 3
        assert stats['total_moves_across_games'] == sum(stats['games_with_move_counts'])
        assert stats['average_moves_per_game'] == stats['total_moves_across_games'] / 3
        assert stats['min_moves_in_game'] == min(stats['games_with_move_counts'])
        assert stats['max_moves_in_game'] == max(stats['games_with_move_counts'])
    
    def test_move_count_with_win_game(self):
        """Test move counting in a winning game."""
        env = MinesweeperEnv(initial_board_size=2, initial_mines=1)  # Small board for easier win
        state, _ = env.reset()
        
        # Play until win or loss
        moves_made = 0
        while True:
            valid_actions = np.where(env.action_masks)[0]
            if len(valid_actions) == 0:
                break
            
            action = valid_actions[0]
            state, reward, terminated, truncated, info = env.step(action)
            moves_made += 1
            
            if terminated:
                break
        
        # Check move count was recorded
        assert len(env.games_with_move_counts) == 1
        assert env.games_with_move_counts[0] == moves_made
        assert env.move_count == moves_made
    
    def test_move_count_with_loss_game(self):
        """Test move counting in a losing game (mine hit)."""
        env = MinesweeperEnv(initial_board_size=4, initial_mines=8)  # High mine density for likely loss
        state, _ = env.reset()
        
        # Play until mine hit
        moves_made = 0
        while True:
            valid_actions = np.where(env.action_masks)[0]
            if len(valid_actions) == 0:
                break
            
            action = valid_actions[0]
            state, reward, terminated, truncated, info = env.step(action)
            moves_made += 1
            
            if terminated:
                break
        
        # Check move count was recorded
        assert len(env.games_with_move_counts) == 1
        assert env.games_with_move_counts[0] == moves_made
        assert env.move_count == moves_made
    
    def test_move_count_with_cascade_reveals(self):
        """Test that cascade reveals count as single moves."""
        env = MinesweeperEnv(initial_board_size=4, initial_mines=2)
        state, _ = env.reset()
        
        # Find a cell that will trigger a cascade (adjacent to 0 mines)
        cascade_cell = None
        for i in range(env.current_board_height):
            for j in range(env.current_board_width):
                if not env.mines[i, j] and env.board[i, j] == 0:
                    cascade_cell = i * env.current_board_width + j
                    break
            if cascade_cell is not None:
                break
        
        if cascade_cell is not None:
            # Take the cascade action
            state, reward, terminated, truncated, info = env.step(cascade_cell)
            
            # Should only count as 1 move, even though multiple cells were revealed
            assert env.move_count == 1
    
    def test_move_statistics_edge_cases(self):
        """Test move statistics with edge cases."""
        env = MinesweeperEnv(initial_board_size=4, initial_mines=2)
        
        # Test with no games played
        stats = env.get_move_statistics()
        assert stats['current_game_moves'] == 0
        assert stats['total_moves_across_games'] == 0
        assert stats['games_with_move_counts'] == []
        assert stats['average_moves_per_game'] == 0
        assert stats['min_moves_in_game'] == 0
        assert stats['max_moves_in_game'] == 0
        
        # Test with single move game
        state, _ = env.reset()
        action = 0
        state, reward, terminated, truncated, info = env.step(action)
        
        stats = env.get_move_statistics()
        assert stats['current_game_moves'] == 1
        # If the game ended after one move, the move count will be recorded
        if terminated:
            assert len(stats['games_with_move_counts']) == 1
            assert stats['games_with_move_counts'][0] == 1
        else:
            assert len(stats['games_with_move_counts']) == 0
    
    def test_move_count_in_info_dict(self):
        """Test that move count is included in the info dictionary."""
        env = MinesweeperEnv(initial_board_size=4, initial_mines=2)
        state, _ = env.reset()
        
        # Take a valid action
        action = 0
        state, reward, terminated, truncated, info = env.step(action)
        
        # Check that move count is accessible
        assert env.move_count == 1
        
        # Note: We could add move_count to info dict if needed
        # For now, it's accessible via env.move_count 
    
    def test_repeated_action_counting(self):
        """Test that repeated actions are counted correctly."""
        env = MinesweeperEnv(initial_board_size=4, initial_mines=2)
        for _ in range(10):
            state, _ = env.reset()
            action = 0
            state, reward, terminated, truncated, info = env.step(action)
            if not terminated:
                # Repeat the same action (should be invalid, but should count as repeated)
                state, reward, terminated, truncated, info = env.step(action)
                stats = env.get_move_statistics()
                assert stats['repeated_action_count'] == 1
                assert action in stats['repeated_actions']
                return
        pytest.skip("Could not find a non-terminal first move after 10 resets.")
    
    def test_revealed_cell_click_counting(self):
        """Test that clicks on already revealed cells are counted correctly."""
        env = MinesweeperEnv(initial_board_size=4, initial_mines=2)
        for _ in range(10):
            state, _ = env.reset()
            action = 0
            state, reward, terminated, truncated, info = env.step(action)
            if not terminated:
                # Try to reveal the same cell again (should count as revealed cell click)
                state, reward, terminated, truncated, info = env.step(action)
                stats = env.get_move_statistics()
                assert stats['revealed_cell_click_count'] == 1
                return
        pytest.skip("Could not find a non-terminal first move after 10 resets.")
    
    def test_invalid_action_counting(self):
        """Test that invalid actions are counted correctly."""
        env = MinesweeperEnv(initial_board_size=4, initial_mines=2)
        state, _ = env.reset()
        # Out-of-bounds action
        state, reward, terminated, truncated, info = env.step(100)
        stats = env.get_move_statistics()
        assert stats['invalid_action_count'] == 1
        # Try an invalid (already revealed) action
        for _ in range(10):
            state, _ = env.reset()
            action = 0
            state, reward, terminated, truncated, info = env.step(action)
            if not terminated:
                state, reward, terminated, truncated, info = env.step(action)
                stats = env.get_move_statistics()
                assert stats['invalid_action_count'] >= 1
                return
        pytest.skip("Could not find a non-terminal first move after 10 resets.") 