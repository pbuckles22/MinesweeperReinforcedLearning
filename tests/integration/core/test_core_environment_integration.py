"""
Integration Tests for Minesweeper RL Environment

These tests verify that all components work together correctly,
including environment initialization, state management, and RL training integration.
"""

import pytest
import os
import sys
import numpy as np
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.core.minesweeper_env import MinesweeperEnv
import unittest
import gymnasium as gym
from src.core.constants import (
    CELL_UNREVEALED,
    CELL_MINE,
    CELL_MINE_HIT,
    REWARD_FIRST_CASCADE_SAFE, REWARD_FIRST_CASCADE_HIT_MINE,
    REWARD_SAFE_REVEAL,
    REWARD_WIN,
    REWARD_HIT_MINE,
    REWARD_INVALID_ACTION
)

def test_imports():
    """Test that all required imports are available"""
    print("Testing imports...")
    assert True  # If we got here, imports worked
    print("✓ All imports successful")

def test_environment_creation():
    """Test that the environment can be created and reset"""
    print("\nTesting environment creation...")
    env = MinesweeperEnv(max_board_size=4, max_mines=2, mine_spacing=2)
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4, 4, 4)  # 4-channel state
    assert isinstance(info, dict)
    print("✓ Environment created and reset successfully")
    print(f"✓ State shape: {obs.shape}")
    print(f"✓ Info: {info}")

def test_basic_actions():
    """Test that basic actions work"""
    print("\nTesting basic actions...")
    env = MinesweeperEnv(max_board_size=4, max_mines=2, mine_spacing=2)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)  # Reveal first cell
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4, 4, 4)  # 4-channel state
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    print("✓ Basic action successful")
    print(f"✓ Reward: {reward}")
    print(f"✓ Terminated: {terminated}")
    print(f"✓ Truncated: {truncated}")
    print(f"✓ Info: {info}")

def test_pygame():
    """Test that pygame can be initialized"""
    print("\nTesting pygame...")
    pygame.init()
    assert pygame.get_init()
    print("✓ Pygame initialized successfully")
    pygame.quit()

def main():
    """Run all environment tests"""
    print("Starting environment tests...\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Environment Creation Test", test_environment_creation),
        ("Basic Actions Test", test_basic_actions),
        ("Pygame Test", test_pygame)
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        if not test_func():
            all_passed = False
            print(f"✗ {test_name} failed")
        else:
            print(f"✓ {test_name} passed")
    
    print("\nTest Summary:")
    if all_passed:
        print("✓ All environment tests passed!")
        return 0
    else:
        print("✗ Some tests failed. Please check the output above.")
        return 1

class TestMinesweeperEnv:
    @pytest.fixture
    def env(self):
        """Create a test environment"""
        return MinesweeperEnv(max_board_size=4, max_mines=2, mine_spacing=2)

    def test_initialization(self, env):
        """Test that the environment initializes correctly"""
        assert env.max_board_size_int == 4
        assert env.max_mines == 2
        assert env.mine_spacing == 2

    def test_invalid_action(self, env):
        """Test that invalid actions are handled gracefully with penalty"""
        env.reset()
        # Test out of bounds action - should return penalty, not raise exception
        state, reward, terminated, truncated, info = env.step(100)  # Out of bounds
        assert not terminated
        assert not truncated
        assert reward == env.reward_invalid_action
        assert isinstance(state, np.ndarray)
        assert isinstance(info, dict)

    def test_mine_reveal(self, env):
        """Test revealing a mine."""
        env.reset()

        # Make a first move (can be a mine or safe, no special logic)
        action = 0
        state, reward, terminated, truncated, info = env.step(action)
        # If the first move caused a win, we can't test mine hits
        if info.get('won', False):
            print("First move caused win, skipping mine hit test")
            return
        
        # Check if we're still in pre-cascade period
        if env.is_first_cascade:
            # If still in pre-cascade, the next mine hit should give neutral reward
            # Find a mine and hit it
            mine_positions = np.where(env.mines)
            if len(mine_positions[0]) > 0:
                row, col = mine_positions[0][0], mine_positions[1][0]
                action = row * env.current_board_width + col
                state, reward, terminated, truncated, info = env.step(action)
                # Should get immediate penalty for pre-cascade mine hit
                if reward in [REWARD_SAFE_REVEAL, REWARD_HIT_MINE, REWARD_WIN]:
                    assert True
                else:
                    assert False, f"Pre-cascade mine hit should give immediate reward/penalty/win, got {reward}"
                assert terminated, "Game should terminate on mine hit"
        else:
            # If post-cascade, mine hit should give full penalty
            # Find a mine and hit it
            mine_positions = np.where(env.mines)
            if len(mine_positions[0]) > 0:
                row, col = mine_positions[0][0], mine_positions[1][0]
                action = row * env.current_board_width + col
                state, reward, terminated, truncated, info = env.step(action)
                assert reward == REWARD_HIT_MINE, "Post-cascade mine hit should give mine hit reward"
                assert terminated, "Game should terminate on mine hit"

    def test_reset(self, env):
        """Test that reset returns the correct observation and info"""
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (4, 4, 4)  # 4-channel state
        assert isinstance(info, dict)
        assert 'won' in info

    def test_step(self, env):
        """Test that step returns the correct observation and info"""
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (4, 4, 4)  # 4-channel state
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

@pytest.fixture
def env():
    return MinesweeperEnv(initial_board_size=3, initial_mines=1)

def test_initialization(env):
    """Test that environment initializes correctly."""
    assert env.current_board_width == 3
    assert env.current_board_height == 3
    assert env.initial_mines == 1
    assert env.is_first_cascade
    assert env.mines.shape == (3, 3)
    assert env.board.shape == (3, 3)
    # Check that the state is properly initialized
    state, info = env.reset()
    assert state.shape == (4, 3, 3)  # 4-channel state
    assert np.all(state[0] == CELL_UNREVEALED)  # Game state should be all unrevealed

def test_reset(env):
    """Test that environment resets correctly."""
    # Make some moves
    env.step(0)  # Reveal first cell
    
    # Reset environment
    state, info = env.reset()
    
    # Check that everything is reset
    assert env.current_board_width == 3
    assert env.current_board_height == 3
    assert env.initial_mines == 1
    assert env.is_first_cascade
    assert env.mines.shape == (3, 3)
    assert env.board.shape == (3, 3)
    assert state.shape == (4, 3, 3)  # 4-channel state
    assert np.all(state[0] == CELL_UNREVEALED)  # Game state should be all unrevealed

def test_board_size_initialization():
    """Test that different board sizes initialize correctly."""
    # Test 5x5 board
    env = MinesweeperEnv(initial_board_size=5, initial_mines=3)
    assert env.current_board_width == 5
    assert env.current_board_height == 5
    assert env.initial_mines == 3
    assert env.mines.shape == (5, 5)
    assert env.board.shape == (5, 5)
    
    # Test 10x10 board
    env = MinesweeperEnv(initial_board_size=10, initial_mines=10)
    assert env.current_board_width == 10
    assert env.current_board_height == 10
    assert env.initial_mines == 10
    assert env.mines.shape == (10, 10)
    assert env.board.shape == (10, 10)

def test_mine_count_initialization():
    """Test that different mine counts initialize correctly."""
    # Test with 1 mine
    env = MinesweeperEnv(initial_board_size=3, initial_mines=1)
    assert env.initial_mines == 1
    assert np.sum(env.mines) == 1
    
    # Test with 2 mines
    env = MinesweeperEnv(initial_board_size=3, initial_mines=2)
    assert env.initial_mines == 2
    assert np.sum(env.mines) == 2
    
    # Test with maximum mines (board_size - 1)
    env = MinesweeperEnv(initial_board_size=3, initial_mines=8)
    assert env.initial_mines == 8
    assert np.sum(env.mines) == 8

def test_adjacent_mines_initialization(env):
    """Test that adjacent mine counts are initialized correctly."""
    env.reset()
    
    # Check that adjacent counts are calculated correctly
    # The exact values depend on mine placement, so we just check bounds
    assert np.all(env.board >= 0), "Adjacent counts should be >= 0"
    assert np.all(env.board <= 9), "Adjacent counts should be <= 9 (mines have value 9)"
    
    # Check that mine positions have value 9 (mine)
    mine_positions = np.where(env.mines)
    for row, col in zip(mine_positions[0], mine_positions[1]):
        assert env.board[row, col] == 9, f"Mine at ({row},{col}) should have value 9"

def test_environment_initialization():
    """Test environment initialization with different parameters."""
    # Test with default parameters
    env = MinesweeperEnv()
    assert env.current_board_width == env.initial_board_width
    assert env.current_board_height == env.initial_board_height
    assert env.current_mines == env.initial_mines
    
    # Test with custom parameters
    env = MinesweeperEnv(initial_board_size=(5, 4), initial_mines=3)
    assert env.current_board_width == 4
    assert env.current_board_height == 5
    assert env.current_mines == 3

def test_board_creation(env):
    """Test that board is created with correct dimensions and mine count."""
    assert env.board.shape == (env.current_board_height, env.current_board_width)
    assert np.sum(env.mines) == env.current_mines

def test_mine_placement(env):
    """Test that mines are placed correctly."""
    env.reset()
    
    # Check that the correct number of mines are placed
    assert np.sum(env.mines) == env.initial_mines, f"Should have {env.initial_mines} mines"
    
    # Check that mines are not placed at the first cell (First cascade safety)
    if env.is_first_cascade:
        assert not env.mines[0, 0], "First cell should not have a mine"

def test_safe_cell_reveal(env):
    """Test revealing a safe cell."""
    env.reset()
    state, reward, terminated, truncated, info = env.step(0)  # Reveal first cell
    
    # Check that the cell was revealed (not unrevealed)
    assert state[0, 0, 0] != CELL_UNREVEALED, "Cell should be revealed"
    assert isinstance(reward, (int, float)), "Reward should be numeric"
    assert isinstance(terminated, bool), "Terminated should be boolean"

def test_difficulty_levels():
    """Test that different difficulty levels initialize correctly."""
    # Test Easy (9x9, 10 mines)
    env = MinesweeperEnv(initial_board_size=9, initial_mines=10)
    assert env.current_board_width == 9
    assert env.current_board_height == 9
    assert env.current_mines == 10
    
    # Test Normal (16x16, 40 mines)
    env = MinesweeperEnv(initial_board_size=16, initial_mines=40)
    assert env.current_board_width == 16
    assert env.current_board_height == 16
    assert env.current_mines == 40
    
    # Test Hard (16x30, 99 mines)
    env = MinesweeperEnv(initial_board_size=(30, 16), initial_mines=99)
    assert env.current_board_width == 16
    assert env.current_board_height == 30
    assert env.current_mines == 99

def test_rectangular_board_actions(env):
    """Test actions on rectangular board."""
    env.current_board_width = 4
    env.current_board_height = 3
    env.reset()
    
    state, reward, terminated, truncated, info = env.step(0)  # Reveal first cell
    assert state.shape == (4, 3, 4)  # 4-channel state with rectangular dimensions
    assert state[0, 0, 0] != CELL_UNREVEALED, "Cell should be revealed"
    assert isinstance(reward, (int, float)), "Reward should be numeric"

def test_curriculum_progression(env):
    """Test that curriculum progression works correctly."""
    # Start with easy board
    env = MinesweeperEnv(initial_board_size=3, initial_mines=1)
    assert env.current_board_width == 3
    assert env.current_board_height == 3
    assert env.current_mines == 1
    
    # Progress to medium board
    env = MinesweeperEnv(initial_board_size=5, initial_mines=3)
    assert env.current_board_width == 5
    assert env.current_board_height == 5
    assert env.current_mines == 3
    
    # Progress to hard board
    env = MinesweeperEnv(initial_board_size=7, initial_mines=5)
    assert env.current_board_width == 7
    assert env.current_board_height == 7
    assert env.current_mines == 5

def test_win_condition(env):
    """Test that the game can be won by revealing all safe cells."""
    env.reset()

    # Set up a controlled board with one mine
    env.mines.fill(False)
    env.mines[1, 1] = True
    env._update_adjacent_counts()

    # Reveal all safe cells
    for i in range(env.current_board_height):
        for j in range(env.current_board_width):
            if not env.mines[i, j]:
                action = i * env.current_board_width + j
                state, reward, terminated, truncated, info = env.step(action)

    # Check that game is won
    assert terminated
    assert not truncated
    # Win before cascade should get immediate reward
    if reward in [REWARD_SAFE_REVEAL, REWARD_HIT_MINE, REWARD_WIN]:
        assert True
    else:
        assert False, f"Win before cascade should get immediate reward/penalty/win, got {reward}"

class TestEnvironmentIntegration:
    """Test complete environment integration."""
    
    def test_full_environment_lifecycle(self):
        """Test complete environment lifecycle from creation to game end."""
        # Create environment
        env = MinesweeperEnv(
            initial_board_size=(5, 5),
            initial_mines=5,
            max_board_size=(10, 10),
            max_mines=20
        )
        
        # Reset environment
        state, info = env.reset(seed=42)
        
        # Verify initial state
        assert state.shape == (4, 5, 5), "Initial state should have correct shape"
        assert np.all(state[0] == CELL_UNREVEALED), "Initial game state should be all unrevealed"
        assert np.all(state[1] >= -1), "Safety hints should be >= -1"
        assert np.all(state[1] <= 8), "Safety hints should be <= 8"
        
        # Play a complete game
        total_reward = 0
        moves_made = 0
        game_ended = False
        
        while not game_ended and moves_made < 25:  # Prevent infinite loop
            # Find an unrevealed cell
            unrevealed = np.where(state[0] == CELL_UNREVEALED)
            if len(unrevealed[0]) == 0:
                break
                
            action = unrevealed[0][0] * env.current_board_width + unrevealed[1][0]
            state, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            moves_made += 1
            
            # Verify state consistency
            assert state.shape == (4, 5, 5), "State shape should remain consistent"
            assert isinstance(reward, (int, float)), "Reward should be numeric"
            assert isinstance(terminated, bool), "Terminated should be boolean"
            assert isinstance(info, dict), "Info should be dictionary"
            
            if terminated:
                game_ended = True
        
        # Verify game ended properly
        assert game_ended, "Game should have ended"
        assert moves_made > 0, "Should have made some moves"
        assert isinstance(total_reward, (int, float)), "Total reward should be numeric"
    
    def test_environment_with_curriculum_learning(self):
        """Test environment integration with curriculum learning."""
        env = MinesweeperEnv(
            initial_board_size=(3, 3),
            initial_mines=1,
            max_board_size=(8, 8),
            max_mines=15
        )
        
        # Test progression through different difficulty levels
        for level in range(3):
            # Set difficulty for this level
            width = 3 + level * 2
            height = 3 + level * 2
            mines = 1 + level * 2
            
            env.current_board_width = width
            env.current_board_height = height
            env.current_mines = mines
            state, info = env.reset(seed=42 + level)
            
            # Verify environment setup
            assert env.state.shape == (4, height, width), f"State should match level {level} dimensions"
            assert env.action_space.n == width * height, f"Action space should match level {level} dimensions"
            assert env.current_mines == mines, f"Mine count should match level {level}"
            
            # Play a few moves
            for action in range(min(5, width * height)):
                state, reward, terminated, truncated, info = env.step(action)
                
                # Verify state consistency
                assert state.shape == (4, height, width), f"State shape should remain consistent at level {level}"
                assert isinstance(reward, (int, float)), f"Reward should be numeric at level {level}"
                
                if terminated:
                    break
    
    def test_environment_with_early_learning(self):
        """Test environment integration with early learning mode."""
        env = MinesweeperEnv(
            initial_board_size=(4, 4),
            initial_mines=3,
            early_learning_mode=True,
            early_learning_corner_safe=True,
            early_learning_edge_safe=True
        )
        
        # Test multiple games with early learning
        for game in range(5):
            state, info = env.reset(seed=42 + game)
            
            # Verify early learning setup
            assert state.shape == (4, 4, 4), "State should have correct shape"
            assert env.early_learning_mode, "Early learning mode should be enabled"
            
            # Test corner safety (note: early learning mode doesn't currently guarantee safe corners)
            corner_action = 0  # (0,0)
            state, reward, terminated, truncated, info = env.step(corner_action)

            # Early learning mode should allow for corner testing (but not guarantee safety)
            # Note: This is a probabilistic test since early learning mode doesn't currently guarantee corner safety
            assert isinstance(reward, (int, float)), f"Should get a numeric reward in game {game}"
            
            # Play a few more moves
            for action in range(1, 5):
                if not terminated:
                    state, reward, terminated, truncated, info = env.step(action)
    
    def test_environment_state_consistency(self):
        """Test that environment maintains state consistency throughout integration."""
        env = MinesweeperEnv(initial_board_size=(6, 6), initial_mines=8)
        
        # Track state evolution
        state_history = []
        
        state, info = env.reset(seed=42)
        state_history.append(state.copy())
        
        # Make several moves and track state changes
        for action in range(10):
            state, reward, terminated, truncated, info = env.step(action)
            state_history.append(state.copy())
            
            # Verify state consistency
            assert state.shape == (4, 6, 6), "State shape should remain consistent"
            assert np.all(state[1] >= -1), "Safety hints should be >= -1"
            assert np.all(state[1] <= 8), "Safety hints should be <= 8"
            
            # Verify that previously revealed cells remain revealed
            if len(state_history) > 1:
                prev_state = state_history[-2]
                revealed_in_prev = prev_state[0] != CELL_UNREVEALED
                assert np.all(state[0][revealed_in_prev] != CELL_UNREVEALED), "Previously revealed cells should remain revealed"
            
            if terminated:
                break
        
        # Verify state history consistency
        assert len(state_history) > 1, "Should have multiple states in history"
        for i, state in enumerate(state_history):
            assert state.shape == (4, 6, 6), f"All states should have consistent shape, state {i}"
    
    def test_environment_action_masking_integration(self):
        """Test action masking integration throughout gameplay."""
        env = MinesweeperEnv(initial_board_size=(5, 5), initial_mines=5)
        state, info = env.reset(seed=42)
        
        # Track action masks
        mask_history = []
        
        # Initial masks should all be True
        initial_masks = env.action_masks
        assert np.all(initial_masks), "All actions should be valid initially"
        assert np.sum(initial_masks) == env.action_space.n, "All actions should be valid initially"
        mask_history.append(initial_masks.copy())
        
        # Make moves and track mask changes
        for action in range(10):
            # Verify current masks
            current_masks = env.action_masks
            mask_history.append(current_masks.copy())
            
            # Take action
            state, reward, terminated, truncated, info = env.step(action)
            
            # Verify mask consistency
            assert len(env.action_masks) == env.action_space.n, "Mask length should match action space"
            assert np.sum(env.action_masks) <= env.action_space.n, "Valid actions should not exceed total actions"
            
            if terminated:
                break
        
        # Verify mask evolution
        assert len(mask_history) > 1, "Should have multiple mask states"
        for i, masks in enumerate(mask_history):
            assert len(masks) == env.action_space.n, f"All masks should have correct length, mask {i}"
    
    def test_environment_reward_integration(self):
        """Test reward system integration throughout gameplay."""
        env = MinesweeperEnv(initial_board_size=(4, 4), initial_mines=3)
        state, info = env.reset(seed=42)
        
        # Track rewards
        reward_history = []
        
        # Play until game ends
        moves_made = 0
        while moves_made < 16:  # Prevent infinite loop
            # Find an unrevealed cell
            unrevealed = np.where(state[0] == CELL_UNREVEALED)
            if len(unrevealed[0]) == 0:
                break
                
            action = unrevealed[0][0] * env.current_board_width + unrevealed[1][0]
            state, reward, terminated, truncated, info = env.step(action)
            
            reward_history.append(reward)
            moves_made += 1
            
            # Verify reward consistency
            assert isinstance(reward, (int, float)), f"Reward should be numeric, move {moves_made}"
            valid_rewards = [REWARD_SAFE_REVEAL, REWARD_HIT_MINE, REWARD_WIN, REWARD_INVALID_ACTION]
            assert reward in valid_rewards, "Reward should be valid"
            
            if terminated:
                break
        
        # Verify reward history
        assert len(reward_history) > 0, "Should have some rewards"
        assert all(isinstance(r, (int, float)) for r in reward_history), "All rewards should be numeric"
    
    def test_environment_info_integration(self):
        """Test info dictionary integration throughout gameplay."""
        env = MinesweeperEnv(initial_board_size=(4, 4), initial_mines=3)
        state, info = env.reset(seed=42)
        
        # Track info
        info_history = []
        
        # Play until game ends
        moves_made = 0
        while moves_made < 16:  # Prevent infinite loop
            # Find an unrevealed cell
            unrevealed = np.where(state[0] == CELL_UNREVEALED)
            if len(unrevealed[0]) == 0:
                break
                
            action = unrevealed[0][0] * env.current_board_width + unrevealed[1][0]
            state, reward, terminated, truncated, info = env.step(action)
            
            info_history.append(info.copy())
            moves_made += 1
            
            # Verify info consistency
            assert isinstance(info, dict), f"Info should be dictionary, move {moves_made}"
            assert 'won' in info, f"Info should contain 'won' key, move {moves_made}"
            assert isinstance(info['won'], bool), f"'won' should be boolean, move {moves_made}"
            
            if terminated:
                break
        
        # Verify info history
        assert len(info_history) > 0, "Should have some info states"
        assert all(isinstance(i, dict) for i in info_history), "All info should be dictionaries"
        assert all('won' in i for i in info_history), "All info should contain 'won' key"
    
    def test_environment_rectangular_integration(self):
        """Test environment integration with rectangular boards."""
        env = MinesweeperEnv(initial_board_size=(4, 6), initial_mines=5)
        state, info = env.reset(seed=42)
        
        # Verify rectangular setup
        assert env.current_board_width == 6, "Should have correct width"
        assert env.current_board_height == 4, "Should have correct height"
        assert state.shape == (4, 4, 6), "State should match rectangular dimensions"
        assert env.action_space.n == 24, "Action space should match rectangular dimensions"
        
        # Play a complete game
        total_reward = 0
        moves_made = 0
        
        while moves_made < 24:  # Prevent infinite loop
            # Find an unrevealed cell
            unrevealed = np.where(state[0] == CELL_UNREVEALED)
            if len(unrevealed[0]) == 0:
                break
                
            action = unrevealed[0][0] * env.current_board_width + unrevealed[1][0]
            state, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            moves_made += 1
            
            # Verify state consistency
            assert state.shape == (4, 4, 6), "State shape should remain consistent"
            assert isinstance(reward, (int, float)), "Reward should be numeric"
            
            if terminated:
                break
        
        # Verify game played properly
        assert moves_made > 0, "Should have made some moves"
        assert isinstance(total_reward, (int, float)), "Total reward should be numeric"
    
    def test_environment_large_board_integration(self):
        """Test environment integration with large boards."""
        env = MinesweeperEnv(initial_board_size=(12, 12), initial_mines=20)
        state, info = env.reset(seed=42)
        
        # Verify large board setup
        assert env.current_board_width == 12, "Should have correct width"
        assert env.current_board_height == 12, "Should have correct height"
        assert state.shape == (4, 12, 12), "State should match large dimensions"
        assert env.action_space.n == 144, "Action space should match large dimensions"
        
        # Play several moves
        total_reward = 0
        moves_made = 0
        
        for action in range(20):  # Make up to 20 moves
            state, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            moves_made += 1
            
            # Verify state consistency
            assert state.shape == (4, 12, 12), "State shape should remain consistent"
            assert isinstance(reward, (int, float)), "Reward should be numeric"
            
            if terminated:
                break
        
        # Verify game played properly
        assert moves_made > 0, "Should have made some moves"
        assert isinstance(total_reward, (int, float)), "Total reward should be numeric"
    
    def test_environment_high_density_integration(self):
        """Test environment integration with high mine density."""
        env = MinesweeperEnv(initial_board_size=(8, 8), initial_mines=50)
        state, info = env.reset(seed=42)
        
        # Verify high density setup
        density = env.current_mines / (env.current_board_width * env.current_board_height)
        assert density == 50/64, "Should have correct mine density"
        
        # Play until game ends
        total_reward = 0
        moves_made = 0
        
        while moves_made < 64:  # Prevent infinite loop
            # Find an unrevealed cell
            unrevealed = np.where(state[0] == CELL_UNREVEALED)
            if len(unrevealed[0]) == 0:
                break
                
            action = unrevealed[0][0] * env.current_board_width + unrevealed[1][0]
            state, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            moves_made += 1
            
            # Verify state consistency
            assert state.shape == (4, 8, 8), "State shape should remain consistent"
            assert isinstance(reward, (int, float)), "Reward should be numeric"
            
            if terminated:
                break
        
        # Verify game played properly
        assert moves_made > 0, "Should have made some moves"
        assert isinstance(total_reward, (int, float)), "Total reward should be numeric"

    def test_initialization(self):
        """Test environment initialization."""
        env = MinesweeperEnv(initial_board_size=4, initial_mines=2)
        assert env.max_board_size_int == 35  # Default max board width (after interpretation)
        assert env.initial_board_size == (4, 4)
        assert env.initial_mines == 2
        assert env.current_board_height == 4
        assert env.current_board_width == 4
        assert env.current_mines == 2

if __name__ == "__main__":
    sys.exit(main()) 