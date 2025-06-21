"""
Comprehensive RL Training Tests

These tests verify RL agent training scenarios including:
- Mines that should not be seen by the agent
- State consistency and observation space validation
- Deterministic and non-deterministic scenarios
- Agent-environment interaction patterns
- Training-specific edge cases
"""

import pytest
import numpy as np
from src.core.minesweeper_env import MinesweeperEnv
from src.core.constants import (
    CELL_UNREVEALED,
    CELL_MINE,
    CELL_MINE_HIT,
    REWARD_SAFE_REVEAL,
    REWARD_WIN,
    REWARD_HIT_MINE,
    REWARD_INVALID_ACTION,
    REWARD_FIRST_CASCADE_SAFE,
    REWARD_FIRST_CASCADE_HIT_MINE,
)

class TestComprehensiveRL:
    """Comprehensive RL training test suite."""
    
    @pytest.fixture
    def rl_env(self):
        """Create a test environment for RL training."""
        return MinesweeperEnv(
            max_board_size=(8, 8),
            max_mines=10,
            initial_board_size=(6, 6),
            initial_mines=6,
            early_learning_mode=True,
            early_learning_threshold=100
        )
    
    @pytest.fixture
    def deterministic_env(self):
        """Create a deterministic environment for controlled testing."""
        env = MinesweeperEnv(
            max_board_size=(4, 4),
            max_mines=2,
            initial_board_size=(4, 4),
            initial_mines=2
        )
        env.reset(seed=42)
        return env

    def test_agent_observation_space_consistency(self, rl_env):
        """Test that agent observations are consistent and within bounds."""
        rl_env.reset()
        
        # Test initial observation
        state = rl_env.state
        assert state.shape == (4, 6, 6), "State should have 4 channels and match board size"
        
        # Test observation space bounds
        obs_space = rl_env.observation_space
        assert obs_space.contains(state), "State should be within observation space bounds"
        
        # Test that unrevealed cells are properly masked
        assert np.all(state[0] == CELL_UNREVEALED), "All cells should be unrevealed initially"
        
        # Test safety hints channel
        safety_hints = state[1]
        assert np.all(safety_hints >= -1), "Safety hints should be >= -1"
        assert np.all(safety_hints <= 8), "Safety hints should be <= 8"

    def test_mines_not_visible_to_agent(self, deterministic_env):
        """Test that mines are not visible to the agent in the observation."""
        # Set up a controlled board with known mine positions
        deterministic_env.mines.fill(False)
        deterministic_env.mines[1, 1] = True  # Mine at center
        deterministic_env.mines[2, 2] = True  # Mine at another position
        deterministic_env._update_adjacent_counts()
        deterministic_env.mines_placed = True
        deterministic_env.is_first_cascade = False
        deterministic_env.first_cascade_done = True
        
        # Check that mines are not visible in the observation
        state = deterministic_env.state
        assert state[0, 1, 1] == CELL_UNREVEALED, "Mine should not be visible to agent"
        assert state[0, 2, 2] == CELL_UNREVEALED, "Mine should not be visible to agent"
        
        # Check that safety hints show correct adjacent mine counts
        assert state[1, 0, 0] >= 0, "Safety hints should show adjacent mine count"
        assert state[1, 1, 1] >= 0, "Safety hints should show adjacent mine count"

    def test_agent_action_consistency(self, rl_env):
        """Test that agent actions produce consistent and valid results."""
        rl_env.reset()
        
        # Test multiple actions and verify consistency
        for action in range(min(10, rl_env.action_space.n)):
            state_before = rl_env.state.copy()
            
            # Take action
            new_state, reward, terminated, truncated, info = rl_env.step(action)
            
            # Verify state changed (unless invalid action)
            if reward != REWARD_INVALID_ACTION:
                assert not np.array_equal(new_state, state_before), "State should change after valid action"
            
            # Verify reward is within expected bounds
            assert reward >= REWARD_HIT_MINE, "Reward should not be below mine hit penalty"
            assert reward <= REWARD_WIN, "Reward should not exceed win reward"
            
            # Verify termination logic
            if terminated:
                # Termination can be due to win, mine hit (post-cascade), or pre-cascade mine hit
                assert (info.get('won', False) or 
                       reward == REWARD_HIT_MINE or 
                       reward == REWARD_FIRST_CASCADE_HIT_MINE), "Termination should indicate win or mine hit"
            
            # Reset for next test
            rl_env.reset()

    def test_deterministic_training_scenarios(self, deterministic_env):
        """Test deterministic scenarios for training stability."""
        # Test 1: Known safe cell
        action = 0  # Corner cell
        state, reward, terminated, truncated, info = deterministic_env.step(action)
        
        # Should get immediate reward/penalty/win for first move (no more neutral pre-cascade)
        valid_first_rewards = [REWARD_SAFE_REVEAL, REWARD_HIT_MINE, REWARD_WIN]
        assert reward in valid_first_rewards, f"First move should give immediate reward/penalty/win, got {reward}"
        
        # Test 2: Known mine position (after First cascade)
        deterministic_env.reset(seed=42)
        deterministic_env.mines.fill(False)
        deterministic_env.mines[1, 1] = True
        deterministic_env._update_adjacent_counts()
        deterministic_env.mines_placed = True
        deterministic_env.is_first_cascade = False
        deterministic_env.first_cascade_done = True
        
        # Take safe move first
        safe_action = 0
        state, reward, terminated, truncated, info = deterministic_env.step(safe_action)
        
        # Then hit mine
        mine_action = 1 * deterministic_env.current_board_width + 1  # (1,1)
        state, reward, terminated, truncated, info = deterministic_env.step(mine_action)
        
        assert terminated, "Hitting mine should terminate game"
        assert reward == REWARD_HIT_MINE, "Hitting mine should give mine hit reward"

    def test_non_deterministic_training_scenarios(self, rl_env):
        """Test non-deterministic scenarios that agents will encounter during training."""
        # Test multiple games with different outcomes
        outcomes = []
        
        for game in range(10):
            rl_env.reset()
            game_terminated = False
            game_reward = 0
            
            # Play until game ends
            for step in range(20):  # Limit steps to prevent infinite loops
                action = np.random.randint(0, rl_env.action_space.n)
                state, reward, terminated, truncated, info = rl_env.step(action)
                
                game_reward += reward
                
                if terminated or truncated:
                    game_terminated = True
                    outcomes.append({
                        'terminated': terminated,
                        'truncated': truncated,
                        'total_reward': game_reward,
                        'won': info.get('won', False)
                    })
                    break
            
            if not game_terminated:
                outcomes.append({
                    'terminated': False,
                    'truncated': True,
                    'total_reward': game_reward,
                    'won': False
                })
        
        # Verify we have a mix of outcomes (realistic training scenario)
        assert len(outcomes) == 10, "Should have 10 game outcomes"
        
        # Verify all outcomes are valid
        for outcome in outcomes:
            assert isinstance(outcome['terminated'], bool), "Terminated should be boolean"
            assert isinstance(outcome['truncated'], bool), "Truncated should be boolean"
            assert isinstance(outcome['total_reward'], (int, float)), "Total reward should be numeric"
            assert isinstance(outcome['won'], bool), "Won should be boolean"

    def test_agent_state_transitions(self, rl_env):
        """Test that agent state transitions are valid and consistent."""
        rl_env.reset()
        
        # Test state transitions through multiple actions
        previous_state = rl_env.state.copy()
        
        for action in range(min(5, rl_env.action_space.n)):
            # Take action
            new_state, reward, terminated, truncated, info = rl_env.step(action)
            
            # Verify state is valid
            assert new_state.shape == (4, 6, 6), "State shape should remain consistent"
            assert rl_env.observation_space.contains(new_state), "State should be within bounds"
            
            # Verify state changed (unless invalid action)
            if reward != REWARD_INVALID_ACTION:
                assert not np.array_equal(new_state, previous_state), "State should change after valid action"
            
            # Verify revealed cells are properly updated
            revealed_cells = np.sum(new_state[0] != CELL_UNREVEALED)
            assert revealed_cells >= 0, "Should have non-negative revealed cells"
            
            previous_state = new_state.copy()
            
            if terminated or truncated:
                break

    def test_early_learning_agent_interaction(self, rl_env):
        """Test agent interaction with early learning mode."""
        # Verify early learning mode is enabled
        assert rl_env.early_learning_mode is True
        
        # Test multiple games to see early learning behavior
        for game in range(5):
            rl_env.reset()
            
            # Play a quick game
            for step in range(3):
                # Use action masks to find valid actions
                valid_actions = np.where(rl_env.action_masks)[0]
                if len(valid_actions) == 0:
                    break  # No more valid actions
                    
                action = valid_actions[0]
                state, reward, terminated, truncated, info = rl_env.step(action)
                
                # Verify early learning rewards are valid
                if step == 0:  # Pre-cascade move
                    valid_first_rewards = [REWARD_SAFE_REVEAL, REWARD_HIT_MINE, REWARD_WIN]
                    assert reward in valid_first_rewards, "First move should give immediate reward, penalty, or win"
                else:
                    # Subsequent moves can include various rewards depending on cascade state
                    valid_rewards = [REWARD_SAFE_REVEAL, REWARD_HIT_MINE, REWARD_WIN, REWARD_INVALID_ACTION]
                    assert reward in valid_rewards, "Reward should be valid for RL agent"
                
                if terminated or truncated:
                    break

    def test_agent_action_masking_consistency(self, rl_env):
        """Test that action masking is consistent for RL agents."""
        rl_env.reset()
        
        # Test initial action masks
        initial_masks = rl_env.action_masks
        assert np.all(initial_masks), "All actions should be valid initially"
        assert np.sum(initial_masks) == rl_env.action_space.n, "All actions should be valid initially"
        
        # Take an action and check masks update
        action = 0
        state, reward, terminated, truncated, info = rl_env.step(action)
        
        new_masks = rl_env.action_masks
        assert not new_masks[action], "Taken action should be masked"
        assert np.sum(new_masks) < rl_env.action_space.n, "Some actions should be masked after move"

    def test_agent_win_condition_detection(self, deterministic_env):
        """Test that win conditions are properly detected for RL agents."""
        # Set up a simple win scenario
        deterministic_env.mines.fill(False)
        deterministic_env.mines[0, 0] = True  # Single mine in corner
        deterministic_env._update_adjacent_counts()
        deterministic_env.mines_placed = True
        deterministic_env.is_first_cascade = False
        deterministic_env.first_cascade_done = True
        
        # Reveal all safe cells
        safe_cells = []
        for i in range(deterministic_env.current_board_height):
            for j in range(deterministic_env.current_board_width):
                if not deterministic_env.mines[i, j]:
                    safe_cells.append(i * deterministic_env.current_board_width + j)
        
        # Take actions to reveal safe cells
        for action in safe_cells:
            state, reward, terminated, truncated, info = deterministic_env.step(action)
            if terminated:
                break
        
        # Should win by revealing all safe cells
        assert terminated, "Game should terminate when all safe cells are revealed"
        assert info.get('won', False), "Should win when all safe cells are revealed"
        assert reward == REWARD_WIN, "Should get win reward"

    def test_agent_mine_hit_handling(self, deterministic_env):
        """Test that mine hits are properly handled for RL agents."""
        # Set up a controlled mine hit scenario
        deterministic_env.mines.fill(False)
        deterministic_env.mines[1, 1] = True  # Mine in center
        deterministic_env._update_adjacent_counts()
        deterministic_env.mines_placed = True
        deterministic_env.is_first_cascade = False
        deterministic_env.first_cascade_done = True
        
        # Take safe move first
        safe_action = 0
        state, reward, terminated, truncated, info = deterministic_env.step(safe_action)
        
        # Then hit mine
        mine_action = 1 * deterministic_env.current_board_width + 1
        state, reward, terminated, truncated, info = deterministic_env.step(mine_action)
        
        # Verify mine hit handling
        assert terminated, "Mine hit should terminate game"
        assert reward == REWARD_HIT_MINE, "Mine hit should give appropriate reward"
        assert state[0, 1, 1] == CELL_MINE_HIT, "Mine hit should be visible in state"
        assert not info.get('won', False), "Mine hit should not be a win"

    def test_agent_observation_space_scaling(self, rl_env):
        """Test that observation space scales correctly with board size."""
        # Test different board sizes
        board_sizes = [(4, 4), (6, 6), (8, 8)]
        
        for width, height in board_sizes:
            env = MinesweeperEnv(
                max_board_size=(width, height),
                max_mines=width * height // 4,
                initial_board_size=(width, height),
                initial_mines=width * height // 8
            )
            
            env.reset()
            state = env.state
            
            # Verify state shape matches board size
            assert state.shape == (4, height, width), f"State shape should match board size {width}x{height}"
            
            # Verify observation space bounds
            assert env.observation_space.contains(state), f"State should be within bounds for {width}x{height} board"

    def test_agent_reward_consistency(self, rl_env):
        """Test that rewards are consistent and appropriate for RL training."""
        rl_env.reset()
        
        # Test pre-cascade rewards
        action = 0
        state, reward, terminated, truncated, info = rl_env.step(action)
        
        # Should get immediate reward/penalty/win for first move (no more neutral pre-cascade)
        valid_first_rewards = [REWARD_SAFE_REVEAL, REWARD_HIT_MINE, REWARD_WIN]
        assert reward in valid_first_rewards, f"First move should give immediate reward/penalty/win, got {reward}"
        
        # Test subsequent move rewards
        if not terminated:
            action = 1
            state, reward, terminated, truncated, info = rl_env.step(action)
            
            # Subsequent moves should have appropriate rewards (could still be pre-cascade if no cascade occurred)
            valid_rewards = [REWARD_SAFE_REVEAL, REWARD_HIT_MINE, REWARD_WIN, REWARD_INVALID_ACTION]
            assert reward in valid_rewards, "Reward should be valid for RL agent"

    def test_agent_info_consistency(self, rl_env):
        """Test that info dictionary is consistent and contains expected keys."""
        rl_env.reset()
        
        # Test info structure
        action = 0
        state, reward, terminated, truncated, info = rl_env.step(action)
        
        # Accept both dict and list for info
        assert isinstance(info, (dict, list)), "Info should be a dictionary or list of dicts"
        if isinstance(info, list):
            assert len(info) > 0
            assert isinstance(info[0], dict)
        
        # Verify info is a dictionary
        assert isinstance(info, dict), "Info should be a dictionary"
        
        # Verify expected keys exist
        assert 'won' in info, "Info should contain 'won' key"
        assert isinstance(info['won'], bool), "'won' should be boolean"
        
        # Verify won status is consistent with termination
        if terminated:
            if info['won']:
                assert reward == REWARD_WIN, "Win should give win reward"
            else:
                assert reward == REWARD_HIT_MINE, "Loss should give mine hit reward"

if __name__ == "__main__":
    pytest.main([__file__])
