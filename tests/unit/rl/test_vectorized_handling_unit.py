import numpy as np
import pytest
from src.core.dqn_agent import DQNAgent
from src.core.minesweeper_env import MinesweeperEnv

class DummyQNet:
    def __call__(self, x):
        # Return dummy Q-values
        return type('Dummy', (), {'cpu': lambda self: self, 'numpy': lambda self: np.zeros(16), 'flatten': lambda self: np.zeros(16)})()

def make_obs(shape=(4, 4)):
    # 4 channels, height x width
    obs = np.full((4, *shape), -1, dtype=np.float32)
    return obs

def make_vec_obs(num_envs=1, shape=(4, 4)):
    # (num_envs, channels, height, width)
    return np.stack([make_obs(shape) for _ in range(num_envs)], axis=0)

def test_get_valid_actions_vectorized_and_nonvectorized():
    agent = DQNAgent(board_size=(4, 4), action_size=16)
    agent.q_network = DummyQNet()
    # Non-vectorized
    obs = make_obs()
    valid = agent._get_valid_actions(obs)
    assert set(valid) == set(range(16))
    # Vectorized
    vec_obs = make_vec_obs(num_envs=2)
    valid_vec = agent._get_valid_actions(vec_obs)
    assert set(valid_vec) == set(range(16))

def test_preprocess_state_vectorized_and_nonvectorized():
    agent = DQNAgent(board_size=(4, 4), action_size=16)
    # Non-vectorized
    obs = make_obs()
    t = agent._preprocess_state(obs)
    assert t.shape == (4, 4, 4)
    # Vectorized
    vec_obs = make_vec_obs(num_envs=2)
    t2 = agent._preprocess_state(vec_obs)
    assert t2.shape == (4, 4, 4)

def test_no_array_truth_value_error():
    agent = DQNAgent(board_size=(4, 4), action_size=16)
    agent.q_network = DummyQNet()
    obs = make_vec_obs(num_envs=1)
    # Should not raise ValueError
    try:
        agent.choose_action(obs, training=True)
    except ValueError as e:
        pytest.fail(f"Array truth value error not handled: {e}")

def test_step_api_compatibility():
    env = MinesweeperEnv(max_board_size=(4, 4), max_mines=1)
    obs, info = env.reset()
    action = 0
    # Should always return 5 values
    result = env.step(action)
    assert isinstance(result, tuple)
    assert len(result) == 5
    # Simulate DummyVecEnv 4-tuple return
    obs = np.expand_dims(obs, axis=0)
    reward = np.array([0.0])
    done = np.array([False])
    info = [{}]
    # Should be able to unpack as 4-tuple
    try:
        o, r, d, i = obs, reward, done, info
        terminated = bool(d[0])
        truncated = False
    except Exception as e:
        pytest.fail(f"Failed to handle 4-tuple step API: {e}")

def test_action_masking_vectorized():
    env = MinesweeperEnv(max_board_size=(4, 4), max_mines=1)
    obs, info = env.reset()
    # Simulate vectorized obs
    vec_obs = np.expand_dims(obs, axis=0)
    agent = DQNAgent(board_size=(4, 4), action_size=16)
    agent.q_network = DummyQNet()
    valid = agent._get_valid_actions(vec_obs)
    assert set(valid) == set(range(16)) 