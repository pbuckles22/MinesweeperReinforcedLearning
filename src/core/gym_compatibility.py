"""
Gymnasium Compatibility Wrapper for Stable Baselines3

This module provides compatibility between gymnasium and Stable Baselines3.
It automatically wraps environments to ensure they work correctly with SB3.
"""

import gymnasium as gym
import numpy as np
from typing import Union, Tuple, Dict, Any, Optional

class GymnasiumCompatibilityWrapper(gym.Wrapper):
    """
    Wrapper to ensure gymnasium environments work with Stable Baselines3.
    
    This wrapper handles the gymnasium API format:
    - reset() -> (obs, info)
    - step() -> (obs, reward, terminated, truncated, info)
    
    And converts it to the format SB3 expects:
    - reset() -> obs
    - step() -> (obs, reward, done, info)
    """
    
    def __init__(self, env):
        super().__init__(env)
        
    def reset(self, **kwargs):
        """Ensure reset returns only the observation."""
        result = self.env.reset(**kwargs)
        
        # Gymnasium API: reset() returns (obs, info)
        if isinstance(result, tuple):
            obs, info = result
            # Store info for potential future use
            self._last_reset_info = info
            return obs
        else:
            return result
    
    def step(self, action):
        """Ensure step returns (obs, reward, done, info)."""
        result = self.env.step(action)
        
        # Gymnasium API: step() returns (obs, reward, terminated, truncated, info)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
            return obs, reward, done, info
        else:
            return result

def wrap_for_sb3(env):
    """
    Wrap an environment to ensure it's compatible with Stable Baselines3.
    
    Args:
        env: The gymnasium environment to wrap
        
    Returns:
        Wrapped environment that's compatible with SB3
    """
    return GymnasiumCompatibilityWrapper(env)

def make_sb3_compatible_env(env_constructor, *args, **kwargs):
    """
    Create an environment that's compatible with Stable Baselines3.
    
    Args:
        env_constructor: Function that creates the environment
        *args, **kwargs: Arguments to pass to env_constructor
        
    Returns:
        Wrapped environment that's compatible with SB3
    """
    env = env_constructor(*args, **kwargs)
    return wrap_for_sb3(env)

def make_vec_env_sb3_compatible(env_constructor, n_envs=1, **kwargs):
    """
    Create a vectorized environment that's compatible with Stable Baselines3.
    
    Args:
        env_constructor: Function that creates the environment
        n_envs: Number of environments to create
        **kwargs: Arguments to pass to env_constructor
        
    Returns:
        Vectorized environment that's compatible with SB3
    """
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    def make_env():
        env = env_constructor(**kwargs)
        return wrap_for_sb3(env)
    
    return DummyVecEnv([make_env for _ in range(n_envs)]) 