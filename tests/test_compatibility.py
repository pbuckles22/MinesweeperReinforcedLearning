#!/usr/bin/env python3
"""
Test Gymnasium Compatibility with Minesweeper Environment
"""

import os
import sys
import torch
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_compatibility():
    print("🧪 Testing Gymnasium Compatibility with Minesweeper Environment")
    print("=" * 60)
    
    try:
        # Test imports
        from src.core.minesweeper_env import MinesweeperEnv
        from src.core.gym_compatibility import wrap_for_sb3, make_sb3_compatible_env, make_vec_env_sb3_compatible
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        
        print("✅ All imports successful")
        
        # Test 1: Direct environment creation
        print("\n🔍 Test 1: Direct environment creation...")
        env = MinesweeperEnv(max_board_size=(4, 4), max_mines=2)
        print(f"   Environment created: {type(env)}")
        
        # Test 2: Check reset format
        print("\n🔍 Test 2: Check reset format...")
        obs = env.reset()
        print(f"   Reset return type: {type(obs)}")
        if isinstance(obs, tuple):
            print(f"   Reset tuple length: {len(obs)}")
            for i, item in enumerate(obs):
                print(f"   Item {i}: {type(item)}")
        else:
            print(f"   Reset returns: {type(obs)}")
        
        # Test 3: Check step format
        print("\n🔍 Test 3: Check step format...")
        action = 0
        result = env.step(action)
        print(f"   Step return length: {len(result)}")
        for i, item in enumerate(result):
            print(f"   Item {i}: {type(item)}")
        
        # Test 4: Test compatibility wrapper
        print("\n🔍 Test 4: Test compatibility wrapper...")
        wrapped_env = wrap_for_sb3(env)
        print(f"   Wrapped environment: {type(wrapped_env)}")
        
        # Test 5: Test wrapped reset
        print("\n🔍 Test 5: Test wrapped reset...")
        wrapped_obs = wrapped_env.reset()
        print(f"   Wrapped reset type: {type(wrapped_obs)}")
        if isinstance(wrapped_obs, np.ndarray):
            print(f"   Wrapped obs shape: {wrapped_obs.shape}")
        
        # Test 6: Test wrapped step
        print("\n🔍 Test 6: Test wrapped step...")
        wrapped_result = wrapped_env.step(action)
        print(f"   Wrapped step length: {len(wrapped_result)}")
        obs, reward, done, info = wrapped_result
        print(f"   Wrapped obs: {type(obs)}, reward: {reward}, done: {done}")
        
        # Test 7: Test with vectorized environment (direct, no wrapper needed)
        print("\n🔍 Test 7: Test with vectorized environment...")
        def make_env():
            return MinesweeperEnv(max_board_size=(4, 4), max_mines=2)
        
        vec_env = DummyVecEnv([make_env])
        print(f"   VecEnv created: {type(vec_env)}")
        
        # Test 8: Test VecEnv reset
        print("\n🔍 Test 8: Test VecEnv reset...")
        vec_obs = vec_env.reset()
        print(f"   VecEnv obs shape: {vec_obs.shape}")
        
        # Test 9: Test VecEnv step
        print("\n🔍 Test 9: Test VecEnv step...")
        vec_action = [0]  # Single action for single environment
        vec_result = vec_env.step(vec_action)
        vec_obs, vec_reward, vec_done, vec_info = vec_result
        print(f"   VecEnv obs: {vec_obs.shape}, reward: {vec_reward}, done: {vec_done}")
        
        # Test 10: Test PPO creation
        print("\n🔍 Test 10: Test PPO creation...")
        model = PPO("MlpPolicy", vec_env, verbose=0)
        print(f"   PPO model created: {type(model)}")
        
        # Cleanup
        env.close()
        wrapped_env.close()
        vec_env.close()
        
        print("\n🎉 All compatibility tests passed!")
        print("✅ Your environment is now compatible with Stable Baselines3")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    if test_compatibility():
        print("\n🚀 Ready to run training with conservative parameters!")
        print("   Run: python test_conservative_training.py")
    else:
        print("\n⚠️  Compatibility issues detected. Please check the error above.")

if __name__ == "__main__":
    main() 