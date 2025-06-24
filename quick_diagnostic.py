#!/usr/bin/env python3
"""
Quick Diagnostic - Test Environment Setup
"""

import os
import sys
import time
import torch
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_device():
    print("🔍 Testing device detection...")
    if torch.backends.mps.is_available():
        print("✅ MPS (M1 GPU) available")
        device = torch.device("mps")
    elif torch.cuda.is_available():
        print("✅ CUDA available")
        device = torch.device("cuda")
    else:
        print("⚠️  Using CPU")
        device = torch.device("cpu")
    
    print(f"Device: {device}")
    return device

def test_environment():
    print("\n🔍 Testing environment creation...")
    try:
        from src.core.minesweeper_env import MinesweeperEnv
        
        start_time = time.time()
        env = MinesweeperEnv(max_board_size=(4, 4), max_mines=2)
        creation_time = time.time() - start_time
        
        print(f"✅ Environment created in {creation_time:.3f}s")
        
        # Test reset
        start_time = time.time()
        obs = env.reset()
        reset_time = time.time() - start_time
        
        print(f"✅ Environment reset in {reset_time:.3f}s")
        print(f"Observation type: {type(obs)}")
        if isinstance(obs, np.ndarray):
            print(f"Observation shape: {obs.shape}")
        elif isinstance(obs, tuple):
            print(f"Observation tuple length: {len(obs)}")
            for i, item in enumerate(obs):
                print(f"  Item {i}: {type(item)}, shape: {getattr(item, 'shape', 'N/A')}")
        
        # Test step
        start_time = time.time()
        action = 0  # First cell
        obs, reward, done, info = env.step(action)
        step_time = time.time() - start_time
        
        print(f"✅ Environment step in {step_time:.3f}s")
        print(f"Reward: {reward}, Done: {done}")
        print(f"Info keys: {list(info.keys()) if info else 'None'}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Environment error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_imports():
    print("\n🔍 Testing imports...")
    try:
        from stable_baselines3 import PPO
        print("✅ Stable Baselines3 imported")
        
        from stable_baselines3.common.vec_env import DummyVecEnv
        print("✅ DummyVecEnv imported")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_vec_env():
    print("\n🔍 Testing vectorized environment...")
    try:
        from src.core.minesweeper_env import MinesweeperEnv
        from stable_baselines3.common.vec_env import DummyVecEnv
        
        def make_env():
            return MinesweeperEnv(max_board_size=(4, 4), max_mines=2)
        
        start_time = time.time()
        vec_env = DummyVecEnv([make_env])
        creation_time = time.time() - start_time
        
        print(f"✅ VecEnv created in {creation_time:.3f}s")
        
        # Test reset
        start_time = time.time()
        obs = vec_env.reset()
        reset_time = time.time() - start_time
        
        print(f"✅ VecEnv reset in {reset_time:.3f}s")
        print(f"VecEnv observation shape: {obs.shape}")
        
        vec_env.close()
        return True
        
    except Exception as e:
        print(f"❌ VecEnv error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 Quick Diagnostic Test")
    print("=" * 30)
    
    # Test device
    device = test_device()
    
    # Test imports
    if not test_imports():
        return
    
    # Test environment
    if not test_environment():
        return
    
    # Test vectorized environment
    if not test_vec_env():
        return
    
    print("\n✅ All tests passed! Environment should work.")
    print("\n💡 If training still hangs, try:")
    print("   1. Smaller batch size (16 instead of 32)")
    print("   2. Fewer timesteps (10,000 instead of 50,000)")
    print("   3. Check if M1 GPU is being used properly")

if __name__ == "__main__":
    main() 