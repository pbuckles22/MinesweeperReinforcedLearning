#!/usr/bin/env python3
"""
M1 GPU Verification Script
Run this to verify your M1 setup is working correctly.
"""

import torch
import numpy as np
from src.core.minesweeper_env import MinesweeperEnv

def test_m1_setup():
    print("🔍 M1 MacBook Setup Verification")
    print("=" * 40)
    
    # 1. Check PyTorch and MPS
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    
    # 2. Test GPU operations
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✅ Using MPS device: {device}")
        
        # Test tensor operations
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        import time
        start_time = time.time()
        z = torch.mm(x, y)
        end_time = time.time()
        
        print(f"✅ GPU matrix multiplication: {end_time - start_time:.3f}s")
        print(f"✅ Result shape: {z.shape}")
    else:
        print("❌ MPS not available, using CPU")
        device = torch.device("cpu")
    
    # 3. Test environment creation
    print("\n🔍 Testing Environment Creation")
    try:
        env = MinesweeperEnv(max_board_size=(8, 8), max_mines=10)
        print("✅ Environment creation successful")
        
        # Test a few steps
        obs, info = env.reset()
        print(f"✅ Environment reset successful, observation shape: {obs.shape}")
        
        # Test action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✅ Environment step successful, reward: {reward}")
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
    
    # 4. Test training imports
    print("\n🔍 Testing Training Imports")
    try:
        from src.core.train_agent import main
        print("✅ Training agent import successful")
    except Exception as e:
        print(f"❌ Training agent import failed: {e}")
    
    print("\n🎉 Setup verification complete!")

if __name__ == "__main__":
    test_m1_setup() 