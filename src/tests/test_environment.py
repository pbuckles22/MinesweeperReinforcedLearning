import os
import sys
import numpy as np
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from minesweeper_env import MinesweeperEnv

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    try:
        import numpy as np
        import pygame
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from minesweeper_env import MinesweeperEnv
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_environment_creation():
    """Test that the environment can be created and reset"""
    print("\nTesting environment creation...")
    try:
        env = MinesweeperEnv(board_size=3, num_mines=1)
        state, _ = env.reset()
        print("✓ Environment created and reset successfully")
        print(f"✓ State shape: {state.shape}")
        return True
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        return False

def test_basic_actions():
    """Test that basic actions work in the environment"""
    print("\nTesting basic actions...")
    try:
        env = MinesweeperEnv(board_size=3, num_mines=1)
        state, _ = env.reset()
        
        # Test a reveal action
        action = 0  # Reveal first cell
        state, reward, terminated, truncated, _ = env.step(action)
        print("✓ Basic action successful")
        print(f"✓ Reward: {reward}")
        print(f"✓ Terminated: {terminated}")
        return True
    except Exception as e:
        print(f"✗ Basic action test failed: {e}")
        return False

def test_pygame():
    """Test that pygame can be initialized"""
    print("\nTesting pygame...")
    try:
        pygame.init()
        pygame.quit()
        print("✓ Pygame initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Pygame initialization failed: {e}")
        return False

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

if __name__ == "__main__":
    sys.exit(main()) 