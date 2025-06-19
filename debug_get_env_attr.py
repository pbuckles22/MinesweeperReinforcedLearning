#!/usr/bin/env python3
"""Debug the get_env_attr method."""

import sys
sys.path.append('src')

from unittest.mock import Mock
from core.train_agent import IterationCallback

def debug_get_env_attr():
    """Debug the get_env_attr method."""
    print("🔍 Debugging get_env_attr method...")
    
    # Create callback
    callback = IterationCallback()
    
    # Create a mock environment with nested wrappers
    inner_env = Mock()
    inner_env.test_attr = "test_value"
    
    middle_env = Mock()
    middle_env.env = inner_env
    
    outer_env = Mock()
    outer_env.env = middle_env
    
    print("✅ Created mock environments")
    
    # Test attribute retrieval
    print("🔍 Testing attribute retrieval...")
    result = callback.get_env_attr(outer_env, "test_attr")
    print(f"✅ Result: {result}")
    assert result == "test_value"
    
    # Test non-existent attribute
    print("🔍 Testing non-existent attribute...")
    result = callback.get_env_attr(outer_env, "non_existent")
    print(f"✅ Result: {result}")
    assert result is None
    
    # Test with environment that has no wrappers
    print("🔍 Testing simple environment...")
    simple_env = Mock()
    simple_env.test_attr = "simple_value"
    result = callback.get_env_attr(simple_env, "test_attr")
    print(f"✅ Result: {result}")
    assert result == "simple_value"
    
    print("🎉 All tests passed!")

if __name__ == "__main__":
    debug_get_env_attr() 