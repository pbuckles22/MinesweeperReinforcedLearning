#!/usr/bin/env python3
"""Simple test to isolate RL test issues."""

import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from unittest.mock import Mock
from core.train_agent import IterationCallback

def test_simple_get_env_attr():
    """Simple test of get_env_attr method."""
    print("🔍 Testing simple get_env_attr...")
    
    # Create callback
    callback = IterationCallback()
    
    # Create a simple mock environment
    simple_env = Mock()
    simple_env.test_attr = "simple_value"
    
    # Test attribute retrieval
    result = callback.get_env_attr(simple_env, "test_attr")
    print(f"✅ Result: {result}")
    assert result == "simple_value"
    
    print("🎉 Simple test passed!")

if __name__ == "__main__":
    test_simple_get_env_attr() 