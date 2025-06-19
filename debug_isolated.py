#!/usr/bin/env python3
"""Isolated test to debug the hanging issue."""

class SimpleEnv:
    """Simple environment class for testing."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

def get_env_attr_simple(env, attr):
    """Simplified version of get_env_attr to test the logic."""
    # Recursively unwrap until the attribute is found or no more wrappers
    seen = set()  # Track seen environments to prevent infinite loops
    current_env = env
    
    while current_env is not None:
        env_id = id(current_env)
        if env_id in seen:
            break  # Prevent infinite loops
        seen.add(env_id)
        
        # Check if current environment has the attribute
        if hasattr(current_env, attr):
            return getattr(current_env, attr)
        
        # Try to get the env attribute, break if it doesn't exist
        try:
            current_env = getattr(current_env, 'env', None)
        except AttributeError:
            break
            
    return None

def test_isolated():
    """Test the isolated get_env_attr function."""
    print("🔍 Testing isolated get_env_attr...")
    
    # Create simple environment objects with nested wrappers
    inner_env = SimpleEnv(test_attr="test_value")
    
    middle_env = SimpleEnv()
    middle_env.env = inner_env
    
    outer_env = SimpleEnv()
    outer_env.env = middle_env
    
    print("✅ Created simple environments")
    
    # Test attribute retrieval
    print("🔍 Testing attribute retrieval...")
    result = get_env_attr_simple(outer_env, "test_attr")
    print(f"✅ Result: {result}")
    assert result == "test_value"
    
    # Test non-existent attribute
    print("🔍 Testing non-existent attribute...")
    result = get_env_attr_simple(outer_env, "non_existent")
    print(f"✅ Result: {result}")
    assert result is None
    
    # Test with environment that has no wrappers
    print("🔍 Testing simple environment...")
    simple_env = SimpleEnv(test_attr="simple_value")
    result = get_env_attr_simple(simple_env, "test_attr")
    print(f"✅ Result: {result}")
    assert result == "simple_value"
    
    print("🎉 Isolated test passed!")

if __name__ == "__main__":
    test_isolated() 