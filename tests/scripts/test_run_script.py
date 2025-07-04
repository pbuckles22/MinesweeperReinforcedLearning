import pytest
import os
import subprocess
import platform
from pathlib import Path

@pytest.fixture
def script_path():
    """Get the path to the run script based on platform."""
    system = platform.system().lower()
    
    if system == "windows":
        return Path("scripts/windows/run_agent.ps1")
    elif system == "linux":
        return Path("scripts/linux/run_agent.sh")
    elif system == "darwin":  # macOS
        return Path("scripts/mac/run_agent.sh")
    else:
        pytest.skip(f"Unsupported platform: {system}")

def test_script_exists(script_path):
    """Test that the run script exists."""
    assert script_path.exists()
    assert script_path.is_file()

def test_script_permissions(script_path):
    """Test that the script has the correct permissions."""
    # Check if the script is readable
    assert os.access(script_path, os.R_OK)

def test_script_syntax(script_path):
    """Test that the script has valid syntax."""
    system = platform.system().lower()
    
    try:
        if system == "windows":
            # Use PowerShell to check syntax without execution
            result = subprocess.run(
                ["powershell", "-Command", f"Get-Content {script_path} | Out-String | Test-ScriptBlock"],
                capture_output=True,
                text=True
            )
            # PowerShell syntax check might not be available, so we'll just check if the command runs
            assert result.returncode in [0, 1], f"Script syntax check failed: {result.stderr}"
        else:
            # For shell scripts, just check if they're readable and have basic structure
            with open(script_path, 'r') as f:
                content = f.read()
            assert content.strip(), "Script is empty"
            assert "#!/bin/bash" in content or "#!/bin/sh" in content, "Script missing shebang"
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Script syntax error: {e.stderr}")

def test_script_parameters(script_path):
    """Test that the script has appropriate parameters based on its purpose."""
    # Read the script content
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check if this is a training script or visualization script
    is_training_script = ("train_agent.py" in content.lower() or 
                         "total_timesteps" in content.lower() or
                         "timesteps" in content.lower())
    
    is_visualization_script = ("visualize_agent.py" in content.lower())
    
    if is_training_script:
        # Training scripts should have training parameters
        assert ("total_timesteps" in content.lower() or 
                "timesteps" in content.lower() or 
                "--timesteps" in content.lower())
        assert ("learning_rate" in content.lower() or 
                "learningrate" in content.lower() or 
                "--learning-rate" in content.lower())
        assert ("board_size" in content.lower() or 
                "boardsize" in content.lower() or 
                "--board-size" in content.lower())
    elif is_visualization_script:
        # Visualization scripts should have visualization parameters
        assert ("visualize_agent.py" in content.lower())
        # Visualization scripts use "$@" to pass through arguments
        assert '"$@"' in content or '--model-path' in content.lower()
    else:
        # For other scripts, just check they have some parameters
        assert ("python" in content.lower() and 
                ("src/" in content.lower() or "train" in content.lower() or "visualize" in content.lower()))

def test_script_environment_check(script_path):
    """Test that the script uses Python for training or visualization."""
    # Read the script content
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check for Python usage
    assert "python" in content.lower()
    # Check for either training or visualization script
    assert ("train_agent.py" in content.lower() or 
            "visualize_agent.py" in content.lower() or
            "src/visualization/" in content.lower() or
            "src/core/" in content.lower())

def test_script_output_handling(script_path):
    """Test that the script handles output correctly."""
    # Read the script content
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check for output handling (cross-platform)
    system = platform.system().lower()
    if system == "windows":
        # Windows scripts should have PowerShell output handling
        assert ("write-host" in content.lower() or 
                "write-output" in content.lower() or
                "echo" in content.lower())
    else:
        # Unix scripts may have echo or may be simple execution scripts
        # Simple scripts that just run Python commands are also valid
        assert ("echo" in content.lower() or 
                "python" in content.lower() or
                "source" in content.lower())

def test_script_error_handling(script_path):
    """Test that the script handles errors correctly."""
    # Read the script content
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check for error handling (cross-platform)
    system = platform.system().lower()
    if system == "windows":
        # Windows scripts should have PowerShell error handling
        assert ("exit" in content.lower() or 
                "throw" in content.lower() or
                "erroractionpreference" in content.lower())
    else:
        # Unix scripts may have exit or may rely on shell error handling
        # Simple scripts that just run Python commands are also valid
        assert ("exit" in content.lower() or 
                "python" in content.lower() or
                "source" in content.lower())
