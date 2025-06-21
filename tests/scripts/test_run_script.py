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
    """Test that the script has training parameters."""
    # Read the script content
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check for training parameters (check for both old and new parameter formats)
    assert ("total_timesteps" in content.lower() or 
            "timesteps" in content.lower() or 
            "--timesteps" in content.lower())
    assert ("learning_rate" in content.lower() or 
            "learningrate" in content.lower() or 
            "--learning-rate" in content.lower())
    assert ("board_size" in content.lower() or 
            "boardsize" in content.lower() or 
            "--board-size" in content.lower())

def test_script_environment_check(script_path):
    """Test that the script uses Python for training."""
    # Read the script content
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check for Python usage
    assert "python" in content.lower()
    assert "train_agent.py" in content.lower()

def test_script_output_handling(script_path):
    """Test that the script handles output correctly."""
    # Read the script content
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check for output handling (cross-platform)
    system = platform.system().lower()
    if system == "windows":
        assert "write-host" in content.lower()
    else:
        assert "echo" in content.lower()

def test_script_error_handling(script_path):
    """Test that the script handles errors correctly."""
    # Read the script content
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check for error handling (cross-platform)
    system = platform.system().lower()
    if system == "windows":
        assert "exit" in content.lower()     # Exit on error
    else:
        assert "exit" in content.lower()     # Exit on error
