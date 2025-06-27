#!/usr/bin/env python3
"""
Pre-Commit Hook Script

This script runs before each commit to ensure:
- Cross-platform test compatibility
- Code coverage thresholds are met
- Code quality standards are maintained
- No breaking changes are introduced

Usage:
    python scripts/pre_commit_hook.py
    # Or as a git hook: .git/hooks/pre-commit
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Any

def get_staged_files() -> List[str]:
    """Get list of staged files."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return result.stdout.strip().split('\n') if result.stdout.strip() else []
    except Exception:
        pass
    return []

def is_test_file(file_path: str) -> bool:
    """Check if file is a test file."""
    return (
        file_path.startswith('tests/') or
        file_path.endswith('_test.py') or
        'test_' in file_path
    )

def is_script_file(file_path: str) -> bool:
    """Check if file is a script file."""
    return (
        file_path.startswith('scripts/') or
        file_path.endswith('.sh') or
        file_path.endswith('.ps1')
    )

def run_quick_tests() -> Dict[str, Any]:
    """Run quick test suite."""
    print("🧪 Running quick tests...")
    
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "pytest",
                "tests/unit/core/test_core_initialization_unit.py",
                "tests/functional/game_flow/test_game_flow_functional.py",
                "-v", "--tb=short", "--maxfail=3"
            ],
            capture_output=True,
            text=True,
            timeout=120  # 2 minutes
        )
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": "Quick tests timed out after 2 minutes"
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Error running quick tests: {str(e)}"
        }

def validate_cross_platform_consistency() -> Dict[str, Any]:
    """Validate cross-platform test consistency."""
    print("🔍 Validating cross-platform consistency...")
    
    try:
        result = subprocess.run(
            [sys.executable, "scripts/validate_cross_platform_tests.py"],
            capture_output=True,
            text=True,
            timeout=60  # 1 minute
        )
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": "Cross-platform validation timed out"
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Error validating cross-platform consistency: {str(e)}"
        }

def check_coverage_threshold() -> Dict[str, Any]:
    """Check if coverage meets minimum threshold."""
    print("📊 Checking coverage threshold...")
    
    try:
        # Run quick coverage check
        result = subprocess.run(
            [sys.executable, "scripts/quick_rl_coverage.py"],
            capture_output=True,
            text=True,
            timeout=180  # 3 minutes
        )
        
        if result.returncode == 0:
            # Check if coverage report exists and meets threshold
            coverage_file = Path("htmlcov/coverage.json")
            if coverage_file.exists():
                import json
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                
                # Extract overall coverage percentage
                total_coverage = coverage_data.get('totals', {}).get('percent_covered', 0)
                threshold = 85  # Minimum 85% coverage
                
                return {
                    "success": total_coverage >= threshold,
                    "coverage": total_coverage,
                    "threshold": threshold,
                    "stdout": f"Coverage: {total_coverage:.1f}% (threshold: {threshold}%)",
                    "stderr": ""
                }
        
        return {
            "success": False,
            "coverage": 0,
            "threshold": 85,
            "stdout": "",
            "stderr": result.stderr
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "coverage": 0,
            "threshold": 85,
            "stdout": "",
            "stderr": "Coverage check timed out"
        }
    except Exception as e:
        return {
            "success": False,
            "coverage": 0,
            "threshold": 85,
            "stdout": "",
            "stderr": f"Error checking coverage: {str(e)}"
        }

def check_code_quality() -> Dict[str, Any]:
    """Check code quality using basic linting."""
    print("🔍 Checking code quality...")
    
    staged_files = get_staged_files()
    python_files = [f for f in staged_files if f.endswith('.py') and not is_test_file(f)]
    
    if not python_files:
        return {"success": True, "stdout": "No Python files to check", "stderr": ""}
    
    try:
        # Basic syntax check
        for file_path in python_files:
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", file_path],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": f"Syntax error in {file_path}: {result.stderr}"
                }
        
        return {
            "success": True,
            "stdout": f"Code quality check passed for {len(python_files)} files",
            "stderr": ""
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Error checking code quality: {str(e)}"
        }

def check_test_script_consistency() -> Dict[str, Any]:
    """Check if test scripts are consistent across platforms."""
    print("🔍 Checking test script consistency...")
    
    staged_files = get_staged_files()
    script_files = [f for f in staged_files if is_script_file(f)]
    
    if not script_files:
        return {"success": True, "stdout": "No script files to check", "stderr": ""}
    
    issues = []
    
    for file_path in script_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check if test script runs pytest
            if 'test' in file_path and 'pytest' not in content:
                issues.append(f"{file_path}: Test script should run pytest")
            
            # Check if script has proper platform-specific elements
            if file_path.endswith('.ps1'):
                if 'Write-Host' not in content:
                    issues.append(f"{file_path}: PowerShell script should use Write-Host")
                if 'venv\\Scripts\\Activate.ps1' not in content:
                    issues.append(f"{file_path}: PowerShell script should activate virtual environment")
            elif file_path.endswith('.sh'):
                if 'source venv/bin/activate' not in content:
                    issues.append(f"{file_path}: Bash script should activate virtual environment")
                if '#!/bin/bash' not in content:
                    issues.append(f"{file_path}: Bash script should have proper shebang")
        
        except Exception as e:
            issues.append(f"{file_path}: Error reading file - {str(e)}")
    
    if issues:
        return {
            "success": False,
            "stdout": "",
            "stderr": "\n".join(issues)
        }
    
    return {
        "success": True,
        "stdout": f"Script consistency check passed for {len(script_files)} files",
        "stderr": ""
    }

def run_pre_commit_checks(strict: bool = False) -> Dict[str, Any]:
    """Run all pre-commit checks."""
    print("🚀 Running pre-commit checks...")
    print("=" * 50)
    
    checks = {
        "quick_tests": run_quick_tests(),
        "cross_platform": validate_cross_platform_consistency(),
        "coverage": check_coverage_threshold(),
        "code_quality": check_code_quality(),
        "script_consistency": check_test_script_consistency()
    }
    
    # Print results
    print("\n📋 Pre-commit Check Results:")
    print("-" * 30)
    
    all_passed = True
    for check_name, result in checks.items():
        status = "✅ PASSED" if result["success"] else "❌ FAILED"
        print(f"{check_name.replace('_', ' ').title()}: {status}")
        
        if not result["success"]:
            all_passed = False
            if result["stderr"]:
                print(f"  Error: {result['stderr']}")
            if result["stdout"]:
                print(f"  Output: {result['stdout']}")
    
    # Summary
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All pre-commit checks passed!")
    else:
        print("❌ Some pre-commit checks failed!")
        if strict:
            print("🚫 Commit blocked due to strict mode")
        else:
            print("⚠️  Commit allowed but issues should be addressed")
    
    return {
        "all_passed": all_passed,
        "checks": checks
    }

def main():
    """Main function for pre-commit hook."""
    parser = argparse.ArgumentParser(description="Pre-Commit Hook")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict mode - block commit on any failure"
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip test execution (for debugging)"
    )
    parser.add_argument(
        "--skip-coverage",
        action="store_true",
        help="Skip coverage check"
    )
    
    args = parser.parse_args()
    
    # Set PYTHONPATH
    os.environ['PYTHONPATH'] = f"src:{os.environ.get('PYTHONPATH', '')}"
    
    # Run checks
    results = run_pre_commit_checks(args.strict)
    
    # Exit with appropriate code
    if results["all_passed"]:
        print("✅ Pre-commit hook completed successfully")
        sys.exit(0)
    else:
        if args.strict:
            print("❌ Pre-commit hook failed - commit blocked")
            sys.exit(1)
        else:
            print("⚠️  Pre-commit hook completed with warnings")
            sys.exit(0)

if __name__ == "__main__":
    main() 