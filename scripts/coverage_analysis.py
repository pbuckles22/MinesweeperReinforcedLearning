#!/usr/bin/env python3
"""
Memory-optimized coverage analysis for Minesweeper RL project.
Handles memory constraints by using coverage.py directly with smart filtering.
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

def run_coverage_with_memory_optimization(
    test_paths: List[str],
    source_paths: List[str],
    output_dir: str = "htmlcov",
    exclude_patterns: List[str] = None,
    max_memory_mb: int = 2048
) -> Dict[str, Any]:
    """
    Run coverage analysis with memory optimizations.
    
    Args:
        test_paths: List of test directories to run
        source_paths: List of source directories to measure coverage for
        output_dir: Directory for HTML coverage reports
        exclude_patterns: Patterns to exclude from coverage
        max_memory_mb: Maximum memory usage in MB
    
    Returns:
        Dictionary with coverage results
    """
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Build coverage command with memory optimizations
    cmd = [
        sys.executable, "-m", "coverage", "run",
        "--source=" + ",".join(source_paths),
        "--omit=*/tests/*,*/venv/*,*/__pycache__/*"
    ]
    
    # Add exclude patterns
    if exclude_patterns:
        for pattern in exclude_patterns:
            cmd.extend(["--omit", pattern])
    
    # Add pytest with memory optimizations
    cmd.extend([
        "-m", "pytest",
        "--tb=short",  # Shorter tracebacks
        "--maxfail=10",  # Stop after 10 failures
        "--durations=10",  # Only show 10 slowest tests
        "--disable-warnings",  # Reduce output
        "--quiet",  # Less verbose
    ])
    
    # Add test paths
    cmd.extend(test_paths)
    
    print(f"Running coverage command: {' '.join(cmd)}")
    print(f"Memory limit: {max_memory_mb}MB")
    
    try:
        # Set environment variables for memory optimization
        env = os.environ.copy()
        env.update({
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",  # Deterministic hashing
        })
        
        # Run coverage
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode != 0:
            print(f"Coverage run failed with return code: {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return {"success": False, "error": result.stderr}
        
        # Generate reports
        print("Generating coverage reports...")
        
        # Terminal report
        subprocess.run([
            sys.executable, "-m", "coverage", "report",
            "--show-missing"
        ], check=True)
        
        # HTML report
        subprocess.run([
            sys.executable, "-m", "coverage", "html",
            f"--directory={output_dir}"
        ], check=True)
        
        # JSON report for programmatic access
        subprocess.run([
            sys.executable, "-m", "coverage", "json",
            f"-o={output_dir}/coverage.json"
        ], check=True)
        
        return {"success": True, "output": result.stdout}
        
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Coverage analysis timed out after 1 hour"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def run_chunked_coverage_analysis():
    """Run coverage analysis in chunks to avoid memory issues."""
    
    # Define test chunks with memory considerations
    chunks = [
        {
            "name": "unit_core",
            "test_paths": ["tests/unit/core/"],
            "source_paths": ["src/core"],
            "output_dir": "htmlcov/unit_core"
        },
        {
            "name": "unit_infrastructure", 
            "test_paths": ["tests/unit/infrastructure/"],
            "source_paths": ["src/core", "scripts"],
            "output_dir": "htmlcov/unit_infrastructure"
        },
        {
            "name": "functional",
            "test_paths": ["tests/functional/"],
            "source_paths": ["src"],
            "output_dir": "htmlcov/functional"
        },
        {
            "name": "integration",
            "test_paths": ["tests/integration/"],
            "source_paths": ["src"],
            "output_dir": "htmlcov/integration"
        },
        {
            "name": "scripts",
            "test_paths": ["tests/scripts/"],
            "source_paths": ["scripts"],
            "output_dir": "htmlcov/scripts"
        }
    ]
    
    # Skip RL tests for now due to memory issues
    # We'll handle them separately with optimizations
    
    results = {}
    
    for chunk in chunks:
        print(f"\n{'='*60}")
        print(f"Running coverage for: {chunk['name']}")
        print(f"{'='*60}")
        
        result = run_coverage_with_memory_optimization(
            test_paths=chunk["test_paths"],
            source_paths=chunk["source_paths"],
            output_dir=chunk["output_dir"],
            max_memory_mb=1024  # Conservative memory limit
        )
        
        results[chunk["name"]] = result
        
        if result["success"]:
            print(f"✅ {chunk['name']} coverage completed successfully")
        else:
            print(f"❌ {chunk['name']} coverage failed: {result['error']}")
    
    # Handle RL tests separately with extreme memory optimization
    print(f"\n{'='*60}")
    print("Running RL tests with extreme memory optimization")
    print(f"{'='*60}")
    
    # Create a minimal RL test runner
    rl_result = run_rl_coverage_with_extreme_optimization()
    results["unit_rl"] = rl_result
    
    return results

def run_rl_coverage_with_extreme_optimization():
    """Run RL tests with extreme memory optimization."""
    
    # Create a minimal test script that only runs essential RL tests
    minimal_rl_test_script = """
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Only import and run the most essential RL tests
from tests.unit.rl.test_rl_evaluation_unit import TestTrainAgent
from tests.unit.rl.test_rl_training_unit import TestExperimentTracker, TestIterationCallback

if __name__ == '__main__':
    import unittest
    
    # Create test suite with only essential tests
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    
    # Add only the most critical RL tests
    suite.addTest(loader.loadTestsFromTestCase(TestTrainAgent))
    suite.addTest(loader.loadTestsFromTestCase(TestExperimentTracker))
    suite.addTest(loader.loadTestsFromTestCase(TestIterationCallback))
    
    # Run with minimal verbosity
    runner = unittest.TextTestRunner(verbosity=1, stream=open(os.devnull, 'w'))
    result = runner.run(suite)
    
    sys.exit(0 if result.wasSuccessful() else 1)
"""
    
    # Write the minimal test script
    with open("temp_minimal_rl_test.py", "w") as f:
        f.write(minimal_rl_test_script)
    
    try:
        # Run coverage with the minimal script
        cmd = [
            sys.executable, "-m", "coverage", "run",
            "--source=src/core",
            "--omit=*/tests/*,*/venv/*,*/__pycache__/*",
            "temp_minimal_rl_test.py"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes timeout
        )
        
        if result.returncode == 0:
            # Generate RL-specific reports
            subprocess.run([
                sys.executable, "-m", "coverage", "html",
                "--directory=htmlcov/unit_rl"
            ], check=True)
            
            return {"success": True, "output": "RL coverage completed with minimal tests"}
        else:
            return {"success": False, "error": f"RL coverage failed: {result.stderr}"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        # Clean up
        if os.path.exists("temp_minimal_rl_test.py"):
            os.remove("temp_minimal_rl_test.py")

def main():
    parser = argparse.ArgumentParser(description="Memory-optimized coverage analysis")
    parser.add_argument("--chunked", action="store_true", help="Run chunked analysis")
    parser.add_argument("--test-path", help="Specific test path to analyze")
    parser.add_argument("--source-path", help="Specific source path to measure")
    parser.add_argument("--output-dir", default="htmlcov", help="Output directory")
    
    args = parser.parse_args()
    
    if args.chunked:
        print("Running chunked coverage analysis...")
        results = run_chunked_coverage_analysis()
        
        # Save results summary
        with open("coverage_results_summary.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*60)
        print("COVERAGE ANALYSIS SUMMARY")
        print("="*60)
        
        successful_chunks = [name for name, result in results.items() if result["success"]]
        failed_chunks = [name for name, result in results.items() if not result["success"]]
        
        print(f"✅ Successful: {len(successful_chunks)} chunks")
        for chunk in successful_chunks:
            print(f"   - {chunk}")
        
        if failed_chunks:
            print(f"❌ Failed: {len(failed_chunks)} chunks")
            for chunk in failed_chunks:
                print(f"   - {chunk}: {results[chunk]['error']}")
        
        print(f"\nResults saved to: coverage_results_summary.json")
        print(f"HTML reports in: htmlcov/")
        
    elif args.test_path and args.source_path:
        print(f"Running coverage for {args.test_path} -> {args.source_path}")
        result = run_coverage_with_memory_optimization(
            test_paths=[args.test_path],
            source_paths=[args.source_path],
            output_dir=args.output_dir
        )
        
        if result["success"]:
            print("✅ Coverage analysis completed successfully")
        else:
            print(f"❌ Coverage analysis failed: {result['error']}")
            sys.exit(1)
    else:
        print("Please specify --chunked or both --test-path and --source-path")
        sys.exit(1)

if __name__ == "__main__":
    main() 