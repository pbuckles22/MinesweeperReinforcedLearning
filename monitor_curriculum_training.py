#!/usr/bin/env python3
"""
Monitor curriculum training progress
"""

import sys
import os
import time
import json
from pathlib import Path

def monitor_training_progress():
    """Monitor the training progress and check for curriculum graduation."""
    print("📊 Monitoring Curriculum Training Progress")
    print("=" * 50)
    
    # Check for training stats file
    stats_file = "training_stats/training_stats.txt"
    
    if not os.path.exists(stats_file):
        print("⏳ Waiting for training to start...")
        print("   Looking for:", stats_file)
        return
    
    print(f"✅ Found training stats file: {stats_file}")
    
    # Monitor the file for updates
    last_size = 0
    last_check = time.time()
    
    while True:
        try:
            # Check if file has been updated
            current_size = os.path.getsize(stats_file)
            
            if current_size > last_size:
                # File has been updated, read the latest content
                with open(stats_file, 'r') as f:
                    lines = f.readlines()
                
                # Find the most recent training info
                latest_info = None
                for line in reversed(lines):
                    if "Stage" in line and ("Results" in line or "Progress" in line):
                        latest_info = line.strip()
                        break
                
                if latest_info:
                    print(f"\n🔄 {latest_info}")
                
                # Look for curriculum progression
                for line in lines[-20:]:  # Check last 20 lines
                    if "Graduated" in line or "Next stage" in line:
                        print(f"\n🎉 CURRICULUM PROGRESSION DETECTED!")
                        print(f"   {line.strip()}")
                        return
                    elif "Target win rate" in line:
                        print(f"   {line.strip()}")
                    elif "Win rate:" in line and "%" in line:
                        print(f"   {line.strip()}")
                
                last_size = current_size
                last_check = time.time()
            
            # Check if training has been inactive for too long
            if time.time() - last_check > 60:  # 1 minute
                print("⏸️  Training appears to be paused or completed")
                break
            
            time.sleep(2)  # Check every 2 seconds
            
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"⚠️  Error monitoring: {e}")
            time.sleep(5)
    
    # Final summary
    print("\n📋 Final Training Summary:")
    try:
        with open(stats_file, 'r') as f:
            content = f.read()
        
        # Look for key metrics
        if "Final evaluation results" in content:
            final_section = content.split("Final evaluation results")[-1]
            print("   Final evaluation results found in training log")
        
        if "Stage" in content and "Results" in content:
            print("   Stage results found in training log")
        
        # Check for graduation
        if "Graduated" in content or "Next stage" in content:
            print("   🎉 CURRICULUM GRADUATION DETECTED!")
        else:
            print("   📚 Agent still in current curriculum stage")
            
    except Exception as e:
        print(f"   ⚠️  Could not read final summary: {e}")

def check_mlflow_experiments():
    """Check MLflow experiments for training progress."""
    print("\n🔍 Checking MLflow Experiments:")
    
    mlruns_dir = "mlruns"
    if not os.path.exists(mlruns_dir):
        print("   No MLflow experiments found")
        return
    
    try:
        # Look for the most recent experiment
        experiment_dirs = [d for d in os.listdir(mlruns_dir) if d.startswith("0")]
        if experiment_dirs:
            latest_experiment = max(experiment_dirs, key=lambda x: os.path.getmtime(os.path.join(mlruns_dir, x)))
            print(f"   Latest experiment: {latest_experiment}")
            
            # Check for runs in the experiment
            runs_dir = os.path.join(mlruns_dir, latest_experiment)
            if os.path.exists(runs_dir):
                run_dirs = [d for d in os.listdir(runs_dir) if d.startswith("0")]
                if run_dirs:
                    latest_run = max(run_dirs, key=lambda x: os.path.getmtime(os.path.join(runs_dir, x)))
                    print(f"   Latest run: {latest_run}")
                    
                    # Check metrics
                    metrics_file = os.path.join(runs_dir, latest_run, "metrics")
                    if os.path.exists(metrics_file):
                        print("   ✅ MLflow metrics available")
                    else:
                        print("   ⏳ MLflow metrics not yet available")
        else:
            print("   No experiments found")
            
    except Exception as e:
        print(f"   ⚠️  Error checking MLflow: {e}")

if __name__ == "__main__":
    print("🚀 Starting Curriculum Training Monitor")
    print("   Press Ctrl+C to stop monitoring")
    print()
    
    # Check MLflow first
    check_mlflow_experiments()
    
    # Monitor training progress
    monitor_training_progress()
    
    print("\n✅ Monitoring completed") 