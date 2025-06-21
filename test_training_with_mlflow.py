#!/usr/bin/env python3
"""
Test Training with MLflow
- Runs a short training session with MLflow logging
- Tests the penalty fix and monitoring capabilities
"""

import os
import subprocess
import time
from src.core.train_agent import main

def run_training_test():
    print("🚀 Starting Test Training with MLflow")
    print("=" * 50)
    
    # Run training with reduced timesteps for testing
    print("📊 Starting training...")
    print("   - Total timesteps: 15,000")
    print("   - Evaluation frequency: 1,500")
    print("   - MLflow logging enabled")
    
    # Override sys.argv to pass arguments to main()
    import sys
    original_argv = sys.argv
    sys.argv = [
        'train_agent.py',
        '--total_timesteps', '15000',
        '--eval_freq', '1500',
        '--n_eval_episodes', '15',
        '--verbose', '1'
    ]
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    finally:
        sys.argv = original_argv
    
    print("\n✅ Training completed!")
    print("📊 To view MLflow UI:")
    print("   mlflow ui")
    print("   Then open http://localhost:5000 in your browser")

if __name__ == "__main__":
    run_training_test() 