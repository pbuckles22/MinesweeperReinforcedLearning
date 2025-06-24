#!/usr/bin/env python3
"""
Analyze training progress and compare results across different training durations.
"""

import json
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_recent_results():
    """Load the most recent training results."""
    results_files = glob.glob("experiments/modular_results_*.json")
    if not results_files:
        print("❌ No results files found")
        return None
    
    # Sort by timestamp (newest first)
    results_files.sort(reverse=True)
    latest_file = results_files[0]
    
    with open(latest_file, 'r') as f:
        return json.load(f)


def analyze_training_progress():
    """Analyze training progress across different runs."""
    print("📊 Training Progress Analysis")
    print("=" * 50)
    
    # Load all results
    results_files = glob.glob("experiments/modular_results_*.json")
    results_files.sort()  # Oldest first
    
    if not results_files:
        print("❌ No results files found")
        return
    
    print(f"Found {len(results_files)} training runs")
    print()
    
    # Analyze each result
    progress_data = []
    
    for i, file_path in enumerate(results_files):
        try:
            with open(file_path, 'r') as f:
                result = json.load(f)
            
            timesteps = result.get('total_timesteps', 0)
            win_rate = result.get('final_win_rate', 0)
            mean_reward = result.get('final_mean_reward', 0)
            timestamp = result.get('timestamp', 'unknown')
            
            progress_data.append({
                'timesteps': timesteps,
                'win_rate': win_rate,
                'mean_reward': mean_reward,
                'timestamp': timestamp,
                'file': file_path
            })
            
            print(f"Run {i+1}: {timesteps:,} timesteps → {win_rate:.1%} win rate (reward: {mean_reward:.1f})")
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️  Skipping corrupted file {file_path}: {e}")
            continue
    
    # Sort by timesteps
    progress_data.sort(key=lambda x: x['timesteps'])
    
    # Calculate improvement
    if len(progress_data) > 1:
        first_run = progress_data[0]
        last_run = progress_data[-1]
        
        win_rate_improvement = last_run['win_rate'] - first_run['win_rate']
        reward_improvement = last_run['mean_reward'] - first_run['mean_reward']
        
        print(f"\n📈 Progress Summary:")
        print(f"   First run: {first_run['win_rate']:.1%} win rate")
        print(f"   Latest run: {last_run['win_rate']:.1%} win rate")
        print(f"   Improvement: {win_rate_improvement:+.1%} win rate")
        print(f"   Reward improvement: {reward_improvement:+.1f}")
    
    return progress_data


def compare_to_human_performance():
    """Compare current performance to human benchmarks."""
    print(f"\n👤 Human Performance Comparison")
    print("=" * 50)
    
    latest_result = load_recent_results()
    if not latest_result:
        return
    
    win_rate = latest_result.get('final_win_rate', 0)
    timesteps = latest_result.get('total_timesteps', 0)
    
    # Human benchmarks for 4x4 with 2 mines
    human_benchmark = 0.85  # 85% win rate
    
    print(f"Current Agent Performance:")
    print(f"   Win Rate: {win_rate:.1%}")
    print(f"   Training Time: {timesteps:,} timesteps")
    print(f"   Human Benchmark: {human_benchmark:.0%}")
    print(f"   Performance Gap: {human_benchmark - win_rate:.1%}")
    
    # Calculate percentage of human performance
    performance_percentage = (win_rate / human_benchmark) * 100
    print(f"   Human Performance: {performance_percentage:.1f}%")
    
    # Recommendations
    print(f"\n🎯 Recommendations:")
    if performance_percentage < 30:
        print(f"   ⚠️  Very early learning phase - need much more training")
    elif performance_percentage < 50:
        print(f"   🔄 Good progress - continue with longer training")
    elif performance_percentage < 70:
        print(f"   🚀 Strong performance - consider hyperparameter tuning")
    else:
        print(f"   🎉 Excellent performance - approaching human level!")


def suggest_next_steps():
    """Suggest next steps based on current performance."""
    print(f"\n🚀 Next Steps Recommendations")
    print("=" * 50)
    
    latest_result = load_recent_results()
    if not latest_result:
        return
    
    win_rate = latest_result.get('final_win_rate', 0)
    timesteps = latest_result.get('total_timesteps', 0)
    
    if win_rate < 0.15:  # Less than 15%
        print("📚 Early Learning Phase:")
        print("   • Run 100,000+ timesteps training")
        print("   • Check if reward structure is working")
        print("   • Verify environment is functioning correctly")
        
    elif win_rate < 0.30:  # 15-30%
        print("🔄 Learning Phase:")
        print("   • Continue with 100,000+ timesteps")
        print("   • Try different learning rates (0.0002, 0.00005)")
        print("   • Increase batch size to 64")
        
    elif win_rate < 0.50:  # 30-50%
        print("🚀 Improving Phase:")
        print("   • Run 200,000+ timesteps")
        print("   • Experiment with hyperparameters")
        print("   • Try curriculum learning")
        
    else:  # 50%+
        print("🎯 Optimization Phase:")
        print("   • Fine-tune hyperparameters")
        print("   • Try different network architectures")
        print("   • Consider advanced techniques")


def main():
    """Main analysis function."""
    print("🔬 Minesweeper RL Training Analysis")
    print("=" * 60)
    
    # Analyze progress
    progress_data = analyze_training_progress()
    
    # Compare to human performance
    compare_to_human_performance()
    
    # Suggest next steps
    suggest_next_steps()
    
    print(f"\n✅ Analysis complete!")


if __name__ == "__main__":
    main() 