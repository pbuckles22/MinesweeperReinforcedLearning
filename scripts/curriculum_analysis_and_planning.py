#!/usr/bin/env python3
"""
Curriculum Analysis and Planning Script

Analyzes the results from the adaptive curriculum training and generates
recommendations for the next training run with realistic targets.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Any


class CurriculumAnalyzer:
    """Analyze curriculum training results and generate recommendations."""
    
    def __init__(self, results_file: str):
        self.results_file = results_file
        self.results = self._load_results()
        
    def _load_results(self) -> Dict[str, Any]:
        """Load training results from JSON file."""
        with open(self.results_file, 'r') as f:
            return json.load(f)
    
    def analyze_performance_trends(self):
        """Analyze performance trends across stages."""
        print("🔍 Performance Analysis")
        print("=" * 60)
        
        stages = self.results['curriculum_results']
        
        for stage in stages:
            stage_num = stage['stage_num']
            board_size = stage['board_size']
            mines = stage['mines']
            target = stage['target_win_rate']
            achieved = stage['achieved_win_rate']
            best = stage['best_eval_win_rate']
            episodes = stage['total_episodes']
            eval_history = stage['eval_history']
            
            print(f"\n📊 Stage {stage_num}: {board_size[0]}×{board_size[1]} Board ({mines} mines)")
            print(f"   Target: {target:.1%}, Achieved: {achieved:.1%}, Best: {best:.1%}")
            print(f"   Episodes: {episodes:,}")
            
            # Analyze evaluation history
            eval_array = np.array(eval_history)
            peak_episode = np.argmax(eval_array) * 500  # 500 episodes per evaluation
            peak_performance = np.max(eval_array)
            
            print(f"   Peak Performance: {peak_performance:.1%} at episode {peak_episode:,}")
            print(f"   Performance Trend: {'↗️ Improving' if achieved > eval_array[0] else '↘️ Declining'}")
            
            # Overfitting analysis
            if len(eval_array) > 10:
                early_avg = np.mean(eval_array[:10])
                late_avg = np.mean(eval_array[-10:])
                overfitting = early_avg - late_avg
                print(f"   Overfitting: {overfitting:.3f} ({'Yes' if overfitting > 0.05 else 'No'})")
    
    def calculate_realistic_targets(self):
        """Calculate realistic targets based on achieved performance."""
        print(f"\n🎯 Realistic Target Recommendations")
        print("=" * 60)
        
        stages = self.results['curriculum_results']
        
        # Calculate performance ratios
        performance_ratios = []
        for stage in stages:
            ratio = stage['achieved_win_rate'] / stage['target_win_rate']
            performance_ratios.append(ratio)
        
        avg_ratio = np.mean(performance_ratios)
        print(f"Average target achievement: {avg_ratio:.1%}")
        
        # Recommend new targets
        print(f"\n📋 Recommended Targets for Next Run:")
        
        # Based on achieved performance, suggest realistic targets
        recommended_targets = [
            ((4, 4), 1, 0.85),    # 85% (achieved 78%, target was 90%)
            ((5, 5), 2, 0.55),    # 55% (achieved 37%, target was 70%)
            ((6, 6), 3, 0.35),    # 35% (achieved 19%, target was 50%)
            ((7, 7), 4, 0.20),    # 20% (new intermediate stage)
            ((8, 8), 6, 0.08),    # 8% (achieved 0.8%, target was 20%)
        ]
        
        for board_size, mines, target in recommended_targets:
            print(f"   {board_size[0]}×{board_size[1]} ({mines} mines): {target:.1%}")
        
        return recommended_targets
    
    def analyze_transfer_learning(self):
        """Analyze transfer learning effectiveness."""
        print(f"\n🔄 Transfer Learning Analysis")
        print("=" * 60)
        
        stages = self.results['curriculum_results']
        
        for i, stage in enumerate(stages):
            if i == 0:
                continue  # Skip first stage (no transfer)
            
            prev_stage = stages[i-1]
            current_stage = stage
            
            prev_board = prev_stage['board_size']
            current_board = current_stage['board_size']
            prev_performance = prev_stage['achieved_win_rate']
            current_performance = current_stage['achieved_win_rate']
            
            print(f"\n📊 Transfer: {prev_board[0]}×{prev_board[1]} → {current_board[0]}×{current_board[1]}")
            print(f"   Previous Performance: {prev_performance:.1%}")
            print(f"   Current Performance: {current_performance:.1%}")
            print(f"   Transfer Success: {'✅' if current_stage['transfer_success'] else '❌'}")
            
            # Calculate performance drop
            if prev_performance > 0:
                drop_ratio = current_performance / prev_performance
                print(f"   Performance Drop: {drop_ratio:.1%} of previous stage")
    
    def identify_optimization_opportunities(self):
        """Identify opportunities for optimization."""
        print(f"\n⚡ Optimization Opportunities")
        print("=" * 60)
        
        stages = self.results['curriculum_results']
        
        # Early stopping analysis
        print("🎯 Early Stopping Opportunities:")
        for stage in stages:
            eval_history = stage['eval_history']
            target = stage['target_win_rate']
            
            # Find when target was first achieved
            target_achieved = [i for i, rate in enumerate(eval_history) if rate >= target]
            
            if target_achieved:
                first_achievement = target_achieved[0] * 500  # episodes
                total_episodes = len(eval_history) * 500
                time_saved = (total_episodes - first_achievement) / total_episodes
                
                print(f"   Stage {stage['stage_num']}: Could save {time_saved:.1%} of training time")
            else:
                print(f"   Stage {stage['stage_num']}: Target never achieved")
        
        # Hyperparameter recommendations
        print(f"\n🔧 Hyperparameter Recommendations:")
        print("   • Increase minimum epsilon from 0.05 to 0.1")
        print("   • Add learning rate decay (reduce by 0.5 every 10k episodes)")
        print("   • Increase early stopping threshold to 2 consecutive achievements")
        print("   • Add intermediate stage (7×7) between 6×6 and 8×8")
    
    def generate_next_curriculum_config(self):
        """Generate configuration for the next curriculum run."""
        print(f"\n🚀 Next Curriculum Configuration")
        print("=" * 60)
        
        recommended_targets = self.calculate_realistic_targets()
        
        config = {
            "curriculum_stages": [
                {
                    "name": "Stage 1: 4x4 Board Mastery",
                    "board_size": (4, 4),
                    "mines": 1,
                    "min_episodes": 3000,
                    "max_episodes": 15000,
                    "target_win_rate": 0.85,
                    "eval_freq": 500,
                    "eval_episodes": 50,
                    "eval_runs": 5,
                    "early_stop_consecutive": 2,
                    "description": "Mastery training on small board"
                },
                {
                    "name": "Stage 2: 5x5 Board Foundation",
                    "board_size": (5, 5),
                    "mines": 2,
                    "min_episodes": 5000,
                    "max_episodes": 20000,
                    "target_win_rate": 0.55,
                    "eval_freq": 500,
                    "eval_episodes": 50,
                    "eval_runs": 5,
                    "early_stop_consecutive": 2,
                    "description": "Transfer learning to medium board"
                },
                {
                    "name": "Stage 3: 6x6 Board Challenge",
                    "board_size": (6, 6),
                    "mines": 3,
                    "min_episodes": 8000,
                    "max_episodes": 25000,
                    "target_win_rate": 0.35,
                    "eval_freq": 500,
                    "eval_episodes": 50,
                    "eval_runs": 5,
                    "early_stop_consecutive": 2,
                    "description": "Advanced training on larger board"
                },
                {
                    "name": "Stage 4: 7x7 Board Intermediate",
                    "board_size": (7, 7),
                    "mines": 5,
                    "min_episodes": 10000,
                    "max_episodes": 30000,
                    "target_win_rate": 0.20,
                    "eval_freq": 500,
                    "eval_episodes": 50,
                    "eval_runs": 5,
                    "early_stop_consecutive": 2,
                    "description": "Intermediate challenge"
                },
                {
                    "name": "Stage 5: 8x8 Board Ultimate",
                    "board_size": (8, 8),
                    "mines": 6,
                    "min_episodes": 12000,
                    "max_episodes": 35000,
                    "target_win_rate": 0.08,
                    "eval_freq": 500,
                    "eval_episodes": 50,
                    "eval_runs": 5,
                    "early_stop_consecutive": 2,
                    "description": "Ultimate challenge on large board"
                }
            ],
            "agent_config": {
                "learning_rate": 0.0001,
                "epsilon_decay": 0.9995,
                "epsilon_min": 0.1,  # Increased from 0.05
                "replay_buffer_size": 100000,
                "batch_size": 32,
                "target_update_freq": 1000,
                "use_double_dqn": True,
                "use_dueling": True,
                "use_prioritized_replay": True
            }
        }
        
        print("📋 Recommended Configuration:")
        for stage in config["curriculum_stages"]:
            board_size = stage["board_size"]
            mines = stage["mines"]
            target = stage["target_win_rate"]
            print(f"   {board_size[0]}×{board_size[1]} ({mines} mines): {target:.1%} target")
        
        print(f"\n🔧 Key Changes:")
        print("   • More realistic targets based on achieved performance")
        print("   • Added 7×7 intermediate stage")
        print("   • Increased minimum epsilon to 0.1")
        print("   • Reduced early stopping threshold to 2 consecutive achievements")
        print("   • Adjusted episode ranges for better time management")
        
        return config
    
    def run_comprehensive_analysis(self):
        """Run comprehensive analysis of the training results."""
        print("🎉 Curriculum Training Analysis")
        print("=" * 70)
        print(f"Results File: {self.results_file}")
        print(f"Total Training Time: {self.results['total_time']/3600:.1f} hours")
        print("=" * 70)
        
        # Run all analyses
        self.analyze_performance_trends()
        self.analyze_transfer_learning()
        self.identify_optimization_opportunities()
        next_config = self.generate_next_curriculum_config()
        
        # Save recommendations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recommendations_file = f"curriculum_recommendations_{timestamp}.json"
        
        with open(recommendations_file, 'w') as f:
            json.dump(next_config, f, indent=2)
        
        print(f"\n💾 Recommendations saved to {recommendations_file}")
        print(f"\n🎯 Next Steps:")
        print(f"   1. Review the analysis above")
        print(f"   2. Update curriculum script with new configuration")
        print(f"   3. Start new training run with realistic targets")
        print(f"   4. Monitor for early stopping opportunities")


def main():
    """Main function to run analysis."""
    # Use the most recent results file
    results_file = "adaptive_curriculum_results_20250626_055457.json"
    
    analyzer = CurriculumAnalyzer(results_file)
    analyzer.run_comprehensive_analysis()


if __name__ == "__main__":
    main() 