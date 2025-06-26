#!/usr/bin/env python3
"""
DQN 8x8 Curriculum Training Script
Comprehensive training for 8x8 Minesweeper with progressive difficulty.
Optimized for overnight training runs.
"""

import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from src.core.minesweeper_env import MinesweeperEnv
from src.core.dqn_agent import DQNAgent, train_dqn_agent, evaluate_dqn_agent

class DQN8x8CurriculumTrainer:
    """Curriculum trainer for 8x8 DQN agent with progressive difficulty."""
    
    def __init__(self, base_episodes=10000, eval_episodes=200):
        self.board_size = (8, 8)
        self.base_episodes = base_episodes
        self.eval_episodes = eval_episodes
        self.action_size = self.board_size[0] * self.board_size[1]  # 64 actions
        
        # 8x8 Curriculum stages: (mine_count, target_win_rate, episodes_multiplier)
        # More realistic targets for 8x8 complexity
        self.curriculum_stages = [
            (5, 0.40, 1.0),     # Stage 1: 5 mines (7.8% density), 40% target
            (8, 0.30, 1.5),     # Stage 2: 8 mines (12.5% density), 30% target
            (12, 0.20, 2.0),    # Stage 3: 12 mines (18.8% density), 20% target
            (16, 0.15, 2.5),    # Stage 4: 16 mines (25% density), 15% target
            (20, 0.10, 3.0),    # Stage 5: 20 mines (31% density), 10% target
        ]
        
        # Results tracking
        self.results = {
            'curriculum_progress': [],
            'stage_results': {},
            'final_evaluation': {},
            'training_time': 0,
            'total_episodes': 0,
            'start_time': datetime.now().isoformat()
        }
        
        # Create results directory
        self.results_dir = Path("training_stats/dqn_8x8_curriculum")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create models directory
        self.models_dir = Path("models/dqn_8x8")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def create_agent(self, stage_name: str) -> DQNAgent:
        """Create a DQN agent optimized for 8x8 boards."""
        # Extract mine count from stage name
        mine_count = int(stage_name.split('_')[1].replace('mines', ''))
        
        # Optimized hyperparameters for 8x8 boards
        if mine_count <= 8:
            learning_rate = 0.0003
            epsilon_decay = 0.9995
            epsilon_min = 0.15
        elif mine_count <= 16:
            learning_rate = 0.0002
            epsilon_decay = 0.9997
            epsilon_min = 0.12
        else:
            learning_rate = 0.0001
            epsilon_decay = 0.9998
            epsilon_min = 0.10
        
        agent = DQNAgent(
            board_size=self.board_size,
            action_size=self.action_size,
            learning_rate=learning_rate,
            epsilon=1.0,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            batch_size=64,
            target_update_freq=500,
            replay_buffer_size=200000,  # Larger buffer for 8x8
            device='cpu'
        )
        
        print(f"🤖 Created DQN agent for {stage_name}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Epsilon decay: {epsilon_decay}")
        print(f"   Epsilon min: {epsilon_min}")
        print(f"   Replay buffer: {agent.replay_buffer.capacity}")
        print(f"   Device: {agent.device}")
        
        return agent
    
    def create_environment(self, mine_count: int) -> MinesweeperEnv:
        """Create environment for specific mine count."""
        env = MinesweeperEnv(
            initial_board_size=self.board_size,
            initial_mines=mine_count
        )
        return env
    
    def train_stage(self, stage_idx: int, mine_count: int, target_win_rate: float, 
                   episodes_multiplier: float, agent: DQNAgent = None) -> dict:
        """Train agent on a specific curriculum stage."""
        stage_name = f"stage_{mine_count}mines"
        episodes = int(self.base_episodes * episodes_multiplier)
        
        print(f"\n🎯 {'='*70}")
        print(f"🎯 CURRICULUM STAGE {stage_idx + 1}: {mine_count} MINES ON 8x8 BOARD")
        print(f"🎯 Target Win Rate: {target_win_rate:.1%}")
        print(f"🎯 Episodes: {episodes:,}")
        print(f"🎯 Mine Density: {mine_count/64:.1%}")
        print(f"🎯 Board Size: {self.board_size}")
        print(f"🎯 {'='*70}")
        
        # Create environment and agent if needed
        env = self.create_environment(mine_count)
        if agent is None:
            agent = self.create_agent(stage_name)
        
        # Train agent
        start_time = time.time()
        training_stats = train_dqn_agent(env, agent, episodes, mine_count, eval_freq=500)
        training_time = time.time() - start_time
        
        # Evaluate agent
        print(f"\n🔍 Evaluating stage {stage_idx + 1}...")
        eval_stats = evaluate_dqn_agent(agent, env, n_episodes=self.eval_episodes)
        
        # Stage results
        stage_result = {
            'stage': stage_idx + 1,
            'mine_count': mine_count,
            'mine_density': mine_count / 64,
            'target_win_rate': target_win_rate,
            'episodes': episodes,
            'training_time': training_time,
            'training_stats': training_stats,
            'evaluation_stats': eval_stats,
            'achieved_win_rate': eval_stats['win_rate'],
            'target_met': eval_stats['win_rate'] >= target_win_rate,
            'agent_epsilon': agent.epsilon,
            'replay_buffer_size': len(agent.replay_buffer)
        }
        
        # Print stage summary
        print(f"\n📊 STAGE {stage_idx + 1} RESULTS:")
        print(f"   Achieved Win Rate: {eval_stats['win_rate']:.1%}")
        print(f"   Target Win Rate: {target_win_rate:.1%}")
        print(f"   Target Met: {'✅ YES' if stage_result['target_met'] else '❌ NO'}")
        print(f"   Training Time: {training_time:.1f}s ({training_time/60:.1f} minutes)")
        print(f"   Final Epsilon: {agent.epsilon:.3f}")
        print(f"   Replay Buffer: {len(agent.replay_buffer)} experiences")
        print(f"   Mean Reward: {eval_stats['mean_reward']:.2f}")
        print(f"   Mean Steps: {eval_stats['mean_length']:.1f}")
        
        return stage_result, agent
    
    def run_curriculum(self) -> dict:
        """Run the complete 8x8 curriculum training."""
        print("🚀 Starting DQN 8x8 Curriculum Training")
        print(f"📋 Board Size: {self.board_size}")
        print(f"📋 Total Stages: {len(self.curriculum_stages)}")
        print(f"📋 Base Episodes: {self.base_episodes:,}")
        print(f"📋 Expected Duration: 6-8 hours")
        print(f"📋 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_time = time.time()
        current_agent = None
        
        for stage_idx, (mine_count, target_win_rate, episodes_multiplier) in enumerate(self.curriculum_stages):
            try:
                stage_start_time = time.time()
                
                # Train on current stage
                stage_result, current_agent = self.train_stage(
                    stage_idx, mine_count, target_win_rate, episodes_multiplier, current_agent
                )
                
                # Store results
                self.results['stage_results'][f"stage_{mine_count}mines"] = stage_result
                self.results['curriculum_progress'].append(stage_result)
                
                # Check if we should continue
                target_met = stage_result['target_met']
                achieved_rate = stage_result['achieved_win_rate']
                target_rate = stage_result['target_win_rate']
                
                if not target_met:
                    # More lenient criteria for 8x8
                    if mine_count <= 8:
                        continue_threshold = 0.20  # 20% for easier stages
                    elif mine_count <= 16:
                        continue_threshold = 0.10  # 10% for medium stages
                    else:
                        continue_threshold = 0.05  # 5% for hard stages
                    
                    if achieved_rate >= continue_threshold:
                        print(f"\n⚠️  Stage {stage_idx + 1} target not met, but performance is sufficient to continue.")
                        print(f"   Achieved: {achieved_rate:.1%}, Target: {target_rate:.1%}")
                        print(f"   Threshold: {continue_threshold:.1%}")
                        print(f"   Continuing to next stage...")
                    else:
                        print(f"\n⚠️  Stage {stage_idx + 1} target not met. Stopping curriculum.")
                        print(f"   Achieved: {achieved_rate:.1%}, Target: {target_rate:.1%}")
                        print(f"   Threshold: {continue_threshold:.1%}")
                        break
                
                # Save agent checkpoint
                checkpoint_path = self.models_dir / f"dqn_8x8_stage_{mine_count}mines.pth"
                current_agent.save_model(str(checkpoint_path))
                print(f"💾 Saved checkpoint: {checkpoint_path}")
                
                # Save intermediate results
                self.save_intermediate_results(stage_idx + 1)
                
                stage_time = time.time() - stage_start_time
                print(f"⏱️  Stage {stage_idx + 1} completed in {stage_time/60:.1f} minutes")
                
            except Exception as e:
                print(f"❌ Error in stage {stage_idx + 1}: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # Final evaluation on all stages
        print(f"\n🔍 Final evaluation on all stages...")
        final_eval = self.final_evaluation(current_agent)
        self.results['final_evaluation'] = final_eval
        
        # Training summary
        total_time = time.time() - start_time
        self.results['training_time'] = total_time
        self.results['total_episodes'] = sum(
            stage['episodes'] for stage in self.results['curriculum_progress']
        )
        self.results['end_time'] = datetime.now().isoformat()
        
        # Save final results
        self.save_results()
        
        # Print final summary
        self.print_final_summary()
        
        return self.results
    
    def save_intermediate_results(self, stage_num: int):
        """Save intermediate results after each stage."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"dqn_8x8_intermediate_stage{stage_num}_{timestamp}.json"
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        serializable_results = convert_numpy(self.results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"💾 Intermediate results saved: {results_file}")
    
    def final_evaluation(self, agent: DQNAgent) -> dict:
        """Evaluate the final agent on all curriculum stages."""
        final_eval = {}
        
        if agent is None:
            print("⚠️  No agent available for final evaluation (training failed early)")
            return final_eval
        
        for mine_count, _, _ in self.curriculum_stages:
            env = self.create_environment(mine_count)
            eval_stats = evaluate_dqn_agent(agent, env, n_episodes=100)
            final_eval[f"{mine_count}mines"] = eval_stats
        
        return final_eval
    
    def save_results(self):
        """Save training results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"dqn_8x8_curriculum_results_{timestamp}.json"
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        serializable_results = convert_numpy(self.results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"💾 Final results saved to: {results_file}")
    
    def print_final_summary(self):
        """Print final training summary."""
        print(f"\n🎉 8x8 CURRICULUM TRAINING COMPLETED!")
        print(f"⏱️  Total Time: {self.results['training_time']/3600:.1f} hours")
        print(f"📊 Total Episodes: {self.results['total_episodes']:,}")
        print(f"📈 Stages Completed: {len(self.results['curriculum_progress'])}")
        
        print(f"\n📋 STAGE SUMMARY:")
        for stage_result in self.results['curriculum_progress']:
            status = "✅" if stage_result['target_met'] else "❌"
            print(f"   {status} Stage {stage_result['stage']}: "
                  f"{stage_result['mine_count']} mines ({stage_result['mine_density']:.1%} density) - "
                  f"{stage_result['achieved_win_rate']:.1%} win rate")
        
        if self.results['final_evaluation']:
            print(f"\n🔍 FINAL EVALUATION:")
            for mine_count, eval_stats in self.results['final_evaluation'].items():
                print(f"   {mine_count}: {eval_stats['win_rate']:.1%} win rate")


def main():
    """Main training function."""
    print("🧪 DQN 8x8 Curriculum Training")
    print("=" * 60)
    print("🚀 Overnight Training Run")
    print("📋 8x8 board with progressive mine density")
    print("⏱️  Expected duration: 6-8 hours")
    print("=" * 60)
    
    # Create trainer
    trainer = DQN8x8CurriculumTrainer(
        base_episodes=10000,
        eval_episodes=200
    )
    
    # Run curriculum
    results = trainer.run_curriculum()
    
    print(f"\n🎯 8x8 training completed! Check results in: {trainer.results_dir}")


if __name__ == "__main__":
    main() 