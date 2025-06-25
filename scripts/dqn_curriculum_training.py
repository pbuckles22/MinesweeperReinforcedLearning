#!/usr/bin/env python3
"""
DQN Curriculum Training Script
Progressive training from easy to hard configurations with DQN agent.
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

class DQNCurriculumTrainer:
    """Curriculum trainer for DQN agent with progressive difficulty."""
    
    def __init__(self, board_size=(4, 4), base_episodes=2000, eval_episodes=100):
        self.board_size = board_size
        self.base_episodes = base_episodes
        self.eval_episodes = eval_episodes
        self.action_size = board_size[0] * board_size[1]
        
        # Curriculum stages: (mine_count, target_win_rate, episodes_multiplier)
        # More realistic targets based on our actual performance
        self.curriculum_stages = [
            (1, 0.45, 1.0),    # Stage 1: 1 mine, 45% target (realistic)
            (2, 0.30, 2.0),    # Stage 2: 2 mines, 30% target (challenging) - 2x episodes
            (3, 0.20, 3.0),    # Stage 3: 3 mines, 20% target (difficult) - 3x episodes
            (4, 0.15, 5.0),    # Stage 4: 4 mines, 15% target (very difficult) - 5x episodes
            (5, 0.10, 7.0),    # Stage 5: 5 mines, 10% target (expert level) - 7x episodes
        ]
        
        # Results tracking
        self.results = {
            'curriculum_progress': [],
            'stage_results': {},
            'final_evaluation': {},
            'training_time': 0,
            'total_episodes': 0
        }
        
        # Create results directory
        self.results_dir = Path("training_stats/dqn_curriculum")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def create_agent(self, stage_name: str) -> DQNAgent:
        """Create a DQN agent with stage-specific hyperparameters."""
        # Extract mine count from stage name (e.g., "stage_1mines" -> 1)
        mine_count = int(stage_name.split('_')[1].replace('mines', ''))
        
        # More conservative learning for harder stages
        if mine_count <= 2:
            learning_rate = 0.0005  # Reduced from 0.001
            epsilon_decay = 0.999   # Slower decay
        elif mine_count <= 3:
            learning_rate = 0.0003
            epsilon_decay = 0.9995
        else:
            learning_rate = 0.0002
            epsilon_decay = 0.9998
        
        agent = DQNAgent(
            board_size=self.board_size,
            action_size=self.action_size,
            learning_rate=learning_rate,
            epsilon=1.0,
            epsilon_decay=epsilon_decay,
            epsilon_min=0.1,  # Higher minimum epsilon for better exploration
            batch_size=32,    # Smaller batch size for more stable learning
            target_update_freq=200,  # More frequent target updates
            device='cpu'
        )
        
        print(f"🤖 Created DQN agent for {stage_name}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Epsilon decay: {epsilon_decay}")
        print(f"   Epsilon min: {agent.epsilon_min}")
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
        
        print(f"\n🎯 {'='*60}")
        print(f"🎯 CURRICULUM STAGE {stage_idx + 1}: {mine_count} MINES")
        print(f"🎯 Target Win Rate: {target_win_rate:.1%}")
        print(f"🎯 Episodes: {episodes}")
        print(f"🎯 Board Size: {self.board_size}")
        print(f"🎯 {'='*60}")
        
        # Create environment and agent if needed
        env = self.create_environment(mine_count)
        if agent is None:
            agent = self.create_agent(stage_name)
        
        # Train agent
        start_time = time.time()
        training_stats = train_dqn_agent(env, agent, episodes, mine_count, eval_freq=100)
        training_time = time.time() - start_time
        
        # Evaluate agent
        print(f"\n🔍 Evaluating stage {stage_idx + 1}...")
        eval_stats = evaluate_dqn_agent(agent, env, n_episodes=self.eval_episodes)
        
        # Stage results
        stage_result = {
            'stage': stage_idx + 1,
            'mine_count': mine_count,
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
        print(f"   Training Time: {training_time:.1f}s")
        print(f"   Final Epsilon: {agent.epsilon:.3f}")
        print(f"   Replay Buffer: {len(agent.replay_buffer)} experiences")
        
        return stage_result, agent
    
    def run_curriculum(self) -> dict:
        """Run the complete curriculum training."""
        print("🚀 Starting DQN Curriculum Training")
        print(f"📋 Board Size: {self.board_size}")
        print(f"📋 Total Stages: {len(self.curriculum_stages)}")
        print(f"📋 Base Episodes: {self.base_episodes}")
        
        start_time = time.time()
        current_agent = None
        
        for stage_idx, (mine_count, target_win_rate, episodes_multiplier) in enumerate(self.curriculum_stages):
            try:
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
                    # Check if performance is close enough to continue
                    # More lenient criteria for progression
                    if mine_count >= 3:
                        # For 3+ mines, continue if above 5% performance
                        continue_threshold = 0.05
                    elif mine_count == 2:
                        # For 2 mines, continue if above 8% performance
                        continue_threshold = 0.08
                    else:
                        # For 1 mine, use 10% threshold
                        continue_threshold = 0.10
                    
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
                checkpoint_path = self.results_dir / f"dqn_stage_{mine_count}mines.pth"
                current_agent.save_model(str(checkpoint_path))
                print(f"💾 Saved checkpoint: {checkpoint_path}")
                
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
        
        # Save results
        self.save_results()
        
        # Print final summary
        self.print_final_summary()
        
        return self.results
    
    def final_evaluation(self, agent: DQNAgent) -> dict:
        """Evaluate the final agent on all curriculum stages."""
        final_eval = {}
        
        if agent is None:
            print("⚠️  No agent available for final evaluation (training failed early)")
            return final_eval
        
        for mine_count, _, _ in self.curriculum_stages:
            env = self.create_environment(mine_count)
            eval_stats = evaluate_dqn_agent(agent, env, n_episodes=50)
            final_eval[f"{mine_count}mines"] = eval_stats
        
        return final_eval
    
    def save_results(self):
        """Save training results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"dqn_curriculum_results_{timestamp}.json"
        
        # Convert numpy types to native Python types for JSON serialization
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
        
        print(f"💾 Results saved to: {results_file}")
    
    def print_final_summary(self):
        """Print final training summary."""
        print(f"\n🎉 CURRICULUM TRAINING COMPLETED!")
        print(f"⏱️  Total Time: {self.results['training_time']:.1f}s")
        print(f"📊 Total Episodes: {self.results['total_episodes']}")
        print(f"📈 Stages Completed: {len(self.results['curriculum_progress'])}")
        
        print(f"\n📋 STAGE SUMMARY:")
        for stage_result in self.results['curriculum_progress']:
            status = "✅" if stage_result['target_met'] else "❌"
            print(f"   {status} Stage {stage_result['stage']}: "
                  f"{stage_result['mine_count']} mines - "
                  f"{stage_result['achieved_win_rate']:.1%} win rate")
        
        if self.results['final_evaluation']:
            print(f"\n🔍 FINAL EVALUATION:")
            for mine_count, eval_stats in self.results['final_evaluation'].items():
                print(f"   {mine_count}: {eval_stats['win_rate']:.1%} win rate")


def main():
    """Main training function."""
    print("🧪 DQN Curriculum Training")
    print("=" * 50)
    
    # Create trainer
    trainer = DQNCurriculumTrainer(
        board_size=(4, 4),
        base_episodes=2000,
        eval_episodes=100
    )
    
    # Run curriculum
    results = trainer.run_curriculum()
    
    print(f"\n🎯 Training completed! Check results in: {trainer.results_dir}")


if __name__ == "__main__":
    main() 